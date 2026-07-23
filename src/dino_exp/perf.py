"""性能测试：多版本 × 多并发的推理基准（吞吐量/延迟/显存），用于模型对比。

并发模型：ThreadPoolExecutor 多 worker 并行调用 score_one（CUDA/torch 计算时
释放 GIL，GPU 上多线程有真实吞吐收益）。切块模型内部已按 tile 批量前向。
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import test_images_with_labels
from dino_exp.infer import load_model_for_version, score_one
from dino_exp.models.registry import Registry


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[idx]


def _sample_images(category: str, cfg: Config, samples: int) -> list[str]:
    """测试集均匀抽样（OK/NG 混合，保持真实分布）。"""
    rows = test_images_with_labels(category, cfg)
    paths = [str(p) for p, _, _ in rows]
    if len(paths) <= samples:
        return paths
    step = len(paths) / samples
    return [paths[int(i * step)] for i in range(samples)]


def run_perf_one(model, image_paths: list[str], concurrency: int, cfg: Config) -> dict:
    """单版本单并发档：吞吐量、延迟分布、GPU 显存增量。"""
    for p in image_paths[:2]:  # 预热
        score_one(model, p, cfg)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    latencies: list[float] = []
    t0 = time.perf_counter()

    def work(p: str) -> float:
        _, _, ms = score_one(model, p, cfg)
        return ms

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = list(pool.map(work, image_paths))
    wall = time.perf_counter() - t0
    lat_sorted = sorted(latencies)
    result = {
        "concurrency": concurrency,
        "images": len(image_paths),
        "wall_s": round(wall, 3),
        "throughput_ips": round(len(image_paths) / wall, 2) if wall > 0 else 0,
        "latency_avg_ms": round(statistics.fmean(latencies), 1) if latencies else 0,
        "latency_p50_ms": round(_percentile(lat_sorted, 0.5), 1),
        "latency_p95_ms": round(_percentile(lat_sorted, 0.95), 1),
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return result


def run_perf(category: str, versions: list[str] | None, concurrency: list[int],
             samples: int, cfg: Config) -> dict:
    """多版本 × 多并发性能对比。versions=None 时用该类别全部版本。"""
    reg = Registry(cfg.models_root)
    versions = versions or reg.list(category)
    if not versions:
        from dino_exp.errors import DinoError

        raise DinoError(f"类别 '{category}' 没有模型版本，请先训练。")
    image_paths = _sample_images(category, cfg, samples)
    report = {"category": category, "samples": len(image_paths),
              "device": "cuda" if torch.cuda.is_available() else "cpu",
              "versions": {}}
    for ver in versions:
        model, threshold, _ = load_model_for_version(category, ver, cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        grid = getattr(model, "train_tile_grid", (1, 1))
        size = getattr(model, "train_image_size", cfg.image_size)
        scheme = getattr(model, "train_tile_scheme", "grid")
        tsize = getattr(model, "train_tile_size", None)
        tile_desc = f"T{tsize}" if scheme == "size" and tsize else f"{grid[0]}x{grid[1]}"
        entry = {"image_size": size, "tile_grid": list(grid), "tile_scheme": scheme,
                 "tile_desc": tile_desc, "runs": []}
        for c in concurrency:
            entry["runs"].append(run_perf_one(model, image_paths, c, cfg))
        report["versions"][ver] = entry
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def format_table(report: dict) -> str:
    """把 run_perf 报告格式化为对比表格文本。"""
    lines = [f"类别: {report['category']}  样本: {report['samples']} 张  设备: {report['device']}", ""]
    header = f"{'版本':<8}{'输入':<7}{'切块':<9}{'并发':<5}{'吞吐(张/s)':<11}{'均':<8}{'p50':<8}{'p95':<8}{'显存(MB)':<10}"
    lines.append(header)
    lines.append("-" * len(header))
    for ver, entry in report["versions"].items():
        tile = entry.get("tile_desc", f"{entry['tile_grid'][0]}x{entry['tile_grid'][1]}")
        for run in entry["runs"]:
            lines.append(
                f"{ver:<8}{entry['image_size']:<7}{tile:<9}{run['concurrency']:<5}"
                f"{run['throughput_ips']:<11}{run['latency_avg_ms']:<8}{run['latency_p50_ms']:<8}"
                f"{run['latency_p95_ms']:<8}{run.get('gpu_mem_peak_mb', '-')}"
            )
    return "\n".join(lines)


def save_perf_report(report: dict, out_dir: str | Path = "results/perf") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"perf_{report['category']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

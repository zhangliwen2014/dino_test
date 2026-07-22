from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from dino_exp.config import Config, resolve_device
from dino_exp.errors import DinoError
from dino_exp.models.dual_bank import DualBankPatchcore, load_banks
from dino_exp.models.registry import Registry


def _preprocess_pil(im: Image.Image, image_size: int) -> torch.Tensor:
    # ToDtype 先于 Resize：与 anomalib PreProcessor 一致在 float 空间 resize，
    # 避免 uint8 空间 resize 的 ≤0.5/255 舍入差（口径 bitwise 对齐）
    tf = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((image_size, image_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(im).unsqueeze(0)


def preprocess_image(path: str | Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as im:
        return _preprocess_pil(im.convert("RGB"), image_size)


def decide_label(score: float, threshold: float) -> str:
    return "NG" if score >= threshold else "OK"


def load_threshold(version_dir: str | Path) -> float:
    p = Path(version_dir) / "metrics.json"
    if not p.exists():
        raise DinoError(f"{p} 不存在，版本可能损坏。请切换/回滚到其他版本。")
    metrics = json.loads(p.read_text(encoding="utf-8"))
    if "threshold" not in metrics:
        raise DinoError(f"{p} 缺少 threshold 字段。请重新训练生成版本。")
    return float(metrics["threshold"])


def heatmap_to_bgr(anomaly_map: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
    amap = anomaly_map.squeeze().cpu().numpy()
    amap = cv2.resize(amap, out_size)
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    return cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _heatmap_name(path: str | Path, version: str) -> str:
    """热力图文件名含父目录名，避免同名不同目录图片的输出互相覆盖。"""
    p = Path(path)
    return f"{p.parent.name}_{p.stem}_{version}_heatmap.png"


def load_model_for_version(category: str, version: str | None, cfg: Config) -> tuple[DualBankPatchcore, float, str]:
    """加载指定（或当前）版本模型 + 阈值。返回 (model, threshold, version)。"""
    reg = Registry(cfg.models_root)
    version = version or reg.current(category)
    if version is None:
        raise DinoError(f"类别 '{category}' 没有任何模型版本。请先 `dino train --category {category}`。")
    vdir = reg.version_dir(category, version)
    from dino_exp.train import build_model

    model = build_model(cfg)
    load_banks(model.model, vdir)
    threshold = load_threshold(vdir)
    model.apply_threshold(threshold)
    model.eval()
    return model, threshold, version


def _to_device(model: DualBankPatchcore, cfg: Config) -> DualBankPatchcore:
    return model.to(resolve_device(cfg))


def _imwrite_unicode(path: Path, img) -> None:
    """cv2.imwrite 在 Windows 不支持非 ASCII 路径（文件名会变乱码）；改用 imencode + 字节写。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise DinoError(f"热力图编码失败: {path}")
    Path(path).write_bytes(buf.tobytes())


def annotate_defects(image_path: str | Path, anomaly_map: torch.Tensor, threshold: float,
                     min_area_ratio: float = 0.001) -> tuple["np.ndarray", list[tuple[int, int, int, int]]]:
    """在原图上标记缺陷区域：取 anomaly map 的「热点核心」（max(校准阈值, 99 百分位)）
    以上的连通域画红框——即使整图偏异（裁切/光照差异）也能聚焦最可疑区域而非整图框死。

    返回 (标注后 BGR 图, [(x, y, w, h), ...])。小于原图 0.1% 面积的噪点框忽略。
    """
    amap = anomaly_map.squeeze().cpu().numpy()
    with Image.open(image_path) as im:
        base = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    amap = cv2.resize(amap, (base.shape[1], base.shape[0]))
    pix_th = max(float(threshold), float(np.percentile(amap, 99.0)))
    mask = (amap >= pix_th).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 去散点
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = base.shape[0] * base.shape[1] * min_area_ratio
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    for x, y, w, h in boxes:
        cv2.rectangle(base, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(base, "NG", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return base, boxes


def _infer_loaded(
    model: DualBankPatchcore,
    threshold: float,
    version: str,
    path: str | Path,
    cfg: Config,
    heatmap_dir: str | Path = "outputs/heatmaps",
) -> dict:
    """对已加载模型的纯推理（不含模型加载），供单图/批量共用。"""
    device = next(model.parameters()).device
    with Image.open(path) as im:  # 一次打开：同时取原图尺寸与预处理输入
        im = im.convert("RGB")
        out_size = im.size  # (W, H)
        tensor = _preprocess_pil(im, cfg.image_size).to(device)
    with torch.no_grad():
        out = model.model(tensor)
    score = float(out.pred_score.item())
    hdir = Path(heatmap_dir)
    hdir.mkdir(parents=True, exist_ok=True)
    heatmap_path = hdir / _heatmap_name(path, version)
    _imwrite_unicode(heatmap_path, heatmap_to_bgr(out.anomaly_map.cpu(), out_size))
    annotated, boxes = annotate_defects(path, out.anomaly_map.cpu(), threshold)
    annotated_path = hdir / _heatmap_name(path, version).replace("_heatmap.png", "_annotated.png")
    _imwrite_unicode(annotated_path, annotated)
    return {
        "label": decide_label(score, threshold),
        "score": score,
        "threshold": threshold,
        "heatmap_path": str(heatmap_path),
        "annotated_path": str(annotated_path),
        "defect_boxes": boxes,
    }


def infer_image(
    path: str | Path,
    version: str | None = None,
    *,
    category: str,
    cfg: Config,
    heatmap_dir: str | Path = "outputs/heatmaps",
) -> dict:
    """单图推理 → {label, score, threshold, heatmap_path}（设计文档 §3.5）。"""
    model, threshold, version = load_model_for_version(category, version, cfg)
    return _infer_loaded(_to_device(model, cfg), threshold, version, path, cfg, heatmap_dir)


def infer_batch(paths: list[str | Path], version: str | None = None, *, category: str, cfg: Config) -> list[dict]:
    model, threshold, version = load_model_for_version(category, version, cfg)  # 批量只加载一次
    model = _to_device(model, cfg)
    return [_infer_loaded(model, threshold, version, p, cfg) for p in paths]


def export_openvino(category: str, version: str | None, cfg: Config) -> Path:
    """按版本快照导出 OpenVINO（设计文档 §3.5）。

    导出物写入 models/<类别>/<版本>/export/；导出时双库状态被烘焙进图，
    再训练后必须对新版本重新调用本函数。失败时抛 DinoError 并提示回退 PyTorch。
    """
    from anomalib.engine import Engine

    model, threshold, version = load_model_for_version(category, version, cfg)
    export_root = Registry(cfg.models_root).version_dir(category, version) / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    try:
        engine = Engine()
        out = engine.export(
            model=model,
            export_type="openvino",
            export_root=export_root,
            input_size=(cfg.image_size, cfg.image_size),
        )
    except Exception as exc:
        raise DinoError(
            f"OpenVINO 导出失败: {exc}。已回退：请继续使用 PyTorch 后端推理"
            "（`pip install \"anomalib[openvino]==2.5.1\"` 后可重试导出）。"
        ) from exc
    # 阈值写入导出 metadata，与 PyTorch 判定口径一致
    (export_root / "threshold.json").write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    return Path(out) if out else export_root

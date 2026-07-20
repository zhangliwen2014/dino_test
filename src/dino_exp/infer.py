from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from dino_exp.config import Config
from dino_exp.errors import DinoError
from dino_exp.models.dual_bank import DualBankPatchcore, load_banks
from dino_exp.models.registry import Registry


def preprocess_image(path: str | Path, image_size: int) -> torch.Tensor:
    tf = T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tensor = preprocess_image(path, cfg.image_size).to(device)
    with torch.no_grad():
        out = model.model(tensor)
    score = float(out.pred_score.item())
    hdir = Path(heatmap_dir)
    hdir.mkdir(parents=True, exist_ok=True)
    heatmap_path = hdir / f"{Path(path).stem}_{version}_heatmap.png"
    with Image.open(path) as im:
        out_size = im.size  # (W, H)
    cv2.imwrite(str(heatmap_path), heatmap_to_bgr(out.anomaly_map.cpu(), out_size))
    return {
        "label": decide_label(score, threshold),
        "score": score,
        "threshold": threshold,
        "heatmap_path": str(heatmap_path),
    }


def infer_batch(paths: list[str | Path], version: str | None = None, *, category: str, cfg: Config) -> list[dict]:
    return [infer_image(p, version, category=category, cfg=cfg) for p in paths]


def export_openvino(category: str, version: str | None, cfg: Config) -> Path:
    """按版本快照导出 OpenVINO（设计文档 §3.5）。

    导出物写入 models/<类别>/<版本>/export/；导出时双库状态被烘焙进图，
    再训练后必须对新版本重新调用本函数。失败时抛 DinoError 并提示回退 PyTorch。
    """
    try:
        from anomalib.engine import Engine
    except ImportError as exc:
        raise DinoError(f"anomalib 导入失败: {exc}。请检查环境后重试。") from exc
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

"""全量/选图验证（FR-3.1/FR-3.2/FR-3.4）：聚合指标、逐图结果、误判过滤、无 NG 降级。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import test_images_with_labels
from dino_exp.infer import decide_label, infer_batch, load_model_for_version, preprocess_image
from dino_exp.models.registry import Registry


def aggregate_metrics(rows: list[dict], threshold: float) -> dict:
    """从逐图 {label_gt, score} 计算图片级 AUROC/AUPR/F1；无 NG 样本时降级。"""
    from torchmetrics import AUROC, AveragePrecision, F1Score

    labels = torch.tensor([r["label_gt"] for r in rows])
    scores = torch.tensor([r["score"] for r in rows])
    if labels.sum().item() == 0:
        return {"degraded": True, "note": "无 NG 测试图，指标降级：仅输出逐图分数"}
    preds = (scores >= threshold).long()
    return {
        "image_AUROC": float(AUROC(task="binary")(scores, labels)),
        "image_AUPR": float(AveragePrecision(task="binary")(scores, labels)),
        "image_F1": float(F1Score(task="binary")(preds, labels)),
    }


def filter_errors(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["label_pred"] != ("NG" if r["label_gt"] == 1 else "OK")]


def save_validation_report(version_dir: str | Path, metrics: dict, rows: list[dict]) -> Path:
    p = Path(version_dir) / "validation.json"
    p.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    return p


def score_test_set(category: str, version: str | None, cfg: Config) -> tuple[list[dict], float, str]:
    """对 test 集逐图推理（raw score），返回 (rows, threshold, version)。"""
    model, threshold, version = load_model_for_version(category, version, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    rows = []
    for path, label_gt, defect_type in test_images_with_labels(category, cfg):
        tensor = preprocess_image(path, cfg.image_size).to(device)
        with torch.no_grad():
            out = model.model(tensor)
        score = float(out.pred_score.item())
        rows.append({
            "path": str(path),
            "label_gt": label_gt,
            "defect_type": defect_type,
            "score": score,
            "label_pred": decide_label(score, threshold),
        })
    return rows, threshold, version


def validate_full(category: str, version: str | None, cfg: Config) -> dict:
    """全量验证（FR-3.1/FR-3.4）：聚合指标 + 逐图结果写入版本目录 validation.json。"""
    rows, threshold, version = score_test_set(category, version, cfg)
    metrics = aggregate_metrics(rows, threshold)
    vdir = Registry(cfg.models_root).version_dir(category, version)
    save_validation_report(vdir, metrics, rows)
    return {"version": version, "metrics": metrics, "rows": rows}


def validate_images(category: str, version: str | None, paths: list[str], cfg: Config) -> list[dict]:
    """选图验证（FR-3.2）：逐图分数/判定/热力图，不出聚合指标。"""
    return infer_batch(list(paths), version, category=category, cfg=cfg)

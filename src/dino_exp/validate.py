"""全量/选图验证（FR-3.1/FR-3.2/FR-3.4）：聚合指标、逐图结果、误判过滤、无 NG 降级。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import dataset_info, mask_path_for, test_images_with_labels
from dino_exp.infer import decide_label, infer_batch, load_model_for_version, preprocess_image
from dino_exp.models.registry import Registry


def aggregate_metrics(rows: list[dict], threshold: float, pixel_pairs: list[tuple] | None = None) -> dict:
    """从逐图 {label_gt, score} 计算图片级 AUROC/AUPR/F1Score；无 NG 样本时降级。

    键名对齐 anomalib evaluator 输出（image_F1Score 等）。pixel_pairs 为
    (anomaly_map (H,W) float, gt_mask (H,W) 0/1 uint8) 列表：非空时追加
    pixel_AUROC/pixel_AUPRO（FR-3.1），为空则不出现 pixel_ 键。
    """
    from torchmetrics import AUROC, AveragePrecision, F1Score

    labels = torch.tensor([r["label_gt"] for r in rows])
    scores = torch.tensor([r["score"] for r in rows])
    if labels.sum().item() == 0:
        return {"degraded": True, "note": "无 NG 测试图，指标降级：仅输出逐图分数"}
    preds = (scores >= threshold).long()
    metrics = {
        "image_AUROC": float(AUROC(task="binary")(scores, labels)),
        "image_AUPR": float(AveragePrecision(task="binary")(scores, labels)),
        "image_F1Score": float(F1Score(task="binary")(preds, labels)),
    }
    if pixel_pairs:
        # 与 train.py build_model 同一挂法：anomalib 指标 v2.5.1 需显式 fields，
        # update 接受任意带同名属性的对象（base.py update 只走 getattr）。
        from types import SimpleNamespace

        from anomalib.metrics import AUPRO, AUROC as AnomalibAUROC

        batch = SimpleNamespace(
            anomaly_map=torch.stack([a for a, _ in pixel_pairs]),
            gt_mask=torch.stack([g for _, g in pixel_pairs]),
        )
        pixel_auroc = AnomalibAUROC(fields=["anomaly_map", "gt_mask"], strict=False)
        pixel_auroc.update(batch)
        pixel_aupro = AUPRO(fields=["anomaly_map", "gt_mask"], strict=False)
        pixel_aupro.update(batch)
        metrics["pixel_AUROC"] = float(pixel_auroc.compute())
        metrics["pixel_AUPRO"] = float(pixel_aupro.compute())
    return metrics


def filter_errors(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["label_pred"] != ("NG" if r["label_gt"] == 1 else "OK")]


def save_validation_report(version_dir: str | Path, metrics: dict, rows: list[dict]) -> Path:
    p = Path(version_dir) / "validation.json"
    p.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    return p


def score_test_set(category: str, version: str | None, cfg: Config) -> tuple[list[dict], float, str, list[tuple]]:
    """对 test 集逐图推理（raw score），返回 (rows, threshold, version, pixel_pairs)。

    pixel_pairs：有 GT mask 的 NG 图收集 (anomaly_map (H,W) float cpu,
    gt_mask (H,W) 0/1 uint8)——gt_mask 按 anomalib 约定读灰度后二值化，
    并 INTER_NEAREST resize 到 anomaly_map 尺寸；无 mask 的数据集为空列表。
    """
    import cv2

    model, threshold, version = load_model_for_version(category, version, cfg)
    info = dataset_info(category, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    rows = []
    pixel_pairs = []
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
        mask_path = mask_path_for(path, defect_type, info)
        if mask_path is not None:
            amap = out.anomaly_map.squeeze().float().cpu()
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy((mask > 127).astype("uint8"))
            pixel_pairs.append((amap, gt_mask))
    return rows, threshold, version, pixel_pairs


def validate_full(category: str, version: str | None, cfg: Config) -> dict:
    """全量验证（FR-3.1/FR-3.4）：聚合指标（有 mask 时含像素级）+ 逐图结果写入版本目录 validation.json。"""
    rows, threshold, version, pixel_pairs = score_test_set(category, version, cfg)
    metrics = aggregate_metrics(rows, threshold, pixel_pairs=pixel_pairs)
    vdir = Registry(cfg.models_root).version_dir(category, version)
    save_validation_report(vdir, metrics, rows)
    return {"version": version, "metrics": metrics, "rows": rows}


def validate_images(category: str, version: str | None, paths: list[str], cfg: Config) -> list[dict]:
    """选图验证（FR-3.2）：逐图分数/判定/热力图，不出聚合指标。"""
    return infer_batch(list(paths), version, category=category, cfg=cfg)

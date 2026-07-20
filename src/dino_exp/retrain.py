from __future__ import annotations

import json

import torch

from dino_exp.config import Config
from dino_exp.datasets import ok_calibration_images
from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.feedback.staging import preview as staging_preview
from dino_exp.infer import load_model_for_version, load_threshold, preprocess_image
from dino_exp.models.dual_bank import extract_embeddings, topk_defect_features
from dino_exp.models.registry import Registry
from dino_exp.train import finalize_version, score_images
from dino_exp.validate import validate_full


def partition_feedback(effective_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    oks = [r for r in effective_rows if r["human_label"] == "ok"]
    ngs = [r for r in effective_rows if r["human_label"] == "ng"]
    return oks, ngs


def auroc_drop_warning(parent: float | None, current: float | None) -> str | None:
    """新版本 AUROC 较父版本下降 > 2 个点时告警（FR-6.5）。"""
    if parent is None or current is None:
        return None
    if parent - current > 0.02:
        return (
            f"新版本 AUROC {current:.4f} 较父版本 {parent:.4f} 下降超过 2 个点。"
            "建议检查反馈质量，必要时 `dino rollback` 回滚到父版本。"
        )
    return None


def preview_retrain(category: str, cfg: Config) -> dict:
    reg = Registry(cfg.models_root)
    current = reg.current(category)
    if current is None:
        raise DinoError(f"类别 '{category}' 无模型版本。请先 `dino train --category {category}`。")
    threshold = load_threshold(reg.version_dir(category, current))
    store = FeedbackStore(cfg.feedback_root, category)
    return {
        "current_version": current,
        **staging_preview(store.staged(), threshold, cfg.suspicious_score_factor),
    }


def retrain(category: str, cfg: Config) -> dict:
    """应用暂存反馈 → 新版本（设计文档 §4 工作流 4）。"""
    pv = preview_retrain(category, cfg)
    if pv["ok"] + pv["ng"] == 0:
        raise DinoError("暂存区为空，拒绝再训练。请先 `dino feedback ...` 添加反馈。")
    parent = pv["current_version"]
    store = FeedbackStore(cfg.feedback_root, category)
    effective_rows = store.apply()  # 消费暂存区（同图最新为准）并归档
    oks, ngs = partition_feedback(effective_rows)

    model, old_threshold, _ = load_model_for_version(category, parent, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    for r in oks:  # OK 反馈 → 钉住并入正常库（不参与 coreset 淘汰/不计入上限）
        feats = extract_embeddings(model.model, preprocess_image(store.images_dir / r["stored_image"], cfg.image_size).to(device))
        model.model.add_normal_features(feats, pinned=True)
    for r in ngs:  # NG 反馈 → top-k 高分 patch 入缺陷库
        feats = topk_defect_features(
            model.model,
            preprocess_image(store.images_dir / r["stored_image"], cfg.image_size).to(device),
            k=cfg.defect_topk,
        )
        model.model.add_defect_features(feats)
    model.model.resample_normal_bank()

    # 阈值重校准并注入（FR-6.5）
    ok_scores = score_images(model, ok_calibration_images(category, cfg), cfg)
    parent_metrics = json.loads(
        (Registry(cfg.models_root).version_dir(category, parent) / "metrics.json").read_text()
    )
    version = finalize_version(
        category, cfg, model,
        ok_scores=ok_scores,
        metrics={},  # 先落盘，验证后补写
        parent=parent,
        feedback_applied=len(effective_rows),
    )
    # 自动验证并与父版本对比（FR-3.4/FR-6.5）
    report = validate_full(category, version, cfg)
    metrics = {**report["metrics"], "threshold": model_threshold(cfg, category, version)}
    vdir = Registry(cfg.models_root).version_dir(category, version)
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    warning = auroc_drop_warning(parent_metrics.get("image_AUROC"), metrics.get("image_AUROC"))
    return {"version": version, "metrics": metrics, "warning": warning, "preview": pv}


def model_threshold(cfg: Config, category: str, version: str) -> float:
    from dino_exp.infer import load_threshold as _lt

    return _lt(Registry(cfg.models_root).version_dir(category, version))

from __future__ import annotations

import json

import torch

from dino_exp.config import Config
from dino_exp.datasets import ok_calibration_images
from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.feedback.staging import effective, preview as staging_preview
from dino_exp.infer import load_model_for_version, load_threshold, preprocess_image
from dino_exp.models.dual_bank import extract_embeddings
from dino_exp.models.registry import Registry
from dino_exp.train import finalize_version
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
    # 生效集合前置计算（纯函数，与 apply() 内部等价）；
    # 先构建版本，全部成功后才消费暂存区——中途失败时反馈仍可重试。
    effective_rows = effective(store.staged())
    oks, ngs = partition_feedback(effective_rows)

    model, old_threshold, _ = load_model_for_version(category, parent, cfg)
    from dino_exp.config import resolve_device

    device = resolve_device(cfg)
    model = model.to(device).eval()

    def _feedback_feats(img_path, k: int | None = None):
        """按模型版本配置（是否切块）提取反馈特征：OK 返回全部 patch 特征；
        NG 返回 top-k 高分 patch 特征。与建库尺度一致。"""
        from PIL import Image as _Img

        from dino_exp.tiles import clamp_grid, should_tile, split_image

        grid = getattr(model, "train_tile_grid", (1, 1))
        overlap = getattr(model, "train_tile_overlap", 0.1)
        input_size = getattr(model, "train_image_size", cfg.image_size)
        with _Img.open(img_path) as im:
            im = im.convert("RGB")
            grid = clamp_grid(grid, im.size[0], im.size[1], input_size)
            if should_tile(im.size[0], im.size[1], input_size, grid):
                parts = [t for t, _ in split_image(im, grid, overlap)]
            else:
                parts = [im]
        embs, scores = [], []
        for part in parts:
            from dino_exp.infer import _preprocess_pil

            tensor = _preprocess_pil(part, input_size).to(device)  # part 是 PIL 图片对象
            emb = extract_embeddings(model.model, tensor)
            if k is not None:
                ps, _ = model.model.nearest_neighbors(emb, n_neighbors=1)
                embs.append(emb)
                scores.append(ps)
            else:
                embs.append(emb)
        feats = torch.cat(embs)
        if k is None:
            return feats
        scores = torch.cat(scores)
        topk = torch.topk(scores, k=min(k, feats.shape[0])).indices
        return feats[topk]

    for r in oks:  # OK 反馈 → 钉住并入正常库（不参与 coreset 淘汰/不计入上限）
        model.model.add_normal_features(_feedback_feats(store.images_dir / r["stored_image"]), pinned=True)
    for r in ngs:  # NG 反馈 → top-k 高分 patch 入缺陷库
        model.model.add_defect_features(
            _feedback_feats(store.images_dir / r["stored_image"], k=cfg.defect_topk))
    model.model.resample_normal_bank()

    # 阈值重校准并注入（FR-6.5，score_one 自动按版本配置切块）
    from dino_exp.infer import score_one

    ok_scores = [score_one(model, p, cfg)[0] for p in ok_calibration_images(category, cfg)]
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
    metrics = {
        **report["metrics"],
        "threshold": model_threshold(cfg, category, version),
        "parent_threshold": old_threshold,  # FR-6.5：新旧阈值同处记录
    }
    vdir = Registry(cfg.models_root).version_dir(category, version)
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    warning = auroc_drop_warning(parent_metrics.get("image_AUROC"), metrics.get("image_AUROC"))
    store.apply()  # 所有步骤成功后才消费暂存区（归档+清空），返回值无需使用
    from dino_exp.logs import get_logger

    get_logger("retrain").info(
        "[%s] 再训练完成: %s → %s（应用反馈 %d 条，AUROC %s → %s）%s",
        category, parent, version, len(effective_rows),
        parent_metrics.get("image_AUROC"), metrics.get("image_AUROC"),
        f" 警告: {warning}" if warning else "",
    )
    return {"version": version, "metrics": metrics, "warning": warning, "preview": pv}


def model_threshold(cfg: Config, category: str, version: str) -> float:
    from dino_exp.infer import load_threshold as _lt

    return _lt(Registry(cfg.models_root).version_dir(category, version))

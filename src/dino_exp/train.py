from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import build_folder, dataset_info, ok_calibration_images
from dino_exp.errors import DinoError
from dino_exp.models.dual_bank import (
    DualBankPatchcore,
    calibrate_threshold,
    save_banks,
)
from dino_exp.models.registry import Registry


def build_model(cfg: Config) -> DualBankPatchcore:
    spec = cfg.backbone_spec
    from anomalib.metrics import AUPR, AUROC, Evaluator, F1Score

    evaluator = Evaluator(test_metrics=[AUROC(), AUPR(), F1Score()])
    return DualBankPatchcore(
        backbone=spec.timm_name,
        layers=cfg.layers,
        image_size=(cfg.image_size, cfg.image_size),
        fusion_weight=cfg.fusion_weight,
        bank_cap_ratio=cfg.bank_cap_ratio,
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
        evaluator=evaluator,
    )


def score_images(model, image_paths: list[Path], cfg: Config) -> list[float]:
    """对图片列表逐张推理，返回原始异常分数（raw pred_score）。"""
    from dino_exp.infer import preprocess_image

    model.eval()
    scores = []
    for p in image_paths:
        tensor = preprocess_image(p, cfg.image_size)
        with torch.no_grad():
            out = model.model(tensor)
        scores.append(float(out.pred_score.item()))
    return scores


def finalize_version(
    category: str,
    cfg: Config,
    model,
    *,
    ok_scores: list[float],
    metrics: dict,
    parent: str | None,
    feedback_applied: int,
) -> str:
    """校准阈值 → 注入模型 → 存版本（banks + config + metrics + meta）。返回版本号。"""
    threshold = calibrate_threshold(ok_scores, sigma=cfg.threshold_sigma)
    model.apply_threshold(threshold)
    with tempfile.TemporaryDirectory() as td:
        bank_path = Path(td) / "normal_bank.pt"
        save_banks(model.model, bank_path)
        version = Registry(cfg.models_root).create_version(
            category,
            normal_bank=bank_path,
            defect_bank=None,  # 缺陷库已含在 save_banks 字典内；registry 的 defect_bank.pt 仅占位
            checkpoint=None,
            config={
                "backbone": cfg.backbone,
                "layers": cfg.layers,
                "image_size": cfg.image_size,
                "coreset_sampling_ratio": cfg.coreset_sampling_ratio,
                "num_neighbors": cfg.num_neighbors,
                "fusion_weight": cfg.fusion_weight,
                "bank_cap_ratio": cfg.bank_cap_ratio,
                "defect_topk": cfg.defect_topk,
                "threshold_sigma": cfg.threshold_sigma,
            },
            metrics={**metrics, "threshold": threshold},
            meta={"parent": parent, "feedback_applied": feedback_applied},
        )
    return version


def train_model(category: str, cfg: Config) -> dict:
    """基础训练（设计文档 §4 工作流 1）。返回新版本指标。"""
    from anomalib.engine import Engine

    info = dataset_info(category, cfg)  # 结构校验前置（NFR-4）
    datamodule = build_folder(category, cfg)
    model = build_model(cfg)
    engine = Engine(default_root_dir=str(Path("results") / category))
    engine.fit(model=model, datamodule=datamodule)
    # 阈值校准（FR-2.3）：OK 校准图 raw 分数 mean+3σ
    ok_scores = score_images(model, ok_calibration_images(category, cfg), cfg)
    # 全量验证（FR-3.4）：degraded 时跳过聚合指标
    if info.degraded:
        metrics: dict = {"degraded": True}
    else:
        results = engine.test(model=model, datamodule=datamodule)
        metrics = {k: float(v) for k, v in results[0].items()}
    version = finalize_version(
        category, cfg, model,
        ok_scores=ok_scores, metrics=metrics, parent=None, feedback_applied=0,
    )
    return {"version": version, "metrics": metrics}

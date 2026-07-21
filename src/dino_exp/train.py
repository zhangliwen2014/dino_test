from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import build_folder, dataset_info, ok_calibration_images
from dino_exp.models.dual_bank import (
    DualBankPatchcore,
    calibrate_threshold,
    save_banks,
)
from dino_exp.models.registry import Registry


def build_model(cfg: Config) -> DualBankPatchcore:
    spec = cfg.backbone_spec
    from anomalib.metrics import AUPR, AUPRO, AUROC, Evaluator, F1Score

    # 挂法对齐 anomalib AnomalibModule.configure_evaluator：fields 显式指定（无默认，
    # 裸构造会抛 ValueError）。像素级 strict=False：无 gt_mask 的批次（degraded 数据集）
    # 静默跳过，compute() 返回 None 而不报错（base.py 已实证）；有 mask 时自动产出
    # pixel_AUROC/pixel_AUPRO（FR-3.1）。
    evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            AUPR(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        ]
    )
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
    device = next(model.model.parameters()).device  # 跟随模型设备（GPU 训练后为 cuda）
    scores = []
    for p in image_paths:
        tensor = preprocess_image(p, cfg.image_size).to(device)
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


def train_model(category: str, cfg: Config, log=None) -> dict:
    """基础训练（设计文档 §4 工作流 1）。返回新版本指标。

    log: 可选阶段日志回调（设计 §3.7，Web UI 经 JobManager 队列轮询展示）。
    """
    from anomalib.engine import Engine

    log = log or (lambda msg: None)
    info = dataset_info(category, cfg)  # 结构校验前置（NFR-4）
    log(f"校验数据集... train/good={info.train_good} test/good={info.test_good}")
    log("加载骨干与数据...")
    datamodule = build_folder(category, cfg)
    model = build_model(cfg)
    engine = Engine(default_root_dir=str(Path("results") / category))
    log("Engine.fit 建库中（CPU 可能需要几分钟）...")
    engine.fit(model=model, datamodule=datamodule)
    log(f"coreset 完成，记忆库 {model.model.memory_bank.shape[0]} 条")
    # 阈值校准（FR-2.3）：OK 校准图 raw 分数 mean+3σ。
    # 校准与注入必须先于 engine.test：否则 PostProcessor 用 F1AdaptiveThreshold
    # 自适应阈值算 image_F1Score，与部署阈值 mean+3σ 口径不一致（偏乐观、
    # 跨版本不可比）。注入后 engine.test 的 F1 与判定/导出共用同一阈值（R2）。
    log("校准阈值...")
    ok_scores = score_images(model, ok_calibration_images(category, cfg), cfg)
    threshold = calibrate_threshold(ok_scores, sigma=cfg.threshold_sigma)
    model.apply_threshold(threshold)  # 幂等（Task 5 已实证）；finalize_version 会用同一 ok_scores 重算并重复注入
    log(f"阈值={threshold:.4f}，注入完成")
    # 全量验证（FR-3.4）：degraded 时跳过聚合指标
    log("全量验证...")
    if info.degraded:
        metrics: dict = {"degraded": True}
    else:
        results = engine.test(model=model, datamodule=datamodule)
        metrics = {k: float(v) for k, v in results[0].items()}
    version = finalize_version(
        category, cfg, model,
        ok_scores=ok_scores, metrics=metrics, parent=None, feedback_applied=0,
    )
    log(f"版本 {version} 已保存")
    return {"version": version, "metrics": metrics}

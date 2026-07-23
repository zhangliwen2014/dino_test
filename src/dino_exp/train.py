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
                # 切块配置随版本保存（推理按此自动切块；从模型属性取已解析的网格）
                "tile_grid": list(getattr(model, "train_tile_grid", (1, 1))),
                "tile_overlap": getattr(model, "train_tile_overlap", cfg.tile_overlap),
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

    from dino_exp.logs import get_logger

    _logger = get_logger("train")
    user_log = log or (lambda msg: None)

    def log(msg: str) -> None:
        _logger.info("[%s] %s", category, msg)  # 持久化到 logs/dino.log
        user_log(msg)  # UI 队列

    info = dataset_info(category, cfg)  # 结构校验前置（NFR-4）
    log(f"校验数据集... train/good={info.train_good} test/good={info.test_good}")

    # 解析切块网格：auto 按首张训练图尺寸推荐；图片小于模型输入则不切
    from PIL import Image as _Img

    from dino_exp.datasets import category_images, test_images_with_labels
    from dino_exp.tiles import auto_grid, parse_tile_mode, should_tile, split_image

    tile_grid = (1, 1)
    train_imgs = []
    if cfg.tile_mode != "off":
        train_imgs = [p for rel, p in category_images(category, cfg) if rel.startswith("train/good")]
        with _Img.open(train_imgs[0]) as im0:
            w0, h0 = im0.size
        if should_tile(w0, h0, cfg.image_size, (2, 2)):
            parsed = parse_tile_mode(cfg.tile_mode)
            tile_grid = auto_grid(w0, h0, cfg.image_size, cfg.tile_target_patch_px) if parsed == "auto" else parsed
            log(f"切块模式 {cfg.tile_mode} → 网格 {tile_grid[0]}x{tile_grid[1]}（图 {w0}x{h0}，重叠 {cfg.tile_overlap}）")
        else:
            log(f"图片 {w0}x{h0} 不大于模型输入 {cfg.image_size}，无需切块")

    from dino_exp.config import resolve_device

    device = resolve_device(cfg)
    if device == "cpu":
        # Lightning 会自动占 GPU；强制 CPU 需在创建 Engine 前屏蔽 CUDA（进程级，本进程专用）
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log(f"加载骨干与数据...（设备: {device}）")
    model = build_model(cfg)
    model.train_tile_grid = tile_grid
    model.train_tile_overlap = cfg.tile_overlap

    if tile_grid != (1, 1):
        # 切块建库：训练/推理同尺度，记忆库特征来自与推理一致的切块
        from dino_exp.infer import _preprocess_pil
        from dino_exp.tiles import clamp_grid

        # 类别内图片尺寸不一时提示（网格按首张图算，其他图钳制密度）
        sizes = set()
        for p in train_imgs[:20]:
            with _Img.open(p) as im:
                sizes.add(im.size)
        if len(sizes) > 1:
            log(f"注意：训练图尺寸不一致（{len(sizes)} 种），网格按首张图计算，其他图自动钳制密度")

        model = model.to(device)
        model.train()
        for p in train_imgs:
            with _Img.open(p) as im:
                im = im.convert("RGB")
                g = clamp_grid(tile_grid, im.size[0], im.size[1], cfg.image_size)
                tiles = split_image(im, g, cfg.tile_overlap)
            batch = torch.cat([_preprocess_pil(t, cfg.image_size) for t, _ in tiles]).to(device)
            with torch.no_grad():
                model.model(batch)  # training 模式自动累积 embedding_store
        model.model.fit_coreset()
        log(f"coreset 完成，记忆库 {model.model.memory_bank.shape[0]} 条")
        engine = None
    else:
        datamodule = build_folder(category, cfg)
        engine = Engine(default_root_dir=str(Path("results") / category))
        log(f"Engine.fit 建库中（{device}{'，可能需要几分钟' if device == 'cpu' else ''}）...")
        engine.fit(model=model, datamodule=datamodule)
        log(f"coreset 完成，记忆库 {model.model.memory_bank.shape[0]} 条")

    # 阈值校准（FR-2.3）：OK 校准图 raw 分数 mean+3σ。
    # 校准与注入必须先于 engine.test：否则 PostProcessor 用 F1AdaptiveThreshold
    # 自适应阈值算 image_F1Score，与部署阈值 mean+3σ 口径不一致（偏乐观、
    # 跨版本不可比）。注入后 engine.test 的 F1 与判定/导出共用同一阈值（R2）。
    log("校准阈值...")
    from dino_exp.infer import score_one

    model.eval()
    ok_scores = [score_one(model, p, cfg)[0] for p in ok_calibration_images(category, cfg)]
    threshold = calibrate_threshold(ok_scores, sigma=cfg.threshold_sigma)
    model.apply_threshold(threshold)  # 幂等（Task 5 已实证）；finalize_version 会用同一 ok_scores 重算并重复注入
    log(f"阈值={threshold:.4f}，注入完成")
    # 全量验证（FR-3.4）：degraded 时跳过聚合指标
    log("全量验证...")
    if tile_grid != (1, 1):
        # 切块模型：用 in-memory score_one 逐图打分（engine.test 不懂切块）
        from dino_exp.validate import aggregate_metrics

        rows = []
        for p, gt, dt in test_images_with_labels(category, cfg):
            s, _, _ = score_one(model, p, cfg)
            rows.append({"label_gt": gt, "score": s})
        metrics = {"degraded": True} if info.degraded else aggregate_metrics(rows, threshold)
    elif info.degraded:
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

"""全量/选图验证（FR-3.1/FR-3.2/FR-3.4）：聚合指标、逐图结果、误判过滤、无 NG 降级。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from dino_exp.config import Config
from dino_exp.datasets import dataset_info, mask_path_for, test_images_with_labels
from dino_exp.infer import decide_label, infer_batch, load_model_for_version, preprocess_image, score_one
from dino_exp.models.registry import Registry


def aggregate_metrics(rows: list[dict], threshold: float, pixel_pairs: list[tuple] | None = None) -> dict:
    """从逐图 {label_gt, score} 计算图片级 AUROC/AUPR/F1Score；无 NG 样本时降级。

    键名对齐 anomalib evaluator 输出（image_F1Score 等）。pixel_pairs 为
    (anomaly_map (H,W) float, gt_mask (H,W) 0/1 uint8) 列表：非空时追加
    pixel_AUROC/pixel_AUPRO（FR-3.1），为空则不出现 pixel_ 键。
    """
    from torchmetrics import AUROC, AveragePrecision, F1Score

    labels = torch.tensor([r["label_gt"] for r in rows])
    raw_scores = torch.tensor([r["score"] for r in rows])
    if labels.sum().item() == 0:
        return {"degraded": True, "note": "无 NG 测试图，指标降级：仅输出逐图分数"}
    # torchmetrics 二分指标按 [0,1] 概率语义处理输入，超界会被钳制导致并列失真；
    # AUROC/AUPR 是秩指标，min-max 归一化不改变结果
    span = raw_scores.max() - raw_scores.min()
    scores = (raw_scores - raw_scores.min()) / (span + 1e-12)
    preds = (raw_scores >= threshold).long()
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


def relabel_rows(rows: list[dict], threshold: float) -> list[dict]:
    """按给定阈值重算每行的判定（阈值界面调整时用，不改动原始分数）。"""
    out = []
    for r in rows:
        r = dict(r)
        r["label_pred"] = decide_label(r["score"], threshold)
        out.append(r)
    return out


def plot_score_distribution(rows: list[dict], threshold: float, out_path: str | Path) -> Path:
    """OK/NG 分数分布直方图 + 阈值线（选阈值的直观依据）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r["score"] for r in rows if r["label_gt"] == 0]
    ng = [r["score"] for r in rows if r["label_gt"] == 1]
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=110)
    bins = 20
    if ok:
        ax.hist(ok, bins=bins, alpha=0.7, color="#16a34a", label=f"OK ({len(ok)})")
    if ng:
        ax.hist(ng, bins=bins, alpha=0.7, color="#dc2626", label=f"NG ({len(ng)})")
    ax.axvline(threshold, color="#111", linestyle="--", linewidth=1.5,
               label=f"threshold={threshold:.2f}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_roc(rows: list[dict], out_path: str | Path) -> Path | None:
    """ROC 曲线（需要同时有 OK 与 NG 标签，否则返回 None）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchmetrics import AUROC

    labels = [r["label_gt"] for r in rows]
    if len(set(labels)) < 2:
        return None
    scores = torch.tensor([r["score"] for r in rows])
    labels_t = torch.tensor(labels)
    auroc = float(AUROC(task="binary")((scores - scores.min()) / (scores.max() - scores.min() + 1e-12), labels_t))
    ths = sorted(set(scores.tolist()), reverse=True)
    tpr, fpr = [1.0], [1.0]
    for t in ths:
        preds = [1 if s >= t else 0 for s in scores.tolist()]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        tpr.append(tp / max(1, sum(labels)))
        fpr.append(fp / max(1, len(labels) - sum(labels)))
    tpr.append(0.0)
    fpr.append(0.0)
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=110)
    ax.plot(fpr, tpr, marker=".", color="#2563eb", label=f"AUROC={auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out)
    plt.close(fig)
    return out


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
    from dino_exp.config import resolve_device

    device = resolve_device(cfg)
    model = model.to(device).eval()
    from dino_exp.infer import _imwrite_unicode, annotate_and_frame

    anno_dir = Path("outputs/annotates") / f"{category}_{version}"
    anno_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pixel_pairs = []
    for path, label_gt, defect_type in test_images_with_labels(category, cfg):
        score, amap_full, _ = score_one(model, path, cfg)  # 自动按版本配置切块
        label_pred = decide_label(score, threshold)
        # 生成标记图（原图+缺陷框+判定外框）与热力图，供验证结果点击预览
        annotated, _ = annotate_and_frame(path, amap_full.cpu(), threshold, label_pred)
        anno_path = anno_dir / f"{Path(path).stem}_anno.png"
        _imwrite_unicode(anno_path, annotated)
        from dino_exp.infer import heatmap_to_bgr

        with Image.open(path) as im:
            orig_size = im.size
        heat_path = anno_dir / f"{Path(path).stem}_heat.png"
        _imwrite_unicode(heat_path, heatmap_to_bgr(amap_full.cpu(), orig_size))
        rows.append({
            "path": str(path),
            "label_gt": label_gt,
            "defect_type": defect_type,
            "score": score,
            "label_pred": label_pred,
            "annotated_path": str(anno_path),
            "heatmap_path": str(heat_path),
        })
        mask_path = mask_path_for(path, defect_type, info)
        if mask_path is not None:
            amap = amap_full.squeeze().float().cpu()
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask = torch.from_numpy((mask > 127).astype("uint8"))
            pixel_pairs.append((amap, gt_mask))
    return rows, threshold, version, pixel_pairs


def validate_full(category: str, version: str | None, cfg: Config) -> dict:
    """全量验证（FR-3.1/FR-3.4）：聚合指标（有 mask 时含像素级）+ 逐图结果写入版本目录 validation.json。

    同时生成分数分布图与 ROC 曲线（有双类标签时）供界面查看/调阈值。
    """
    rows, threshold, version, pixel_pairs = score_test_set(category, version, cfg)
    metrics = aggregate_metrics(rows, threshold, pixel_pairs=pixel_pairs)
    vdir = Registry(cfg.models_root).version_dir(category, version)
    save_validation_report(vdir, metrics, rows)
    dist_path = plot_score_distribution(rows, threshold, vdir / "validation_dist.png")
    roc_path = plot_roc(rows, vdir / "validation_roc.png")
    return {"version": version, "metrics": metrics, "rows": rows,
            "dist_plot": str(dist_path), "roc_plot": str(roc_path) if roc_path else None}


def validate_images(category: str, version: str | None, paths: list[str], cfg: Config) -> list[dict]:
    """选图验证（FR-3.2）：逐图分数/判定/热力图，不出聚合指标。"""
    return infer_batch(list(paths), version, category=category, cfg=cfg)

import json

import torch

from dino_exp.validate import aggregate_metrics, filter_errors, save_validation_report


def test_aggregate_metrics_computes_auroc_f1():
    # 4 OK(score 低) + 4 NG(score 高)，完美可分
    rows = [
        {"label_gt": 0, "score": s} for s in [0.1, 0.2, 0.3, 0.4]
    ] + [
        {"label_gt": 1, "score": s} for s in [0.6, 0.7, 0.8, 0.9]
    ]
    m = aggregate_metrics(rows, threshold=0.5)
    assert m["image_AUROC"] == 1.0
    assert m["image_F1Score"] == 1.0


def test_aggregate_metrics_degraded_returns_none_metrics():
    rows = [{"label_gt": 0, "score": 0.1}, {"label_gt": 0, "score": 0.2}]
    m = aggregate_metrics(rows, threshold=0.5)
    assert m["degraded"] is True
    assert "image_AUROC" not in m


def _ng_rows() -> list[dict]:
    return [
        {"label_gt": 0, "score": 0.1},
        {"label_gt": 1, "score": 0.9},
    ]


def test_aggregate_metrics_with_pixel_pairs_adds_pixel_metrics():
    # 假像素数据：anomaly_map 高分区恰为 gt_mask 缺陷区 → 像素指标应接近完美
    gt = torch.zeros(8, 8, dtype=torch.uint8)
    gt[2:5, 2:5] = 1
    amap = gt.float()  # 缺陷区 1.0、背景 0.0
    m = aggregate_metrics(_ng_rows(), threshold=0.5, pixel_pairs=[(amap, gt), (amap, gt)])
    assert m["pixel_AUROC"] == 1.0
    assert 0.9 < m["pixel_AUPRO"] <= 1.0


def test_aggregate_metrics_without_pixel_pairs_omits_pixel_keys():
    m = aggregate_metrics(_ng_rows(), threshold=0.5)
    assert "pixel_AUROC" not in m
    assert "pixel_AUPRO" not in m
    # 空列表同样不出现（无 mask 的数据集）
    m = aggregate_metrics(_ng_rows(), threshold=0.5, pixel_pairs=[])
    assert "pixel_AUROC" not in m


def test_filter_errors():
    rows = [
        {"path": "a.png", "label_gt": 0, "label_pred": "OK"},
        {"path": "b.png", "label_gt": 1, "label_pred": "OK"},  # 漏报
        {"path": "c.png", "label_gt": 0, "label_pred": "NG"},  # 误报
    ]
    errs = filter_errors(rows)
    assert [r["path"] for r in errs] == ["b.png", "c.png"]


def test_save_validation_report(tmp_path):
    save_validation_report(tmp_path, {"image_AUROC": 0.9}, [{"path": "a"}])
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["metrics"]["image_AUROC"] == 0.9
    assert len(report["rows"]) == 1

import json

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
    assert m["image_F1"] == 1.0


def test_aggregate_metrics_degraded_returns_none_metrics():
    rows = [{"label_gt": 0, "score": 0.1}, {"label_gt": 0, "score": 0.2}]
    m = aggregate_metrics(rows, threshold=0.5)
    assert m["degraded"] is True
    assert "image_AUROC" not in m


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

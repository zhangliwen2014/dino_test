import pytest
import torch

from dino_exp.errors import DinoError
from dino_exp.infer import decide_label, heatmap_to_bgr, load_threshold


def test_decide_label():
    assert decide_label(1.5, 1.0) == "NG"
    assert decide_label(0.5, 1.0) == "OK"
    assert decide_label(1.0, 1.0) == "NG"  # ≥ 阈值判 NG


def test_load_threshold(tmp_path):
    import json

    (tmp_path / "metrics.json").write_text(json.dumps({"threshold": 1.23}))
    assert load_threshold(tmp_path) == 1.23


def test_load_threshold_missing_raises(tmp_path):
    with pytest.raises(DinoError, match="threshold"):
        load_threshold(tmp_path)


def test_heatmap_to_bgr_shape_and_range():
    import numpy as np

    amap = torch.rand(1, 1, 16, 16)
    bgr = heatmap_to_bgr(amap, out_size=(64, 64))
    assert bgr.shape == (64, 64, 3)
    assert bgr.dtype == np.uint8


def test_heatmap_name_includes_parent_dir(tmp_path):
    """同名不同目录的图片产生不同热力图文件名，避免互相覆盖。"""
    from dino_exp.infer import _heatmap_name

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pa = tmp_path / "a" / "x.png"
    pb = tmp_path / "b" / "x.png"
    assert _heatmap_name(pa, "v001") == "a_x_v001_heatmap.png"
    assert _heatmap_name(pb, "v001") == "b_x_v001_heatmap.png"
    assert _heatmap_name(pa, "v001") != _heatmap_name(pb, "v001")


def test_infer_batch_loads_model_once(monkeypatch):
    """批量推理只加载一次模型。"""
    import dino_exp.infer as infer_mod

    calls = {"n": 0}

    class FakeModel:
        def to(self, device):
            return self

    def fake_load(category, version, cfg):
        calls["n"] += 1
        return FakeModel(), 1.0, "v001"

    monkeypatch.setattr(infer_mod, "load_model_for_version", fake_load)
    monkeypatch.setattr(infer_mod, "_infer_loaded", lambda *a, **k: {"label": "OK"})
    from dino_exp.config import Config

    results = infer_mod.infer_batch(["a.png", "b.png", "c.png"], category="x", cfg=Config())
    assert calls["n"] == 1
    assert results == [{"label": "OK"}] * 3


def test_imwrite_unicode_path(tmp_path):
    """中文路径写热力图不产生乱码文件名（cv2.imwrite 在 Windows 的已知缺陷）。"""
    import numpy as np

    from dino_exp.infer import _imwrite_unicode

    target = tmp_path / "热力图_测试.png"
    _imwrite_unicode(target, np.zeros((8, 8, 3), dtype=np.uint8))
    assert target.exists()
    assert target.read_bytes()[:4] == b"\x89PNG"


def test_annotate_defects_draws_box(tmp_path):
    import numpy as np
    from PIL import Image as _Img

    from dino_exp.infer import annotate_defects

    img_path = tmp_path / "img.png"
    _Img.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8)).save(img_path)
    amap = torch.zeros(1, 1, 100, 100)
    amap[0, 0, 40:60, 40:60] = 10.0  # 中央一块高异常区
    annotated, boxes = annotate_defects(img_path, amap, threshold=5.0)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert 35 <= x <= 45 and 35 <= y <= 45 and w >= 15 and h >= 15
    assert annotated.shape == (100, 100, 3)
    # OK 图（全部低于阈值）无框
    _, boxes2 = annotate_defects(img_path, torch.zeros(1, 1, 100, 100), threshold=5.0)
    assert boxes2 == []

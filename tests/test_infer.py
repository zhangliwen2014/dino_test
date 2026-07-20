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

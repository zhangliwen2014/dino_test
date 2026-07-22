import pytest
import torch
from PIL import Image

from dino_exp.config import Config, load_config
from dino_exp.errors import DinoError
from dino_exp.tiles import (
    auto_grid,
    clamp_grid,
    parse_tile_mode,
    should_tile,
    split_image,
    stitch_anomaly_maps,
)


def test_parse_tile_mode():
    assert parse_tile_mode("off") == (1, 1)
    assert parse_tile_mode("auto") == "auto"
    assert parse_tile_mode("3x3") == (3, 3)
    with pytest.raises(ValueError, match="未知 tile 模式"):
        parse_tile_mode("5x5")


def test_auto_grid_small_image_no_tile_needed():
    # 小图：2x2 就满足目标 → 最小网格
    assert auto_grid(400, 300, 224, target_patch_px=20) == (2, 2)


def test_auto_grid_large_image_dense():
    # 2048×1536 @224：2x2=64px, 3x3=42.7, 4x4=32, 6x6=21.3, 8x8=16
    assert auto_grid(2048, 1536, 224, target_patch_px=20) == (8, 8)   # 6x6=21.3>20 → 8x8
    assert auto_grid(2048, 1536, 224, target_patch_px=40) == (4, 4)   # 3x3=42.7>40 → 4x4
    assert auto_grid(2048, 1536, 224, target_patch_px=50) == (3, 3)   # 42.7≤50 → 3x3


def test_split_image_grid_and_overlap():
    im = Image.new("RGB", (100, 80))
    tiles = split_image(im, (2, 2), overlap=0.0)
    assert len(tiles) == 4
    # 无重叠时恰好铺满
    boxes = [b for _, b in tiles]
    assert (0, 0, 50, 40) in boxes and (50, 40, 100, 80) in boxes
    # 有重叠时边缘块向外延伸但不越界
    tiles_ov = split_image(im, (2, 2), overlap=0.2)
    for _, (x0, y0, x1, y1) in tiles_ov:
        assert x0 >= 0 and y0 >= 0 and x1 <= 100 and y1 <= 80


def test_stitch_core_assignment_no_duplication():
    # 2x2 网格：每块 amap 全是常数，拼接后各象限等于对应块的值
    im_w, im_h = 100, 80
    boxes = [(0, 0, 55, 45), (45, 0, 100, 45), (0, 35, 55, 80), (45, 35, 100, 80)]
    vals = [1.0, 2.0, 3.0, 4.0]
    amaps = [torch.full((1, 8, 8), v) for v in vals]
    merged = stitch_anomaly_maps(amaps, boxes, (2, 2), (im_w, im_h))
    assert merged.shape == (1, 1, im_h, im_w)
    # 四个象限中心点应等于对应块值（核心归属，无重复计分）
    assert merged[0, 0, 20, 25].item() == pytest.approx(1.0)
    assert merged[0, 0, 20, 75].item() == pytest.approx(2.0)
    assert merged[0, 0, 60, 25].item() == pytest.approx(3.0)
    assert merged[0, 0, 60, 75].item() == pytest.approx(4.0)


def test_clamp_grid_limits_density_for_small_image():
    # 300×300 图、image_size 224：min_tile=112 → 最多 2x2
    assert clamp_grid((8, 8), 300, 300, 224) == (2, 2)
    assert clamp_grid((1, 1), 300, 300, 224) == (1, 1)
    assert clamp_grid((4, 4), 2048, 1536, 224) == (4, 4)


def test_should_tile_only_when_image_larger_than_model():
    assert should_tile(2048, 1536, 224, (2, 2)) is True
    assert should_tile(200, 200, 224, (2, 2)) is False
    assert should_tile(2048, 1536, 224, (1, 1)) is False


def test_config_tile_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tile_mode: 5x5\n", encoding="utf-8")
    with pytest.raises(DinoError, match="tile_mode"):
        load_config(p)
    p.write_text("tile_mode: auto\ntile_overlap: 0.6\n", encoding="utf-8")
    with pytest.raises(DinoError, match="tile_overlap"):
        load_config(p)
    p.write_text("tile_mode: 3x3\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.tile_mode == "3x3"

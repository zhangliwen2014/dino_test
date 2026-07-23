"""固定尺寸切块（tile_scheme="size"）单元测试：T 公式、滑窗起点、中点归属、
pad/小图放大边界、score_one 新旧 scheme 分流、训练建库分支、版本配置还原。"""

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from dino_exp.config import Config
from dino_exp.tiles import compute_tile_size, split_fixed, stitch_fixed, upscale_small


def test_compute_tile_size():
    assert compute_tile_size(20, 224, 14) == 320   # DINOv2 patch14@224 → 16 patch/边 → T=20×16
    assert compute_tile_size(20, 224, 16) == 280   # DINOv3 patch16@224 → 14 patch/边 → T=20×14


def test_split_fixed_sliding_starts_anchored():
    im = Image.new("RGB", (2048, 1536))
    tiles, pad = split_fixed(im, 320, overlap=0.15)  # stride=272
    assert pad == (0, 0)
    xs = sorted({b[0] for _, b in tiles})
    ys = sorted({b[1] for _, b in tiles})
    assert xs[0] == 0 and ys[0] == 0
    assert xs[-1] == 2048 - 320 == 1728  # 末块贴边锚定 W-T
    assert ys[-1] == 1536 - 320 == 1216
    for a, b in zip(xs, xs[1:-1]):       # 除末段贴边外，相邻起点间距恒为 stride
        assert b - a == 272
    for t, (x0, y0, x1, y1) in tiles:    # 每块尺寸恒为 T×T，无畸形边缘块
        assert t.size == (320, 320)
        assert (x1 - x0, y1 - y0) == (320, 320)


def _expected_owners(width, starts, tile):
    """独立复算中点归属：bounds=相邻块中心中点，返回每像素的归属块值（从 1 起）。"""
    centers = [s + tile / 2 for s in starts]
    bounds = [0] + [int((centers[i - 1] + centers[i]) / 2) for i in range(1, len(starts))] + [width]
    owners = np.zeros(width, dtype=np.float32)
    for i in range(len(starts)):
        owners[bounds[i]:bounds[i + 1]] = float(i + 1)
    return owners


@pytest.mark.parametrize("width", [864, 865, 1135, 592])  # (W-T) mod stride = 0 / 1 / 271 / 0（k=1）
def test_stitch_fixed_midpoint_ownership_exactly_once(width):
    T, overlap = 320, 0.15  # stride=272；864=320+2×272，1135-320=815=2×272+271
    im = Image.new("RGB", (width, T))  # y 轴单段，只验 x 归属
    tiles, pad = split_fixed(im, T, overlap, margin=40)
    assert pad == (0, 0)
    xs = sorted({b[0] for _, b in tiles})
    assert xs[-1] == width - T           # 各种余数下末块都贴边锚定
    amaps = [torch.full((1, 8, 8), float(i + 1)) for i in range(len(tiles))]
    merged = stitch_fixed(amaps, [b for _, b in tiles], (width, T))
    assert merged.shape == (1, 1, T, width)
    got = merged[0, 0, T // 2].numpy()
    # 与独立复算的中点归属逐像素一致 → 每个像素恰好归属一个块（无重叠无空洞）
    np.testing.assert_array_equal(got, _expected_owners(width, xs, T))


def test_split_fixed_long_strip_pads_short_side():
    im = Image.new("RGB", (1000, 100))   # 长条图：x 可切，y < T → pad
    tiles, pad = split_fixed(im, 320, overlap=0.15, margin=40)
    assert pad == (0, 220)               # 短边 reflect pad 到 T
    for t, (x0, y0, x1, y1) in tiles:
        assert t.size == (320, 320)      # 块保持正方形
    xs = sorted({b[0] for _, b in tiles})
    assert xs[-1] == 1000 - 320
    # pad 的图先拼后裁回原始 out_size，全覆盖无空洞
    amaps = [torch.ones(1, 8, 8) for _ in tiles]
    merged = stitch_fixed(amaps, [b for _, b in tiles], (1000, 100))
    assert merged.shape == (1, 1, 100, 1000)
    assert torch.all(merged == 1.0)


def test_split_fixed_both_axes_small_returns_single_tile():
    im = Image.new("RGB", (350, 330))    # 两轴 < T+margin=360 且 ≥ T → 单块整图
    tiles, pad = split_fixed(im, 320, overlap=0.15, margin=40)
    assert len(tiles) == 1
    assert pad == (0, 0)
    assert tiles[0][1] == (0, 0, 350, 330)


def test_upscale_small_only_when_both_axes_small():
    T, margin = 320, 40
    out = upscale_small(Image.new("RGB", (300, 200)), T, margin)  # 小图 → 短边放大到 T
    assert out.size == (480, 320)
    im2 = Image.new("RGB", (1000, 100))   # 长条图（一轴 ≥ T+margin）不放大，走 pad
    assert upscale_small(im2, T, margin).size == (1000, 100)
    im3 = Image.new("RGB", (350, 330))    # 短边已 ≥ T → 不动
    assert upscale_small(im3, T, margin).size == (350, 330)


# ---------- score_one 新旧 scheme 分流（fake 模型） ----------


class _FakeOut:
    def __init__(self, n):
        self.pred_score = torch.arange(1, n + 1, dtype=torch.float32)
        self.anomaly_map = torch.ones(n, 1, 8, 8)


class _FakeCore:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(int(batch.shape[0]))
        return _FakeOut(batch.shape[0])


class _FakeModel:
    def __init__(self, **attrs):
        self.model = _FakeCore()
        self.train_image_size = 64
        for k, v in attrs.items():
            setattr(self, k, v)

    def parameters(self):
        return iter([torch.zeros(1)])


def _write_img(path, size):
    Image.new("RGB", size, (128, 64, 32)).save(path)


def test_score_one_size_scheme_tiles(tmp_path):
    from dino_exp.infer import score_one

    p = tmp_path / "a.png"
    _write_img(p, (200, 200))
    model = _FakeModel(train_tile_scheme="size", train_tile_size=80,
                       train_tile_overlap=0.15, train_tile_target_patch_px=20)
    score, amap, _ = score_one(model, p, Config())
    # stride=68, margin=2P=40：200 ≥ T+40=120 → 两轴滑窗，起点 0,68,120 → 3x3=9 块一次前向
    assert model.model.batch_sizes == [9]
    assert score == 9.0  # max(各块分数)
    assert amap.shape == (1, 1, 200, 200)


def test_score_one_size_scheme_small_image_upscale(tmp_path):
    from dino_exp.infer import score_one

    p = tmp_path / "s.png"
    _write_img(p, (60, 30))  # 两轴 < T+2P=120 且短边 < T → 放大到 160x80
    model = _FakeModel(train_tile_scheme="size", train_tile_size=80,
                       train_tile_overlap=0.15, train_tile_target_patch_px=20)
    score, amap, _ = score_one(model, p, Config())
    # 放大后 160 ≥ 120 → x 轴滑窗（起点 0,68,80），y 单段 → 3 块
    assert model.model.batch_sizes == [3]
    assert amap.shape == (1, 1, 80, 160)  # 热图按放大后（未 pad）尺寸输出


def test_score_one_grid_scheme_unchanged(tmp_path):
    from dino_exp.infer import score_one

    p = tmp_path / "g.png"
    _write_img(p, (100, 100))
    model = _FakeModel(train_tile_grid=(2, 2), train_tile_overlap=0.1)  # 无 scheme 属性 → 旧 grid
    score, amap, _ = score_one(model, p, Config())
    assert model.model.batch_sizes == [4]  # 2x2=4 块
    assert score == 4.0
    assert amap.shape == (1, 1, 100, 100)


def test_score_one_grid_scheme_small_image_direct(tmp_path):
    from dino_exp.infer import score_one

    p = tmp_path / "d.png"
    _write_img(p, (50, 50))  # 小于模型输入 64 → 整图直推
    model = _FakeModel(train_tile_grid=(2, 2))
    score, amap, _ = score_one(model, p, Config())
    assert model.model.batch_sizes == [1]
    assert score == 1.0


def test_tile_images_for_model_scheme_dispatch():
    """retrain 反馈特征提取的切块辅助：按版本 scheme 分流，与 score_one 同口径。"""
    from dino_exp.infer import tile_images_for_model

    cfg = Config()
    im_big = Image.new("RGB", (200, 200))
    m_size = _FakeModel(train_tile_scheme="size", train_tile_size=80,
                        train_tile_overlap=0.15, train_tile_target_patch_px=20)
    assert len(tile_images_for_model(m_size, im_big, cfg)) == 9
    m_grid = _FakeModel(train_tile_grid=(2, 2))
    assert len(tile_images_for_model(m_grid, im_big, cfg)) == 4
    m_off = _FakeModel()
    assert tile_images_for_model(m_off, im_big, cfg) == [im_big]


# ---------- 版本配置还原（load_model_for_version） ----------


def _make_version(tmp_path, config_dict):
    from dino_exp.models.registry import Registry

    cfg = Config(models_root=tmp_path / "models")
    bank = tmp_path / "b.pt"
    torch.save({"memory_bank": torch.randn(4, 2), "defect_bank": torch.empty(0),
                "pinned_count": 0, "base_bank_size": 4}, bank)
    Registry(cfg.models_root).create_version(
        "c", normal_bank=bank, defect_bank=None, checkpoint=None,
        config=config_dict, metrics={"threshold": 1.0}, meta={"parent": None})
    return cfg


def _patch_build_model(monkeypatch):
    class _Fake:
        def __init__(self):
            self.model = type("M", (), {})()  # load_banks 直接赋属性即可

        def apply_threshold(self, t):
            self.threshold = t

        def eval(self):
            return self

    monkeypatch.setattr("dino_exp.train.build_model", lambda c: _Fake())


def test_load_model_restores_size_scheme(tmp_path, monkeypatch):
    import dino_exp.infer as infer_mod

    cfg = _make_version(tmp_path, {"backbone": "dinov2_vits14", "layers": ["blocks.11"],
                                   "image_size": 224, "tile_scheme": "size",
                                   "tile_size": 320, "tile_overlap": 0.15,
                                   "tile_target_patch_px": 20})
    _patch_build_model(monkeypatch)
    model, _, _ = infer_mod.load_model_for_version("c", None, cfg)
    assert model.train_tile_scheme == "size"
    assert model.train_tile_size == 320
    assert model.train_tile_overlap == 0.15
    assert model.train_tile_target_patch_px == 20


def test_load_model_defaults_grid_scheme_for_old_versions(tmp_path, monkeypatch):
    import dino_exp.infer as infer_mod

    cfg = _make_version(tmp_path, {"tile_grid": [2, 2], "tile_overlap": 0.1})  # 旧版本无 tile_scheme
    _patch_build_model(monkeypatch)
    model, _, _ = infer_mod.load_model_for_version("c", None, cfg)
    assert model.train_tile_scheme == "grid"  # 缺省按 grid 兼容
    assert model.train_tile_grid == (2, 2)
    assert not hasattr(model, "train_tile_size")


# ---------- 训练侧：auto → 固定尺寸建库，NxN → 旧网格（finalize 持久化） ----------


class _TrainFakeCore:
    def __init__(self):
        self.memory_bank = torch.randn(9, 4)
        self.defect_bank = torch.empty(0)
        self.pinned_count = torch.tensor([0])
        self.base_bank_size = 9
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(int(batch.shape[0]))

    def fit_coreset(self):
        pass


class _TrainFakeModel:
    def __init__(self):
        self.model = _TrainFakeCore()

    def to(self, device):
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def apply_threshold(self, t):
        self.threshold = t

    def parameters(self):
        return iter([torch.zeros(1)])


def _run_train_with_fakes(tmp_path, monkeypatch, tile_mode):
    import dino_exp.train as tr
    from dino_exp.datasets import DatasetInfo
    from pathlib import Path

    cfg = Config(data_root=tmp_path / "data", models_root=tmp_path / "models",
                 tile_mode=tile_mode, image_size=64, tile_overlap=0.15)
    img = tmp_path / "data" / "c" / "train" / "good" / "a.png"
    img.parent.mkdir(parents=True)
    _write_img(img, (200, 200))  # T=20×(64//14)=80；200 ≥ T+2P=120 → 可切

    monkeypatch.setattr(tr, "dataset_info",
                        lambda c, cfg: DatasetInfo("c", Path("d"), 1, 1, {"broken": 1}))
    monkeypatch.setattr("dino_exp.datasets.category_images",
                        lambda c, cfg: [("train/good/a.png", img)])
    monkeypatch.setattr("dino_exp.datasets.test_images_with_labels",
                        lambda c, cfg: [(img, 1, "broken")])
    created = []
    monkeypatch.setattr(tr, "build_model",
                        lambda c: created.append(_TrainFakeModel()) or created[-1])
    monkeypatch.setattr(tr, "ok_calibration_images", lambda c, cfg: [img])
    monkeypatch.setattr("dino_exp.infer.score_one", lambda m, p, c: (1.0, None, 0.0))
    monkeypatch.setattr("dino_exp.validate.aggregate_metrics",
                        lambda rows, th: {"image_AUROC": 1.0})
    result = tr.train_model("c", cfg)
    saved = yaml.safe_load(
        (cfg.models_root / "c" / result["version"] / "config.yaml").read_text(encoding="utf-8"))
    return created[0], saved


def test_train_auto_uses_fixed_size_scheme(tmp_path, monkeypatch):
    model, saved = _run_train_with_fakes(tmp_path, monkeypatch, "auto")
    assert model.train_tile_scheme == "size"
    assert model.train_tile_size == 80
    assert model.model.batch_sizes == [9]  # 3x3=9 块一次前向建库
    assert saved["tile_scheme"] == "size"
    assert saved["tile_size"] == 80
    assert saved["tile_overlap"] == 0.15
    assert saved["tile_target_patch_px"] == 20


def test_train_explicit_grid_keeps_legacy_scheme(tmp_path, monkeypatch):
    model, saved = _run_train_with_fakes(tmp_path, monkeypatch, "3x3")
    assert model.train_tile_scheme == "grid"
    assert model.train_tile_grid == (3, 3)
    assert model.model.batch_sizes == [9]  # 旧比例网格 3x3=9 块
    assert saved["tile_scheme"] == "grid"
    assert saved["tile_grid"] == [3, 3]
    assert saved["tile_size"] is None

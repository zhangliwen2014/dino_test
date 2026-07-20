import pytest

from dino_exp.config import Config
from dino_exp.datasets import (
    build_folder,
    dataset_info,
    import_images,
    list_datasets,
    ok_calibration_images,
    test_images_with_labels,
)
from dino_exp.errors import DinoError


def _mkimg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")  # 内容不重要，存在即可


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_root=tmp_path / "data")
    for i in range(5):
        _mkimg(c.data_root / "bottle" / "train" / "good" / f"t{i}.png")
    for i in range(2):
        _mkimg(c.data_root / "bottle" / "test" / "good" / f"g{i}.png")
    for i in range(3):
        _mkimg(c.data_root / "bottle" / "test" / "broken" / f"b{i}.png")
    for i in range(1):
        _mkimg(c.data_root / "bottle" / "test" / "contamination" / f"c{i}.png")
    _mkimg(c.data_root / "bottle" / "mask" / "broken" / "b0_mask.png")
    return c


def test_dataset_info_counts_and_defect_types(cfg):
    info = dataset_info("bottle", cfg)
    assert info.train_good == 5
    assert info.test_good == 2
    assert info.defect_types == {"broken": 3, "contamination": 1}
    assert info.has_test_good is True
    assert info.degraded is False


def test_dataset_info_missing_train_good_raises(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    (cfg.data_root / "x" / "test" / "good").mkdir(parents=True)
    with pytest.raises(DinoError, match="train/good"):
        dataset_info("x", cfg)


def test_dataset_info_degraded_when_no_ng(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    d = cfg.data_root / "cat"
    (d / "train" / "good").mkdir(parents=True)
    (d / "test" / "good").mkdir(parents=True)
    (d / "train" / "good" / "a.png").write_bytes(b"x")
    info = dataset_info("cat", cfg)
    assert info.degraded is True
    assert info.defect_types == {}


def test_import_images_ok_and_ng(cfg):
    src = cfg.data_root / "incoming"
    src.mkdir()
    (src / "a.png").write_bytes(b"x")
    (src / "b.png").write_bytes(b"x")
    paths = import_images([src / "a.png"], "bottle", "ok", None, cfg)
    assert paths[0].parent.name == "good" and "train" not in str(paths[0])
    paths = import_images([src / "b.png"], "bottle", "ng", "scratch", cfg)
    assert paths[0].parent.name == "scratch"
    info = dataset_info("bottle", cfg)
    assert info.defect_types["scratch"] == 1


def test_import_images_ng_without_defect_type_raises(cfg):
    src = cfg.data_root / "incoming2"
    src.mkdir()
    (src / "a.png").write_bytes(b"x")
    with pytest.raises(DinoError, match="缺陷类型"):
        import_images([src / "a.png"], "bottle", "ng", None, cfg)


def test_list_datasets(cfg):
    rows = list_datasets(cfg)
    assert len(rows) == 1 and rows[0].category == "bottle"


def test_ok_calibration_images_uses_test_good(cfg):
    imgs = ok_calibration_images("bottle", cfg)
    assert len(imgs) == 2 and all("test" in str(p) for p in imgs)


def test_ok_calibration_images_fallback_split_20pct(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    for i in range(10):
        _mkimg(cfg.data_root / "c" / "train" / "good" / f"{i}.png")
    for i in range(2):
        _mkimg(cfg.data_root / "c" / "test" / "broken" / f"{i}.png")  # 无 test/good
    imgs = ok_calibration_images("c", cfg)
    assert len(imgs) == 2  # 10 × 20%
    # 确定性：两次调用结果一致
    assert imgs == ok_calibration_images("c", cfg)


def test_test_images_with_labels(cfg):
    rows = test_images_with_labels("bottle", cfg)
    labels = {p.name: (lbl, dt) for p, lbl, dt in rows}
    assert labels["g0.png"] == (0, "good")
    assert labels["b0.png"] == (1, "broken")
    assert labels["c0.png"] == (1, "contamination")


def test_build_folder_passes_defect_dir_list(cfg):
    dm = build_folder("bottle", cfg)
    # Folder 接受 Sequence[Path]；验证 abnormal_dir 覆盖全部缺陷类型
    assert isinstance(dm.abnormal_dir, list)
    assert len(dm.abnormal_dir) == 2
    assert dm.normal_split_ratio == 0.0  # 有 test/good 时不切分


def test_build_folder_split_ratio_when_no_test_good(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    for i in range(5):
        _mkimg(cfg.data_root / "c" / "train" / "good" / f"{i}.png")
        _mkimg(cfg.data_root / "c" / "test" / "broken" / f"{i}.png")
    dm = build_folder("c", cfg)
    assert dm.normal_split_ratio == 0.2

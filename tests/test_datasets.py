import pytest

from dino_exp.config import Config
from dino_exp.datasets import (
    build_folder,
    dataset_info,
    import_images,
    list_datasets,
    mask_path_for,
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


def test_mask_path_for(cfg):
    info = dataset_info("bottle", cfg)
    root = info.root
    # 有 mask 的 NG 图：按 MVTec 约定 mask/<defect_type>/<stem>_mask.png
    assert mask_path_for(root / "test" / "broken" / "b0.png", "broken", info) == (
        root / "mask" / "broken" / "b0_mask.png"
    )
    # 无 mask 文件的 NG 图 / good 图 → None
    assert mask_path_for(root / "test" / "broken" / "b1.png", "broken", info) is None
    assert mask_path_for(root / "test" / "good" / "g0.png", "good", info) is None


def test_build_folder_passes_defect_dir_list(cfg):
    dm = build_folder("bottle", cfg)
    # Folder 接受 Sequence[Path]；验证 abnormal_dir 覆盖全部缺陷类型
    assert isinstance(dm.abnormal_dir, list)
    assert len(dm.abnormal_dir) == 2
    # 有 test/good：不从 train/good 切分，train 数不变
    dm.setup("fit")
    assert len(dm.train_data) == 5


def test_build_folder_split_ratio_when_no_test_good(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    for i in range(5):
        _mkimg(cfg.data_root / "c" / "train" / "good" / f"{i}.png")
        _mkimg(cfg.data_root / "c" / "test" / "broken" / f"{i}.png")
    dm = build_folder("c", cfg)
    # 无 test/good：anomalib 按 test_split_ratio=0.2 从 train/good 切出 test 正常部分
    # floor(5×0.8)=4 / floor(5×0.2)=1 → train 4 + test 正常 1
    dm.setup("fit")
    assert len(dm.train_data) == 4
    normal_in_test = dm.test_data.samples[dm.test_data.samples.label_index == 0]
    assert len(normal_in_test) == 1
    # 确定性：同 seed 两次切分结果一致
    dm2 = build_folder("c", cfg)
    dm2.setup("fit")
    assert list(normal_in_test.image_path) == list(
        dm2.test_data.samples[dm2.test_data.samples.label_index == 0].image_path
    )


def test_list_datasets_skips_incomplete_category(cfg):
    bad = cfg.data_root / "broken_cat"
    (bad / "test" / "good").mkdir(parents=True)
    with pytest.warns(UserWarning, match="跳过不完整类别 'broken_cat'"):
        rows = list_datasets(cfg)
    assert [r.category for r in rows] == ["bottle"]


def test_build_folder_warns_when_mask_partial(cfg):
    # bottle 只有 broken 有 mask，contamination 缺 mask → 整体弃用并告警
    with pytest.warns(UserWarning, match="contamination 缺少 mask 目录.*整体弃用"):
        dm = build_folder("bottle", cfg)
    assert dm.mask_dir is None


def test_import_images_atomic_on_conflict(cfg):
    src = cfg.data_root / "incoming3"
    src.mkdir()
    (src / "new.png").write_bytes(b"x")
    (src / "g0.png").write_bytes(b"x")  # 与 test/good/g0.png 重名
    with pytest.raises(DinoError, match="目标已存在"):
        import_images([src / "new.png", src / "g0.png"], "bottle", "ok", None, cfg)
    # 第一张也不应落盘（全量校验先于拷贝）
    assert not (cfg.data_root / "bottle" / "test" / "good" / "new.png").exists()


def test_import_mvtec_existing_dest_fails_fast_without_download(tmp_path, monkeypatch):
    """目标已存在且无 --force：先报错，不触发 4.9GB 下载。"""
    from dino_exp import datasets as ds

    cfg = Config(data_root=tmp_path / "data")
    (cfg.data_root / "bottle").mkdir(parents=True)
    called = []
    monkeypatch.setattr("anomalib.data.MVTecAD", lambda *a, **k: called.append(1))
    with pytest.raises(DinoError, match="--force"):
        ds.import_mvtec("bottle", cfg)
    assert not called  # 未触发下载（先校验后下载）


def test_import_mvtec_force_replaces_existing(tmp_path, monkeypatch):
    """--force：删除已存在目录（含中断半成品）后重新导入。"""
    from dino_exp import datasets as ds

    cfg = Config(data_root=tmp_path / "data")
    (cfg.data_root / "bottle" / "stale").mkdir(parents=True)  # 上次中断的半成品
    src = tmp_path / "mvtec" / "bottle"
    (src / "train" / "good").mkdir(parents=True)
    (src / "train" / "good" / "t0.png").write_bytes(b"x")
    (src / "test" / "good").mkdir(parents=True)
    (src / "test" / "good" / "g0.png").write_bytes(b"x")
    monkeypatch.setattr(ds, "MVTEC_ROOT", tmp_path / "mvtec")

    class FakeDM:
        def __init__(self, *a, **k):
            pass

        def prepare_data(self):
            pass

    monkeypatch.setattr("anomalib.data.MVTecAD", FakeDM)
    dest = ds.import_mvtec("bottle", cfg, force=True)
    assert not (dest / "stale").exists()  # 半成品已清除
    assert (dest / "train" / "good" / "t0.png").exists()

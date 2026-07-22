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


def test_import_images_ng_without_defect_type_defaults_unknown(cfg):
    src = cfg.data_root / "incoming2"
    src.mkdir()
    (src / "a.png").write_bytes(b"x")
    paths = import_images([src / "a.png"], "bottle", "ng", None, cfg)
    assert paths[0].parent.name == "unknown"  # 缺省缺陷类型 → test/unknown/


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


def test_list_datasets_includes_incomplete_category_with_error(cfg):
    bad = cfg.data_root / "broken_cat"
    (bad / "test" / "good").mkdir(parents=True)
    rows = list_datasets(cfg)
    assert [r.category for r in rows] == ["bottle", "broken_cat"]
    bad_row = rows[1]
    assert bad_row.error is not None and "train/good" in bad_row.error
    assert bad_row.train_good == 0
    assert rows[0].error is None  # 完整类别无 error


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


def test_import_images_split_train(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.png").write_bytes(b"x")
    paths = import_images([src / "a.png"], "c", "ok", None, cfg, split="train")
    assert paths[0] == cfg.data_root / "c" / "train" / "good" / "a.png"
    assert paths[0].exists()


def test_import_images_split_auto_80_20(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    src = tmp_path / "in"
    src.mkdir()
    for i in range(10):
        (src / f"{i:02d}.png").write_bytes(b"x")
    paths = import_images(sorted(src.iterdir()), "c", "ok", None, cfg, split="auto")
    train = [p for p in paths if p.parent.parent.name == "train"]
    test = [p for p in paths if p.parent.parent.name == "test"]
    assert len(train) == 8 and len(test) == 2  # 10 × 80% = 8
    # 确定性：按文件名排序，前 8 张进 train
    assert all(int(p.stem) < 8 for p in train)
    assert all(int(p.stem) >= 8 for p in test)


def test_import_images_split_invalid_raises(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    with pytest.raises(DinoError, match="train/test/auto"):
        import_images([src], "c", "ok", None, cfg, split="bad")


def test_category_images_returns_relative_and_absolute(cfg):
    from dino_exp.datasets import category_images

    imgs = category_images("bottle", cfg)
    rels = [rel for rel, _ in imgs]
    assert "train/good/t0.png" in rels
    assert "test/good/g0.png" in rels
    assert "test/broken/b0.png" in rels
    assert rels == sorted(rels)
    abs_map = dict(imgs)
    assert abs_map["test/good/g0.png"].is_absolute() or str(abs_map["test/good/g0.png"]).startswith(str(cfg.data_root))


def test_delete_category(tmp_path):
    from dino_exp.datasets import delete_category

    cfg = Config(data_root=tmp_path / "data")
    d = cfg.data_root / "c" / "train" / "good"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")
    delete_category("c", cfg)
    assert not (cfg.data_root / "c").exists()


def test_delete_category_missing_raises(tmp_path):
    from dino_exp.datasets import delete_category

    cfg = Config(data_root=tmp_path / "data")
    cfg.data_root.mkdir()
    with pytest.raises(DinoError, match="不存在"):
        delete_category("ghost", cfg)


def test_delete_category_rejects_traversal(tmp_path):
    from dino_exp.datasets import delete_category

    cfg = Config(data_root=tmp_path / "data")
    cfg.data_root.mkdir()
    for bad in ["../x", "a/b", "a\\b", "", ".."]:
        with pytest.raises(DinoError, match="非法类别名"):
            delete_category(bad, cfg)


def test_fix_category_moves_80_20(tmp_path):
    from dino_exp.datasets import fix_category

    cfg = Config(data_root=tmp_path / "data")
    good = cfg.data_root / "c" / "test" / "good"
    good.mkdir(parents=True)
    for i in range(10):
        (good / f"{i:02d}.png").write_bytes(b"x")
    r = fix_category("c", cfg)
    assert r["moved_to_train"] == 8 and r["kept_in_test"] == 2
    assert len(list((cfg.data_root / "c" / "train" / "good").iterdir())) == 8
    assert len(list(good.iterdir())) == 2
    # 修复后类别完整
    from dino_exp.datasets import dataset_info

    assert dataset_info("c", cfg).train_good == 8


def test_fix_category_already_complete(tmp_path):
    from dino_exp.datasets import fix_category

    cfg = Config(data_root=tmp_path / "data")
    d = cfg.data_root / "c" / "train" / "good"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")
    r = fix_category("c", cfg)
    assert r["moved_to_train"] == 0 and "无需处理" in r["note"]


def test_fix_category_no_images_raises(tmp_path):
    from dino_exp.datasets import fix_category

    cfg = Config(data_root=tmp_path / "data")
    (cfg.data_root / "c").mkdir(parents=True)
    with pytest.raises(DinoError, match="无法自动修复"):
        fix_category("c", cfg)


def test_import_images_from_zip_auto_split(tmp_path):
    import zipfile

    cfg = Config(data_root=tmp_path / "data")
    zpath = tmp_path / "imgs.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(10):
            zf.writestr(f"sub/dir/{i:02d}.png", b"x")  # zip 内含嵌套目录
        zf.writestr("readme.txt", "not an image")  # 非图片应被忽略
    paths = import_images([zpath], "c", "ok", None, cfg, split="auto")
    train = [p for p in paths if p.parent.parent.name == "train"]
    test = [p for p in paths if p.parent.parent.name == "test"]
    assert len(paths) == 10 and len(train) == 8 and len(test) == 2


def test_import_images_zip_no_images_raises(tmp_path):
    import zipfile

    cfg = Config(data_root=tmp_path / "data")
    zpath = tmp_path / "empty.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.txt", "x")
    with pytest.raises(DinoError, match="没有图片"):
        import_images([zpath], "c", "ok", None, cfg)


def test_import_images_invalid_zip_raises(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DinoError, match="不是有效的 zip"):
        import_images([bad], "c", "ok", None, cfg)


def test_category_cache_invalidation(tmp_path):
    """导入后缓存失效，类别列表立即反映新类别（目录仍是唯一事实源）。"""
    from dino_exp.datasets import dataset_categories

    cfg = Config(data_root=tmp_path / "data")
    cfg.data_root.mkdir()
    assert dataset_categories(cfg) == []
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    import_images([src], "newcat", "ok", None, cfg, split="train")
    assert "newcat" in dataset_categories(cfg)

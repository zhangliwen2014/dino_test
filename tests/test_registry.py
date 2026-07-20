import json

import pytest
import torch

from dino_exp.errors import DinoError
from dino_exp.models.registry import Registry


def _banks(tmp_path):
    nb = tmp_path / "nb.pt"
    torch.save(torch.randn(8, 4), nb)
    return nb


def _create(reg, tmp_path, parent=None, metrics=None):
    return reg.create_version(
        "bottle",
        normal_bank=_banks(tmp_path),
        defect_bank=None,
        checkpoint=None,
        config={"backbone": "dinov2_vits14"},
        metrics=metrics or {"image_AUROC": 0.95, "threshold": 1.5},
        meta={"parent": parent, "feedback_applied": 0},
    )


def test_create_first_version(tmp_path):
    reg = Registry(tmp_path / "models")
    v = _create(reg, tmp_path)
    assert v == "v001"
    assert reg.current("bottle") == "v001"
    d = reg.version_dir("bottle", "v001")
    assert (d / "normal_bank.pt").exists()
    assert (d / "defect_bank.pt").exists()
    assert json.loads((d / "metrics.json").read_text())["threshold"] == 1.5
    assert json.loads((d / "meta.json").read_text())["parent"] is None
    # current 是普通文本指针文件，不是符号链接
    cur = tmp_path / "models" / "bottle" / "current"
    assert cur.is_file() and not cur.is_symlink()
    assert cur.read_text().strip() == "v001"


def test_versions_increment_and_parent(tmp_path):
    reg = Registry(tmp_path / "models")
    v1 = _create(reg, tmp_path)
    v2 = _create(reg, tmp_path, parent=v1, metrics={"image_AUROC": 0.96, "threshold": 1.6})
    assert v2 == "v002"
    assert reg.list("bottle") == ["v001", "v002"]
    assert reg.current("bottle") == "v002"


def test_switch_and_rollback(tmp_path):
    reg = Registry(tmp_path / "models")
    v1 = _create(reg, tmp_path)
    _create(reg, tmp_path, parent=v1)
    reg.switch("bottle", "v001")
    assert reg.current("bottle") == "v001"
    reg.rollback("bottle", "v002")
    assert reg.current("bottle") == "v002"


def test_switch_missing_version_raises(tmp_path):
    reg = Registry(tmp_path / "models")
    _create(reg, tmp_path)
    with pytest.raises(DinoError, match="不存在"):
        reg.switch("bottle", "v009")


def test_failed_write_does_not_break_current(tmp_path):
    reg = Registry(tmp_path / "models")
    v1 = _create(reg, tmp_path)
    # 用一个会失败的 normal_bank 路径模拟写入中途失败
    with pytest.raises(DinoError, match="写入失败"):
        reg.create_version(
            "bottle",
            normal_bank=tmp_path / "does_not_exist.pt",
            defect_bank=None, checkpoint=None,
            config={}, metrics={}, meta={"parent": v1, "feedback_applied": 0},
        )
    assert reg.current("bottle") == "v001"  # 指针未被破坏
    assert reg.list("bottle") == ["v001"]   # 半成品已清理
    assert not list((tmp_path / "models" / "bottle").glob(".tmp-*"))


def test_read_paths_do_not_create_dirs(tmp_path):
    """list/current/version_dir 是纯读路径：不为 typo 类别名造空目录（UI 轮询安全）。"""
    reg = Registry(tmp_path / "models")
    assert reg.list("typo") == []
    assert reg.current("typo") is None
    with pytest.raises(DinoError, match="不存在"):
        reg.version_dir("typo", "v001")
    assert not (tmp_path / "models" / "typo").exists()
    assert not (tmp_path / "models").exists()

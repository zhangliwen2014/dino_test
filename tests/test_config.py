import pytest

from dino_exp.config import (
    BACKBONE_ALIASES,
    Config,
    load_config,
    resolve_backbone,
    validate_image_size,
)
from dino_exp.errors import DinoError


def test_alias_table_contains_defaults():
    spec = resolve_backbone("dinov2_vits14")
    assert spec.timm_name == "vit_small_patch14_dinov2.lvd142m"
    assert spec.patch_size == 14
    assert spec.default_layers == ["blocks.11"]  # spike(Task 2)确认后可改
    spec3 = resolve_backbone("dinov3_vits16")
    assert spec3.timm_name == "vit_small_patch16_dinov3.lvd1689m"
    assert spec3.patch_size == 16


def test_resolve_backbone_accepts_raw_timm_name():
    spec = resolve_backbone("vit_small_patch14_dinov2.lvd142m")
    assert spec.patch_size == 14


def test_resolve_backbone_unknown_raises_with_hint():
    with pytest.raises(DinoError, match="可用别名"):
        resolve_backbone("resnet18")


def test_validate_image_size_ok():
    validate_image_size(224, 14)  # 16*14
    validate_image_size(518, 14)


def test_validate_image_size_bad_raises_with_fix_hint():
    with pytest.raises(DinoError, match="14 的整数倍"):
        validate_image_size(256, 14)
    with pytest.raises(DinoError, match="16 的整数倍"):
        validate_image_size(220, 16)  # 224 恰为 14*16，改用 220


def test_load_config_defaults(tmp_path):
    cfg = load_config(tmp_path / "nonexistent.yaml")  # 文件不存在 → 全默认
    assert cfg.backbone == "dinov2_vits14"
    assert cfg.image_size == 224
    assert cfg.coreset_sampling_ratio == 0.1
    assert cfg.fusion_weight == 0.5
    assert cfg.bank_cap_ratio == 1.5
    assert cfg.defect_topk == 10
    assert cfg.threshold_sigma == 3.0


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("image_size: 224\nfusion_weight: 0.7\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.fusion_weight == 0.7
    assert cfg.image_size == 224


def test_load_config_rejects_bad_combo(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("image_size: 256\n", encoding="utf-8")  # 256 不是 14 的倍数
    with pytest.raises(DinoError):
        load_config(p)


def test_alias_table_complete():
    expected = {
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
        "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16",
    }
    assert set(BACKBONE_ALIASES) == expected

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


def test_load_config_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 默认路径 config/default.yaml 不存在 → 全默认
    cfg = load_config(None)
    assert cfg.backbone == "dinov2_vits14"
    assert cfg.image_size == 224
    assert cfg.coreset_sampling_ratio == 0.1
    assert cfg.fusion_weight == 0.5
    assert cfg.bank_cap_ratio == 1.5
    assert cfg.defect_topk == 10
    assert cfg.threshold_sigma == 3.0


def test_load_config_explicit_missing_path_raises(tmp_path):
    with pytest.raises(DinoError, match="不存在"):
        load_config(tmp_path / "nonexistent.yaml")


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


def test_load_config_rejects_unknown_key_colliding_with_property(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("backbone_spec: x\n", encoding="utf-8")  # 撞到 property 名，hasattr 判不出
    with pytest.raises(DinoError, match="未知键"):
        load_config(p)


def test_load_config_rejects_bad_layer_format(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("layers: [blocks.11x]\n", encoding="utf-8")
    with pytest.raises(DinoError, match="blocks"):
        load_config(p)


def test_load_config_rejects_non_list_layers(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("layers: blocks.11\n", encoding="utf-8")
    with pytest.raises(DinoError, match="字符串列表"):
        load_config(p)


def test_load_config_rejects_layer_beyond_depth(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("backbone: dinov2_vits14\nlayers: [blocks.23]\n", encoding="utf-8")  # s 深度仅 12
    with pytest.raises(DinoError, match="深度 12"):
        load_config(p)


def test_load_config_rejects_out_of_range_values(tmp_path):
    for body in [
        "coreset_sampling_ratio: 0",
        "coreset_sampling_ratio: 1.5",
        "fusion_weight: -0.1",
        "defect_topk: 0",
        "num_neighbors: 0",
        "train_batch_size: 0",
        "eval_batch_size: 0",
    ]:
        p = tmp_path / "c.yaml"
        p.write_text(body + "\n", encoding="utf-8")
        with pytest.raises(DinoError, match="非法"):
            load_config(p)


def test_validate_image_size_non_positive():
    with pytest.raises(DinoError, match="正整数"):
        validate_image_size(0, 14)
    with pytest.raises(DinoError, match="正整数"):
        validate_image_size(-224, 14)


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(DinoError, match="mapping"):
        load_config(p)


def test_setup_logging_creates_file(tmp_path):
    import logging

    import dino_exp.logs as logs_mod

    # 重置全局状态（其他测试可能已初始化过默认日志目录）
    logger = logging.getLogger("dino_exp")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logs_mod._configured = False

    log_file = logs_mod.setup_logging(tmp_path / "logs")
    logs_mod.get_logger("test").info("hello-log-test")
    for h in logger.handlers:
        h.flush()
    assert log_file.exists()
    assert "hello-log-test" in log_file.read_text(encoding="utf-8")


def test_resolve_device_auto(monkeypatch):
    import torch

    from dino_exp.config import Config, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(Config(device="auto")) == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(Config(device="auto")) == "cpu"


def test_resolve_device_explicit(monkeypatch):
    import torch

    from dino_exp.config import Config, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(Config(device="cpu")) == "cpu"
    assert resolve_device(Config(device="cuda")) == "cuda"


def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    import torch

    from dino_exp.config import Config, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DinoError, match="CPU 版"):
        resolve_device(Config(device="cuda"))


def test_load_config_device_default_and_validate(tmp_path):
    from dino_exp.config import Config, load_config

    assert Config().device == "auto"
    p = tmp_path / "c.yaml"
    p.write_text("device: tpu\n", encoding="utf-8")
    with pytest.raises(DinoError, match="auto/cpu/cuda"):
        load_config(p)

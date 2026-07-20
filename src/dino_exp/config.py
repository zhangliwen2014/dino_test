from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from dino_exp.errors import DinoError

_LAYER_RE = re.compile(r"^blocks\.\d+$")


@dataclass(frozen=True)
class BackboneSpec:
    alias: str
    timm_name: str
    patch_size: int
    default_layers: list[str]
    depth: int


BACKBONE_ALIASES: dict[str, BackboneSpec] = {
    "dinov2_vits14": BackboneSpec("dinov2_vits14", "vit_small_patch14_dinov2.lvd142m", 14, ["blocks.11"], 12),
    "dinov2_vitb14": BackboneSpec("dinov2_vitb14", "vit_base_patch14_dinov2.lvd142m", 14, ["blocks.11"], 12),
    "dinov2_vitl14": BackboneSpec("dinov2_vitl14", "vit_large_patch14_dinov2.lvd142m", 14, ["blocks.23"], 24),
    "dinov3_vits16": BackboneSpec("dinov3_vits16", "vit_small_patch16_dinov3.lvd1689m", 16, ["blocks.11"], 12),
    "dinov3_vitb16": BackboneSpec("dinov3_vitb16", "vit_base_patch16_dinov3.lvd1689m", 16, ["blocks.11"], 12),
    "dinov3_vitl16": BackboneSpec("dinov3_vitl16", "vit_large_patch16_dinov3.lvd1689m", 16, ["blocks.23"], 24),
}


def resolve_backbone(alias_or_timm: str) -> BackboneSpec:
    if alias_or_timm in BACKBONE_ALIASES:
        return BACKBONE_ALIASES[alias_or_timm]
    for spec in BACKBONE_ALIASES.values():
        if spec.timm_name == alias_or_timm:
            return spec
    raise DinoError(
        f"未知骨干 '{alias_or_timm}'。可用别名: {', '.join(sorted(BACKBONE_ALIASES))}，"
        "或直接传上表中的 timm 模型名。"
    )


def validate_image_size(image_size: int, patch_size: int) -> None:
    if image_size <= 0:
        raise DinoError(f"输入尺寸 {image_size} 必须为正整数。请改为 patch 尺寸 {patch_size} 的整数倍（如 {patch_size * 16}）。")
    if image_size % patch_size != 0:
        raise DinoError(
            f"输入尺寸 {image_size} 不是骨干 patch 尺寸 {patch_size} 的整数倍。"
            f"请改为 {patch_size} 的整数倍（如 {(image_size // patch_size) * patch_size} 或 {(image_size // patch_size + 1) * patch_size}）。"
        )


@dataclass
class Config:
    backbone: str = "dinov2_vits14"
    layers: list[str] = field(default_factory=lambda: ["blocks.11"])
    image_size: int = 224
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    fusion_weight: float = 0.5
    bank_cap_ratio: float = 1.5
    defect_topk: int = 10
    threshold_sigma: float = 3.0
    suspicious_score_factor: float = 3.0
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0
    data_root: Path = Path("data")
    models_root: Path = Path("models")
    feedback_root: Path = Path("feedback")

    @property
    def backbone_spec(self) -> BackboneSpec:
        return resolve_backbone(self.backbone)


def _validate_layers(layers: object, spec: BackboneSpec) -> None:
    if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
        raise DinoError(f"layers 须为字符串列表（如 [blocks.11]），得到 {layers!r}。请对照 config/default.yaml 修正。")
    for layer in layers:
        if not _LAYER_RE.match(layer):
            raise DinoError(f"ViT 骨干 layers 须为 'blocks.<int>' 形式，得到 '{layer}'。请改为如 blocks.11。")
        if int(layer.split(".")[1]) >= spec.depth:
            raise DinoError(
                f"{layer} 超出 {spec.alias} 深度 {spec.depth}，请改用 blocks.0~blocks.{spec.depth - 1}。"
            )


def _validate_numeric_ranges(cfg: Config) -> None:
    checks = [
        ("coreset_sampling_ratio", cfg.coreset_sampling_ratio, lambda v: 0 < v <= 1, "须在 (0, 1] 区间（如 0.1）"),
        ("fusion_weight", cfg.fusion_weight, lambda v: v >= 0, "须 >= 0（如 0.5）"),
        ("defect_topk", cfg.defect_topk, lambda v: v >= 1, "须 >= 1（如 10）"),
        ("num_neighbors", cfg.num_neighbors, lambda v: v >= 1, "须 >= 1（如 9）"),
        ("train_batch_size", cfg.train_batch_size, lambda v: v >= 1, "须 >= 1（如 16）"),
        ("eval_batch_size", cfg.eval_batch_size, lambda v: v >= 1, "须 >= 1（如 16）"),
    ]
    for name, value, ok, hint in checks:
        if not ok(value):
            raise DinoError(f"配置项 {name}={value!r} 非法，{hint}。")


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        p = Path("config/default.yaml")
        if not p.exists():
            return cfg  # 默认文件不存在 → 全默认
    else:
        p = Path(path)
        if not p.exists():
            raise DinoError(f"配置文件 {p} 不存在，请检查路径。")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DinoError(f"配置文件 {p} 顶层须为键值映射（mapping），得到 {type(raw).__name__}。请对照 config/default.yaml 修正。")
    valid = {f.name for f in fields(cfg)}
    for key, value in raw.items():
        if key not in valid:
            raise DinoError(f"配置文件 {p} 含未知键 '{key}'。请对照 config/default.yaml 修正。")
        if key.endswith("_root"):
            value = Path(value)
        setattr(cfg, key, value)
    spec = cfg.backbone_spec  # 校验骨干别名
    validate_image_size(cfg.image_size, spec.patch_size)
    _validate_layers(cfg.layers, spec)
    _validate_numeric_ranges(cfg)
    return cfg

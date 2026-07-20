# DINO 无监督异常检测试验环境 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows 本机构建基于 DINOv2/DINOv3 特征的无监督工业异常检测试验环境：OK 图训练（冻结骨干 + PatchCore 记忆库 + 缺陷原型库双库方案）→ OK/NG 判定 + 热力图 → 人工反馈 → 增量再训练 → 版本化可回滚，CLI + Gradio 双入口。
**Architecture:** 交互层（click CLI + Gradio 四页签，薄封装）→ 应用层（datasets / validation / feedback / registry / train / retrain / infer）→ 引擎层（anomalib Engine + DualBankPatchcore 子类）→ 推理后端（PyTorch 优先，OpenVINO 版本快照可选）。设计文档：`docs/superpowers/specs/2026-07-20-dino-anomaly-env-design.md`。
**Tech Stack:** anomalib==2.5.1, timm, torch, click, gradio, pytest, PyYAML

## 已核实的 anomalib v2.5.1 API 要点（本计划代码的依据）

以下来自 `lib/v2.5.1` 标签源码逐行核实，执行时不得凭记忆替换：

- `Patchcore(backbone, layers, pre_trained, coreset_sampling_ratio, num_neighbors, precision, pre_processor, post_processor, evaluator, visualizer)`；内部 `self.model = PatchcoreModel(layers, backbone, pre_trained, num_neighbors)`。
- `PatchcoreModel`：`memory_bank` 为注册 buffer；`forward` 训练态把 embedding 追加到 `self.embedding_store`；eval 态返回 `InferenceBatch(pred_score=, anomaly_map=)`；打分路径 `forward → nearest_neighbors(embedding, n_neighbors=1) → compute_anomaly_score`；`subsample_embedding` 自 2.1 起 deprecated，用 `KCenterGreedy(embedding=..., sampling_ratio=...).sample_coreset() -> torch.Tensor`。
- Lightning 层 `Patchcore.fit()` 由 `MemoryBankMixin.on_validation_start/on_train_epoch_end` 自动触发；`configure_pre_processor(image_size=(h, w))` 为 classmethod。
- ViT 骨干：`TimmFeatureExtractor` 对名字含 "vit" 的骨干自动走 `forward_intermediates`，默认 NCHW reshape 输出 4D 特征图；layers 写 `blocks.<int>`。
- **指标**：v2.5.1 的 `Engine.__init__` 只有 `callbacks / logger / default_root_dir`——设计文档 §1 中 `Engine(image_metrics=[...])` 的写法在 2.5.1 不存在。指标经 `anomalib.metrics.Evaluator(val_metrics=, test_metrics=)` 传入模型构造函数（`Patchcore(evaluator=...)`）。本计划按核实后的写法实现（设计文档此行可在后续修订）。
- **阈值注入**：`PostProcessor` 内部用 `_image_threshold_metric = F1AdaptiveThreshold(fields=["pred_score","gt_label"], strict=False)` 与 `_pixel_threshold_metric = F1AdaptiveThreshold(fields=["anomaly_map","gt_mask"], strict=False)`，validation epoch 结束时把 `compute()` 拷入 `_image_threshold`/`_pixel_threshold` buffer（`image_threshold` property 读取该 buffer）。注入 mean+3σ = 把两个 metric 换成 `anomalib.metrics.ManualThreshold(default_value=t, fields=..., strict=False)` 并同时直接写 buffer。
- `Folder(name, root, normal_dir, abnormal_dir=None, normal_test_dir=None, mask_dir=None, normal_split_ratio=0.2, train_batch_size=32, eval_batch_size=32, num_workers=8, ...)`；`normal_dir/abnormal_dir/normal_test_dir/mask_dir` 均接受 `str | Path | Sequence`，多缺陷目录直接传路径列表（这就是设计文档"manifest"的具体实现形式，无需硬链接视图）。
- `engine.export(model, export_type="openvino", export_root=..., input_size=(h, w), ...)`，`export_type` 接受字符串；OpenVINO 需 `pip install "anomalib[openvino]"`。
- `AnomalyDINO(encoder_name="vit_small_patch14_dinov2", ...)`（MemoryBankMixin 子类，spike 对照用）。
- `MVTecAD(root="./datasets/MVTecAD", category="bottle", ...)`，`prepare_data()` 时自动下载解压。

## 全局约定

- 包名 `dino_exp`，src 布局；所有路径操作用 `pathlib.Path`；Windows 默认 `num_workers=0`。
- 应用层统一异常：`DinoError(Exception)`，报错信息附修复建议（NFR-4）。
- 每个 Task 完成后按 Step 中的命令提交；commit message 前缀 `feat:`/`test:`/`chore:`。
- 单元测试不实例化真实骨干（不下载权重）；双库逻辑抽成纯函数测试，nn.Module 方法为薄封装。真实模型只出现在 spike 脚本与 `pytest -m slow` 冒烟测试中。

---

## Task 1: 项目脚手架与配置层

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.yaml`
- Create: `src/dino_exp/__init__.py`
- Create: `src/dino_exp/errors.py`
- Create: `src/dino_exp/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
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
        validate_image_size(224, 16)


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
```

- [ ] **Step 2: 运行确认失败**

```bash
pip install -e ".[dev]" 2>/dev/null || true   # pyproject 尚未存在时跳过
python -m pytest tests/test_config.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp'`（或收集失败）。

- [ ] **Step 3: 写 `pyproject.toml`、`config/default.yaml`、`errors.py`**

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "dino-exp"
version = "0.1.0"
description = "DINO 无监督异常检测试验环境"
requires-python = ">=3.10,<3.13"
dependencies = [
    "anomalib[cpu]==2.5.1",
    "click>=8.1",
    "gradio>=5.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
openvino = ["anomalib[openvino]==2.5.1"]

[project.scripts]
dino = "dino_exp.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = ["slow: 需要真实骨干权重与较长运行时间的测试"]
```

`config/default.yaml`：

```yaml
# DINO 异常检测默认配置
backbone: dinov2_vits14        # 别名，见 dino_exp.config.BACKBONE_ALIASES
layers: [blocks.11]            # ViT 特征层，spike(scripts/spike_backbone.py)确认后写回
image_size: 224                # 必须为骨干 patch 尺寸整数倍(DINOv2=14, DINOv3=16)
coreset_sampling_ratio: 0.1
num_neighbors: 9
fusion_weight: 0.5             # 缺陷库加分权重 w
bank_cap_ratio: 1.5            # 正常库上限 = 基础版本库大小 × 此倍数(钉住特征除外)
defect_topk: 10                # NG 反馈取 top-k 高分 patch
threshold_sigma: 3.0           # 阈值 = mean + sigma * std
suspicious_score_factor: 3.0   # 护栏: OK 反馈 score > 此倍数×阈值 视为可疑
train_batch_size: 16
eval_batch_size: 16
num_workers: 0                 # Windows 默认 0
data_root: data
models_root: models
feedback_root: feedback
```

`src/dino_exp/errors.py`：

```python
class DinoError(Exception):
    """应用层统一异常；message 必须附带修复建议（NFR-4）。"""
```

`src/dino_exp/__init__.py`：空文件。

- [ ] **Step 4: 写 `src/dino_exp/config.py` 最小实现**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dino_exp.errors import DinoError


@dataclass(frozen=True)
class BackboneSpec:
    alias: str
    timm_name: str
    patch_size: int
    default_layers: list[str]


BACKBONE_ALIASES: dict[str, BackboneSpec] = {
    "dinov2_vits14": BackboneSpec("dinov2_vits14", "vit_small_patch14_dinov2.lvd142m", 14, ["blocks.11"]),
    "dinov2_vitb14": BackboneSpec("dinov2_vitb14", "vit_base_patch14_dinov2.lvd142m", 14, ["blocks.11"]),
    "dinov2_vitl14": BackboneSpec("dinov2_vitl14", "vit_large_patch14_dinov2.lvd142m", 14, ["blocks.23"]),
    "dinov3_vits16": BackboneSpec("dinov3_vits16", "vit_small_patch16_dinov3.lvd1689m", 16, ["blocks.11"]),
    "dinov3_vitb16": BackboneSpec("dinov3_vitb16", "vit_base_patch16_dinov3.lvd1689m", 16, ["blocks.11"]),
    "dinov3_vitl16": BackboneSpec("dinov3_vitl16", "vit_large_patch16_dinov3.lvd1689m", 16, ["blocks.23"]),
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


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else Path("config/default.yaml")
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for key, value in raw.items():
            if not hasattr(cfg, key):
                raise DinoError(f"配置文件 {p} 含未知键 '{key}'。请对照 config/default.yaml 修正。")
            if key.endswith("_root"):
                value = Path(value)
            setattr(cfg, key, value)
    spec = cfg.backbone_spec  # 校验骨干别名
    validate_image_size(cfg.image_size, spec.patch_size)
    for layer in cfg.layers:
        if not layer.startswith("blocks."):
            raise DinoError(f"ViT 骨干 layers 须为 'blocks.<int>' 形式，得到 '{layer}'。请改为如 blocks.11。")
    return cfg
```

- [ ] **Step 5: 安装并运行测试确认通过**

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
python -m pytest tests/test_config.py -v
```

预期：8 个测试全过。注：首次 `pip install anomalib[cpu]==2.5.1` 较慢（torch CPU 轮），见文末「风险与依赖安装备注」。

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml config/default.yaml src/dino_exp/__init__.py src/dino_exp/errors.py src/dino_exp/config.py tests/test_config.py
git commit -m "feat: 项目脚手架与配置层（骨干别名表、image-size patch 倍数校验）"
```

---

## Task 2: Spike — 验证 Patchcore + DINOv2 骨干与层选择

**Files:**
- Create: `scripts/spike_backbone.py`

说明：spike 是一次性验证脚本（非 pytest），目的是回答两个问题并给出明确出口：(1) `Patchcore(backbone="vit_small_patch14_dinov2.lvd142m", layers=["blocks.9"] / ["blocks.11"])` 在 v2.5.1 能否跑通、特征维度多少；(2) AnomalyDINO 对照 baseline 能否跑通。出口 = 把确认后的默认层写回 `config/default.yaml` 与 `BACKBONE_ALIASES`。

- [ ] **Step 1: 写 `scripts/spike_backbone.py`**

```python
"""Spike: 验证 anomalib v2.5.1 Patchcore + DINOv2 骨干的层选择与可运行性。

运行（需联网下载 ~90MB DINOv2 权重，CPU 可跑）:
    python scripts/spike_backbone.py

预期输出（形状随 image_size=224 固定）:
    [Patchcore blocks.9]  feature shape: torch.Size([1, 384, 16, 16])  -> OK (4D)
    [Patchcore blocks.11] feature shape: torch.Size([1, 384, 16, 16])  -> OK (4D)
    [Patchcore blocks.9+11] embedding dim: 768 (= 384*2)
    [AnomalyDINO] encoder forward OK, feature shape: ...
    SPIKE RESULT: PASS
出口: 将选定的默认层写回 config/default.yaml 的 layers 与 dino_exp.config.BACKBONE_ALIASES。
"""

import torch
from anomalib.models import Patchcore


def check_patchcore(layers: list[str]) -> None:
    model = Patchcore(
        backbone="vit_small_patch14_dinov2.lvd142m",
        layers=layers,
        pre_trained=True,
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats = model.model.feature_extractor(x)
    for name, f in feats.items():
        assert f.dim() == 4, f"层 {name} 输出不是 4D 特征图: {f.shape}"
        print(f"[Patchcore {name}]  feature shape: {tuple(f.shape)}  -> OK (4D)")
    # 验证 generate_embedding + AvgPool 通路（Patchcore forward 的训练分支）
    emb = model.model(x)  # training 模式下返回 (B*H*W, D) embedding
    print(f"[Patchcore {'+'.join(layers)}] embedding dim: {emb.shape[-1]}")


def check_anomaly_dino() -> None:
    from anomalib.models import AnomalyDINO

    model = AnomalyDINO(encoder_name="vit_small_patch14_dinov2")
    x = torch.randn(1, 3, 224, 224)
    model.train()
    with torch.no_grad():
        out = model.model(x)
    print(f"[AnomalyDINO] encoder forward OK, output type: {type(out).__name__}")


if __name__ == "__main__":
    check_patchcore(["blocks.9"])
    check_patchcore(["blocks.11"])
    check_patchcore(["blocks.9", "blocks.11"])
    check_anomaly_dino()
    print("SPIKE RESULT: PASS")
```

- [ ] **Step 2: 运行 spike**

```bash
python scripts/spike_backbone.py
```

预期：打印 4D 特征形状（`(1, 384, 16, 16)`），末行 `SPIKE RESULT: PASS`。若某层报层名错误，用 `python -c "import timm; m=timm.create_model('vit_small_patch14_dinov2.lvd142m', features_only=True); print(m.feature_info.module_name())"` 列出可用层名后调整候选。

- [ ] **Step 3: 写回默认层并提交**

人工判断：单取 `blocks.11` 且 embedding 维度 384 满足 Patchcore 通路即保持默认；若 spike 显示 `blocks.9` 更稳（如末层异常），改 `config/default.yaml` 的 `layers` 与 `config.py` 中 `BACKBONE_ALIASES` 对应 `default_layers`，同步更新 `tests/test_config.py::test_alias_table_contains_defaults` 的断言。然后：

```bash
git add scripts/spike_backbone.py config/default.yaml src/dino_exp/config.py tests/test_config.py
git commit -m "test: spike 验证 Patchcore+DINOv2 骨干层选择并写回默认层"
```

---

## Task 3: 数据集管理 `datasets.py`

**Files:**
- Create: `src/dino_exp/datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 写失败测试 `tests/test_datasets.py`**

```python
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
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_datasets.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.datasets'`。

- [ ] **Step 3: 写 `src/dino_exp/datasets.py`**

```python
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dino_exp.config import Config
from dino_exp.errors import DinoError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MVTEC_ROOT = Path("datasets/MVTecAD")  # anomalib 自动下载根目录


@dataclass
class DatasetInfo:
    category: str
    root: Path
    train_good: int
    test_good: int
    defect_types: dict[str, int] = field(default_factory=dict)
    has_masks: bool = False

    @property
    def has_test_good(self) -> bool:
        return self.test_good > 0

    @property
    def degraded(self) -> bool:
        """无 NG 测试图：指标降级（仅阈值校准+逐图分数）。"""
        return sum(self.defect_types.values()) == 0


def _imgs(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def dataset_info(category: str, cfg: Config) -> DatasetInfo:
    root = cfg.data_root / category
    train_good = _imgs(root / "train" / "good")
    if not train_good:
        raise DinoError(
            f"类别 '{category}' 缺少 train/good 图片（{root / 'train' / 'good'}）。"
            "请先 `dino dataset download --category ...` 或 `dino dataset import ...` 导入 OK 训练图。"
        )
    test_dir = root / "test"
    defect_types = {
        d.name: len(_imgs(d))
        for d in sorted(test_dir.iterdir())
        if d.is_dir() and d.name != "good" and _imgs(d)
    } if test_dir.is_dir() else {}
    mask_root = root / "mask"
    has_masks = mask_root.is_dir() and any(_imgs(d) for d in mask_root.iterdir() if d.is_dir())
    return DatasetInfo(
        category=category,
        root=root,
        train_good=len(train_good),
        test_good=len(_imgs(test_dir / "good")),
        defect_types=defect_types,
        has_masks=has_masks,
    )


def list_datasets(cfg: Config) -> list[DatasetInfo]:
    if not cfg.data_root.is_dir():
        return []
    return [dataset_info(d.name, cfg) for d in sorted(cfg.data_root.iterdir()) if d.is_dir()]


def import_images(srcs: list[str | Path], category: str, label: str, defect_type: str | None, cfg: Config) -> list[Path]:
    label = label.lower()
    if label not in {"ok", "ng"}:
        raise DinoError(f"label 只能是 ok/ng，得到 '{label}'。")
    if label == "ng" and not defect_type:
        raise DinoError("NG 图片必须指定缺陷类型名（--defect-type），如 scratch。")
    dest_dir = cfg.data_root / category / "test" / ("good" if label == "ok" else defect_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for src in srcs:
        src = Path(src)
        if src.suffix.lower() not in IMG_EXTS:
            raise DinoError(f"不支持的图片格式 '{src.suffix}'。支持: {sorted(IMG_EXTS)}")
        dest = dest_dir / src.name
        if dest.exists():
            raise DinoError(f"目标已存在: {dest}。请重命名来源图片或先删除旧文件。")
        shutil.copy2(src, dest)
        out.append(dest)
    return out


def import_mvtec(category: str, cfg: Config) -> Path:
    """调用 anomalib MVTecAD 自动下载，再拷贝/重命名为统一目录规范。"""
    from anomalib.data import MVTecAD

    MVTecAD(root=str(MVTEC_ROOT), category=category).prepare_data()  # 触发下载解压
    src = MVTEC_ROOT / category
    if not src.is_dir():
        raise DinoError(f"MVTec 下载后未找到 {src}。请检查网络后重试 `dino dataset download`。")
    dest = cfg.data_root / category
    if dest.exists():
        raise DinoError(f"目标已存在: {dest}。如需重新导入请先删除该目录。")
    shutil.copytree(src / "train", dest / "train")
    shutil.copytree(src / "test", dest / "test")
    if (src / "ground_truth").is_dir():
        shutil.copytree(src / "ground_truth", dest / "mask")
    return dest


def ok_calibration_images(category: str, cfg: Config, ratio: float = 0.2) -> list[Path]:
    """OK 校准图：优先 test/good；缺失时从 train/good 确定性切 20%（FR-2.3）。"""
    info = dataset_info(category, cfg)
    good_test = _imgs(info.root / "test" / "good")
    if good_test:
        return good_test
    train = _imgs(info.root / "train" / "good")
    n = max(1, int(len(train) * ratio))
    rng = random.Random(42)  # 固定种子保证可复现
    return sorted(rng.sample(train, n))


def test_images_with_labels(category: str, cfg: Config) -> list[tuple[Path, int, str]]:
    """(path, label 0=OK/1=NG, defect_type) 全量 test 清单。"""
    info = dataset_info(category, cfg)
    rows = [(p, 0, "good") for p in _imgs(info.root / "test" / "good")]
    for dt in info.defect_types:
        rows += [(p, 1, dt) for p in _imgs(info.root / "test" / dt)]
    return rows


def build_folder(category: str, cfg: Config):
    """构造 anomalib Folder datamodule。

    多缺陷类型：Folder 的 abnormal_dir/mask_dir 接受 Sequence，直接传缺陷子目录
    路径列表（即设计文档"manifest"的实现形式）。缺 test/good 时 normal_split_ratio=0.2。
    """
    from anomalib.data import Folder

    info = dataset_info(category, cfg)
    defect_dirs = [f"test/{dt}" for dt in info.defect_types]
    mask_dirs = [f"mask/{dt}" for dt in info.defect_types if (info.root / "mask" / dt).is_dir()]
    kwargs: dict = {}
    if defect_dirs:
        kwargs["abnormal_dir"] = defect_dirs  # Sequence[str]：多缺陷类型合并（v2.5.1 已核实支持）
        # 仅当每个缺陷类型都有 mask 目录时才传 mask（保证与 abnormal_dir 对齐）
        kwargs["mask_dir"] = mask_dirs if len(mask_dirs) == len(defect_dirs) else None
    if info.has_test_good:
        kwargs["normal_test_dir"] = "test/good"
    return Folder(
        name=category,
        root=info.root,
        normal_dir="train/good",
        normal_split_ratio=0.0 if info.has_test_good else 0.2,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        **kwargs,
    )
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_datasets.py -v
```

预期：12 个测试全过（`build_folder` 两个用例会实例化 anomalib Folder，但不下载骨干、不读图片内容）。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/datasets.py tests/test_datasets.py
git commit -m "feat: 数据集管理（目录校验/导入/MVTec 下载/多缺陷 Folder 映射/降级标记）"
```

---

## Task 4: 版本库 `models/registry.py`

**Files:**
- Create: `src/dino_exp/models/__init__.py`
- Create: `src/dino_exp/models/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: 写失败测试 `tests/test_registry.py`**

```python
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
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_registry.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.models'`。

- [ ] **Step 3: 写 `src/dino_exp/models/__init__.py`（空）与 `src/dino_exp/models/registry.py`**

```python
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from dino_exp.errors import DinoError

_VERSION_RE = re.compile(r"^v(\d{3,})$")


class Registry:
    """模型版本库。目录结构见设计文档 §3.4；current 为指针文件（非符号链接）。

    原子性：所有内容先写入 <exp>/.tmp-<v>，成功后 os.rename 一次性替换，
    current 指针文件最后更新（同样走临时文件 + os.replace）。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _exp_dir(self, experiment: str) -> Path:
        d = self.root / experiment
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list(self, experiment: str) -> list[str]:
        d = self._exp_dir(experiment)
        return sorted(p.name for p in d.iterdir() if p.is_dir() and _VERSION_RE.match(p.name))

    def current(self, experiment: str) -> str | None:
        cur = self._exp_dir(experiment) / "current"
        return cur.read_text(encoding="utf-8").strip() if cur.exists() else None

    def version_dir(self, experiment: str, version: str) -> Path:
        d = self._exp_dir(experiment) / version
        if not d.is_dir():
            raise DinoError(f"版本 {experiment}/{version} 不存在。可用版本: {self.list(experiment)}")
        return d

    def _next_version(self, experiment: str) -> str:
        nums = [int(_VERSION_RE.match(v).group(1)) for v in self.list(experiment)]
        return f"v{(max(nums) + 1) if nums else 1:03d}"

    def create_version(
        self,
        experiment: str,
        *,
        normal_bank: str | Path,
        defect_bank: str | Path | None,
        checkpoint: str | Path | None,
        config: dict,
        metrics: dict,
        meta: dict,
    ) -> str:
        exp = self._exp_dir(experiment)
        version = self._next_version(experiment)
        tmp = exp / f".tmp-{version}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        try:
            shutil.copy2(normal_bank, tmp / "normal_bank.pt")
            if defect_bank is not None:
                shutil.copy2(defect_bank, tmp / "defect_bank.pt")
            else:
                torch.save(torch.empty(0), tmp / "defect_bank.pt")
            if checkpoint is not None:
                shutil.copy2(checkpoint, tmp / "checkpoint.ckpt")
            (tmp / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            full_meta = {"created_at": datetime.now(timezone.utc).isoformat(), **meta}
            (tmp / "meta.json").write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
            os.rename(tmp, exp / version)  # 同分区原子替换
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise DinoError(f"版本写入失败: {exc}。当前版本未受影响，请检查磁盘空间与路径后重试。") from exc
        self.switch(experiment, version)  # current 指针最后更新
        return version

    def switch(self, experiment: str, version: str) -> None:
        self.version_dir(experiment, version)  # 校验存在
        exp = self._exp_dir(experiment)
        tmp = exp / ".current.tmp"
        tmp.write_text(version + "\n", encoding="utf-8")
        os.replace(tmp, exp / "current")

    def rollback(self, experiment: str, version: str) -> None:
        """回滚 = 切换 current 指针；历史版本目录从不被修改。"""
        self.switch(experiment, version)
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_registry.py -v
```

预期：5 个测试全过。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/models/__init__.py src/dino_exp/models/registry.py tests/test_registry.py
git commit -m "feat: 模型版本库（原子写入、current 指针文件、切换/回滚）"
```

---

## Task 5: 双库模型 `models/dual_bank.py`

**Files:**
- Create: `src/dino_exp/models/dual_bank.py`
- Test: `tests/test_dual_bank.py`

设计依据（设计文档 §3.2 覆盖方法清单）：打分在 `PatchcoreModel.forward → nearest_neighbors(n_neighbors=1)`，加分注入 patch 分数层面；`fit()` 用 `KCenterGreedy` 而非 deprecated 的 `subsample_embedding`；阈值经 PostProcessor 注入。单元测试只测纯函数（不实例化骨干）。

- [ ] **Step 1: 写失败测试 `tests/test_dual_bank.py`**

```python
import pytest
import torch

from dino_exp.models.dual_bank import (
    apply_defect_boost,
    calibrate_threshold,
    coreset_cap,
    merge_pinned,
)


def test_boost_applies_when_defect_closer():
    d_n = torch.tensor([1.0, 2.0, 3.0])
    d_d = torch.tensor([0.5, 2.5, 3.0])  # patch0/patch1 离缺陷库更近
    out = apply_defect_boost(d_n, d_d, w=0.5)
    assert out[0].item() == pytest.approx(1.5)   # 1.0 * 1.5
    assert out[1].item() == pytest.approx(3.0)   # 2.0 * 1.5
    assert out[2].item() == pytest.approx(3.0)   # 不加分（相等不加）


def test_boost_zero_weight_is_noop():
    d_n = torch.tensor([1.0, 2.0])
    d_d = torch.tensor([0.1, 0.1])
    assert torch.equal(apply_defect_boost(d_n, d_d, w=0.0), d_n)


def test_coreset_cap_formula():
    # cap = 基础库大小 × 倍数 - 钉住数，至少 1
    assert coreset_cap(base_size=1000, cap_ratio=1.5, pinned=0) == 1500
    assert coreset_cap(base_size=1000, cap_ratio=1.5, pinned=600) == 900
    assert coreset_cap(base_size=10, cap_ratio=1.5, pinned=20) == 1


def test_merge_pinned_inserts_after_pinned_region():
    bank = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    # 前 2 行是钉住区
    new = torch.full((1, 2), 99.0)
    merged, pinned_count = merge_pinned(bank, pinned_count=2, new_feats=new, pinned=True)
    assert pinned_count == 3
    assert torch.equal(merged[2], torch.tensor([99.0, 99.0]))  # 插入在钉住区尾部
    assert torch.equal(merged[0], bank[0]) and torch.equal(merged[-1], bank[-1])


def test_merge_unpinned_appends_at_end():
    bank = torch.zeros(3, 2)
    merged, pinned_count = merge_pinned(bank, pinned_count=1, new_feats=torch.ones(2, 2), pinned=False)
    assert pinned_count == 1
    assert merged.shape == (5, 2)
    assert torch.equal(merged[-1], torch.ones(2))


def test_calibrate_threshold_mean_plus_3sigma():
    # 1..9: mean=5, 样本 std≈2.7386 → t ≈ 5 + 3*2.7386
    scores = [float(i) for i in range(1, 10)]
    t = calibrate_threshold(scores, sigma=3.0)
    assert t == pytest.approx(5 + 3 * 2.7386, abs=1e-3)


def test_calibrate_threshold_single_score():
    assert calibrate_threshold([2.5], sigma=3.0) == 2.5  # std=0


def test_calibrate_threshold_empty_raises():
    from dino_exp.errors import DinoError

    with pytest.raises(DinoError, match="校准"):
        calibrate_threshold([], sigma=3.0)
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_dual_bank.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.models.dual_bank'`。

- [ ] **Step 3: 写 `src/dino_exp/models/dual_bank.py`**

```python
"""DualBankPatchcore：Patchcore 子类 + 缺陷原型库（设计文档 §3.2）。

分层：
- 纯函数（apply_defect_boost / coreset_cap / merge_pinned / calibrate_threshold）：
  全部业务规则，单元测试只覆盖这层，不依赖真实骨干。
- DualBankPatchcoreModel(PatchcoreModel)：defect_bank buffer、钉住区管理、
  nearest_neighbors 注入加分（仅 n_neighbors==1 的 patch 打分支路，不影响
  compute_anomaly_score 内部的 support-sample 查询）。
- DualBankPatchcore(Patchcore)：__init__ 替换 self.model；fit() 直接用
  KCenterGreedy（PatchcoreModel.subsample_embedding 自 anomalib 2.1 起 deprecated）；
  apply_threshold() 把 mean+3σ 注入 PostProcessor（ManualThreshold），使指标、
  判定、导出 metadata 共用同一阈值。
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch
from anomalib.metrics import ManualThreshold
from anomalib.models.components import KCenterGreedy
from anomalib.models.image.patchcore.lightning_model import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from dino_exp.errors import DinoError

# ---------------- 纯函数（单测目标） ----------------


def apply_defect_boost(patch_scores: torch.Tensor, defect_dists: torch.Tensor, w: float) -> torch.Tensor:
    """缺陷库加分：d_d < d_n 的 patch 分数 ×(1+w)；只加分不减分。"""
    boost = (defect_dists < patch_scores).to(patch_scores.dtype) * w
    return patch_scores * (1.0 + boost)


def coreset_cap(base_size: int, cap_ratio: float, pinned: int) -> int:
    """非钉住区容量 = 基础版本库大小 × 固定倍数 - 钉住数（至少 1）。"""
    return max(1, int(base_size * cap_ratio) - pinned)


def merge_pinned(
    memory_bank: torch.Tensor, pinned_count: int, new_feats: torch.Tensor, pinned: bool
) -> tuple[torch.Tensor, int]:
    """合并新特征，维持 [钉住区..., 非钉住区...] 布局；返回 (新库, 新钉住数)。"""
    if pinned:
        merged = torch.cat([memory_bank[:pinned_count], new_feats, memory_bank[pinned_count:]])
        return merged, pinned_count + len(new_feats)
    return torch.cat([memory_bank, new_feats]), pinned_count


def calibrate_threshold(ok_scores: list[float], sigma: float = 3.0) -> float:
    """mean + sigma × 样本标准差；单样本时 std=0。"""
    if not ok_scores:
        raise DinoError("校准集为空，无法计算阈值。请检查数据集 test/good（或 train/good 切分）是否存在图片。")
    mean = statistics.fmean(ok_scores)
    std = statistics.stdev(ok_scores) if len(ok_scores) > 1 else 0.0
    return mean + sigma * std


# ---------------- 模型层（薄封装） ----------------


class DualBankPatchcoreModel(PatchcoreModel):
    def __init__(
        self,
        *,
        layers,
        backbone,
        pre_trained: bool = True,
        num_neighbors: int = 9,
        fusion_weight: float = 0.5,
        bank_cap_ratio: float = 1.5,
        coreset_sampling_ratio: float = 0.1,
    ) -> None:
        super().__init__(layers=layers, backbone=backbone, pre_trained=pre_trained, num_neighbors=num_neighbors)
        self.fusion_weight = fusion_weight
        self.bank_cap_ratio = bank_cap_ratio
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.register_buffer("defect_bank", torch.empty(0))
        self.register_buffer("pinned_count", torch.zeros(1, dtype=torch.long))
        self.base_bank_size: int | None = None  # 基础版本库大小；持久化在 meta.json

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int):
        patch_scores, locations = super().nearest_neighbors(embedding, n_neighbors)
        # 仅 patch 打分支路（forward 固定 n_neighbors=1）注入缺陷库加分，
        # 不影响 compute_anomaly_score 内部的 support-sample 查询（n>1）。
        if n_neighbors == 1 and self.defect_bank.dim() == 2 and self.defect_bank.shape[0] > 0:
            d_def = self.euclidean_dist(embedding, self.defect_bank).min(dim=1).values
            patch_scores = apply_defect_boost(patch_scores, d_def, self.fusion_weight)
        return patch_scores, locations

    @property
    def _pinned(self) -> int:
        return int(self.pinned_count.item())

    def add_normal_features(self, feats: torch.Tensor, pinned: bool = False) -> None:
        feats = feats.to(self.memory_bank.device, self.memory_bank.dtype)
        merged, count = merge_pinned(self.memory_bank, self._pinned, feats, pinned)
        self.memory_bank = merged
        self.pinned_count = torch.tensor([count], dtype=torch.long, device=merged.device)
        self.resample_normal_bank()

    def add_defect_features(self, feats: torch.Tensor) -> None:
        feats = feats.to(self.memory_bank.device, self.memory_bank.dtype)
        self.defect_bank = feats if self.defect_bank.numel() == 0 else torch.cat([self.defect_bank, feats])

    def fit_coreset(self) -> None:
        """初始建库：vstack embedding_store → coreset 采样 → memory_bank。"""
        if not self.embedding_store:
            raise DinoError("embedding_store 为空，无法 coreset。请确认 Engine.fit 已遍历训练集。")
        feats = torch.vstack(self.embedding_store)
        self.embedding_store.clear()
        ratio = self.coreset_sampling_ratio
        if ratio < 1.0 and int(len(feats) * ratio) >= 1:
            feats = KCenterGreedy(embedding=feats, sampling_ratio=ratio).sample_coreset()
        self.memory_bank = feats
        self.pinned_count = torch.zeros(1, dtype=torch.long, device=feats.device)
        self.base_bank_size = feats.shape[0]

    def resample_normal_bank(self) -> None:
        """非钉住区超过上限时 coreset 重采样淘汰；钉住区豁免且不计入上限。"""
        if self.base_bank_size is None:
            return
        p = self._pinned
        pinned_part = self.memory_bank[:p]
        unpinned = self.memory_bank[p:]
        cap = coreset_cap(self.base_bank_size, self.bank_cap_ratio, p)
        if len(unpinned) <= cap:
            return
        sampled = KCenterGreedy(embedding=unpinned, sampling_ratio=cap / len(unpinned)).sample_coreset()
        self.memory_bank = torch.cat([pinned_part, sampled.to(pinned_part.device)])


class DualBankPatchcore(Patchcore):
    def __init__(
        self,
        *,
        backbone: str,
        layers,
        image_size: tuple[int, int] = (224, 224),
        fusion_weight: float = 0.5,
        bank_cap_ratio: float = 1.5,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        pre_trained: bool = True,
        evaluator=True,
    ) -> None:
        pre_processor = Patchcore.configure_pre_processor(image_size=image_size)
        super().__init__(
            backbone=backbone,
            layers=layers,
            pre_trained=False,  # 父类构造的 PatchcoreModel 随即被替换，避免重复加载权重
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
        self.model = DualBankPatchcoreModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
            fusion_weight=fusion_weight,
            bank_cap_ratio=bank_cap_ratio,
            coreset_sampling_ratio=coreset_sampling_ratio,
        )

    def fit(self) -> None:
        """MemoryBankMixin 钩子在 train epoch 结束时调用；直接用 KCenterGreedy。"""
        self.model.fit_coreset()

    def apply_threshold(self, threshold: float) -> None:
        """把 mean+3σ 阈值注入 PostProcessor（指标/判定/导出共用同一阈值）。"""
        t = float(threshold)
        pp = self.post_processor
        pp._image_threshold_metric = ManualThreshold(
            default_value=t, fields=["pred_score", "gt_label"], strict=False
        )
        pp._pixel_threshold_metric = ManualThreshold(
            default_value=t, fields=["anomaly_map", "gt_mask"], strict=False
        )
        pp._image_threshold = torch.tensor(t)
        pp._pixel_threshold = torch.tensor(t)


# ---------------- 特征提取辅助（训练/再训练/反馈共用） ----------------


def extract_embeddings(model: DualBankPatchcoreModel, images: torch.Tensor) -> torch.Tensor:
    """对已预处理图片张量提取 patch embedding，返回 (B*H*W, D)。"""
    with torch.no_grad():
        features = model.feature_extractor(images)
    features = {k: model.feature_pooler(v) for k, v in features.items()}
    embedding = model.generate_embedding(features)
    return model.reshape_embedding(embedding)


def topk_defect_features(model: DualBankPatchcoreModel, image: torch.Tensor, k: int) -> torch.Tensor:
    """取单图异常分数 top-k 的 patch 特征（NG 反馈入缺陷库）。"""
    embedding = extract_embeddings(model, image)
    if model.memory_bank.numel() == 0:
        raise DinoError("正常记忆库为空，无法计算 patch 分数。请先完成基础训练。")
    patch_scores, _ = model.nearest_neighbors(embedding, n_neighbors=1)
    k = min(k, embedding.shape[0])
    idx = torch.topk(patch_scores, k=k).indices
    return embedding[idx]


def save_banks(model: DualBankPatchcoreModel, path: str | Path) -> None:
    """统一存储格式：normal_bank.pt 为字典（含缺陷库与钉住元数据）。"""
    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "defect_bank": model.defect_bank.cpu(),
            "pinned_count": int(model.pinned_count.item()),
            "base_bank_size": model.base_bank_size,
        },
        path,
    )


def load_banks(model: DualBankPatchcoreModel, bank_dir: str | Path) -> None:
    """从版本目录恢复双库。

    存储约定：normal_bank.pt 为 save_banks 字典格式（主数据源）；
    defect_bank.pt 在无缺陷库时是 registry 写入的 torch.empty(0) 占位张量，
    若是 2D 张量（独立存档的缺陷库）则优先使用。
    """
    bank_dir = Path(bank_dir)
    nb = torch.load(bank_dir / "normal_bank.pt", map_location="cpu", weights_only=True)
    db_path = bank_dir / "defect_bank.pt"
    defect = (
        torch.load(db_path, map_location="cpu", weights_only=True)
        if db_path.exists() else torch.empty(0)
    )
    model.memory_bank = nb["memory_bank"]
    model.defect_bank = defect if defect.dim() == 2 else nb["defect_bank"]
    model.pinned_count = torch.tensor([nb["pinned_count"]], dtype=torch.long)
    model.base_bank_size = nb["base_bank_size"] if nb["base_bank_size"] is not None else nb["memory_bank"].shape[0]
```

存储格式约定（跨 Task 一致）：registry 的 `normal_bank.pt` 一律为 `save_banks` 字典格式；`defect_bank.pt` 在无缺陷库时为 `torch.empty(0)` 占位。Task 6 `finalize_version` 与 Task 7 `load_model_for_version` 均按此约定读写。

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_dual_bank.py -v
```

预期：8 个纯函数测试全过（不实例化 DualBankPatchcoreModel，无权重下载）。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/models/dual_bank.py tests/test_dual_bank.py
git commit -m "feat: 双库模型（缺陷库加分/钉住豁免/coreset 上限/阈值注入 PostProcessor）"
```

---

## Task 6: 训练管线 `train.py`

**Files:**
- Create: `src/dino_exp/train.py`
- Create: `src/dino_exp/validate.py`（Task 8 实现，本 Task 先建空壳函数供调用）
- Test: `tests/test_train.py`

单元测试不跑真实训练；测试「校准分数收集 → 阈值 → 版本落盘」的编排逻辑，模型侧用假实现注入。

- [ ] **Step 1: 写失败测试 `tests/test_train.py`**

```python
import json

import pytest
import torch

from dino_exp.config import Config
from dino_exp.train import finalize_version


def _mk_dataset(cfg, n_train=6, n_test_good=2, n_ng=2):
    d = cfg.data_root / "bottle"
    for i in range(n_train):
        (d / "train" / "good").mkdir(parents=True, exist_ok=True)
        (d / "train" / "good" / f"t{i}.png").write_bytes(b"x")
    for i in range(n_test_good):
        (d / "test" / "good").mkdir(parents=True, exist_ok=True)
        (d / "test" / "good" / f"g{i}.png").write_bytes(b"x")
    for i in range(n_ng):
        (d / "test" / "broken").mkdir(parents=True, exist_ok=True)
        (d / "test" / "broken" / f"b{i}.png").write_bytes(b"x")


class FakeModel:
    """模拟训练后的 DualBankPatchcore（仅有 finalize 需要的接口）。"""

    def __init__(self):
        self.model = self  # lightning.model 约定
        self.memory_bank = torch.randn(50, 8)
        self.defect_bank = torch.empty(0)
        self.pinned_count = torch.tensor([0])
        self.base_bank_size = 50
        self.applied = None

    def apply_threshold(self, t):
        self.applied = t


def test_finalize_version_saves_banks_threshold_and_meta(tmp_path):
    cfg = Config(data_root=tmp_path / "data", models_root=tmp_path / "models")
    _mk_dataset(cfg)
    model = FakeModel()
    ok_scores = [1.0, 1.2, 0.8, 1.1]
    version = finalize_version(
        "bottle", cfg, model, ok_scores=ok_scores,
        metrics={"image_AUROC": 0.95}, parent=None, feedback_applied=0,
    )
    assert version == "v001"
    vdir = cfg.models_root / "bottle" / "v001"
    assert (vdir / "normal_bank.pt").exists()
    metrics = json.loads((vdir / "metrics.json").read_text())
    # mean=1.025, std≈0.1708 → t≈1.537
    assert metrics["threshold"] == pytest.approx(1.537, abs=1e-3)
    assert metrics["image_AUROC"] == 0.95
    assert model.applied == metrics["threshold"]  # 阈值已注入模型
    meta = json.loads((vdir / "meta.json").read_text())
    assert meta["parent"] is None and meta["feedback_applied"] == 0
    banks = torch.load(vdir / "normal_bank.pt", weights_only=True)
    assert banks["memory_bank"].shape == (50, 8)
    assert banks["base_bank_size"] == 50
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_train.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.train'`。

- [ ] **Step 3: 写 `src/dino_exp/train.py` 与 `src/dino_exp/validate.py` 空壳**

`src/dino_exp/train.py`：

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import build_folder, dataset_info, ok_calibration_images
from dino_exp.errors import DinoError
from dino_exp.models.dual_bank import (
    DualBankPatchcore,
    calibrate_threshold,
    save_banks,
)
from dino_exp.models.registry import Registry


def build_model(cfg: Config) -> DualBankPatchcore:
    spec = cfg.backbone_spec
    from anomalib.metrics import AUPR, AUROC, Evaluator, F1Score

    evaluator = Evaluator(test_metrics=[AUROC(), AUPR(), F1Score()])
    return DualBankPatchcore(
        backbone=spec.timm_name,
        layers=cfg.layers,
        image_size=(cfg.image_size, cfg.image_size),
        fusion_weight=cfg.fusion_weight,
        bank_cap_ratio=cfg.bank_cap_ratio,
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
        evaluator=evaluator,
    )


def score_images(model, image_paths: list[Path], cfg: Config) -> list[float]:
    """对图片列表逐张推理，返回原始异常分数（raw pred_score）。"""
    from dino_exp.infer import preprocess_image

    model.eval()
    scores = []
    for p in image_paths:
        tensor = preprocess_image(p, cfg.image_size)
        with torch.no_grad():
            out = model.model(tensor)
        scores.append(float(out.pred_score.item()))
    return scores


def finalize_version(
    category: str,
    cfg: Config,
    model,
    *,
    ok_scores: list[float],
    metrics: dict,
    parent: str | None,
    feedback_applied: int,
) -> str:
    """校准阈值 → 注入模型 → 存版本（banks + config + metrics + meta）。返回版本号。"""
    threshold = calibrate_threshold(ok_scores, sigma=cfg.threshold_sigma)
    model.apply_threshold(threshold)
    with tempfile.TemporaryDirectory() as td:
        bank_path = Path(td) / "normal_bank.pt"
        save_banks(model.model, bank_path)
        version = Registry(cfg.models_root).create_version(
            category,
            normal_bank=bank_path,
            defect_bank=None,  # 缺陷库已含在 save_banks 字典内；registry 的 defect_bank.pt 仅占位
            checkpoint=None,
            config={
                "backbone": cfg.backbone,
                "layers": cfg.layers,
                "image_size": cfg.image_size,
                "coreset_sampling_ratio": cfg.coreset_sampling_ratio,
                "num_neighbors": cfg.num_neighbors,
                "fusion_weight": cfg.fusion_weight,
                "bank_cap_ratio": cfg.bank_cap_ratio,
                "defect_topk": cfg.defect_topk,
                "threshold_sigma": cfg.threshold_sigma,
            },
            metrics={**metrics, "threshold": threshold},
            meta={"parent": parent, "feedback_applied": feedback_applied},
        )
    return version


def train_model(category: str, cfg: Config) -> dict:
    """基础训练（设计文档 §4 工作流 1）。返回新版本指标。"""
    from anomalib.engine import Engine

    info = dataset_info(category, cfg)  # 结构校验前置（NFR-4）
    datamodule = build_folder(category, cfg)
    model = build_model(cfg)
    engine = Engine(default_root_dir=str(Path("results") / category))
    engine.fit(model=model, datamodule=datamodule)
    # 阈值校准（FR-2.3）：OK 校准图 raw 分数 mean+3σ
    ok_scores = score_images(model, ok_calibration_images(category, cfg), cfg)
    # 全量验证（FR-3.4）：degraded 时跳过聚合指标
    if info.degraded:
        metrics: dict = {"degraded": True}
    else:
        results = engine.test(model=model, datamodule=datamodule)
        metrics = {k: float(v) for k, v in results[0].items()}
    version = finalize_version(
        category, cfg, model,
        ok_scores=ok_scores, metrics=metrics, parent=None, feedback_applied=0,
    )
    return {"version": version, "metrics": metrics}
```

`src/dino_exp/validate.py`（空壳，Task 8 填充）：

```python
"""全量/选图验证。实现见 Task 8。"""
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_train.py -v
```

预期：1 个测试通过。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/train.py src/dino_exp/validate.py tests/test_train.py
git commit -m "feat: 训练管线（Engine.fit→校准→注入阈值→版本落盘→自动验证）"
```

---

## Task 7: 推理 `infer.py`（含 OpenVINO 版本快照导出）

**Files:**
- Create: `src/dino_exp/infer.py`
- Test: `tests/test_infer.py`

- [ ] **Step 1: 写失败测试 `tests/test_infer.py`**

```python
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
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_infer.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.infer'`。

- [ ] **Step 3: 写 `src/dino_exp/infer.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms.v2 as T

from dino_exp.config import Config
from dino_exp.errors import DinoError
from dino_exp.models.dual_bank import DualBankPatchcore, load_banks
from dino_exp.models.registry import Registry


def preprocess_image(path: str | Path, image_size: int) -> torch.Tensor:
    tf = T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def decide_label(score: float, threshold: float) -> str:
    return "NG" if score >= threshold else "OK"


def load_threshold(version_dir: str | Path) -> float:
    p = Path(version_dir) / "metrics.json"
    if not p.exists():
        raise DinoError(f"{p} 不存在，版本可能损坏。请切换/回滚到其他版本。")
    metrics = json.loads(p.read_text(encoding="utf-8"))
    if "threshold" not in metrics:
        raise DinoError(f"{p} 缺少 threshold 字段。请重新训练生成版本。")
    return float(metrics["threshold"])


def heatmap_to_bgr(anomaly_map: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
    amap = anomaly_map.squeeze().cpu().numpy()
    amap = cv2.resize(amap, out_size)
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    return cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)


def load_model_for_version(category: str, version: str | None, cfg: Config) -> tuple[DualBankPatchcore, float, str]:
    """加载指定（或当前）版本模型 + 阈值。返回 (model, threshold, version)。"""
    reg = Registry(cfg.models_root)
    version = version or reg.current(category)
    if version is None:
        raise DinoError(f"类别 '{category}' 没有任何模型版本。请先 `dino train --category {category}`。")
    vdir = reg.version_dir(category, version)
    from dino_exp.train import build_model

    model = build_model(cfg)
    load_banks(model.model, vdir)
    threshold = load_threshold(vdir)
    model.apply_threshold(threshold)
    model.eval()
    return model, threshold, version


def infer_image(
    path: str | Path,
    version: str | None = None,
    *,
    category: str,
    cfg: Config,
    heatmap_dir: str | Path = "outputs/heatmaps",
) -> dict:
    """单图推理 → {label, score, threshold, heatmap_path}（设计文档 §3.5）。"""
    model, threshold, version = load_model_for_version(category, version, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tensor = preprocess_image(path, cfg.image_size).to(device)
    with torch.no_grad():
        out = model.model(tensor)
    score = float(out.pred_score.item())
    hdir = Path(heatmap_dir)
    hdir.mkdir(parents=True, exist_ok=True)
    heatmap_path = hdir / f"{Path(path).stem}_{version}_heatmap.png"
    with Image.open(path) as im:
        out_size = im.size  # (W, H)
    cv2.imwrite(str(heatmap_path), heatmap_to_bgr(out.anomaly_map.cpu(), out_size))
    return {
        "label": decide_label(score, threshold),
        "score": score,
        "threshold": threshold,
        "heatmap_path": str(heatmap_path),
    }


def infer_batch(paths: list[str | Path], version: str | None = None, *, category: str, cfg: Config) -> list[dict]:
    return [infer_image(p, version, category=category, cfg=cfg) for p in paths]


def export_openvino(category: str, version: str | None, cfg: Config) -> Path:
    """按版本快照导出 OpenVINO（设计文档 §3.5）。

    导出物写入 models/<类别>/<版本>/export/；导出时双库状态被烘焙进图，
    再训练后必须对新版本重新调用本函数。失败时抛 DinoError 并提示回退 PyTorch。
    """
    try:
        from anomalib.engine import Engine
    except ImportError as exc:
        raise DinoError(f"anomalib 导入失败: {exc}。请检查环境后重试。") from exc
    model, threshold, version = load_model_for_version(category, version, cfg)
    export_root = Registry(cfg.models_root).version_dir(category, version) / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    try:
        engine = Engine()
        out = engine.export(
            model=model,
            export_type="openvino",
            export_root=export_root,
            input_size=(cfg.image_size, cfg.image_size),
        )
    except Exception as exc:
        raise DinoError(
            f"OpenVINO 导出失败: {exc}。已回退：请继续使用 PyTorch 后端推理"
            "（`pip install \"anomalib[openvino]==2.5.1\"` 后可重试导出）。"
        ) from exc
    # 阈值写入导出 metadata，与 PyTorch 判定口径一致
    (export_root / "threshold.json").write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    return Path(out) if out else export_root
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_infer.py -v
```

预期：4 个测试全过（不加载真实模型）。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/infer.py tests/test_infer.py
git commit -m "feat: 推理（单图/批量/热力图/设备自适应/OpenVINO 版本快照导出）"
```

---

## Task 8: 验证 `validate.py`

**Files:**
- Modify: `src/dino_exp/validate.py`（替换 Task 6 的空壳）
- Test: `tests/test_validate.py`

- [ ] **Step 1: 写失败测试 `tests/test_validate.py`**

```python
import json

from dino_exp.validate import aggregate_metrics, filter_errors, save_validation_report


def test_aggregate_metrics_computes_auroc_f1():
    # 4 OK(score 低) + 4 NG(score 高)，完美可分
    rows = [
        {"label_gt": 0, "score": s} for s in [0.1, 0.2, 0.3, 0.4]
    ] + [
        {"label_gt": 1, "score": s} for s in [0.6, 0.7, 0.8, 0.9]
    ]
    m = aggregate_metrics(rows, threshold=0.5)
    assert m["image_AUROC"] == 1.0
    assert m["image_F1"] == 1.0


def test_aggregate_metrics_degraded_returns_none_metrics():
    rows = [{"label_gt": 0, "score": 0.1}, {"label_gt": 0, "score": 0.2}]
    m = aggregate_metrics(rows, threshold=0.5)
    assert m["degraded"] is True
    assert "image_AUROC" not in m


def test_filter_errors():
    rows = [
        {"path": "a.png", "label_gt": 0, "label_pred": "OK"},
        {"path": "b.png", "label_gt": 1, "label_pred": "OK"},  # 漏报
        {"path": "c.png", "label_gt": 0, "label_pred": "NG"},  # 误报
    ]
    errs = filter_errors(rows)
    assert [r["path"] for r in errs] == ["b.png", "c.png"]


def test_save_validation_report(tmp_path):
    save_validation_report(tmp_path, {"image_AUROC": 0.9}, [{"path": "a"}])
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["metrics"]["image_AUROC"] == 0.9
    assert len(report["rows"]) == 1
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_validate.py -x
```

预期：`ImportError: cannot import name 'aggregate_metrics'`。

- [ ] **Step 3: 写 `src/dino_exp/validate.py`（全量替换空壳）**

```python
from __future__ import annotations

import json
from pathlib import Path

import torch

from dino_exp.config import Config
from dino_exp.datasets import dataset_info, test_images_with_labels
from dino_exp.infer import decide_label, infer_image, load_model_for_version, preprocess_image
from dino_exp.models.registry import Registry


def aggregate_metrics(rows: list[dict], threshold: float) -> dict:
    """从逐图 {label_gt, score} 计算图片级 AUROC/AUPR/F1；无 NG 样本时降级。"""
    from torchmetrics import AUROC, AveragePrecision, F1Score

    labels = torch.tensor([r["label_gt"] for r in rows])
    scores = torch.tensor([r["score"] for r in rows])
    if labels.sum().item() == 0:
        return {"degraded": True, "note": "无 NG 测试图，指标降级：仅输出逐图分数"}
    preds = (scores >= threshold).long()
    return {
        "image_AUROC": float(AUROC(task="binary")(scores, labels)),
        "image_AUPR": float(AveragePrecision(task="binary")(scores, labels)),
        "image_F1": float(F1Score(task="binary")(preds, labels)),
    }


def filter_errors(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["label_pred"] != ("NG" if r["label_gt"] == 1 else "OK")]


def save_validation_report(version_dir: str | Path, metrics: dict, rows: list[dict]) -> Path:
    p = Path(version_dir) / "validation.json"
    p.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    return p


def score_test_set(category: str, version: str | None, cfg: Config) -> tuple[list[dict], float, str]:
    """对 test 集逐图推理（raw score），返回 (rows, threshold, version)。"""
    model, threshold, version = load_model_for_version(category, version, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    rows = []
    for path, label_gt, defect_type in test_images_with_labels(category, cfg):
        tensor = preprocess_image(path, cfg.image_size).to(device)
        with torch.no_grad():
            out = model.model(tensor)
        score = float(out.pred_score.item())
        rows.append({
            "path": str(path),
            "label_gt": label_gt,
            "defect_type": defect_type,
            "score": score,
            "label_pred": decide_label(score, threshold),
        })
    return rows, threshold, version


def validate_full(category: str, version: str | None, cfg: Config) -> dict:
    """全量验证（FR-3.1/FR-3.4）：聚合指标 + 逐图结果写入版本目录 validation.json。"""
    rows, threshold, version = score_test_set(category, version, cfg)
    metrics = aggregate_metrics(rows, threshold)
    vdir = Registry(cfg.models_root).version_dir(category, version)
    save_validation_report(vdir, metrics, rows)
    return {"version": version, "metrics": metrics, "rows": rows}


def validate_images(category: str, version: str | None, paths: list[str], cfg: Config) -> list[dict]:
    """选图验证（FR-3.2）：逐图分数/判定/热力图，不出聚合指标。"""
    return [infer_image(p, version, category=category, cfg=cfg) for p in paths]
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_validate.py -v
```

预期：4 个测试全过。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/validate.py tests/test_validate.py
git commit -m "feat: 验证（全量聚合指标+逐图结果/选图/误判过滤/无 NG 降级）"
```

---

## Task 9: 反馈存储 `feedback/store.py` + `staging.py`

**Files:**
- Create: `src/dino_exp/feedback/__init__.py`
- Create: `src/dino_exp/feedback/store.py`
- Create: `src/dino_exp/feedback/staging.py`
- Test: `tests/test_feedback.py`

- [ ] **Step 1: 写失败测试 `tests/test_feedback.py`**

```python
import pytest

from dino_exp.config import Config
from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.feedback.staging import conflicts, effective, preview


@pytest.fixture
def store(tmp_path):
    cfg = Config(feedback_root=tmp_path / "feedback")
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    return FeedbackStore(cfg.feedback_root, "bottle"), img


def _fb(img, label, score=0.5, ts="2026-07-20T10:00:00"):
    return {
        "image_path": str(img), "model_version": "v001",
        "prediction": "NG", "score": score,
        "human_label": label, "defect_type": None, "timestamp": ts,
    }


def test_stage_persists_jsonl_and_image_copy(store):
    s, img = store
    rec = s.stage(_fb(img, "ok"))
    assert rec["id"]
    staged = s.staged()
    assert len(staged) == 1
    assert (s.images_dir / rec["stored_image"]).exists()


def test_remove_single_entry(store):
    s, img = store
    r1 = s.stage(_fb(img, "ok"))
    r2 = s.stage(_fb(img, "ng", ts="2026-07-20T11:00:00"))
    assert s.remove(r1["id"]) is True
    assert [r["id"] for r in s.staged()] == [r2["id"]]
    assert s.remove("nonexistent") is False


def test_effective_latest_wins(store):
    s, img = store
    s.stage(_fb(img, "ng", ts="2026-07-20T10:00:00"))
    s.stage(_fb(img, "ok", ts="2026-07-20T12:00:00"))  # 更新的一条生效
    eff = effective(s.staged())
    assert len(eff) == 1 and eff[0]["human_label"] == "ok"


def test_conflicts_detects_label_flip(store):
    s, img = store
    s.stage(_fb(img, "ng", ts="2026-07-20T10:00:00"))
    s.stage(_fb(img, "ok", ts="2026-07-20T12:00:00"))
    conf = conflicts(s.staged())
    assert len(conf) == 1 and str(img) in conf[0]


def test_preview_bidirectional_suspicious(store, tmp_path):
    s, img = store
    img2 = tmp_path / "y.png"
    img2.write_bytes(b"y")
    s.stage(_fb(img, "ok", score=5.0))    # OK 但 score > 3×1.0 → 可疑
    s.stage(_fb(img2, "ng", score=0.5))   # NG 但 score < 阈值 → 可疑（漏报型）
    pv = preview(s.staged(), threshold=1.0, factor=3.0)
    assert pv["ok"] == 1 and pv["ng"] == 1
    assert len(pv["suspicious"]) == 2


def test_apply_archives_and_clears(store):
    s, img = store
    s.stage(_fb(img, "ok"))
    applied = s.apply()
    assert len(applied) == 1
    assert s.staged() == []
    assert s.applied_file.exists() and len(s.applied_file.read_text().strip().splitlines()) == 1


def test_apply_empty_raises(store):
    s, _ = store
    with pytest.raises(DinoError, match="暂存区为空"):
        s.apply()
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_feedback.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.feedback'`。

- [ ] **Step 3: 写 `src/dino_exp/feedback/__init__.py`（空）、`store.py`、`staging.py`**

`src/dino_exp/feedback/store.py`：

```python
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dino_exp.errors import DinoError


class FeedbackStore:
    """反馈持久化：staged.jsonl（暂存）+ applied.jsonl（归档）+ images/（图片拷贝）。"""

    def __init__(self, root: str | Path, experiment: str):
        self.dir = Path(root) / experiment
        self.images_dir = self.dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.staged_file = self.dir / "staged.jsonl"
        self.applied_file = self.dir / "applied.jsonl"

    @staticmethod
    def _read(file: Path) -> list[dict]:
        if not file.exists():
            return []
        return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write(file: Path, rows: list[dict]) -> None:
        file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def stage(self, record: dict) -> dict:
        src = Path(record["image_path"])
        if not src.exists():
            raise DinoError(f"反馈图片不存在: {src}。请确认路径后重试。")
        fid = uuid.uuid4().hex[:12]
        stored = f"{fid}{src.suffix}"
        shutil.copy2(src, self.images_dir / stored)
        row = {
            "id": fid,
            "stored_image": stored,
            "defect_type": record.get("defect_type"),
            "timestamp": record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            **{k: record[k] for k in ("image_path", "model_version", "prediction", "score", "human_label")},
        }
        if row["human_label"] not in {"ok", "ng"}:
            raise DinoError(f"human_label 只能是 ok/ng，得到 '{row['human_label']}'。")
        rows = self._read(self.staged_file)
        rows.append(row)
        self._write(self.staged_file, rows)
        return row

    def staged(self) -> list[dict]:
        return self._read(self.staged_file)

    def remove(self, feedback_id: str) -> bool:
        rows = self._read(self.staged_file)
        kept = [r for r in rows if r["id"] != feedback_id]
        if len(kept) == len(rows):
            return False
        self._write(self.staged_file, kept)
        return True

    def apply(self) -> list[dict]:
        from dino_exp.feedback.staging import effective

        rows = self.staged()
        if not rows:
            raise DinoError("暂存区为空，拒绝再训练。请先 `dino feedback ...` 添加反馈。")
        applied = self._read(self.applied_file) + rows
        self._write(self.applied_file, applied)
        self._write(self.staged_file, [])
        return effective(rows)  # 生效集合：同图最新一条为准
```

`src/dino_exp/feedback/staging.py`：

```python
from __future__ import annotations


def _by_image(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["image_path"], []).append(r)
    return groups


def effective(rows: list[dict]) -> list[dict]:
    """同一张图多次反馈以 timestamp 最新一条为准。"""
    return [max(g, key=lambda r: r["timestamp"]) for g in _by_image(rows).values()]


def conflicts(rows: list[dict]) -> list[str]:
    """同图先后标记了不同标签的冲突记录（列入预览交人工裁决）。"""
    return [
        img for img, g in _by_image(rows).items()
        if len({r["human_label"] for r in g}) > 1
    ]


def preview(rows: list[dict], threshold: float, factor: float = 3.0) -> dict:
    """再训练预览：生效数量 + 双向可疑项 + 同图冲突。"""
    eff = effective(rows)
    suspicious = [
        r for r in eff
        if (r["human_label"] == "ok" and r["score"] > factor * threshold)
        or (r["human_label"] == "ng" and r["score"] < threshold)
    ]
    return {
        "ok": sum(1 for r in eff if r["human_label"] == "ok"),
        "ng": sum(1 for r in eff if r["human_label"] == "ng"),
        "suspicious": suspicious,
        "conflicts": conflicts(rows),
    }
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_feedback.py -v
```

预期：7 个测试全过。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/feedback/ tests/test_feedback.py
git commit -m "feat: 反馈存储与暂存区（最新为准/冲突裁决/单条删除/双向可疑/应用归档）"
```

---

## Task 10: 再训练管线 `retrain.py`

**Files:**
- Create: `src/dino_exp/retrain.py`
- Test: `tests/test_retrain.py`

单元测试聚焦编排：OK 反馈 → 钉住并入、NG 反馈 → top-k、AUROC 对比告警；特征提取与模型用 fake 注入。

- [ ] **Step 1: 写失败测试 `tests/test_retrain.py`**

```python
import json

import pytest
import torch

from dino_exp.config import Config
from dino_exp.retrain import auroc_drop_warning, partition_feedback


def test_partition_feedback():
    eff = [
        {"human_label": "ok", "stored_image": "a.png"},
        {"human_label": "ng", "stored_image": "b.png", "defect_type": "scratch"},
        {"human_label": "ng", "stored_image": "c.png", "defect_type": None},
    ]
    oks, ngs = partition_feedback(eff)
    assert len(oks) == 1 and len(ngs) == 2


def test_auroc_drop_warning_triggers_over_2_points():
    assert auroc_drop_warning(parent=0.95, current=0.92) is not None  # 降 3 个点
    assert auroc_drop_warning(parent=0.95, current=0.94) is None     # 降 1 个点
    assert auroc_drop_warning(parent=0.95, current=0.96) is None     # 上升


def test_auroc_drop_warning_degraded_skips():
    assert auroc_drop_warning(parent=None, current=None) is None


def test_retrain_end_to_end_with_fakes(tmp_path, monkeypatch):
    """用 fake 模型与 fake 反馈跑通：预览→应用→新版本→对比告警。"""
    from dino_exp import retrain as rt

    cfg = Config(
        data_root=tmp_path / "data", models_root=tmp_path / "models",
        feedback_root=tmp_path / "feedback",
    )
    # 造数据集（校验用）
    for i in range(4):
        (cfg.data_root / "c" / "train" / "good").mkdir(parents=True, exist_ok=True)
        (cfg.data_root / "c" / "train" / "good" / f"{i}.png").write_bytes(b"x")
    # 造父版本 v001
    from dino_exp.models.registry import Registry

    bank = tmp_path / "b.pt"
    torch.save({"memory_bank": torch.randn(20, 4), "defect_bank": torch.empty(0),
                "pinned_count": 0, "base_bank_size": 20}, bank)
    Registry(cfg.models_root).create_version(
        "c", normal_bank=bank, defect_bank=None, checkpoint=None,
        config={}, metrics={"image_AUROC": 0.95, "threshold": 1.0},
        meta={"parent": None, "feedback_applied": 0},
    )
    # 造反馈：1 OK（高分误报）+ 1 NG
    from dino_exp.feedback.store import FeedbackStore

    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    img2 = tmp_path / "g.png"
    img2.write_bytes(b"y")
    store = FeedbackStore(cfg.feedback_root, "c")
    # 注意：OK 与 NG 反馈必须用不同图片——同图多条会被 effective() 折叠为最新一条
    store.stage({"image_path": str(img), "model_version": "v001", "prediction": "NG",
                 "score": 5.0, "human_label": "ok", "defect_type": None,
                 "timestamp": "2026-07-20T10:00:00"})
    store.stage({"image_path": str(img2), "model_version": "v001", "prediction": "OK",
                 "score": 0.1, "human_label": "ng", "defect_type": "s",
                 "timestamp": "2026-07-20T11:00:00"})

    class FakeInner:
        def __init__(self):
            self.memory_bank = torch.randn(20, 4)
            self.defect_bank = torch.empty(0)
            self.pinned_count = torch.tensor([0])
            self.base_bank_size = 20
            self.bank_cap_ratio = 1.5
            self.coreset_sampling_ratio = 0.1

        def add_normal_features(self, feats, pinned=False):
            from dino_exp.models.dual_bank import merge_pinned

            self.memory_bank, c = merge_pinned(self.memory_bank, int(self.pinned_count), feats, pinned)
            self.pinned_count = torch.tensor([c])

        def add_defect_features(self, feats):
            self.defect_bank = feats if self.defect_bank.numel() == 0 else torch.cat([self.defect_bank, feats])

        def resample_normal_bank(self):
            pass

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()
            self.threshold = None

        def apply_threshold(self, t):
            self.threshold = t

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(rt, "load_model_for_version", lambda *a, **k: (FakeModel(), 1.0, "v001"))
    monkeypatch.setattr(rt, "extract_embeddings", lambda m, t: torch.randn(16, 4))
    monkeypatch.setattr(rt, "topk_defect_features", lambda m, t, k: torch.randn(k, 4))
    monkeypatch.setattr(rt, "preprocess_image", lambda p, s: torch.randn(1, 3, 8, 8))
    monkeypatch.setattr(rt, "ok_calibration_images", lambda c, cfg: [img])
    monkeypatch.setattr(rt, "score_images", lambda m, ps, cfg: [0.9, 1.1, 1.0])
    monkeypatch.setattr(rt, "validate_full", lambda c, v, cfg: {"version": v, "metrics": {"image_AUROC": 0.92}, "rows": []})

    result = rt.retrain("c", cfg)
    assert result["version"] == "v002"
    assert result["warning"] is not None and "回滚" in result["warning"]  # 0.95→0.92 降 3 点
    meta = json.loads((cfg.models_root / "c" / "v002" / "meta.json").read_text())
    assert meta["parent"] == "v001" and meta["feedback_applied"] == 2
    assert store.staged() == []  # 暂存区已清空
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_retrain.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.retrain'`。

- [ ] **Step 3: 写 `src/dino_exp/retrain.py`**

```python
from __future__ import annotations

import json

import torch

from dino_exp.config import Config
from dino_exp.datasets import ok_calibration_images
from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.feedback.staging import preview as staging_preview
from dino_exp.infer import load_model_for_version, load_threshold, preprocess_image
from dino_exp.models.dual_bank import extract_embeddings, topk_defect_features
from dino_exp.models.registry import Registry
from dino_exp.train import finalize_version, score_images
from dino_exp.validate import validate_full


def partition_feedback(effective_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    oks = [r for r in effective_rows if r["human_label"] == "ok"]
    ngs = [r for r in effective_rows if r["human_label"] == "ng"]
    return oks, ngs


def auroc_drop_warning(parent: float | None, current: float | None) -> str | None:
    """新版本 AUROC 较父版本下降 > 2 个点时告警（FR-6.5）。"""
    if parent is None or current is None:
        return None
    if parent - current > 0.02:
        return (
            f"新版本 AUROC {current:.4f} 较父版本 {parent:.4f} 下降超过 2 个点。"
            "建议检查反馈质量，必要时 `dino rollback` 回滚到父版本。"
        )
    return None


def preview_retrain(category: str, cfg: Config) -> dict:
    reg = Registry(cfg.models_root)
    current = reg.current(category)
    if current is None:
        raise DinoError(f"类别 '{category}' 无模型版本。请先 `dino train --category {category}`。")
    threshold = load_threshold(reg.version_dir(category, current))
    store = FeedbackStore(cfg.feedback_root, category)
    return {
        "current_version": current,
        **staging_preview(store.staged(), threshold, cfg.suspicious_score_factor),
    }


def retrain(category: str, cfg: Config) -> dict:
    """应用暂存反馈 → 新版本（设计文档 §4 工作流 4）。"""
    pv = preview_retrain(category, cfg)
    if pv["ok"] + pv["ng"] == 0:
        raise DinoError("暂存区为空，拒绝再训练。请先 `dino feedback ...` 添加反馈。")
    parent = pv["current_version"]
    store = FeedbackStore(cfg.feedback_root, category)
    effective_rows = store.apply()  # 消费暂存区（同图最新为准）并归档
    oks, ngs = partition_feedback(effective_rows)

    model, old_threshold, _ = load_model_for_version(category, parent, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    for r in oks:  # OK 反馈 → 钉住并入正常库（不参与 coreset 淘汰/不计入上限）
        feats = extract_embeddings(model.model, preprocess_image(store.images_dir / r["stored_image"], cfg.image_size).to(device))
        model.model.add_normal_features(feats, pinned=True)
    for r in ngs:  # NG 反馈 → top-k 高分 patch 入缺陷库
        feats = topk_defect_features(
            model.model,
            preprocess_image(store.images_dir / r["stored_image"], cfg.image_size).to(device),
            k=cfg.defect_topk,
        )
        model.model.add_defect_features(feats)
    model.model.resample_normal_bank()

    # 阈值重校准并注入（FR-6.5）
    ok_scores = score_images(model, ok_calibration_images(category, cfg), cfg)
    parent_metrics = json.loads(
        (Registry(cfg.models_root).version_dir(category, parent) / "metrics.json").read_text()
    )
    version = finalize_version(
        category, cfg, model,
        ok_scores=ok_scores,
        metrics={},  # 先落盘，验证后补写
        parent=parent,
        feedback_applied=len(effective_rows),
    )
    # 自动验证并与父版本对比（FR-3.4/FR-6.5）
    report = validate_full(category, version, cfg)
    metrics = {**report["metrics"], "threshold": model_threshold(cfg, category, version)}
    vdir = Registry(cfg.models_root).version_dir(category, version)
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    warning = auroc_drop_warning(parent_metrics.get("image_AUROC"), metrics.get("image_AUROC"))
    return {"version": version, "metrics": metrics, "warning": warning, "preview": pv}


def model_threshold(cfg: Config, category: str, version: str) -> float:
    from dino_exp.infer import load_threshold as _lt

    return _lt(Registry(cfg.models_root).version_dir(category, version))
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_retrain.py -v
```

预期：4 个测试全过。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/retrain.py tests/test_retrain.py
git commit -m "feat: 再训练管线（钉住并入/top-k 缺陷库/阈值重校准/父版本对比告警）"
```

---

## Task 11: CLI `cli.py`

**Files:**
- Create: `src/dino_exp/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试 `tests/test_cli.py`**

```python
from click.testing import CliRunner

from dino_exp.cli import main


def test_help_lists_all_commands():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["dataset", "train", "validate", "test", "feedback", "retrain",
                "versions", "rollback", "export", "unstage", "ui"]:
        assert cmd in result.output


def test_dataset_list_empty(tmp_path, monkeypatch):
    from dino_exp.config import Config

    monkeypatch.setattr("dino_exp.cli.load_config", lambda p=None: Config(data_root=tmp_path / "d"))
    result = CliRunner().invoke(main, ["dataset", "list"])
    assert result.exit_code == 0
    assert "无数据集" in result.output or "category" in result.output.lower()


def test_feedback_requires_label():
    result = CliRunner().invoke(main, ["feedback", "--image", "x.png"])
    assert result.exit_code != 0  # --label 必填


def test_retrain_prints_preview_and_aborts_without_yes(tmp_path, monkeypatch):
    from dino_exp.config import Config

    monkeypatch.setattr("dino_exp.cli.load_config", lambda p=None: Config(
        data_root=tmp_path / "d", models_root=tmp_path / "m", feedback_root=tmp_path / "f"))
    # cli 在命令函数内 `from dino_exp.retrain import preview_retrain`（延迟导入），
    # 因此补丁目标是源模块属性，调用时生效
    monkeypatch.setattr("dino_exp.retrain.preview_retrain", lambda c, cfg: {
        "current_version": "v001", "ok": 1, "ng": 2, "suspicious": [], "conflicts": []})
    result = CliRunner().invoke(main, ["retrain", "--category", "bottle"], input="n\n")
    assert result.exit_code == 0
    assert "v001" in result.output and "取消" in result.output
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_cli.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.cli'`。

- [ ] **Step 3: 写 `src/dino_exp/cli.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import click

from dino_exp.config import load_config
from dino_exp.errors import DinoError


@click.group()
@click.option("--config", "config_path", default=None, help="配置文件路径，默认 config/default.yaml")
@click.pass_context
def main(ctx, config_path):
    """dino — DINO 无监督异常检测试验环境 CLI。"""
    ctx.obj = load_config(config_path)


def _err(fn):
    """统一异常出口：DinoError → 友好报错（含修复建议），退出码 2。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except DinoError as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


# ---------- dataset ----------

@main.group()
def dataset():
    """数据集管理。"""


@dataset.command("list")
@click.pass_obj
@_err
def dataset_list(cfg):
    from dino_exp.datasets import list_datasets

    rows = list_datasets(cfg)
    if not rows:
        click.echo("无数据集。请先 download 或 import。")
        return
    for info in rows:
        defects = ", ".join(f"{k}:{v}" for k, v in info.defect_types.items()) or "-"
        click.echo(f"{info.category}\ttrain/good={info.train_good}\ttest/good={info.test_good}\t缺陷=[{defects}]"
                   f"{'\t[降级:无NG图]' if info.degraded else ''}")


@dataset.command("download")
@click.option("--category", required=True)
@click.pass_obj
@_err
def dataset_download(cfg, category):
    from dino_exp.datasets import import_mvtec

    dest = import_mvtec(category, cfg)
    click.echo(f"已下载并转换: {dest}")


@dataset.command("import")
@click.option("--category", required=True)
@click.option("--label", type=click.Choice(["ok", "ng"]), required=True)
@click.option("--defect-type", default=None)
@click.argument("images", nargs=-1, required=True)
@click.pass_obj
@_err
def dataset_import(cfg, category, label, defect_type, images):
    from dino_exp.datasets import import_images

    paths = import_images(list(images), category, label, defect_type, cfg)
    click.echo(f"已导入 {len(paths)} 张到 {paths[0].parent}")


@dataset.command("preview")
@click.option("--category", required=True)
@click.pass_obj
@_err
def dataset_preview(cfg, category):
    from dino_exp.datasets import dataset_info

    info = dataset_info(category, cfg)
    click.echo(json.dumps({
        "category": info.category, "train_good": info.train_good,
        "test_good": info.test_good, "defect_types": info.defect_types,
        "has_masks": info.has_masks, "degraded": info.degraded,
    }, ensure_ascii=False, indent=2))


# ---------- train / validate / test ----------

@main.command()
@click.option("--category", required=True)
@click.option("--backbone", default=None, help="骨干别名，默认取配置")
@click.option("--coreset", "coreset", type=float, default=None)
@click.option("--image-size", type=int, default=None)
@click.pass_obj
@_err
def train(cfg, category, backbone, coreset, image_size):
    from dino_exp.config import validate_image_size
    from dino_exp.train import train_model

    if backbone:
        cfg.backbone = backbone
        cfg.layers = cfg.backbone_spec.default_layers
    if coreset:
        cfg.coreset_sampling_ratio = coreset
    if image_size:
        validate_image_size(image_size, cfg.backbone_spec.patch_size)
        cfg.image_size = image_size
    result = train_model(category, cfg)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@main.command()
@click.option("--category", required=True)
@click.option("--version", default=None)
@click.option("--full", "full", is_flag=True)
@click.option("--images", nargs=-1, default=None)
@click.option("--errors-only", is_flag=True)
@click.pass_obj
@_err
def validate(cfg, category, version, full, images, errors_only):
    from dino_exp.validate import filter_errors, validate_full, validate_images

    if images:
        rows = validate_images(category, version, list(images), cfg)
        for r in rows:
            click.echo(f"{r['label']}\tscore={r['score']:.4f}\t{r['heatmap_path']}")
        return
    report = validate_full(category, version, cfg)
    click.echo(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    rows = filter_errors(report["rows"]) if errors_only else report["rows"]
    for r in rows:
        click.echo(f"{r['label_pred']}\tscore={r['score']:.4f}\tgt={r['defect_type']}\t{r['path']}")


@main.command(name="test")
@click.option("--category", required=True)
@click.option("--image", "images", multiple=True, required=True)
@click.option("--version", default=None)
@click.pass_obj
@_err
def test_cmd(cfg, category, images, version):
    from dino_exp.infer import infer_batch

    for r in infer_batch(list(images), version, category=category, cfg=cfg):
        click.echo(f"{r['label']}\tscore={r['score']:.4f}\tthreshold={r['threshold']:.4f}\t{r['heatmap_path']}")


# ---------- feedback / retrain / versions ----------

@main.command()
@click.option("--category", required=True)
@click.option("--image", required=True)
@click.option("--label", type=click.Choice(["ok", "ng"]), required=True)
@click.option("--defect-type", default=None)
@click.option("--score", type=float, default=None, help="预测分数（来自 dino test 输出）")
@click.option("--prediction", default=None)
@click.pass_obj
@_err
def feedback(cfg, category, image, label, defect_type, score, prediction):
    from dino_exp.feedback.store import FeedbackStore
    from dino_exp.models.registry import Registry

    version = Registry(cfg.models_root).current(category)
    rec = FeedbackStore(cfg.feedback_root, category).stage({
        "image_path": image, "model_version": version,
        "prediction": prediction or "unknown", "score": score if score is not None else 0.0,
        "human_label": label, "defect_type": defect_type,
    })
    click.echo(f"已暂存反馈 {rec['id']}（{label}）。")


@main.command()
@click.option("--category", required=True)
@click.option("--yes", is_flag=True, help="跳过确认直接执行")
@click.pass_obj
@_err
def retrain(cfg, category, yes):
    from dino_exp.retrain import preview_retrain
    from dino_exp.retrain import retrain as do_retrain

    pv = preview_retrain(category, cfg)
    click.echo(json.dumps(
        {"current_version": pv["current_version"], "ok": pv["ok"], "ng": pv["ng"],
         "suspicious": len(pv["suspicious"]), "conflicts": pv["conflicts"]},
        ensure_ascii=False, indent=2))
    if pv["suspicious"]:
        click.echo(f"警告: {len(pv['suspicious'])} 条可疑反馈（OK 高分 / NG 低分），请先确认。")
    if pv["conflicts"]:
        click.echo(f"警告: 同图冲突 {len(pv['conflicts'])} 起，将以最新一条为准。")
    if not yes and not click.confirm("确认执行再训练？"):
        click.echo("已取消。")
        return
    result = do_retrain(category, cfg)
    click.echo(json.dumps({k: v for k, v in result.items() if k != "preview"}, ensure_ascii=False, indent=2, default=str))
    if result["warning"]:
        click.echo(f"⚠ {result['warning']}")


@main.command()
@click.option("--category", required=True)
@click.pass_obj
@_err
def versions(cfg, category):
    from dino_exp.models.registry import Registry

    reg = Registry(cfg.models_root)
    cur = reg.current(category)
    for v in reg.list(category):
        click.echo(f"{'*' if v == cur else ' '} {v}")


@main.command()
@click.option("--category", required=True)
@click.argument("version")
@click.pass_obj
@_err
def rollback(cfg, category, version):
    from dino_exp.models.registry import Registry

    Registry(cfg.models_root).rollback(category, version)
    click.echo(f"已回滚到 {version}。")


@main.command()
@click.option("--category", required=True)
@click.option("--version", default=None, help="缺省为当前版本；导出物写入该版本 export/ 目录")
@click.pass_obj
@_err
def export(cfg, category, version):
    """导出当前版本的 OpenVINO 快照（再训练后需对新版本重新导出）。"""
    from dino_exp.infer import export_openvino

    out = export_openvino(category, version, cfg)
    click.echo(f"已导出: {out}")


@main.command()
@click.option("--category", required=True)
@click.argument("feedback_id")
@click.pass_obj
@_err
def unstage(cfg, category, feedback_id):
    """从暂存区单条删除反馈（FR-5.6）。"""
    from dino_exp.feedback.store import FeedbackStore

    if FeedbackStore(cfg.feedback_root, category).remove(feedback_id):
        click.echo(f"已删除反馈 {feedback_id}。")
    else:
        raise DinoError(f"暂存区无反馈 {feedback_id}。可用 `dino retrain --category {category}` 预览现有暂存反馈。")


@main.command()
@click.pass_obj
@_err
def ui(cfg):
    from dino_exp.webui.app import launch

    launch(cfg)
```

- [ ] **Step 4: 运行确认通过**

```bash
python -m pytest tests/test_cli.py -v
```

预期：4 个测试全过。

- [ ] **Step 5: 提交**

```bash
git add src/dino_exp/cli.py tests/test_cli.py
git commit -m "feat: CLI（dataset/train/validate/test/feedback/retrain/versions/rollback/ui）"
```

---

## Task 12: Web UI（Gradio 四页签）

**Files:**
- Create: `src/dino_exp/webui/__init__.py`
- Create: `src/dino_exp/webui/jobs.py`
- Create: `src/dino_exp/webui/app.py`
- Create: `src/dino_exp/webui/dataset_tab.py`
- Create: `src/dino_exp/webui/train_tab.py`
- Create: `src/dino_exp/webui/validate_tab.py`
- Create: `src/dino_exp/webui/test_tab.py`
- Test: `tests/test_webui_jobs.py`

UI 为薄封装（FR-7），全部调用应用层函数。后台机制（设计 §3.7）：训练在 worker 线程执行，进度/日志写内存队列，UI 以 `gr.Timer` 轮询。

- [ ] **Step 1: 写失败测试 `tests/test_webui_jobs.py`（只测任务队列，不起 Gradio）**

```python
import time

from dino_exp.webui.jobs import JobManager


def test_job_runs_and_collects_logs():
    jm = JobManager()

    def work(log):
        log("step 1")
        log("step 2")
        return {"version": "v001"}

    jid = jm.start("train", work)
    for _ in range(50):
        if jm.status(jid)["state"] == "done":
            break
        time.sleep(0.05)
    st = jm.status(jid)
    assert st["state"] == "done"
    assert st["result"] == {"version": "v001"}
    assert "step 1" in "".join(st["logs"])


def test_job_error_captured():
    jm = JobManager()

    def bad(log):
        raise RuntimeError("boom")

    jid = jm.start("train", bad)
    for _ in range(50):
        if jm.status(jid)["state"] != "running":
            break
        time.sleep(0.05)
    st = jm.status(jid)
    assert st["state"] == "error"
    assert "boom" in st["error"]


def test_one_job_per_kind():
    jm = JobManager()
    jm.start("train", lambda log: time.sleep(2))
    try:
        jm.start("train", lambda log: None)
        raise AssertionError("应当拒绝并发的同类任务")
    except Exception as exc:
        assert "进行中" in str(exc)
```

- [ ] **Step 2: 运行确认失败**

```bash
python -m pytest tests/test_webui_jobs.py -x
```

预期：`ModuleNotFoundError: No module named 'dino_exp.webui'`。

- [ ] **Step 3: 写 `src/dino_exp/webui/jobs.py` 与 `__init__.py`（空）**

```python
"""后台任务管理：worker 线程 + 内存日志队列；UI 用 gr.Timer 轮询 status()。"""

from __future__ import annotations

import threading
import traceback
import uuid
from queue import Queue

from dino_exp.errors import DinoError


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn) -> str:
        with self._lock:
            for j in self._jobs.values():
                if j["kind"] == kind and j["state"] == "running":
                    raise DinoError(f"已有 {kind} 任务进行中，请等待完成后再启动。")
            jid = uuid.uuid4().hex[:8]
            queue: Queue = Queue()
            self._jobs[jid] = {"kind": kind, "state": "running", "queue": queue,
                               "logs": [], "result": None, "error": None}

        def run():
            try:
                result = fn(queue.put)
                self._jobs[jid].update(state="done", result=result)
            except Exception:
                self._jobs[jid].update(state="error", error=traceback.format_exc(limit=5))

        threading.Thread(target=run, daemon=True).start()
        return jid

    def status(self, jid: str) -> dict:
        job = self._jobs[jid]
        while not job["queue"].empty():
            job["logs"].append(str(job["queue"].get()))
        return {k: v for k, v in job.items() if k != "queue"}
```

- [ ] **Step 4: 写四个页签与 `app.py`**

`src/dino_exp/webui/dataset_tab.py`：

```python
import gradio as gr

from dino_exp.datasets import dataset_info, import_images, import_mvtec, list_datasets
from dino_exp.errors import DinoError


def build(cfg):
    with gr.Tab("数据集"):
        out = gr.Dataframe(headers=["类别", "train/good", "test/good", "缺陷类型", "降级"],
                           label="数据集列表")

        def refresh():
            return [[i.category, i.train_good, i.test_good,
                     ", ".join(f"{k}:{v}" for k, v in i.defect_types.items()) or "-",
                     "是" if i.degraded else "否"] for i in list_datasets(cfg)]

        with gr.Row():
            cat_dl = gr.Textbox(label="MVTec 类别名", placeholder="bottle")
            btn_dl = gr.Button("下载 MVTec")
            dl_msg = gr.Textbox(label="结果", interactive=False)

        def do_download(cat):
            try:
                return str(import_mvtec(cat, cfg))
            except DinoError as exc:
                return f"错误: {exc}"

        btn_dl.click(do_download, cat_dl, dl_msg)

        with gr.Row():
            cat_im = gr.Textbox(label="类别名")
            label_im = gr.Radio(["ok", "ng"], value="ok", label="标签")
            dt_im = gr.Textbox(label="缺陷类型（NG 必填）")
            files = gr.File(file_count="multiple", label="选择图片")
            btn_im = gr.Button("导入")
            im_msg = gr.Textbox(label="结果", interactive=False)

        def do_import(cat, label, dt, fs):
            try:
                paths = import_images([f.name for f in fs], cat, label, dt or None, cfg)
                return f"已导入 {len(paths)} 张"
            except DinoError as exc:
                return f"错误: {exc}"

        btn_im.click(do_import, [cat_im, label_im, dt_im, files], im_msg)
        cat_info = gr.JSON(label="类别详情")
        cat_im.change(lambda c: dataset_info(c, cfg).__dict__ if c else {}, cat_im, cat_info)
        gr.Timer(5.0).tick(refresh, outputs=out)
```

`src/dino_exp/webui/train_tab.py`：

```python
import gradio as gr

from dino_exp.errors import DinoError
from dino_exp.train import train_model
from dino_exp.webui.jobs import JobManager


def build(cfg, jm: JobManager):
    with gr.Tab("训练"):
        cat = gr.Textbox(label="类别名", placeholder="bottle")
        backbone = gr.Dropdown(
            ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
             "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
            value=cfg.backbone, label="骨干")
        coreset = gr.Slider(0.01, 1.0, value=cfg.coreset_sampling_ratio, label="coreset 采样率")
        image_size = gr.Number(value=cfg.image_size, label="输入尺寸（patch 整数倍）")
        btn = gr.Button("开始训练", variant="primary")
        status = gr.Textbox(label="状态", interactive=False)
        logs = gr.Textbox(label="日志", interactive=False, lines=12)
        result = gr.JSON(label="验证指标（训练完成自动全量验证）")
        state_jid = gr.State(None)

        def start(c, bb, cs, sz):
            run_cfg = type(cfg)(**{**cfg.__dict__, "backbone": bb,
                                   "coreset_sampling_ratio": float(cs), "image_size": int(sz)})
            run_cfg.layers = run_cfg.backbone_spec.default_layers
            try:
                jid = jm.start("train", lambda log: train_model(c, run_cfg))
                return jid, f"训练已启动: {jid}", ""
            except DinoError as exc:
                return None, f"错误: {exc}", ""

        def poll(jid):
            if not jid:
                return "未启动", "", None
            st = jm.status(jid)
            text = "".join(st["logs"])
            if st["state"] == "done":
                return "完成", text, st["result"]
            if st["state"] == "error":
                return f"失败:\n{st['error']}", text, None
            return "运行中...", text, None

        btn.click(start, [cat, backbone, coreset, image_size], [state_jid, status, logs])
        gr.Timer(1.0).tick(poll, state_jid, [status, logs, result])
```

`src/dino_exp/webui/validate_tab.py`：

```python
import gradio as gr

from dino_exp.models.registry import Registry
from dino_exp.validate import filter_errors, validate_full, validate_images


def build(cfg):
    with gr.Tab("验证"):
        cat = gr.Textbox(label="类别名")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)
        errors_only = gr.Checkbox(label="只看误判", value=False)
        btn_full = gr.Button("全量验证", variant="primary")
        metrics = gr.JSON(label="聚合指标")
        rows_out = gr.Dataframe(headers=["判定", "分数", "GT", "路径"], label="逐图结果")

        cat.change(lambda c: gr.update(choices=Registry(cfg.models_root).list(c)),
                   cat, version)

        def do_full(c, v, eo):
            report = validate_full(c, v or None, cfg)
            rows = filter_errors(report["rows"]) if eo else report["rows"]
            table = [[r["label_pred"], round(r["score"], 4), r["defect_type"], r["path"]] for r in rows]
            return report["metrics"], table

        btn_full.click(do_full, [cat, version, errors_only], [metrics, rows_out])

        gr.Markdown("### 选图验证")
        files = gr.File(file_count="multiple", label="上传图片")
        btn_sel = gr.Button("验证所选")
        sel_out = gr.Dataframe(headers=["判定", "分数", "热力图"], label="结果")

        def do_sel(c, v, fs):
            rows = validate_images(c, v or None, [f.name for f in fs], cfg)
            return [[r["label"], round(r["score"], 4), r["heatmap_path"]] for r in rows]

        btn_sel.click(do_sel, [cat, version, files], sel_out)
```

`src/dino_exp/webui/test_tab.py`：

```python
import gradio as gr

from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.infer import infer_image
from dino_exp.models.registry import Registry
from dino_exp.retrain import preview_retrain, retrain


def build(cfg):
    with gr.Tab("测试与反馈"):
        cat = gr.Textbox(label="类别名")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)
        cat.change(lambda c: gr.update(choices=Registry(cfg.models_root).list(c)),
                   cat, version)
        img = gr.Image(type="filepath", label="上传图片")
        btn = gr.Button("测试", variant="primary")
        label_out = gr.Textbox(label="判定", interactive=False)
        score_out = gr.Number(label="异常分数")
        heat_out = gr.Image(label="热力图")
        state_score = gr.State(0.0)
        state_pred = gr.State("")

        def do_test(c, v, path):
            r = infer_image(path, v or None, category=c, cfg=cfg)
            return r["label"], r["score"], r["heatmap_path"], r["score"], r["label"]

        btn.click(do_test, [cat, version, img],
                  [label_out, score_out, heat_out, state_score, state_pred])

        gr.Markdown("### 反馈")
        fb_label = gr.Radio(["ok", "ng"], value="ok", label="实际标签")
        fb_dt = gr.Textbox(label="缺陷类型（NG 可填）")
        fb_btn = gr.Button("提交反馈")
        fb_msg = gr.Textbox(label="结果", interactive=False)

        def do_feedback(c, path, label, dt, score, pred):
            if not path:
                return "请先测试一张图片"
            try:
                rec = FeedbackStore(cfg.feedback_root, c).stage({
                    "image_path": path,
                    "model_version": Registry(cfg.models_root).current(c),
                    "prediction": pred, "score": float(score),
                    "human_label": label, "defect_type": dt or None,
                })
                return f"已暂存 {rec['id']}"
            except DinoError as exc:
                return f"错误: {exc}"

        fb_btn.click(do_feedback,
                     [cat, img, fb_label, fb_dt, state_score, state_pred], fb_msg)

        gr.Markdown("### 再训练与版本")
        pv_btn = gr.Button("预览暂存区")
        pv_out = gr.JSON(label="预览")
        rt_btn = gr.Button("执行再训练", variant="primary")
        rt_out = gr.JSON(label="结果")
        pv_btn.click(lambda c: preview_retrain(c, cfg), cat, pv_out)
        rt_btn.click(lambda c: {k: v for k, v in retrain(c, cfg).items() if k != "preview"},
                     cat, rt_out)

        ver_out = gr.Dataframe(headers=["版本", "当前"], label="版本列表")
        rb_ver = gr.Textbox(label="回滚到版本")
        rb_btn = gr.Button("回滚")
        rb_msg = gr.Textbox(label="结果", interactive=False)

        def list_versions(c):
            reg = Registry(cfg.models_root)
            cur = reg.current(c)
            return [[v, "✓" if v == cur else ""] for v in reg.list(c)]

        def do_rollback(c, v):
            try:
                Registry(cfg.models_root).rollback(c, v)
                return f"已回滚到 {v}"
            except DinoError as exc:
                return f"错误: {exc}"

        gr.Timer(5.0).tick(list_versions, cat, ver_out)
        rb_btn.click(do_rollback, [cat, rb_ver], rb_msg)
```

`src/dino_exp/webui/app.py`：

```python
import gradio as gr

from dino_exp.config import Config
from dino_exp.webui import dataset_tab, test_tab, train_tab, validate_tab
from dino_exp.webui.jobs import JobManager


def launch(cfg: Config) -> None:
    jm = JobManager()
    with gr.Blocks(title="DINO 异常检测试验环境") as demo:
        gr.Markdown("# DINO 无监督异常检测试验环境")
        dataset_tab.build(cfg)
        train_tab.build(cfg, jm)
        validate_tab.build(cfg)
        test_tab.build(cfg)
    demo.queue(default_concurrency_limit=2).launch()


if __name__ == "__main__":
    from dino_exp.config import load_config

    launch(load_config())
```

- [ ] **Step 5: 运行测试 + 手动冒烟 UI**

```bash
python -m pytest tests/test_webui_jobs.py -v   # 3 个测试全过
python -m dino_exp.webui.app                    # 手动打开 http://127.0.0.1:7860 检查四页签渲染
```

- [ ] **Step 6: 提交**

```bash
git add src/dino_exp/webui/ tests/test_webui_jobs.py
git commit -m "feat: Gradio 四页签 UI（数据集/训练/验证/测试与反馈，后台任务+日志轮询）"
```

---

## Task 13: 集成冒烟测试 + bottle 验收基准

**Files:**
- Create: `tests/test_smoke.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 写 `tests/conftest.py` 与 `tests/test_smoke.py`**

`tests/conftest.py`：

```python
import numpy as np
import pytest
from PIL import Image

from dino_exp.config import Config


def _save_png(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (56, 56, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def smoke_cfg(tmp_path):
    """5 张 56x56 小图的合成数据集（56 = 14×4，满足 patch 倍数校验）。"""
    cfg = Config(
        data_root=tmp_path / "data", models_root=tmp_path / "models",
        feedback_root=tmp_path / "feedback", image_size=56,
        coreset_sampling_ratio=0.5, train_batch_size=2, eval_batch_size=2,
        defect_topk=3,
    )
    for i in range(3):
        _save_png(cfg.data_root / "toy" / "train" / "good" / f"t{i}.png", seed=i)
    _save_png(cfg.data_root / "toy" / "test" / "good" / "g0.png", seed=10)
    _save_png(cfg.data_root / "toy" / "test" / "broken" / "b0.png", seed=11)
    return cfg
```

`tests/test_smoke.py`：

```python
"""全链路冒烟：train→validate→test→feedback→retrain→rollback（设计 §6）。

需联网下载 DINOv2 ViT-S 权重（首次 ~90MB），CPU 运行约 2-5 分钟。
运行: python -m pytest tests/test_smoke.py -m slow -v
"""

import json

import pytest
import torch

from dino_exp.feedback.store import FeedbackStore
from dino_exp.infer import infer_image
from dino_exp.models.registry import Registry
from dino_exp.retrain import retrain
from dino_exp.train import train_model
from dino_exp.validate import validate_full

pytestmark = pytest.mark.slow


def test_full_pipeline(smoke_cfg):
    cfg = smoke_cfg
    # 1. 训练 → v001
    r1 = train_model("toy", cfg)
    assert r1["version"] == "v001"
    assert (cfg.models_root / "toy" / "v001" / "normal_bank.pt").exists()
    # 2. 全量验证
    report = validate_full("toy", "v001", cfg)
    assert "threshold" not in report["metrics"]  # threshold 在 metrics.json
    assert (cfg.models_root / "toy" / "v001" / "validation.json").exists()
    # 3. 单图测试
    out = infer_image(cfg.data_root / "toy" / "test" / "good" / "g0.png",
                      category="toy", cfg=cfg, heatmap_dir=cfg.models_root / "hm")
    assert out["label"] in {"OK", "NG"}
    assert json.loads((cfg.models_root / "toy" / "v001" / "metrics.json").read_text())["threshold"] == out["threshold"]
    # 4. 反馈（OK 一张 + NG 一张）
    store = FeedbackStore(cfg.feedback_root, "toy")
    store.stage({"image_path": str(cfg.data_root / "toy" / "test" / "good" / "g0.png"),
                 "model_version": "v001", "prediction": out["label"],
                 "score": out["score"], "human_label": "ok", "defect_type": None})
    store.stage({"image_path": str(cfg.data_root / "toy" / "test" / "broken" / "b0.png"),
                 "model_version": "v001", "prediction": "OK",
                 "score": 0.0, "human_label": "ng", "defect_type": "broken"})
    # 5. 再训练 → v002（钉住 + 缺陷库填充 + 阈值重校准）
    r2 = retrain("toy", cfg)
    assert r2["version"] == "v002"
    banks = torch.load(cfg.models_root / "toy" / "v002" / "normal_bank.pt",
                       weights_only=True)
    assert banks["pinned_count"] > 0  # 钉住特征已入库
    assert banks["defect_bank"].shape[0] > 0  # 缺陷库已填充
    # 6. 回滚 → current 回到 v001
    Registry(cfg.models_root).rollback("toy", "v001")
    assert Registry(cfg.models_root).current("toy") == "v001"
```

- [ ] **Step 2: 运行冒烟测试**

```bash
python -m pytest tests/test_smoke.py -m slow -v
```

预期：约 2-5 分钟（首次含权重下载），1 个测试通过。失败时按报错对照「已核实的 API 要点」排查（重点：`Evaluator` 指标字段、`ManualThreshold` 注入、`Folder` Sequence 参数）。

- [ ] **Step 3: bottle 验收基准（手动执行，写运行说明）**

```bash
# 1. 下载 bottle 类别（约 60MB）
dino dataset download --category bottle
# 2. 训练（CPU 约 ≤10 分钟，GPU 更快）
dino train --category bottle
# 3. 全量验证 → 验收: image_AUROC ≥ 0.90（需求 §1 成功标准 2）
dino validate --category bottle --full
# 4. 误报反馈案例（成功标准 3）：挑一张被判 NG 的 good 图反馈 OK，再训练后复测该图
dino validate --category bottle --full --errors-only
dino feedback --category bottle --image data/bottle/test/good/000.png --label ok --score <上一步分数> --prediction NG
dino retrain --category bottle --yes
dino test --category bottle --image data/bottle/test/good/000.png   # 期望翻转为 OK
dino versions --category bottle && dino rollback --category bottle v001
```

通过标准：`validate` 输出 `image_AUROC ≥ 0.90`；反馈再训练后误报图判定翻转为 OK 且新 AUROC 下降 ≤ 2 个点（CLI 会打印对比与告警）。

- [ ] **Step 4: 提交**

```bash
git add tests/conftest.py tests/test_smoke.py
git commit -m "test: 全链路冒烟测试（slow 标记）与 bottle 验收基准说明"
```

---

## 风险与依赖安装备注

- **安装（CPU）**：`python -m venv .venv && .venv/Scripts/activate` → `pip install -e ".[dev]"`（等价于 `pip install "anomalib[cpu]==2.5.1" click gradio PyYAML pytest`）。anomalib extras 还有 `[cu126]`（NVIDIA GPU）：`pip install "anomalib[cu126]==2.5.1"`。**必须锁 2.5.1**（ViT 骨干支持自该版本起，需求 §5）。
- **OpenVINO（可选）**：`pip install "anomalib[openvino]==2.5.1"`；导出为版本快照，再训练后需重新导出（Task 7 `export_openvino`）。
- **DINOv2 权重**：首次实例化骨干时 timm 自动从 HuggingFace 下载（ViT-S 约 90MB），缓存在用户目录；离线环境需手动下载后放入 timm/HF 缓存。
- **DINOv3 权重**：Meta 官方为 gated，需 `huggingface-cli login`（HF token）或接受 timm 镜像的许可；默认骨干 DINOv2 不受影响（需求 §5/§7）。
- **Windows 注意**：`current` 为普通文本指针文件（非符号链接，免管理员权限）；`DataLoader num_workers=0`（配置默认）；所有路径经 `pathlib`；anomalib 结果目录默认 `results/`（Engine `default_root_dir`）。
- **已知 API 风险点**（冒烟测试首要排查项）：
  1. `Evaluator(test_metrics=[AUROC(), AUPR(), F1Score()])` 的默认 fields 是否直接链通 Patchcore 输出（若不链通，需显式 `fields=["pred_score", "gt_label"]`）。
  2. `ManualThreshold(default_value=t, fields=..., strict=False)` 注入 PostProcessor 私有属性 `_image_threshold_metric/_pixel_threshold_metric`（2.5.1 已核实字段名，锁版本可控）。
  3. `Folder(abnormal_dir=[Path,...])` Sequence 路径在 Windows 下的归一化。
  4. anomalib 指标输出键名（`image_AUROC` 等）以 `engine.test` 实际返回为准，再训练对比读取 `metrics.get("image_AUROC")`，键缺失时退化为无告警。

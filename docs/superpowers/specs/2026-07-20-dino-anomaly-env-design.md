# DINO 无监督异常检测试验环境 — 设计文档

- 日期：2026-07-20
- 状态：待用户确认
- 对应需求文档：`docs/requirements/2026-07-20-dino-anomaly-env-requirements.md`

## 1. 技术选型（已确认的决策）

| 决策点 | 结论 |
|---|---|
| 核心任务 | 无监督缺陷/异常检测（仅 OK 图训练） |
| 训练方式 | 冻结骨干 + PatchCore 特征记忆库 |
| 技术路线 | anomalib（≥ v2.5）做模型引擎，外围应用层自研 |
| 骨干 | timm 加载，DINOv2/DINOv3、ViT-S/B/L 可配置，默认 `vit_small_patch14_dinov2.lvd142m` |
| 设备 | 自动检测 CUDA → CPU；OpenVINO 为可选推理后端 |
| 交互 | CLI（click）+ Web UI（Gradio 四页签），同一应用层 API |
| 数据 | MVTec AD（按类别下载）+ 自有图片，统一目录规范 |
| 输出 | OK/NG 判定 + 异常分数 + 热力图 |
| 反馈 | 双库方案（正常记忆库扩充 + 缺陷原型库） |
| 版本 | 全量版本化 + 可回滚 |

关键事实（2026-07 调研确认）：

- anomalib v2.5.1 支持 timm DINOv2/DINOv3 骨干；`Patchcore(backbone=..., layers=...)` 可传 timm 模型名或自定义 `nn.Module`。
- `Patchcore` 继承 `MemoryBankMixin`，`model.memory_bank` 为公开 `torch.Tensor`，可读写、可子类化扩展。
- `engine.export(export_type="onnx" | "openvino")` 内置导出；OpenVINO 需 `pip install "anomalib[openvino]"`。
- 指标内置：`Engine(image_metrics=["AUROC", ...], pixel_metrics=["AUROC", "AUPRO"])`。
- anomalib 支持 Windows；硬件 extras：`[cpu]` / `[cu126]` 等。

## 2. 分层架构

```
┌─────────────────────────────────────────────┐
│ 交互层   CLI (click)  +  Web UI (Gradio)     │  薄封装，无业务逻辑
├─────────────────────────────────────────────┤
│ 应用层   ExperimentManager                   │
│          ├─ datasets    数据集管理            │
│          ├─ validation  全量/选图验证         │
│          ├─ feedback    反馈存储 + 暂存区      │
│          └─ registry    模型版本库 + 回滚      │
├─────────────────────────────────────────────┤
│ 引擎层   anomalib Engine                     │
│          └─ DualBankPatchcore（子类化          │
│             Patchcore，双库打分）              │
├─────────────────────────────────────────────┤
│ 推理后端 PyTorch (CUDA→CPU 自动) / OpenVINO   │
└─────────────────────────────────────────────┘
```

## 3. 模块设计

### 3.1 `datasets.py` — 数据集管理

- 目录规范：`data/<类别>/{train/good, test/good, test/<缺陷类型>, mask/<缺陷类型>?}`。
- `import_mvtec(category)`：调用 anomalib `MVTecAD` 下载并转换规范。
- `import_images(src, category, label, defect_type?)`：自有图片归入规范目录。
- `list_datasets()`：类别、split 计数、缺陷类型统计。
- 加载统一封装为 anomalib `Folder` datamodule；缺 `test/good` 时从 `train/good` 切 20% 作校准集。

### 3.2 `models/dual_bank.py` — DualBankPatchcore（核心定制）

子类化 `anomalib.models.Patchcore`：

- **正常记忆库**：训练（fit）时按标准 PatchCore 流程提取 patch 特征 → coreset 采样（采样率可配，默认 10%）。
- **缺陷原型库**：新增张量属性，初始为空；仅由 NG 反馈再训练填充。
- **打分**：
  - 无缺陷库时 = 标准 PatchCore：patch 分数 ≈ 正常库最近邻距离 × 孤立度权重；图片分数 = max(patch 分数)。
  - 有缺陷库时：patch 同时查正常库距离 dₙ 与缺陷库距离 d_d；若 d_d < dₙ，分数上调：`score *= (1 + w)`，w 默认 0.5 可配。缺陷库只加分不减分。
- **增量接口**：`add_normal_features(feats)` / `add_defect_features(feats)`，供再训练调用；正常库并入后重新 coreset 采样（上限默认原库 1.5 倍）。
- **阈值**：`calibrate_threshold(ok_scores)`，默认 mean + 3σ，保存于版本元数据。

### 3.3 `feedback/store.py` + `staging.py` — 反馈

- 每条反馈（JSONL + 图片拷贝）：`{id, image_path, stored_image, model_version, prediction, score, human_label(ok/ng), defect_type?, timestamp}`。
- 暂存区 = 当前版本下未应用的反馈集合；`stage()` 写入，`preview()` 汇总（含可疑项检查：OK 反馈但 score > 3×阈值），`apply()` 消费并归档。
- 反馈不改变当前模型，仅再训练时生效。

### 3.4 `models/registry.py` — 版本库

```
models/<实验名>/
├── current -> v002          # 指针文件
└── v002/
    ├── checkpoint.ckpt      # anomalib checkpoint
    ├── normal_bank.pt       # 正常记忆库
    ├── defect_bank.pt       # 缺陷原型库（可空）
    ├── config.yaml          # 骨干、参数快照
    ├── metrics.json         # 验证指标 + 阈值
    └── meta.json            # 父版本、创建时间、应用的反馈数
```

- `create_version(parent?)` / `list()` / `switch(version)` / `rollback(version)`。
- 所有写入先落临时目录，成功后原子替换；`current` 指针最后更新。

### 3.5 `infer.py` — 推理

- `infer_image(path, version?) -> {label, score, threshold, heatmap_path}`。
- `infer_batch(...)`；设备自动选择（`cuda` → `cpu`），可选 `backend=openvino`（导出失败时回退 PyTorch 并提示）。

### 3.6 `cli.py` — 命令行

```
dino dataset list|download|import|preview
dino train --category bottle --backbone dinov2_vits14 [--coreset 0.1 --image-size 256]
dino validate [--version v002] [--full | --images a.jpg b.jpg] [--errors-only]
dino test --image x.jpg [--version v002]
dino feedback --image x.jpg --label ok|ng [--defect-type crack]
dino retrain [--yes]      # 应用暂存反馈 → 新版本（默认先预览需确认）
dino versions / dino rollback v001
dino ui                   # 启动 Gradio
```

### 3.7 `webui/` — Gradio 四页签

| 页签 | 功能 |
|---|---|
| 数据集 | 列表/统计/缩略图；MVTec 按类下载；自有图片导入（选标签/缺陷类型） |
| 训练 | 选类别+骨干+参数 → 后台训练（进度/日志）→ 完成自动跳验证结果 |
| 验证 | 全量验证（指标卡片+逐图列表，可按误判过滤）/ 选图验证（勾选或上传）；结果图可直接反馈 |
| 测试与反馈 | 上传/浏览图片 → 分数+判定+热力图 → 一键 OK/NG 反馈；再训练预览与触发；版本切换/回滚 |

## 4. 核心工作流

1. **训练**：选类别 → 校验数据集 → 加载 Folder datamodule → anomalib Engine.fit（DualBankPatchcore，骨干仅前向）→ coreset → 校准阈值 → 存新版本 → 自动全量验证 → 报告指标。
2. **验证**：选版本 → 全量（聚合指标+逐图结果，可过滤误判）或选图（逐图分数/热力图）→ 误判可直接反馈。
3. **测试反馈**：上传/选图 → 推理 → 显示结果 → 人工标记 OK/NG → 写入暂存区。
4. **再训练**：预览暂存区（数量+可疑项）→ 确认 → OK 反馈特征并入正常库（重采样）、NG 反馈高分 patch 并入缺陷库 → 阈值重校准 → 存新版本（父版本指针）→ 自动验证并与父版本对比 → 超阈值下降告警提示回滚。

## 5. 错误处理

- 启动校验：torch/CUDA 状态、DINO 权重缓存（DINOv3 如需 HF token 给出引导）、数据集结构合规性——早发现，报错附修复建议。
- 权重下载失败 → 提示手动下载与放置路径。
- 再训练暂存区为空 → 拒绝执行。
- 版本操作原子化，失败不破坏当前版本。
- OpenVINO 导出/推理失败 → 回退 PyTorch CPU 并明确提示。

## 6. 测试策略

- **单元测试**（不依赖真实模型，构造假特征向量）：双库打分融合逻辑、孤立度加权、coreset 上限、护栏规则（3×阈值可疑判定）、反馈存取、暂存区状态机、版本注册表保存/切换/回滚原子性。
- **集成冒烟测试**：5 张小图 + ViT-S 骨干，CPU 跑通 train→validate→test→feedback→retrain→rollback 全链路。
- **验收基准**：MVTec AD bottle 类别全量验证，图片级 AUROC ≥ 0.90。

## 7. 项目结构

```
dino/
├── pyproject.toml            # anomalib[cpu|cu126]、gradio、click、pytest
├── config/default.yaml       # 骨干、输入尺寸、coreset 率、融合权重 w、库上限、阈值规则
├── src/dino_exp/
│   ├── cli.py
│   ├── config.py
│   ├── datasets.py
│   ├── models/{dual_bank.py, registry.py}
│   ├── feedback/{store.py, staging.py}
│   ├── infer.py
│   └── webui/{app.py, dataset_tab.py, train_tab.py, validate_tab.py, test_tab.py}
├── data/  models/  feedback/   # 均 gitignore
├── tests/
└── docs/
```

## 8. 实施顺序（概要，详见实现计划）

1. 环境搭建 + anomalib/Patchcore+DINO 骨干 spike 验证（最高风险优先）。
2. 数据集管理 + 基础训练 + 版本库 + CLI。
3. 推理 + 热力图 + 验证。
4. 反馈存储 + 双库再训练 + 回滚。
5. Gradio 四页签 UI。
6. 冒烟测试 + bottle 验收基准。

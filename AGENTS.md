# AGENTS.md — dino 项目指南（面向后续 Agent）

DINO 无监督异常检测试验环境：anomalib 2.5.1（PatchCore + DINOv2/v3 冻结骨干）为引擎，自研应用层实现「训练 → 验证 → 测试 → OK/NG 反馈 → 增量再训练 → 版本化回滚」闭环，CLI + Gradio UI 双入口。用户使用手册见 `README.md`；需求/设计/计划文档见 `docs/`。

## 构建与测试

```bash
# 环境（Python 3.10–3.12，Windows）
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[dev]"

# 测试（一律用 venv 的 python，cwd 为仓库根）
.venv/Scripts/python.exe -m pytest tests/ -q          # 86 passed, 1 skipped（日常）
.venv/Scripts/python.exe -m pytest tests/test_smoke.py -m slow -v   # 全链路冒烟，约 1 分钟

# CLI 冒烟
.venv/Scripts/python.exe -m dino_exp.cli --help
```

- `slow` marker 的测试（真实骨干、真实 Engine.fit）默认被 conftest 钩子跳过，需 `-m slow` 显式运行。
- 单元测试**不得**实例化真实骨干/下载权重：双库逻辑测纯函数，模型层用 `DualBankPatchcoreModel(layers=["layer1"], backbone="resnet18", pre_trained=False)` 小模型，编排层用 fake/monkeypatch。
- DINOv2 权重已缓存于本机 HF cache；新环境下载需 `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 在本网络不可达）。

## 代码结构（src 布局，包名 `dino_exp`）

```
src/dino_exp/
├── config.py            # Config dataclass + 骨干别名表（BACKBONE_ALIASES，含 depth）+ 各类校验
├── errors.py            # DinoError：应用层统一异常，message 必须附修复建议（NFR-4）
├── datasets.py          # 目录规范扫描/导入/MVTec/build_folder（anomalib Folder 构造）
├── train.py             # build_model（Evaluator 挂法！）/train_model(log=回调)/finalize_version/score_images
├── validate.py          # aggregate_metrics（图片级+像素级）/score_test_set/validate_full/validate_images
├── infer.py             # preprocess_image/infer_image/infer_batch/export_openvino/load_model_for_version
├── retrain.py           # 再训练编排（先建版本后消费暂存区！）
├── models/
│   ├── dual_bank.py     # 核心：DualBankPatchcore(Model)、纯函数层、save/load_banks
│   └── registry.py      # 版本库：原子写入、current 指针文件（非 symlink）
├── feedback/{store.py,staging.py}  # JSONL 暂存/归档、effective/conflicts/preview
├── cli.py               # click 薄封装，应用层 import 全部延迟到命令函数内
└── webui/               # Gradio 四页签薄封装 + jobs.py（后台任务队列）
```

## 必须遵守的约定

1. **阈值单一来源**：mean+3σ 校准 → 注入 PostProcessor（`apply_threshold`，幂等）→ metrics.json → 判定/导出共用。不要在应用层另算阈值。engine.test 必须在阈值注入**之后**跑（否则 F1 口径漂移）。
2. **存储格式**：`normal_bank.pt` 一律为 `save_banks` 字典 `{memory_bank, defect_bank, pinned_count, base_bank_size}`；registry 的 `defect_bank.pt` 只是 `torch.empty(0)` 占位。`load_banks` 在 `model.to(device)` **之前**调用。
3. **metrics.json 键名**：`image_AUROC/image_AUPR/image_F1Score/pixel_AUROC/pixel_AUPRO/threshold`（+retrain 的 `parent_threshold`）。改键名必须全链路同步。
4. **钉住（pin）语义**：OK 反馈特征插入 memory_bank 前段钉住区，不参与 coreset 淘汰、不计入上限（上限=基础版本库×1.5）。这是「反馈后判定翻转」的机制保证，不要破坏。
5. **缺陷库只加分不减分**：`d_d < d_n` 时 patch 分数 ×(1+w)，注入在 `nearest_neighbors` 的 `n_neighbors==1` 支路（support-sample 查询不受影响）。
6. **再训练顺序**：先用 `staging.effective(store.staged())` 算生效集合 → 建版本 → 全部成功后才 `store.apply()` 消费暂存区（失败不丢反馈）。不要调回先 apply。
7. **UI/CLI 都是薄封装**：业务逻辑只进应用层（datasets/train/validate/infer/retrain/feedback/registry），双入口调用同一函数。
8. **原子写**：版本库（tmp 目录+rename，current 最后更新）、反馈 JSONL（tmp+os.replace）、数据导入（先全量校验后拷贝）。新写文件路径遵循同样标准。
9. **git**：`data/ models/ feedback/ results/ outputs/ datasets/` 均已 gitignore（根锚定），运行时产物不入库；commit 前缀 `feat:/fix:/test:/docs:/chore:`。

## anomalib 2.5.1 关键事实（已源码核实，勿凭记忆改动）

- `Patchcore(backbone=, layers=, ...)`；ViT 骨干 layers 写 `blocks.<int>`（默认 blocks.11，spike 验证过）；timm 名如 `vit_small_patch14_dinov2.lvd142m`。
- 打分路径：`PatchcoreModel.forward → nearest_neighbors(n_neighbors=1) → compute_anomaly_score`；`memory_bank` 是注册 buffer（`model.model.memory_bank`）；coreset 用 `KCenterGreedy`（`subsample_embedding` 已 deprecated）。
- **Evaluator**：指标必须显式 fields+prefix，如 `AUROC(fields=["pred_score","gt_label"], prefix="image_", strict=False)`，裸 `AUROC()` 运行期必崩；像素级 `fields=["anomaly_map","gt_mask"]`。
- **阈值注入**：`ManualThreshold(default_value=t)`（**无** fields/strict 参数）替换 PostProcessor 的 `_image_threshold_metric/_pixel_threshold_metric` 并直写 buffer。
- `Folder(abnormal_dir=...)` 接受 Sequence（多缺陷目录合并）；`normal_split_ratio` 是**死参数**，真实切分走 `test_split_ratio` + `seed=42`；校准集单一事实源 = `dm.setup("fit")` 后 `dm.val_data.samples` 过滤 `label_index==0`。
- eval 模式 `model.model(tensor)` 返回 `InferenceBatch(pred_score=raw 未归一化, anomaly_map=)`——与 mean+3σ 阈值同口径。
- 预处理口径：float 空间 resize（ToDtype→Resize→Normalize ImageNet），`preprocess_image` 已对齐，勿改顺序。

## 环境坑（本机实测）

- 网络：huggingface.co 不可达 → `HF_ENDPOINT=https://hf-mirror.com`；MVTec 官方源是 4.9GB 全量包，可用镜像单类别替代。
- Windows 终端 GBK：rich 日志可能 UnicodeEncodeError → 设 `PYTHONUTF8=1`。
- 杀 Gradio 进程：Git Bash `kill` 可能杀不掉，`netstat -ano | grep 7860` 找 PID 后 `taskkill //PID <pid> //F`。
- OpenVINO 导出是**版本快照**（库烘焙进图），再训练后需对新版本重新导出；`anomalib[openvino]` 为可选依赖，导出未端到端实测。

## 当前状态（2026-07-20）

分支 `feat/dino-exp-impl`（待合并 main）：13 个计划任务全部完成，验收全过（bottle image_AUROC=1.0；误报反馈翻转 NG→OK 且 AUROC 不降；CLI+UI 全流程各走通一遍）。Known issues 见 README 文末与终审报告（UI 缺 export/unstage 入口、验证页直接反馈未做、OpenVINO 未实测、DINOv3 gated 需 token）。

# DINO 无监督异常检测试验环境 — 使用手册

基于 DINOv2/DINOv3 特征的无监督工业异常检测：只用 OK（良品）图片训练，自动判定测试图 OK/NG 并输出异常热力图；识别错误时人工反馈，反馈累积后增量再训练，生成可回滚的新版本模型。

- 引擎：anomalib 2.5.1（PatchCore + DINO 骨干，冻结权重，无需 GPU 训练）
- 入口：CLI（`dino` 命令）+ Web UI（Gradio 四页签），功能一一对应
- 平台：Windows（CPU 可跑全流程，有 NVIDIA GPU 自动加速）

---

## 0. 检测原理（DINO 增强版 PatchCore）

本环境采用的无监督异常检测方法是 **PatchCore 算法框架 + DINO 特征提取器**的组合，外加自研的双库反馈机制：

**算法层 — PatchCore**
- 训练时只用 OK 图：把每张图切成 patch，提取特征后存入**正常记忆库**（coreset 采样压缩，控制库大小）
- 判定时对测试图的每个 patch 找它在记忆库中的最近邻，**距离越远越异常**；最异常 patch 的分数经孤立度加权后成为图片分数
- 输出：图片级 OK/NG 判定 + 异常分数 + 像素级热力图（缺陷定位）

**特征层 — DINOv2/DINOv3（替换原版骨干）**
- 原版 PatchCore 用 WideResNet50（ImageNet 监督预训练 CNN）；本环境换成 **DINO（自监督预训练 ViT）**，对工业缺陷这类分布外区域更敏感（AnomalyDINO、Dinomaly 等同路线工作的共同选择）
- 骨干**冻结不训练**，所以建库只需前向推理，CPU 也能跑，有 GPU 自动加速

**反馈机制（自研双库）**
- OK 反馈 → 特征**钉住**并入正常记忆库（不被淘汰、不计入库上限，保证误报图翻转）
- NG 反馈 → 高分 patch 特征入**缺陷原型库**，此后相似缺陷会被加分检出（只加分不减分）
- 阈值自动校准（OK 校准图分数的 mean+3σ），训练/判定/导出三处共用同一阈值

> 备选方案说明：spike 阶段曾用 AnomalyDINO（纯 DINO 特征比对，training-free）做对照 baseline，未选为主方案——PatchCore 的打分/热力图生态更成熟，双库扩展更顺。

---

## 1. 环境搭建

```bash
# 需要 Python 3.10–3.12
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# 可选：NVIDIA GPU 用户改装 CUDA 版 torch（实测 RTX 3060 + cu126 可用，约 2-3GB 下载）
.venv/Scripts/python.exe -m pip install "torch==2.13.0+cu126" "torchvision==0.28.0+cu126" --index-url https://download.pytorch.org/whl/cu126
# 验证：.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 装好后无需改任何代码：训练/推理自动检测并使用 GPU（设备自适应）

# 可选：OpenVINO 推理加速
.venv/Scripts/python.exe -m pip install "anomalib[openvino]==2.5.1"
```

**网络说明**：首次使用会从 HuggingFace 下载 DINOv2 权重（约 90MB，只下一次）。无法直连 huggingface.co 时先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Git Bash
# 或 Windows CMD: set HF_ENDPOINT=https://hf-mirror.com
```

**Windows 控制台中文乱码/编码报错**：建议设 `PYTHONUTF8=1`（GBK 终端下 rich 日志可能报编码错误）。

验证安装：

```bash
.venv/Scripts/python.exe -m pytest tests/ -q      # 86 passed, 1 skipped
.venv/Scripts/python.exe -m dino_exp.cli --help    # 或装好后直接用 dino --help
```

> 下文命令均可用 `dino ...`（安装后的入口）或 `.venv/Scripts/python.exe -m dino_exp.cli ...` 两种形式。

## 1.5 便捷脚本（推荐）

`scripts/` 下提供三组开箱即用的脚本（`.bat` 供 Windows 双击/CMD，`.sh` 供 Git Bash；均自动使用 `.venv` 并预设 HF 镜像与 UTF-8 环境，`-h` 查看帮助）：

| 脚本 | 用途 | 示例 |
|---|---|---|
| `scripts/run_ui.bat` / `.sh` | 启动 Web 界面 | `scripts\run_ui.bat --port 8080` |
| `scripts/dino_cli.bat` / `.sh` | 命令行入口（透传所有 dino 命令） | `scripts\dino_cli.bat train --category bottle` |
| `scripts/run_tests.bat` / `.sh` | 运行测试 | `scripts\run_tests.bat --slow` |

- `run_tests`：无参数=快速单元测试（约 10 秒）；`--slow`=全链路冒烟（约 1 分钟）；`--all`=全部。
- `dino_cli`：无参数或 `-h` 显示常用流程速查表 + 完整命令帮助。
- `run_ui`：默认仅本机访问（127.0.0.1:7860），`--port` 可换端口；**局域网访问**用 `scripts\run_ui.bat --host 0.0.0.0`，然后局域网内其他机器访问 `http://<本机IP>:7860`（注意：UI 无鉴权，局域网内任何人都可操作，仅在可信网络使用；本机 IP 用 `ipconfig` 查看）。

## 2. 快速开始（10 分钟跑通全流程）

```bash
# 1. 准备数据：下载 MVTec AD 的 bottle 类别（或放自己的图，见 §3）
dino dataset download --category bottle

# 2. 训练基础模型（CPU 约 2 分钟；自动全量验证并输出指标）
dino train --category bottle

# 3. 查看验证结果（图片级 + 像素级指标）
dino validate --category bottle --full

# 4. 单图测试：判定 + 分数 + 热力图
dino test --category bottle --image data/bottle/test/good/000.png

# 5. 发现误判？反馈真实标签
dino feedback --category bottle --image data/bottle/test/good/018.png --label ok

# 6. 再训练（先预览暂存反馈，确认后生成新版本 v002）
dino retrain --category bottle --yes

# 7. 复测同一张图 → 判定应已翻转；查看/回滚版本
dino test --category bottle --image data/bottle/test/good/018.png
dino versions --category bottle
dino rollback --category bottle v001
```

Web UI 方式：

```bash
dino ui   # 打开 http://127.0.0.1:7860
```

四个页签：**数据集**（列表/下载/导入/图片画廊浏览）→ **训练**（选类别+骨干+参数，后台运行带日志）→ **验证**（全量指标+逐图结果+误判过滤；选图验证可从数据集直接选图或上传）→ **测试与反馈**（从数据集选图或上传测试、标 OK/NG 反馈、预览/执行再训练、版本回滚）。所有错误提示默认为一句话摘要（含修复建议），需要排查时可展开「错误详情」查看完整堆栈。

## 3. 数据准备

### 目录规范

```
data/<类别>/
├── train/good/          # OK 训练图（必需，训练只用这些）
├── test/good/           # OK 测试图（阈值校准用；缺失时自动从 train/good 切 20%）
├── test/<缺陷类型>/      # NG 测试图（如 broken、scratch；可多个缺陷类型）
└── mask/<缺陷类型>/      # 缺陷掩码（可选；有则输出像素级指标 pixel_AUROC/AUPRO）
```

### 导入自己的图片

```bash
# OK 图：--split 决定去向后（NG 图始终进 test/<缺陷类型>）
dino dataset import --category 我的产品 --label ok --split auto 图1.jpg 图2.jpg
#   --split auto  按文件名 8:2 自动分入 train/good（训练集）与 test/good（阈值校准集），从零建数据集推荐
#   --split train 全部进 train/good
#   --split test  全部进 test/good（默认）

# NG 图（缺陷类型选填，默认进 test/unknown/；指定则按类型归档）
dino dataset import --category 我的产品 --label ng --defect-type 划痕 ng1.jpg
```

> **注意**：训练必须有 `train/good` 图片。只导入到 test/good 的类别会被标记为不完整（缺训练图）。多类型/多工位场景：每个「类型+工位」建一个类别（如 `blister_silian_15_e0`），各自训练独立模型，效果最好。

UI 中：数据集页签 → 填类别名、选标签、选 OK 图去向、选文件 → 导入；「查看图片」可浏览类别下的图片画廊。

### 查看数据集

```bash
dino dataset list                 # 所有类别概览（不完整类别会标注原因）
dino dataset preview --category bottle   # 单类别详情（JSON）
dino dataset fix --category 我的产品     # 自动修复：test/good 的 OK 图按 8:2 整理出 train/good
dino dataset delete --category 我的产品  # 删除整个类别（有确认；--yes 跳过）
```

> **提示**：只有 OK 图、没有 NG 测试图时为「降级」模式——可训练和逐图打分，但不输出 AUROC/F1 等聚合指标。缺 train/good 的类别为「不完整」，用 `dataset fix` 自动整理或在 UI「类别管理」中一键修复/删除。

## 4. 训练

```bash
dino train --category bottle                          # 默认配置
dino train --category bottle --backbone dinov2_vitb14 # 换骨干
dino train --category bottle --coreset 0.05 --image-size 518
dino train --category bottle --tiles auto             # 切块模式（大图小缺陷）
```

**切块（tiling）**：针对大图上的小缺陷（如 2000×1500 图上的 10-15px 黑点）。原图切成小块分别提特征，等效提升分辨率：

- `--tiles off`（默认）不切；`auto`（推荐）按**固定尺寸 T×T** 切块——T = P×(image_size/patch)，全图统一块尺寸、无 resize 畸变、一次 batch 前向；显式 `2x2/3x3/4x4/6x6/8x8` 为旧的比例网格方案（向后兼容）
- `--tile-px N`：auto 模式的目标 patch 覆盖像素 P（默认 20；2048×1536 图建议 16 → T=256，分辨率与旧 8x8 相当）
- 切块配置（方案/T/重叠率）**随模型版本保存**，推理时自动按版本配置切块——训练后无需再管
- 边界：图片小于切块要求时自动走整图推理（小图放大对齐）；长条图 pad 保方形；边缘缺陷由 15% 重叠覆盖，拼接用重叠区中点归属避免重复计分
- 代价：推理时间约 × 块数（2048×1536 约 42 块/图，GPU 一次 batch 处理）

**输入尺寸参考**（2000×1500 原图，DINOv2）：224 不切≈识别 160px 缺陷；896 不切≈40-60px；224+固定切块 T=256≈10-15px（实测 10px 黑点检出并红框标注）

**骨干别名**（`--backbone`）：

| 别名 | 说明 |
|---|---|
| `dinov2_vits14`（默认） | DINOv2 ViT-S，最快，CPU 首选 |
| `dinov2_vitb14` / `dinov2_vitl14` | DINOv2 ViT-B/L，更准更慢 |
| `dinov3_vits16` / `dinov3_vitb16` / `dinov3_vitl16` | DINOv3（权重 gated，需 HF token，见 §8） |

- `--image-size` 默认 224，必须是 patch 的整数倍（DINOv2=14，DINOv3=16），不满足会报错并给出建议值。
- 每次训练生成一个新版本（v001、v002…），存于 `models/<类别>/vNNN/`（特征库+配置+指标+元数据）。
- 阈值自动校准：OK 校准图分数的 mean+3σ，训练/判定/导出口径一致。
- 训练完成自动跑全量验证，指标写入版本目录的 `metrics.json` 与 `validation.json`。

## 5. 验证

```bash
dino validate --category bottle --full               # 全量：聚合指标 + 逐图结果
dino validate --category bottle --full --errors-only # 只看误判（误报/漏报）
dino validate --category bottle --version v001       # 指定版本
dino validate --category bottle --images a.jpg --images b.jpg  # 选图验证（不出聚合指标）
```

指标说明：`image_AUROC`（≥0.90 为良好）、`image_AUPR`、`image_F1Score`；有 mask 时另有 `pixel_AUROC`/`pixel_AUPRO`（缺陷定位质量）。`threshold` 为当前判定线。

## 6. 测试与反馈

```bash
# 测试（可多张）
dino test --category bottle --image x.jpg
# 输出：判定(OK/NG)  分数  阈值  热力图路径

# 反馈：实际是 OK（误判为 NG 的误报）
dino feedback --category bottle --image x.jpg --label ok

# 反馈：实际是 NG（漏报），可带缺陷类型
dino feedback --category bottle --image y.jpg --label ng --defect-type 划痕

# 撤销某条暂存反馈
dino unstage --category bottle <反馈id>
```

- 反馈进入**暂存区**，不影响当前模型，下次再训练时生效（保证测试结果可复现）。
- UI 中更直观：测试与反馈页签 → 上传图片 → 测试 → 看热力图 → 选「实际标签」→ 提交反馈。

## 7. 再训练与版本管理

```bash
dino retrain --category bottle          # 先显示预览（OK/NG 数量、可疑项、冲突），询问确认
dino retrain --category bottle --yes    # 跳过确认
```

再训练做什么：

- OK 反馈 → 特征**钉住**并入正常记忆库（不被淘汰，保证误报图翻转）
- NG 反馈 → 高分 top-k patch 特征入**缺陷原型库**（此后相似缺陷会被加分检出）
- 阈值自动重校准；生成新版本并自动全量验证，与父版本对比 AUROC
- 若新 AUROC 下降超过 2 个点，打印告警并建议回滚

**护栏**：OK 反馈但分数远超阈值（>3×）、或 NG 反馈但分数低于阈值（漏报型），预览中会标为「可疑」请二次确认——防止错误标签污染特征库。

```bash
dino versions --category bottle        # 版本列表（* 为当前）
dino rollback --category bottle v001   # 回滚（只切指针，历史版本不删）
dino export --category bottle          # 导出当前版本的 OpenVINO 快照（可选）
```

> OpenVINO 导出物是**版本快照**：特征库烘焙进模型文件，再训练出新版本后需重新 `dino export`。

## 8. 常见问题

**DINOv3 权重下载失败 / 401**
Meta 官方 DINOv3 权重是 gated 的：先到 HuggingFace 对应页面同意条款，然后 `huggingface-cli login` 配置 token。默认骨干 DINOv2 无此限制。

**huggingface.co 连不上**
设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试（见 §1）。

**MVTec 下载太慢**
anomalib 官方源是全量 4.9GB 压缩包。也可从镜像站下载单类别后按 §3 目录规范手动放置。

**训练/推理慢**
CPU + ViT-S 下：单图推理约 1-3 秒，bottle（250 图）建库约 2 分钟。有 NVIDIA GPU 会自动使用；或 `pip install "anomalib[openvino]==2.5.1"` 后 `dino export` 用 OpenVINO 加速。

**Windows 终端乱码或 UnicodeEncodeError**
设 `PYTHONUTF8=1` 后重跑。

**出错后在哪里查日志？**
所有关键操作（训练阶段、反馈、再训练、版本创建）和错误（含完整堆栈）都写入 `logs/dino.log`（滚动切割 5MB×3，CLI 和 UI 共用）。排查问题先看这个文件。

**再训练提示「暂存区为空」**
先 `dino feedback ...` 添加反馈。反馈只在再训练时生效。

**下载中断后重跑报「目标已存在」**
上次中断留下了半成品目录。加 `--force` 重新下载（会先删除旧目录）：`dino dataset download --category bottle --force`。

## 9. 配置文件

`config/default.yaml`（也可用 `dino --config 路径.yaml` 指定）：骨干、layers、image_size、coreset 采样率、融合权重 w（缺陷库加分强度，默认 0.5）、库上限倍数（默认 1.5×）、NG 反馈 top-k（默认 10）、阈值 sigma（默认 3.0）、可疑因子（默认 3.0）、各根目录。未知键或非法值会在启动时报错并给出修复建议。

## 10. 目录说明

```
data/        数据集（git 不跟踪）
models/      模型版本库 models/<类别>/vNNN/（git 不跟踪）
feedback/    反馈数据（git 不跟踪）
outputs/     热力图输出（git 不跟踪）
results/     anomalib 训练日志（git 不跟踪）
config/      配置文件
scripts/     便捷脚本（run_ui / dino_cli / run_tests，.bat + .sh，-h 查看帮助）
src/dino_exp/  源代码
tests/       测试（pytest；-m slow 跑全链路冒烟，约 1 分钟）
docs/        需求/设计/计划文档
```

---

*参考指标（本机 CPU 实测，MVTec bottle）：image_AUROC=1.0，pixel_AUROC≈0.95，误报反馈再训练后判定翻转且 AUROC 无下降。*

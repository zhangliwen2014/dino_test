#!/usr/bin/env bash
# dino_cli.sh — DINO 异常检测试验环境命令行入口（Git Bash 用）
set -euo pipefail
cd "$(dirname "$0")/.."

show_help() {
    cat <<'EOF'
用法: scripts/dino_cli.sh <命令> [参数...]

全部功能与 Web UI 一一对应。常用流程:

  1. 准备数据
     scripts/dino_cli.sh dataset download --category bottle     # 下载 MVTec 类别
     scripts/dino_cli.sh dataset import --category 产品A --label ok 图1.jpg   # 导入自己的图
     scripts/dino_cli.sh dataset list                           # 查看数据集

  2. 训练（生成基础模型 v001）
     scripts/dino_cli.sh train --category bottle
     可选: --backbone dinov2_vitb14  --coreset 0.05  --image-size 518

  3. 验证（指标 + 逐图结果）
     scripts/dino_cli.sh validate --category bottle --full
     scripts/dino_cli.sh validate --category bottle --full --errors-only   # 只看误判

  4. 测试（判定 + 分数 + 热力图）
     scripts/dino_cli.sh test --category bottle --image x.jpg

  5. 反馈（发现误判时标记真实标签）
     scripts/dino_cli.sh feedback --category bottle --image x.jpg --label ok
     scripts/dino_cli.sh feedback --category bottle --image y.jpg --label ng --defect-type 划痕
     scripts/dino_cli.sh unstage --category bottle <反馈id>     # 撤销某条反馈

  6. 再训练（应用反馈 → 新版本）
     scripts/dino_cli.sh retrain --category bottle              # 先预览再确认
     scripts/dino_cli.sh retrain --category bottle --yes        # 跳过确认

  7. 版本管理
     scripts/dino_cli.sh versions --category bottle
     scripts/dino_cli.sh rollback --category bottle v001

  其他:
     scripts/dino_cli.sh export --category bottle               # OpenVINO 快照导出
     scripts/dino_cli.sh ui                                     # 启动 Web 界面
     scripts/dino_cli.sh <命令> --help                          # 查看单个命令的详细参数

说明: 自动使用 .venv，并预设 HF_ENDPOINT 镜像与 PYTHONUTF8。
EOF
}

PY=".venv/Scripts/python.exe"
if [[ ! -x "$PY" ]]; then
    echo "未找到 .venv，请先运行: python -m venv .venv && .venv/Scripts/python.exe -m pip install -e \".[dev]\"" >&2
    exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUTF8=1

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    echo "---------------- dino --help ----------------"
    "$PY" -m dino_exp.cli --help
    exit 0
fi

exec "$PY" -m dino_exp.cli "$@"

#!/usr/bin/env bash
# run_ui.sh — 启动 DINO 异常检测试验环境 Web 界面（Git Bash 用）
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
用法: scripts/run_ui.sh [--port 端口] [--host 地址]

启动 Web 界面（Gradio 四页签：数据集 / 训练 / 验证 / 测试与反馈）。
启动后在浏览器打开提示的地址（默认 http://127.0.0.1:7860）。

选项:
  --port N    监听端口（默认 7860，端口被占用时换一个）
  --host IP   绑定地址（默认 127.0.0.1 仅本机；0.0.0.0 允许局域网访问——
              注意 UI 无鉴权，局域网内任何人都可操作，仅在可信网络使用）
  -h, --help  显示本帮助

说明:
  - 自动使用 .venv 虚拟环境，并预设 HF_ENDPOINT 镜像与 PYTHONUTF8
  - 停止：在运行窗口按 Ctrl+C
  - 首次使用需先准备数据并训练，例如：
      scripts/dino_cli.sh dataset download --category bottle
      scripts/dino_cli.sh train --category bottle
EOF
    exit 0
fi

PORT=7860
HOST="127.0.0.1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        *) echo "未知参数: $1（用 -h 查看帮助）" >&2; exit 2 ;;
    esac
done

PY=".venv/Scripts/python.exe"
[[ -x "$PY" ]] || { echo "未找到 .venv，请先运行: python -m venv .venv && .venv/Scripts/python.exe -m pip install -e \".[dev]\"" >&2; exit 1; }

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUTF8=1

echo "启动 Web UI: http://${HOST}:${PORT}  （Ctrl+C 停止）"
"$PY" -m dino_exp.cli ui --port "$PORT" --host "$HOST"

#!/usr/bin/env bash
# run_tests.sh — 测试运行脚本（Git Bash 用）
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
用法: scripts/run_tests.sh [选项] [-- 额外 pytest 参数]

选项:
  （无参数）    快速测试：全部单元测试（约 10 秒，86 个）
  --slow        只跑全链路冒烟测试（真实骨干，约 1 分钟：
                train→validate→test→feedback→retrain→rollback）
  --all         全部测试 = 单元测试 + 冒烟测试
  -v            详细输出（可与其他选项组合）
  -h, --help    显示本帮助

示例:
  scripts/run_tests.sh                    # 日常开发用
  scripts/run_tests.sh --slow             # 改完训练/推理链路后验证
  scripts/run_tests.sh --all -v           # 交付前完整验证
  scripts/run_tests.sh -- tests/test_dual_bank.py -x   # 只跑某个文件（-- 后透传 pytest）
EOF
    exit 0
fi

PY=".venv/Scripts/python.exe"
[[ -x "$PY" ]] || { echo "未找到 .venv，请先运行: python -m venv .venv && .venv/Scripts/python.exe -m pip install -e \".[dev]\"" >&2; exit 1; }

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUTF8=1

MODE="quick"
VERBOSE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slow) MODE="slow"; shift ;;
        --all)  MODE="all";  shift ;;
        -v)     VERBOSE="-v"; shift ;;
        --)     shift; EXTRA=("$@"); break ;;
        *)      echo "未知参数: $1（用 -h 查看帮助）" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    quick) echo "== 快速测试（单元测试，slow 冒烟默认跳过）==" ; "$PY" -m pytest tests/ -q $VERBOSE ${EXTRA[@]+"${EXTRA[@]}"} ;;
    slow)  echo "== 全链路冒烟测试（真实骨干，约 1 分钟）=="     ; "$PY" -m pytest tests/test_smoke.py -m slow $VERBOSE ${EXTRA[@]+"${EXTRA[@]}"} ;;
    all)   echo "== 全部测试（单元 + 冒烟）=="                  ; "$PY" -m pytest tests/ -q -m "slow or not slow" $VERBOSE ${EXTRA[@]+"${EXTRA[@]}"} ;;
esac

@echo off
rem run_tests.bat — 测试运行脚本（Windows 双击/CMD 用）
setlocal
cd /d "%~dp0\.."

if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="/?" goto :help
goto :run

:help
echo 用法: scripts\run_tests.bat [选项] [-- 额外 pytest 参数]
echo.
echo 选项:
echo   （无参数）    快速测试：全部单元测试（约 10 秒，86 个）
echo   --slow        只跑全链路冒烟测试（真实骨干，约 1 分钟）
echo   --all         全部测试 = 单元测试 + 冒烟测试
echo   -v            详细输出（可与其他选项组合）
echo   -h, --help    显示本帮助
echo.
echo 示例:
echo   scripts\run_tests.bat                    日常开发用
echo   scripts\run_tests.bat --slow             改完训练/推理链路后验证
echo   scripts\run_tests.bat --all -v           交付前完整验证
echo   scripts\run_tests.bat -- tests/test_dual_bank.py -x
exit /b 0

:run
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo 未找到 .venv，请先运行: python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -e ".[dev]" >&2
    exit /b 1
)

if not defined HF_ENDPOINT set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1

set MODE=quick
set VERBOSE=
set EXTRA=
:parse
if "%~1"=="" goto :exec
if "%~1"=="--slow" (set MODE=slow & shift & goto :parse)
if "%~1"=="--all"  (set MODE=all  & shift & goto :parse)
if "%~1"=="-v"     (set VERBOSE=-v & shift & goto :parse)
if "%~1"=="--"     (shift & set EXTRA=%* & goto :exec)
echo 未知参数: %~1（用 -h 查看帮助） >&2
exit /b 2

:exec
if "%MODE%"=="slow" goto :slow
if "%MODE%"=="all"  goto :all
echo == 快速测试（单元测试，slow 冒烟默认跳过）==
"%PY%" -m pytest tests/ -q %VERBOSE% %EXTRA%
exit /b %ERRORLEVEL%

:slow
echo == 全链路冒烟测试（真实骨干，约 1 分钟）==
"%PY%" -m pytest tests/test_smoke.py -m slow %VERBOSE% %EXTRA%
exit /b %ERRORLEVEL%

:all
echo == 全部测试（单元 + 冒烟）==
"%PY%" -m pytest tests/ -q -m "slow or not slow" %VERBOSE% %EXTRA%
exit /b %ERRORLEVEL%

@echo off
rem run_ui.bat — 启动 DINO 异常检测试验环境 Web 界面（Windows 双击/CMD 用）
setlocal
cd /d "%~dp0\.."

if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="/?" goto :help
goto :run

:help
echo 用法: scripts\run_ui.bat [--port 端口] [--host 地址]
echo.
echo 启动 Web 界面（Gradio 四页签：数据集 / 训练 / 验证 / 测试与反馈）。
echo 启动后在浏览器打开提示的地址（默认 http://127.0.0.1:7860）。
echo.
echo 选项:
echo   --port N    监听端口（默认 7860，端口被占用时换一个）
echo   --host IP   绑定地址（默认 127.0.0.1 仅本机；0.0.0.0 允许局域网访问——
echo               注意 UI 无鉴权，局域网内任何人都可操作，仅在可信网络使用）
echo   -h, --help  显示本帮助
echo.
echo 说明:
echo   - 自动使用 .venv 虚拟环境，并预设 HF_ENDPOINT 镜像与 PYTHONUTF8
echo   - 停止：在运行窗口按 Ctrl+C 或直接关闭窗口
echo   - 首次使用需先准备数据并训练，例如：
echo       scripts\dino_cli.bat dataset download --category bottle
echo       scripts\dino_cli.bat train --category bottle
exit /b 0

:run
set PORT=7860
set HOST=127.0.0.1
:parse
if "%~1"=="" goto :exec
if "%~1"=="--port" (
    if "%~2"=="" (echo 缺少端口号 & exit /b 2)
    set PORT=%~2
    shift & shift & goto :parse
)
if "%~1"=="--host" (
    if "%~2"=="" (echo 缺少绑定地址 & exit /b 2)
    set HOST=%~2
    shift & shift & goto :parse
)
echo 未知参数: %~1（用 -h 查看帮助） >&2
exit /b 2

:exec
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo 未找到 .venv，请先运行: python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -e ".[dev]" >&2
    exit /b 1
)

if not defined HF_ENDPOINT set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1

echo 启动 Web UI: http://%HOST%:%PORT%  （Ctrl+C 停止）
"%PY%" -m dino_exp.cli ui --port %PORT% --host %HOST%

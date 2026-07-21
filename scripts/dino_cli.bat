@echo off
rem dino_cli.bat — DINO 异常检测试验环境命令行入口（Windows 双击/CMD 用）
setlocal
cd /d "%~dp0\.."

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo 未找到 .venv，请先运行: python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -e ".[dev]" >&2
    exit /b 1
)

if not defined HF_ENDPOINT set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUTF8=1

if "%~1"=="" goto :help
if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="/?" goto :help
goto :run

:help
echo 用法: scripts\dino_cli.bat ^<命令^> [参数...]
echo.
echo 全部功能与 Web UI 一一对应。常用流程:
echo.
echo   1. 准备数据
echo      scripts\dino_cli.bat dataset download --category bottle
echo      scripts\dino_cli.bat dataset import --category 产品A --label ok 图1.jpg
echo      scripts\dino_cli.bat dataset list
echo.
echo   2. 训练（生成基础模型 v001）
echo      scripts\dino_cli.bat train --category bottle
echo      可选: --backbone dinov2_vitb14  --coreset 0.05  --image-size 518
echo.
echo   3. 验证（指标 + 逐图结果）
echo      scripts\dino_cli.bat validate --category bottle --full
echo      scripts\dino_cli.bat validate --category bottle --full --errors-only
echo.
echo   4. 测试（判定 + 分数 + 热力图）
echo      scripts\dino_cli.bat test --category bottle --image x.jpg
echo.
echo   5. 反馈（发现误判时标记真实标签）
echo      scripts\dino_cli.bat feedback --category bottle --image x.jpg --label ok
echo      scripts\dino_cli.bat feedback --category bottle --image y.jpg --label ng --defect-type 划痕
echo      scripts\dino_cli.bat unstage --category bottle ^<反馈id^>
echo.
echo   6. 再训练（应用反馈 -^> 新版本）
echo      scripts\dino_cli.bat retrain --category bottle
echo      scripts\dino_cli.bat retrain --category bottle --yes
echo.
echo   7. 版本管理
echo      scripts\dino_cli.bat versions --category bottle
echo      scripts\dino_cli.bat rollback --category bottle v001
echo.
echo   其他:
echo      scripts\dino_cli.bat export --category bottle
echo      scripts\dino_cli.bat ui
echo      scripts\dino_cli.bat ^<命令^> --help     查看单个命令的详细参数
echo.
echo ---------------- dino --help ----------------
"%PY%" -m dino_exp.cli --help
exit /b 0

:run
"%PY%" -m dino_exp.cli %*

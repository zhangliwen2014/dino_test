"""统一日志：logs/dino.log 滚动文件 + 控制台。CLI 与 Web UI 启动时调用 setup_logging()。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path("logs")
_LOGGER_NAME = "dino_exp"
_configured = False


def setup_logging(log_dir: str | Path = _LOG_DIR, level: int = logging.INFO) -> Path:
    """初始化全局日志（幂等）。返回日志文件路径。

    文件滚动切割：单文件 5MB × 3 个备份。所有 dino_exp 模块经 get_logger 输出。
    """
    global _configured
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dino.log"
    if _configured:
        return log_file

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("日志系统初始化完成: %s", log_file.resolve())
    _configured = True
    return log_file


def get_logger(name: str) -> logging.Logger:
    """获取 dino_exp 子 logger（如 get_logger('train') → dino_exp.train）。"""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")

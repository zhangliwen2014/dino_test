"""Web UI 共用工具。"""

from __future__ import annotations

import traceback

from dino_exp.errors import DinoError


def error_summary(exc: Exception) -> str:
    """一句话错误摘要（面向用户）：DinoError 含修复建议；其他异常给类型+简述。"""
    if isinstance(exc, DinoError):
        return f"错误: {exc}"
    return f"错误: {type(exc).__name__}: {exc}（可在下方「错误详情」查看堆栈）"


def error_detail(exc: Exception) -> str:
    """详细错误信息（面向排查）：完整堆栈。DinoError 信息已自足，无需堆栈。"""
    if isinstance(exc, DinoError):
        return "（应用层错误，修复建议见摘要，无堆栈）"
    return traceback.format_exc()


def error_pair(exc: Exception) -> tuple[str, str]:
    """(摘要, 详情)，供 UI 分别填入友好提示框与可折叠详情框。"""
    return error_summary(exc), error_detail(exc)


def error_text(exc: Exception) -> str:
    """兼容旧调用：单文本场景 = 摘要 + 详情拼接。"""
    s, d = error_pair(exc)
    return s if isinstance(exc, DinoError) else f"{s}\n\n{d}"

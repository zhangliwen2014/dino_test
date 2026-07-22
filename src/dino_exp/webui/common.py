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


def category_choices(cfg) -> list[str]:
    """data_root 下现有类别名列表（供下拉选择；5 秒 TTL 缓存）。"""
    from dino_exp.datasets import dataset_categories

    return dataset_categories(cfg)


def verdict_html(label: str, score: float, threshold: float) -> str:
    """OK/NG 彩色判定徽章（绿=OK 红=NG），测试页/验证页共用。"""
    if not label:
        return ""
    color = "#16a34a" if label == "OK" else "#dc2626"
    return (
        f"<div style='display:inline-block;padding:8px 24px;border-radius:8px;"
        f"background:{color};color:#fff;font-size:26px;font-weight:700'>{label}</div>"
        f"<div style='margin-top:6px;color:#666'>异常分数 {score:.4f} / 阈值 {threshold:.4f}</div>"
    )


def verdict_summary_html(rows: list[dict]) -> str:
    """多图结果汇总：单图用大徽章，多图用 OK/NG 计数。"""
    if not rows:
        return ""
    if len(rows) == 1:
        r = rows[0]
        return verdict_html(r["label"], r["score"], r.get("threshold", 0.0))
    ok = sum(1 for r in rows if r["label"] == "OK")
    ng = len(rows) - ok
    return (
        f"<div style='font-size:18px'>共 {len(rows)} 张："
        f"<span style='color:#16a34a;font-weight:700'>OK {ok}</span> ／ "
        f"<span style='color:#dc2626;font-weight:700'>NG {ng}</span></div>"
    )


def category_dropdown(cfg, *, allow_custom: bool = False, refresh: float = 5.0, **kw):
    """类别下拉框：choices 来自现有数据集并定时刷新；allow_custom=True 时允许输入新类别。
    默认选中第一个类别，保证首次加载时下游（图片列表等）有数据。"""
    import gradio as gr

    choices = category_choices(cfg)
    dd = gr.Dropdown(choices=choices, value=choices[0] if choices else None,
                     allow_custom_value=allow_custom, filterable=True, **kw)
    if refresh:
        gr.Timer(refresh).tick(lambda: gr.update(choices=category_choices(cfg)), None, dd)
    return dd

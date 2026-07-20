from __future__ import annotations


def _by_image(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["image_path"], []).append(r)
    return groups


def effective(rows: list[dict]) -> list[dict]:
    """同一张图多次反馈以 timestamp 最新一条为准。"""
    return [max(g, key=lambda r: r["timestamp"]) for g in _by_image(rows).values()]


def conflicts(rows: list[dict]) -> list[str]:
    """同图先后标记了不同标签的冲突记录（列入预览交人工裁决）。"""
    return [
        img for img, g in _by_image(rows).items()
        if len({r["human_label"] for r in g}) > 1
    ]


def preview(rows: list[dict], threshold: float, factor: float = 3.0) -> dict:
    """再训练预览：生效数量 + 双向可疑项 + 同图冲突。"""
    eff = effective(rows)
    suspicious = [
        r for r in eff
        if (r["human_label"] == "ok" and r["score"] > factor * threshold)
        or (r["human_label"] == "ng" and r["score"] < threshold)
    ]
    return {
        "ok": sum(1 for r in eff if r["human_label"] == "ok"),
        "ng": sum(1 for r in eff if r["human_label"] == "ng"),
        "suspicious": suspicious,
        "conflicts": conflicts(rows),
    }

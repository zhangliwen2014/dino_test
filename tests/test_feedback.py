import pytest

from dino_exp.config import Config
from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.feedback.staging import conflicts, effective, preview


@pytest.fixture
def store(tmp_path):
    cfg = Config(feedback_root=tmp_path / "feedback")
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    return FeedbackStore(cfg.feedback_root, "bottle"), img


def _fb(img, label, score=0.5, ts="2026-07-20T10:00:00"):
    return {
        "image_path": str(img), "model_version": "v001",
        "prediction": "NG", "score": score,
        "human_label": label, "defect_type": None, "timestamp": ts,
    }


def test_stage_persists_jsonl_and_image_copy(store):
    s, img = store
    rec = s.stage(_fb(img, "ok"))
    assert rec["id"]
    staged = s.staged()
    assert len(staged) == 1
    assert (s.images_dir / rec["stored_image"]).exists()


def test_remove_single_entry(store):
    s, img = store
    r1 = s.stage(_fb(img, "ok"))
    r2 = s.stage(_fb(img, "ng", ts="2026-07-20T11:00:00"))
    assert s.remove(r1["id"]) is True
    assert [r["id"] for r in s.staged()] == [r2["id"]]
    assert s.remove("nonexistent") is False


def test_effective_latest_wins(store):
    s, img = store
    s.stage(_fb(img, "ng", ts="2026-07-20T10:00:00"))
    s.stage(_fb(img, "ok", ts="2026-07-20T12:00:00"))  # 更新的一条生效
    eff = effective(s.staged())
    assert len(eff) == 1 and eff[0]["human_label"] == "ok"


def test_conflicts_detects_label_flip(store):
    s, img = store
    s.stage(_fb(img, "ng", ts="2026-07-20T10:00:00"))
    s.stage(_fb(img, "ok", ts="2026-07-20T12:00:00"))
    conf = conflicts(s.staged())
    assert len(conf) == 1 and str(img) in conf[0]


def test_preview_bidirectional_suspicious(store, tmp_path):
    s, img = store
    img2 = tmp_path / "y.png"
    img2.write_bytes(b"y")
    s.stage(_fb(img, "ok", score=5.0))    # OK 但 score > 3×1.0 → 可疑
    s.stage(_fb(img2, "ng", score=0.5))   # NG 但 score < 阈值 → 可疑（漏报型）
    pv = preview(s.staged(), threshold=1.0, factor=3.0)
    assert pv["ok"] == 1 and pv["ng"] == 1
    assert len(pv["suspicious"]) == 2


def test_apply_archives_and_clears(store):
    s, img = store
    s.stage(_fb(img, "ok"))
    applied = s.apply()
    assert len(applied) == 1
    assert s.staged() == []
    assert s.applied_file.exists() and len(s.applied_file.read_text().strip().splitlines()) == 1


def test_apply_empty_raises(store):
    s, _ = store
    with pytest.raises(DinoError, match="暂存区为空"):
        s.apply()

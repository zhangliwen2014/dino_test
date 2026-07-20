import json

import pytest
import torch

from dino_exp.config import Config
from dino_exp.retrain import auroc_drop_warning, partition_feedback


def test_partition_feedback():
    eff = [
        {"human_label": "ok", "stored_image": "a.png"},
        {"human_label": "ng", "stored_image": "b.png", "defect_type": "scratch"},
        {"human_label": "ng", "stored_image": "c.png", "defect_type": None},
    ]
    oks, ngs = partition_feedback(eff)
    assert len(oks) == 1 and len(ngs) == 2


def test_auroc_drop_warning_triggers_over_2_points():
    assert auroc_drop_warning(parent=0.95, current=0.92) is not None  # 降 3 个点
    assert auroc_drop_warning(parent=0.95, current=0.94) is None     # 降 1 个点
    assert auroc_drop_warning(parent=0.95, current=0.96) is None     # 上升


def test_auroc_drop_warning_degraded_skips():
    assert auroc_drop_warning(parent=None, current=None) is None


def _setup_retrain_fixture(tmp_path, monkeypatch):
    """构造 fake 再训练环境：数据集、父版本 v001、1 OK + 1 NG 反馈、fake 模型补丁。"""
    from dino_exp import retrain as rt

    cfg = Config(
        data_root=tmp_path / "data", models_root=tmp_path / "models",
        feedback_root=tmp_path / "feedback",
    )
    # 造数据集（校验用）
    for i in range(4):
        (cfg.data_root / "c" / "train" / "good").mkdir(parents=True, exist_ok=True)
        (cfg.data_root / "c" / "train" / "good" / f"{i}.png").write_bytes(b"x")
    # 造父版本 v001
    from dino_exp.models.registry import Registry

    bank = tmp_path / "b.pt"
    torch.save({"memory_bank": torch.randn(20, 4), "defect_bank": torch.empty(0),
                "pinned_count": 0, "base_bank_size": 20}, bank)
    Registry(cfg.models_root).create_version(
        "c", normal_bank=bank, defect_bank=None, checkpoint=None,
        config={}, metrics={"image_AUROC": 0.95, "threshold": 1.0},
        meta={"parent": None, "feedback_applied": 0},
    )
    # 造反馈：1 OK（高分误报）+ 1 NG
    from dino_exp.feedback.store import FeedbackStore

    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    img2 = tmp_path / "g.png"
    img2.write_bytes(b"y")
    store = FeedbackStore(cfg.feedback_root, "c")
    # 注意：OK 与 NG 反馈必须用不同图片——同图多条会被 effective() 折叠为最新一条
    store.stage({"image_path": str(img), "model_version": "v001", "prediction": "NG",
                 "score": 5.0, "human_label": "ok", "defect_type": None,
                 "timestamp": "2026-07-20T10:00:00"})
    store.stage({"image_path": str(img2), "model_version": "v001", "prediction": "OK",
                 "score": 0.1, "human_label": "ng", "defect_type": "s",
                 "timestamp": "2026-07-20T11:00:00"})

    class FakeInner:
        def __init__(self):
            self.memory_bank = torch.randn(20, 4)
            self.defect_bank = torch.empty(0)
            self.pinned_count = torch.tensor([0])
            self.base_bank_size = 20
            self.bank_cap_ratio = 1.5
            self.coreset_sampling_ratio = 0.1

        def add_normal_features(self, feats, pinned=False):
            from dino_exp.models.dual_bank import merge_pinned

            self.memory_bank, c = merge_pinned(self.memory_bank, int(self.pinned_count), feats, pinned)
            self.pinned_count = torch.tensor([c])

        def add_defect_features(self, feats):
            self.defect_bank = feats if self.defect_bank.numel() == 0 else torch.cat([self.defect_bank, feats])

        def resample_normal_bank(self):
            pass

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()
            self.threshold = None

        def apply_threshold(self, t):
            self.threshold = t

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(rt, "load_model_for_version", lambda *a, **k: (FakeModel(), 1.0, "v001"))
    monkeypatch.setattr(rt, "extract_embeddings", lambda m, t: torch.randn(16, 4))
    monkeypatch.setattr(rt, "topk_defect_features", lambda m, t, k: torch.randn(k, 4))
    monkeypatch.setattr(rt, "preprocess_image", lambda p, s: torch.randn(1, 3, 8, 8))
    monkeypatch.setattr(rt, "ok_calibration_images", lambda c, cfg: [img])
    monkeypatch.setattr(rt, "score_images", lambda m, ps, cfg: [0.9, 1.1, 1.0])
    monkeypatch.setattr(rt, "validate_full", lambda c, v, cfg: {"version": v, "metrics": {"image_AUROC": 0.92}, "rows": []})
    return rt, cfg, store


def test_retrain_end_to_end_with_fakes(tmp_path, monkeypatch):
    """用 fake 模型与 fake 反馈跑通：预览→应用→新版本→对比告警。"""
    rt, cfg, store = _setup_retrain_fixture(tmp_path, monkeypatch)

    result = rt.retrain("c", cfg)
    assert result["version"] == "v002"
    assert result["warning"] is not None and "回滚" in result["warning"]  # 0.95→0.92 降 3 点
    meta = json.loads((cfg.models_root / "c" / "v002" / "meta.json").read_text())
    assert meta["parent"] == "v001" and meta["feedback_applied"] == 2
    assert store.staged() == []  # 暂存区已清空


def test_retrain_failure_preserves_staged_feedback(tmp_path, monkeypatch):
    """validate_full 失败时：暂存区不被消费，反馈可重试（评审修复）。"""
    rt, cfg, store = _setup_retrain_fixture(tmp_path, monkeypatch)

    def boom(c, v, cfg):
        raise RuntimeError("validation crashed")

    monkeypatch.setattr(rt, "validate_full", boom)
    with pytest.raises(RuntimeError, match="validation crashed"):
        rt.retrain("c", cfg)
    assert len(store.staged()) == 2  # 暂存区未清空，反馈未丢

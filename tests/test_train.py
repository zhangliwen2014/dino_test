import json

import pytest
import torch

from dino_exp.config import Config
from dino_exp.train import finalize_version


def _mk_dataset(cfg, n_train=6, n_test_good=2, n_ng=2):
    d = cfg.data_root / "bottle"
    for i in range(n_train):
        (d / "train" / "good").mkdir(parents=True, exist_ok=True)
        (d / "train" / "good" / f"t{i}.png").write_bytes(b"x")
    for i in range(n_test_good):
        (d / "test" / "good").mkdir(parents=True, exist_ok=True)
        (d / "test" / "good" / f"g{i}.png").write_bytes(b"x")
    for i in range(n_ng):
        (d / "test" / "broken").mkdir(parents=True, exist_ok=True)
        (d / "test" / "broken" / f"b{i}.png").write_bytes(b"x")


class FakeModel:
    """模拟训练后的 DualBankPatchcore（仅有 finalize 需要的接口）。"""

    def __init__(self):
        self.model = self  # lightning.model 约定
        self.memory_bank = torch.randn(50, 8)
        self.defect_bank = torch.empty(0)
        self.pinned_count = torch.tensor([0])
        self.base_bank_size = 50
        self.applied = None

    def apply_threshold(self, t):
        self.applied = t

    def eval(self):
        return self


def test_finalize_version_saves_banks_threshold_and_meta(tmp_path):
    cfg = Config(data_root=tmp_path / "data", models_root=tmp_path / "models")
    _mk_dataset(cfg)
    model = FakeModel()
    ok_scores = [1.0, 1.2, 0.8, 1.1]
    version = finalize_version(
        "bottle", cfg, model, ok_scores=ok_scores,
        metrics={"image_AUROC": 0.95}, parent=None, feedback_applied=0,
    )
    assert version == "v001"
    vdir = cfg.models_root / "bottle" / "v001"
    assert (vdir / "normal_bank.pt").exists()
    metrics = json.loads((vdir / "metrics.json").read_text())
    # mean=1.025, std≈0.1708 → t≈1.537
    assert metrics["threshold"] == pytest.approx(1.537, abs=1e-3)
    assert metrics["image_AUROC"] == 0.95
    assert model.applied == metrics["threshold"]  # 阈值已注入模型
    meta = json.loads((vdir / "meta.json").read_text())
    assert meta["parent"] is None and meta["feedback_applied"] == 0
    banks = torch.load(vdir / "normal_bank.pt", weights_only=True)
    assert banks["memory_bank"].shape == (50, 8)
    assert banks["base_bank_size"] == 50


def test_train_model_emits_stage_logs(tmp_path, monkeypatch):
    """log 回调收到各阶段日志（设计 §3.7 训练日志通道，Web UI 轮询展示）。"""
    from pathlib import Path

    from dino_exp import train as tr
    from dino_exp.datasets import DatasetInfo

    cfg = Config(data_root=tmp_path / "data", models_root=tmp_path / "models")
    monkeypatch.setattr(tr, "dataset_info",
                        lambda c, cfg: DatasetInfo("c", Path("d"), 4, 2, {"broken": 2}))
    monkeypatch.setattr(tr, "build_folder", lambda c, cfg: object())
    monkeypatch.setattr(tr, "build_model", lambda cfg: FakeModel())
    monkeypatch.setattr(tr, "ok_calibration_images", lambda c, cfg: [Path("x.png")])
    # 新版校准走 score_one（train_model 内 from dino_exp.infer import score_one）
    monkeypatch.setattr("dino_exp.infer.score_one", lambda m, p, c: (1.0, None))

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def fit(self, model, datamodule):
            pass

        def test(self, model, datamodule):
            return [{"image_AUROC": 0.9}]

    import anomalib.engine

    monkeypatch.setattr(anomalib.engine, "Engine", FakeEngine)

    records = []
    result = tr.train_model("c", cfg, log=records.append)
    assert result["version"] == "v001"
    text = "\n".join(records)
    assert "校验数据集" in text and "train/good=4" in text
    assert "Engine.fit 建库中" in text
    assert "coreset 完成，记忆库 50 条" in text
    assert "阈值=" in text and "注入完成" in text
    assert "全量验证" in text
    assert "版本 v001 已保存" in text

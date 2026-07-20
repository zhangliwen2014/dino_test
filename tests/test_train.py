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

import pytest
import torch

from dino_exp.config import Config
from dino_exp.perf import _percentile, format_table, run_perf, run_perf_one


class _FakeModel:
    def __init__(self):
        self.train_tile_grid = (2, 2)
        self.train_image_size = 224

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


def test_percentile():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0  # idx=int(4*0.5)=2 → 上中位数
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert _percentile([], 0.5) == 0.0


def test_run_perf_one_metrics(monkeypatch):
    import dino_exp.perf as perf_mod

    monkeypatch.setattr(perf_mod, "score_one", lambda m, p, c: (1.0, None, 10.0))
    r = run_perf_one(_FakeModel(), ["a.png"] * 4, concurrency=2, cfg=Config())
    assert r["images"] == 4
    assert r["concurrency"] == 2
    assert r["latency_avg_ms"] == 10.0
    assert r["latency_p50_ms"] == 10.0
    assert r["throughput_ips"] > 0
    assert r["wall_s"] >= 0


def test_run_perf_report_structure(tmp_path, monkeypatch):
    import dino_exp.perf as perf_mod
    from dino_exp.models.registry import Registry

    cfg = Config(models_root=tmp_path / "models", data_root=tmp_path / "data")
    # 造数据集与两个版本
    for i in range(3):
        d = cfg.data_root / "c" / "train" / "good"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"t{i}.png").write_bytes(b"x")
        d2 = cfg.data_root / "c" / "test" / "good"
        d2.mkdir(parents=True, exist_ok=True)
        (d2 / f"g{i}.png").write_bytes(b"x")
    bank = tmp_path / "b.pt"
    torch.save({"memory_bank": torch.randn(4, 2), "defect_bank": torch.empty(0),
                "pinned_count": 0, "base_bank_size": 4}, bank)
    reg = Registry(cfg.models_root)
    reg.create_version("c", normal_bank=bank, defect_bank=None, checkpoint=None,
                       config={}, metrics={"threshold": 1.0}, meta={"parent": None})
    reg.create_version("c", normal_bank=bank, defect_bank=None, checkpoint=None,
                       config={}, metrics={"threshold": 1.0}, meta={"parent": "v001"})

    monkeypatch.setattr(perf_mod, "load_model_for_version",
                        lambda c, v, cfg: (_FakeModel(), 1.0, v))
    monkeypatch.setattr(perf_mod, "score_one", lambda m, p, c: (1.0, None, 5.0))
    report = run_perf("c", None, [1, 4], 2, cfg)
    assert set(report["versions"]) == {"v001", "v002"}
    for entry in report["versions"].values():
        assert len(entry["runs"]) == 2
        assert entry["runs"][0]["latency_avg_ms"] == 5.0
    text = format_table(report)
    assert "v001" in text and "v002" in text and "吞吐" in text

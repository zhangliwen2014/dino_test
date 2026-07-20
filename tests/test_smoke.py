"""全链路冒烟：train→validate→test→feedback→retrain→rollback（设计 §6）。

需联网下载 DINOv2 ViT-S 权重（首次 ~90MB），CPU 运行约 2-5 分钟。
运行: python -m pytest tests/test_smoke.py -m slow -v
"""

import json

import pytest
import torch

from dino_exp.feedback.store import FeedbackStore
from dino_exp.infer import infer_image
from dino_exp.models.registry import Registry
from dino_exp.retrain import retrain
from dino_exp.train import train_model
from dino_exp.validate import validate_full

pytestmark = pytest.mark.slow


def test_full_pipeline(smoke_cfg):
    cfg = smoke_cfg
    # 1. 训练 → v001
    r1 = train_model("toy", cfg)
    assert r1["version"] == "v001"
    assert (cfg.models_root / "toy" / "v001" / "normal_bank.pt").exists()
    # 2. 全量验证
    report = validate_full("toy", "v001", cfg)
    assert "threshold" not in report["metrics"]  # threshold 在 metrics.json
    assert (cfg.models_root / "toy" / "v001" / "validation.json").exists()
    # 3. 单图测试
    out = infer_image(cfg.data_root / "toy" / "test" / "good" / "g0.png",
                      category="toy", cfg=cfg, heatmap_dir=cfg.models_root / "hm")
    assert out["label"] in {"OK", "NG"}
    assert json.loads((cfg.models_root / "toy" / "v001" / "metrics.json").read_text())["threshold"] == out["threshold"]
    # 4. 反馈（OK 一张 + NG 一张）
    store = FeedbackStore(cfg.feedback_root, "toy")
    store.stage({"image_path": str(cfg.data_root / "toy" / "test" / "good" / "g0.png"),
                 "model_version": "v001", "prediction": out["label"],
                 "score": out["score"], "human_label": "ok", "defect_type": None})
    store.stage({"image_path": str(cfg.data_root / "toy" / "test" / "broken" / "b0.png"),
                 "model_version": "v001", "prediction": "OK",
                 "score": 0.0, "human_label": "ng", "defect_type": "broken"})
    # 5. 再训练 → v002（钉住 + 缺陷库填充 + 阈值重校准）
    r2 = retrain("toy", cfg)
    assert r2["version"] == "v002"
    banks = torch.load(cfg.models_root / "toy" / "v002" / "normal_bank.pt",
                       weights_only=True)
    assert banks["pinned_count"] > 0  # 钉住特征已入库
    assert banks["defect_bank"].shape[0] > 0  # 缺陷库已填充
    # 6. 回滚 → current 回到 v001
    Registry(cfg.models_root).rollback("toy", "v001")
    assert Registry(cfg.models_root).current("toy") == "v001"

import numpy as np
import pytest
from PIL import Image

from dino_exp.config import Config


def _save_png(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (56, 56, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def smoke_cfg(tmp_path):
    """5 张 56x56 小图的合成数据集（56 = 14×4，满足 patch 倍数校验）。"""
    cfg = Config(
        data_root=tmp_path / "data", models_root=tmp_path / "models",
        feedback_root=tmp_path / "feedback", image_size=56,
        coreset_sampling_ratio=0.5, train_batch_size=2, eval_batch_size=2,
        defect_topk=3,
    )
    for i in range(3):
        _save_png(cfg.data_root / "toy" / "train" / "good" / f"t{i}.png", seed=i)
    _save_png(cfg.data_root / "toy" / "test" / "good" / "g0.png", seed=10)
    _save_png(cfg.data_root / "toy" / "test" / "broken" / "b0.png", seed=11)
    return cfg


def pytest_collection_modifyitems(config, items):
    """slow 测试（真实骨干权重、分钟级运行）默认跳过；显式 -m slow 时运行。"""
    if config.getoption("-m"):
        return  # 用户显式给了 marker 表达式，不干预
    skip = pytest.mark.skip(reason="slow 测试默认跳过，用 -m slow 显式运行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

import pytest
import torch

from dino_exp.models.dual_bank import (
    DualBankPatchcoreModel,
    apply_defect_boost,
    calibrate_threshold,
    coreset_cap,
    load_banks,
    merge_pinned,
    save_banks,
)


def test_boost_applies_when_defect_closer():
    d_n = torch.tensor([1.0, 2.0, 3.0])
    d_d = torch.tensor([0.5, 1.5, 3.0])  # patch0/patch1 离缺陷库更近（计划原值 2.5 系笔误：2.5 > 2.0 与"更近"矛盾）
    out = apply_defect_boost(d_n, d_d, w=0.5)
    assert out[0].item() == pytest.approx(1.5)   # 1.0 * 1.5
    assert out[1].item() == pytest.approx(3.0)   # 2.0 * 1.5
    assert out[2].item() == pytest.approx(3.0)   # 不加分（相等不加）


def test_boost_zero_weight_is_noop():
    d_n = torch.tensor([1.0, 2.0])
    d_d = torch.tensor([0.1, 0.1])
    assert torch.equal(apply_defect_boost(d_n, d_d, w=0.0), d_n)


def test_coreset_cap_formula():
    # cap = 基础库大小 × 倍数 - 钉住数，至少 1
    assert coreset_cap(base_size=1000, cap_ratio=1.5, pinned=0) == 1500
    assert coreset_cap(base_size=1000, cap_ratio=1.5, pinned=600) == 900
    assert coreset_cap(base_size=10, cap_ratio=1.5, pinned=20) == 1


def test_merge_pinned_inserts_after_pinned_region():
    bank = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    # 前 2 行是钉住区
    new = torch.full((1, 2), 99.0)
    merged, pinned_count = merge_pinned(bank, pinned_count=2, new_feats=new, pinned=True)
    assert pinned_count == 3
    assert torch.equal(merged[2], torch.tensor([99.0, 99.0]))  # 插入在钉住区尾部
    assert torch.equal(merged[0], bank[0]) and torch.equal(merged[-1], bank[-1])


def test_merge_unpinned_appends_at_end():
    bank = torch.zeros(3, 2)
    merged, pinned_count = merge_pinned(bank, pinned_count=1, new_feats=torch.ones(2, 2), pinned=False)
    assert pinned_count == 1
    assert merged.shape == (5, 2)
    assert torch.equal(merged[-1], torch.ones(2))


def test_calibrate_threshold_mean_plus_3sigma():
    # 1..9: mean=5, 样本 std≈2.7386 → t ≈ 5 + 3*2.7386
    scores = [float(i) for i in range(1, 10)]
    t = calibrate_threshold(scores, sigma=3.0)
    assert t == pytest.approx(5 + 3 * 2.7386, abs=1e-3)


def test_calibrate_threshold_single_score():
    assert calibrate_threshold([2.5], sigma=3.0) == 2.5  # std=0


def test_calibrate_threshold_empty_raises():
    from dino_exp.errors import DinoError

    with pytest.raises(DinoError, match="校准"):
        calibrate_threshold([], sigma=3.0)


# ---------------- 模型层测试（resnet18 layer1，CPU 秒级，pre_trained=False 不下载权重） ----------------

LAYER1_DIM = 64  # resnet18 layer1 经 AvgPool2d(3,1,1) 池化后的通道数（已探明）


def _make_model(fusion_weight: float = 0.5, bank_cap_ratio: float = 1.5) -> DualBankPatchcoreModel:
    model = DualBankPatchcoreModel(
        layers=["layer1"],
        backbone="resnet18",
        pre_trained=False,
        fusion_weight=fusion_weight,
        bank_cap_ratio=bank_cap_ratio,
    )
    model.eval()
    return model


def test_boost_injected_only_when_defect_bank_present():
    torch.manual_seed(0)
    model = _make_model()
    query = torch.randn(4, LAYER1_DIM)
    model.memory_bank = query + 10.0  # 正常库距离恒远大于 0
    base_scores, _ = model.nearest_neighbors(query, n_neighbors=1)
    # 缺陷库为空：分数不变
    again, _ = model.nearest_neighbors(query, n_neighbors=1)
    assert torch.equal(base_scores, again)
    # 缺陷库含与 query[0] 完全相同的特征 → d_def=0 < d_n → 该 patch ×(1+w)
    model.defect_bank = query[0:1].clone()
    boosted, _ = model.nearest_neighbors(query, n_neighbors=1)
    assert boosted[0].item() == pytest.approx(base_scores[0].item() * 1.5)
    assert torch.all(boosted >= base_scores)  # 只加分不减分


def test_boost_not_applied_for_multi_neighbor_queries():
    torch.manual_seed(1)
    model = _make_model()
    query = torch.randn(4, LAYER1_DIM)
    model.memory_bank = torch.randn(8, LAYER1_DIM)
    model.defect_bank = query[0:1].clone()
    k = min(9, model.memory_bank.shape[0])
    with_defect, _ = model.nearest_neighbors(query, n_neighbors=k)
    model.defect_bank = torch.empty(0)
    without_defect, _ = model.nearest_neighbors(query, n_neighbors=k)
    assert torch.equal(with_defect, without_defect)  # support-sample 查询支路不受影响


def test_pinned_exempt_and_unpinned_resampled_to_cap():
    torch.manual_seed(2)
    model = _make_model(bank_cap_ratio=1.5)
    model.memory_bank = torch.randn(4, LAYER1_DIM)
    model.base_bank_size = 4  # 总量上限 = 4 × 1.5 = 6
    pinned = torch.full((2, LAYER1_DIM), 7.0)
    model.add_normal_features(pinned, pinned=True)
    assert model._pinned == 2
    assert torch.equal(model.memory_bank[:2], pinned)
    assert model.memory_bank.shape[0] == 6  # 非钉住 4 = cap(6-2)，未触发重采样
    model.add_normal_features(torch.randn(5, LAYER1_DIM), pinned=False)
    # 非钉住 9 > cap 4 → 重采样淘汰到 4；总行数 = 钉住 2 + cap 4
    assert model._pinned == 2
    assert torch.equal(model.memory_bank[:2], pinned)  # 钉住区豁免
    assert model.memory_bank.shape[0] == 2 + 4


def test_add_defect_features_accumulates():
    model = _make_model()
    feats = torch.randn(3, LAYER1_DIM)
    model.add_defect_features(feats)  # 空库首次：直接赋值
    assert model.defect_bank.shape == (3, LAYER1_DIM)
    model.add_defect_features(feats)  # 再次：cat
    assert model.defect_bank.shape == (6, LAYER1_DIM)


def test_save_load_banks_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = _make_model()
    model.memory_bank = torch.randn(6, LAYER1_DIM)
    model.defect_bank = torch.randn(2, LAYER1_DIM)
    model.pinned_count = torch.tensor([3], dtype=torch.long)
    model.base_bank_size = 4
    save_banks(model, tmp_path / "normal_bank.pt")

    restored = _make_model()
    load_banks(restored, tmp_path)
    assert torch.equal(restored.memory_bank, model.memory_bank)
    assert torch.equal(restored.defect_bank, model.defect_bank)
    assert int(restored.pinned_count.item()) == 3
    assert restored.base_bank_size == 4
    for key in ("memory_bank", "defect_bank", "pinned_count"):
        assert key in restored.state_dict()


def test_resample_cap_one_no_float_crash():
    # 回归：cap=1 且非钉住区 49 行时 49*(1/49)=0.9999...→int→0，曾使 KCenterGreedy 抛 ValueError
    torch.manual_seed(4)
    model = _make_model(bank_cap_ratio=1.5)
    pinned = torch.full((5, LAYER1_DIM), 7.0)
    model.memory_bank = torch.cat([pinned, torch.randn(49, LAYER1_DIM)])
    model.pinned_count = torch.tensor([5], dtype=torch.long)
    model.base_bank_size = 4  # 上限 6；cap = 6 - 5 = 1
    model.resample_normal_bank()
    assert torch.equal(model.memory_bank[:5], pinned)
    assert model.memory_bank.shape[0] == 5 + 1

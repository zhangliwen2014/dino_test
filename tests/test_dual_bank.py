import pytest
import torch

from dino_exp.models.dual_bank import (
    apply_defect_boost,
    calibrate_threshold,
    coreset_cap,
    merge_pinned,
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

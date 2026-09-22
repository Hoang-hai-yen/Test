"""Unit tests for aero_eyes.stages.stage1's fuse_prototype() and
select_calibration_frames() -- pulled out of run_stage1() as standalone,
array-only functions specifically so they're testable without mocking the
whole Stage 1 pipeline (segmentation/feature-extraction/video I/O). No
torch import needed (these are pure numpy), unlike most of this project's
model-facing tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.stages.stage1 import fuse_prototype, select_calibration_frames


def _masks(*ratios: float, size: int = 100) -> list[np.ndarray]:
    """Build fake boolean masks whose foreground area ratio is exactly
    `ratios[i]` (rounded to the nearest /size)."""
    out = []
    for r in ratios:
        m = np.zeros(size, dtype=bool)
        m[: int(round(r * size))] = True
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# fuse_prototype
# ---------------------------------------------------------------------------

def test_fuse_mean_matches_mask_area_weighted_average():
    per_ref = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    masks = _masks(0.5, 0.5, 0.5)  # equal weights -> plain mean
    out = fuse_prototype(per_ref, masks, "mean")
    assert np.allclose(out, per_ref.mean(axis=0))


def test_fuse_mean_upweights_higher_mask_ratio():
    per_ref = np.array([[10.0, 0.0], [0.0, 10.0]])
    masks = _masks(0.9, 0.1)  # ref 0 much more trusted than ref 1
    out = fuse_prototype(per_ref, masks, "mean")
    # Should be much closer to ref 0 than a plain unweighted mean [5,5] would be.
    assert out[0] > 5.0
    assert out[1] < 5.0


def test_fuse_max_is_elementwise_max():
    per_ref = np.array([[1.0, 5.0], [3.0, 2.0]])
    out = fuse_prototype(per_ref, _masks(0.5, 0.5), "max")
    assert np.allclose(out, [3.0, 5.0])


def test_fuse_concat_then_pca_returns_correct_dim():
    per_ref = np.random.default_rng(0).normal(size=(3, 8))
    out = fuse_prototype(per_ref, _masks(0.5, 0.5, 0.5), "concat_then_pca")
    assert out.shape == (8,)


def test_fuse_unknown_raises():
    per_ref = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="Unknown fusion method"):
        fuse_prototype(per_ref, _masks(0.5), "bogus")


def test_fuse_agreement_weighted_downweights_outlier_ref():
    """Two refs pointing the same direction, one pointing very differently
    (an outlier) -- agreement_weighted should pull the fused prototype
    CLOSER to the consensus pair than plain mean-pooling would."""
    consensus_dir = np.array([1.0, 0.0, 0.0])
    outlier_dir = np.array([0.0, 1.0, 0.0])
    per_ref = np.stack([consensus_dir, consensus_dir * 0.9, outlier_dir])
    masks = _masks(0.5, 0.5, 0.5)  # equal mask confidence -- isolate the agreement effect

    mean_out = fuse_prototype(per_ref, masks, "mean")
    agreement_out = fuse_prototype(per_ref, masks, "agreement_weighted", agreement_weighted_epsilon=20.0)

    # Both should point more toward consensus_dir than outlier_dir, but
    # agreement_weighted should do so MORE strongly (smaller component
    # along the outlier's axis, relative to its own norm).
    mean_outlier_frac = mean_out[1] / np.linalg.norm(mean_out)
    agreement_outlier_frac = agreement_out[1] / np.linalg.norm(agreement_out)
    assert agreement_outlier_frac < mean_outlier_frac


def test_fuse_agreement_weighted_epsilon_zero_matches_mask_weighted_mean():
    """epsilon=0 -> every agreement weight is equal -> combined_weights
    reduces exactly to the mask-area weights alone (same as fusion="mean")."""
    per_ref = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, -5.0]])
    masks = _masks(0.7, 0.2, 0.5)
    mean_out = fuse_prototype(per_ref, masks, "mean")
    agreement_out = fuse_prototype(per_ref, masks, "agreement_weighted", agreement_weighted_epsilon=0.0)
    assert np.allclose(mean_out, agreement_out)


def test_fuse_agreement_weighted_identical_refs_equals_mean():
    per_ref = np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    masks = _masks(0.5, 0.5, 0.5)
    out = fuse_prototype(per_ref, masks, "agreement_weighted", agreement_weighted_epsilon=50.0)
    assert np.allclose(out, [3.0, 4.0])


# ---------------------------------------------------------------------------
# select_calibration_frames
# ---------------------------------------------------------------------------

def test_select_calibration_frames_keeps_least_similar():
    prototype = np.array([1.0, 0.0])
    pool_idxs = [0, 10, 20, 30]
    pool_frames = ["f0", "f10", "f20", "f30"]
    pool_feats = np.array([
        [1.0, 0.0],    # sim=1.0 -- very target-like, should be excluded
        [0.0, 1.0],    # sim=0.0
        [-1.0, 0.0],   # sim=-1.0 -- most different, should be kept
        [0.1, 0.995],  # sim~0.0995
    ])
    kept_idxs, kept_frames = select_calibration_frames(pool_idxs, pool_frames, pool_feats, prototype, n=2)
    assert kept_idxs == [10, 20]  # ascending frame-index order; idx 0 (most target-like) excluded
    assert kept_frames == ["f10", "f20"]


def test_select_calibration_frames_n_exceeds_pool_returns_all():
    prototype = np.array([1.0, 0.0])
    pool_idxs = [0, 5]
    pool_frames = ["a", "b"]
    pool_feats = np.array([[1.0, 0.0], [0.0, 1.0]])
    kept_idxs, kept_frames = select_calibration_frames(pool_idxs, pool_frames, pool_feats, prototype, n=10)
    assert kept_idxs == [0, 5]
    assert kept_frames == ["a", "b"]


def test_select_calibration_frames_unnormalized_prototype_still_correct():
    """prototype passed in is NOT unit-norm -- function must normalize it
    internally rather than assume the caller already did."""
    prototype = np.array([100.0, 0.0])  # same direction as [1,0], huge norm
    pool_idxs = [0, 1]
    pool_frames = ["a", "b"]
    pool_feats = np.array([[1.0, 0.0], [-1.0, 0.0]])
    kept_idxs, _ = select_calibration_frames(pool_idxs, pool_frames, pool_feats, prototype, n=1)
    assert kept_idxs == [1]  # the anti-aligned one, not the aligned one

"""Unit tests for compute_adaptive_threshold -- the 2 knobs added to guard
stage3.adaptive_threshold against dynamic_prototype skewing it:
adaptive_threshold_anchor_to_original_refs and adaptive_threshold_robust.
See Stage3Config's own docstrings for the full rationale (narrow
dynamic_prototype additions can inflate OTHER candidates' scores via
max-pooling, dragging mean/std -- and so the threshold -- up for everyone).
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.config import Stage3Config
from aero_eyes.stages.stage3 import compute_adaptive_threshold


def test_anchor_to_original_refs_ignores_dynamic_prototype_inflation():
    """all_sims (post dynamic_prototype) has a few candidates inflated by
    max-pooling against a narrow appended reference -- anchoring to the
    ORIGINAL (pre-dynamic_prototype) distribution must ignore that
    inflation entirely and reproduce the threshold as if dynamic_prototype
    never ran."""
    rng = np.random.default_rng(0)
    original = rng.normal(loc=0.1, scale=0.02, size=200)

    inflated = original.copy()
    inflated[:5] = 0.9  # a handful of candidates spiked by a narrow dynamic ref

    s3_unanchored = Stage3Config(
        adaptive_threshold=True, adaptive_z_score=1.0, adaptive_min_floor=0.0,
        adaptive_threshold_anchor_to_original_refs=False,
    )
    thresh_unanchored, center_unanchored, _, _ = compute_adaptive_threshold(
        inflated, original, "cosine", s3_unanchored,
    )

    s3_anchored = Stage3Config(
        adaptive_threshold=True, adaptive_z_score=1.0, adaptive_min_floor=0.0,
        adaptive_threshold_anchor_to_original_refs=True,
    )
    thresh_anchored, center_anchored, _, _ = compute_adaptive_threshold(
        inflated, original, "cosine", s3_anchored,
    )

    # Unanchored: mean/std computed on the INFLATED distribution -- pulled
    # well above the original's own mean by the 5 spiked outliers.
    assert center_unanchored == pytest.approx(float(inflated.mean()))
    assert thresh_unanchored > float(original.mean()) + 3 * float(original.std())

    # Anchored: mean/std computed on the ORIGINAL distribution only --
    # completely unaffected by the inflated candidates, reproduces exactly
    # what the threshold would have been without dynamic_prototype.
    assert center_anchored == pytest.approx(float(original.mean()))
    assert thresh_anchored == pytest.approx(float(original.mean()) + 1.0 * float(original.std()))
    assert thresh_anchored < thresh_unanchored


def test_robust_median_mad_resists_outliers():
    """A tight cluster of true scores + a few extreme outliers (same shape
    as a narrow dynamic_prototype addition inflating a handful of
    candidates): mean+std is dragged far above the cluster; median+MAD
    stays close to it."""
    cluster = np.full(100, 0.10)
    with_outliers = cluster.copy()
    with_outliers[:3] = 0.95  # 3 extreme outliers among 100

    s3_plain = Stage3Config(adaptive_threshold=True, adaptive_z_score=1.0, adaptive_min_floor=0.0,
                             adaptive_threshold_robust=False)
    thresh_plain, center_plain, _, label_plain = compute_adaptive_threshold(
        with_outliers, with_outliers, "cosine", s3_plain,
    )

    s3_robust = Stage3Config(adaptive_threshold=True, adaptive_z_score=1.0, adaptive_min_floor=0.0,
                              adaptive_threshold_robust=True)
    thresh_robust, center_robust, _, label_robust = compute_adaptive_threshold(
        with_outliers, with_outliers, "cosine", s3_robust,
    )

    assert label_plain == "mean/std"
    assert label_robust == "median/MAD"

    # mean+std: 3 outliers among 100 already pull the threshold well past
    # the cluster's own value (0.10).
    assert thresh_plain > 0.15

    # median+MAD: median is still exactly the cluster value (97/100 points
    # sit there), MAD is 0 (>97% of points are identical to the median) --
    # threshold stays AT the cluster value, not dragged up by the outliers.
    assert center_robust == pytest.approx(0.10)
    assert thresh_robust == pytest.approx(0.10, abs=1e-6)
    assert thresh_robust < thresh_plain


def test_otsu_splits_cleanly_separated_bimodal_distribution():
    """A clearly bimodal all_sims (a big FP cluster near 0.1, a smaller TP
    cluster near 0.8) -- Otsu should land the threshold in the gap between
    them, correctly separating the two clusters."""
    rng = np.random.default_rng(0)
    fp_cluster = rng.normal(loc=0.10, scale=0.01, size=300)
    tp_cluster = rng.normal(loc=0.80, scale=0.01, size=50)
    all_sims = np.concatenate([fp_cluster, tp_cluster])

    s3 = Stage3Config(adaptive_threshold=True, adaptive_min_floor=0.0, adaptive_threshold_method="otsu")
    thresh, _, _, label = compute_adaptive_threshold(all_sims, all_sims, "cosine", s3)

    assert label == "otsu"
    assert 0.10 < thresh < 0.80, f"threshold {thresh} should fall in the gap between the 2 clusters"
    # Sanity: applying it actually recovers both clusters correctly.
    assert (fp_cluster < thresh).all()
    assert (tp_cluster >= thresh).all()


def test_gmm_bimodal_thresholds_between_the_two_components():
    rng = np.random.default_rng(1)
    fp_cluster = rng.normal(loc=0.10, scale=0.01, size=300)
    tp_cluster = rng.normal(loc=0.80, scale=0.01, size=50)
    all_sims = np.concatenate([fp_cluster, tp_cluster])

    s3 = Stage3Config(adaptive_threshold=True, adaptive_min_floor=0.0, adaptive_threshold_method="gmm")
    thresh, _, _, label = compute_adaptive_threshold(all_sims, all_sims, "cosine", s3)

    assert label == "gmm_bimodal"
    assert 0.10 < thresh < 0.80
    assert (fp_cluster < thresh).all()
    assert (tp_cluster >= thresh).all()


def test_gmm_falls_back_to_percentile_on_unimodal_distribution():
    """A low-FP sample: all_sims is really just ONE cluster of mostly true
    positives (the case that broke a fixed z multiplier in the first
    place) -- forcing a 2-cluster split onto this is meaningless, must
    fall back to the permissive percentile cut instead."""
    rng = np.random.default_rng(2)
    all_sims = rng.normal(loc=0.75, scale=0.02, size=200)  # one tight cluster, no second mode

    s3 = Stage3Config(
        adaptive_threshold=True, adaptive_min_floor=0.0, adaptive_threshold_method="gmm",
        adaptive_gmm_fallback_percentile=20.0,
    )
    thresh, _, _, label = compute_adaptive_threshold(all_sims, all_sims, "cosine", s3)

    assert label == "gmm_unimodal_fallback"
    assert thresh == pytest.approx(float(np.percentile(all_sims, 20.0)))
    # Permissive: most of the (already-good) cluster should clear it.
    assert (all_sims >= thresh).mean() >= 0.75


def test_otsu_and_gmm_fall_back_to_z_score_below_min_samples():
    """Too few candidates for a distribution-shape method to be reliable
    -- both otsu and gmm must fall back to the plain z_score path."""
    all_sims = np.array([0.1, 0.2, 0.15, 0.9, 0.85])  # 5 points, well below default min_samples=20

    for method in ("otsu", "gmm"):
        s3 = Stage3Config(
            adaptive_threshold=True, adaptive_z_score=1.0, adaptive_min_floor=0.0,
            adaptive_threshold_method=method, adaptive_threshold_min_samples=20,
        )
        thresh, center, spread, label = compute_adaptive_threshold(all_sims, all_sims, "cosine", s3)
        assert label == "mean/std"
        assert thresh == pytest.approx(float(all_sims.mean()) + 1.0 * float(all_sims.std()))


def test_adaptive_min_floor_still_applies_to_otsu_and_gmm():
    """The hard floor must still clamp otsu/gmm's own threshold, same as
    it already does for z_score -- a degenerate distribution (all sims
    tiny/negative) shouldn't be able to push the effective threshold below
    the floor just because a different method computed it."""
    all_sims = np.full(50, -0.5)  # degenerate: identical, very low similarity

    s3_otsu = Stage3Config(adaptive_threshold=True, adaptive_min_floor=0.05, adaptive_threshold_method="otsu")
    thresh_otsu, *_ = compute_adaptive_threshold(all_sims, all_sims, "cosine", s3_otsu)
    assert thresh_otsu == pytest.approx(0.05)

    s3_gmm = Stage3Config(adaptive_threshold=True, adaptive_min_floor=0.05, adaptive_threshold_method="gmm")
    thresh_gmm, *_ = compute_adaptive_threshold(all_sims, all_sims, "cosine", s3_gmm)
    assert thresh_gmm == pytest.approx(0.05)

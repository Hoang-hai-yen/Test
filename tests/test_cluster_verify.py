"""Unit tests for cluster_verify_candidates -- the DAVE (arXiv:2404.16622)
module (ii)-style shared primitive used by both
stage3.verification_method="cluster" and
stage123_geco2.dynamic_prototype.cluster_verification. See
ClusterVerificationConfig's own docstring (aero_eyes/config.py) for the full
rationale versus a global scalar threshold.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.config import ClusterVerificationConfig
from aero_eyes.utils.cluster_verify import _self_tuning_n_clusters, cluster_verify_candidates


def _make_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _two_cluster_scene(rng, d=16, noise=0.02, n_tp=6, n_fp=6, n_ref=3):
    """A tight TP cluster around exemplar_center, a tight FP (confuser)
    cluster around a DIFFERENT, well-separated confuser_center, and
    exemplars drawn from the SAME neighborhood as the TP cluster."""
    exemplar_center = np.zeros(d)
    exemplar_center[0] = 1.0
    confuser_center = np.zeros(d)
    confuser_center[1] = 1.0

    tp = _make_unit(exemplar_center[None, :] + noise * rng.normal(size=(n_tp, d)))
    fp = _make_unit(confuser_center[None, :] + noise * rng.normal(size=(n_fp, d)))
    cand = np.concatenate([tp, fp], axis=0)
    ref = _make_unit(exemplar_center[None, :] + (noise / 2) * rng.normal(size=(n_ref, d)))
    return cand, ref, n_tp, n_fp


@pytest.mark.parametrize("cluster_method", ["hdbscan", "spectral"])
def test_separates_tp_from_fp_cluster(cluster_method):
    """Well-separated TP/FP clusters: both backends must keep exactly the
    exemplar-containing cluster and reject the confuser cluster."""
    rng = np.random.default_rng(0)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)

    cfg = ClusterVerificationConfig(
        enabled=True, cluster_method=cluster_method, min_cluster_size=2,
    )
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == f"cluster_{cluster_method}"
    assert keep_mask[:n_tp].all(), "every TP candidate should cluster with an exemplar"
    assert not keep_mask[n_tp:].any(), "every FP candidate should be rejected"


def test_survives_overlapping_global_similarity_ranges():
    """Constructed so a GLOBAL scalar threshold (z-score/Otsu/GMM, or any
    single cosine cutoff) CANNOT separate TP from FP -- the confuser (FP)
    cluster's similarity to the pooled exemplar mean is actually HIGHER than
    every single TP candidate's own similarity, because the 2 real object
    viewpoints (A and B) are represented by 3 exemplars unevenly (2 near A,
    1 near B) and the confuser (C) happens to sit geometrically between them,
    closer to the pooled mean than either genuine viewpoint alone. No
    threshold on pooled similarity could ever keep the TPs and reject the
    FP here (keeping high scorers keeps the confuser and drops half the real
    matches). Cluster-based verification doesn't pool at all -- it clusters
    every candidate directly against EVERY individual exemplar -- so both TP
    sub-clusters (each containing at least one exemplar) are correctly kept
    and the exemplar-free FP cluster is correctly rejected.
    """
    rng = np.random.default_rng(0)
    d = 16
    view_a = np.zeros(d)
    view_a[0] = 1.0
    view_b_raw = 0.6 * view_a.copy()
    view_b_raw[3] = 0.8
    view_b = view_b_raw / np.linalg.norm(view_b_raw)
    confuser_raw = 0.5 * view_a + 0.5 * view_b
    confuser = confuser_raw / np.linalg.norm(confuser_raw)

    noise = 0.02
    tp_a = _make_unit(view_a[None, :] + noise * rng.normal(size=(4, d)))
    tp_b = _make_unit(view_b[None, :] + noise * rng.normal(size=(4, d)))
    fp = _make_unit(confuser[None, :] + noise * rng.normal(size=(8, d)))
    # 3 exemplars, unevenly split across the 2 genuine viewpoints (2 near A,
    # 1 near B) -- exactly DAVE's k=3 exemplar setup, just not uniformly
    # covering every real appearance variation.
    ref = _make_unit(np.stack([view_a, view_a, view_b]) + (noise / 2) * rng.normal(size=(3, d)))

    cand = np.concatenate([tp_a, tp_b, fp], axis=0)
    pooled_ref = ref.mean(axis=0)
    pooled_ref /= np.linalg.norm(pooled_ref)
    sims = cand @ pooled_ref
    # Sanity-check the premise: the confuser cluster's LOWEST pooled
    # similarity must still exceed EVERY genuine TP candidate's own pooled
    # similarity -- i.e. no threshold on this pooled score could work.
    assert sims[8:].min() > sims[:8].max(), (
        "test setup didn't actually make the confuser out-score every real match"
    )

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", min_cluster_size=2)
    keep_mask, _ = cluster_verify_candidates(cand, ref, cfg)

    assert keep_mask[:8].all(), "both genuine-viewpoint TP sub-clusters should be verified"
    assert not keep_mask[8:].any(), "the exemplar-free confuser cluster should be rejected"


def test_fallback_below_min_candidates():
    rng = np.random.default_rng(2)
    cand, ref, _, _ = _two_cluster_scene(rng)
    small_cand = cand[:2]

    calls = []

    def fallback(cand_feats, ref_feats):
        calls.append((cand_feats.shape, ref_feats.shape))
        return np.array([True, False])

    cfg = ClusterVerificationConfig(enabled=True, min_candidates_for_cluster=4)
    keep_mask, method_label = cluster_verify_candidates(
        small_cand, ref, cfg, fallback_keep_mask_fn=fallback,
    )

    assert method_label == "cluster_fallback_threshold"
    assert keep_mask.tolist() == [True, False]
    assert len(calls) == 1  # fallback actually invoked, not clustering


def test_fallback_required_below_min_candidates_without_fn_raises():
    rng = np.random.default_rng(3)
    cand, ref, _, _ = _two_cluster_scene(rng)
    cfg = ClusterVerificationConfig(enabled=True, min_candidates_for_cluster=4)

    with pytest.raises(ValueError):
        cluster_verify_candidates(cand[:2], ref, cfg, fallback_keep_mask_fn=None)


def test_no_candidates_returns_empty_mask():
    ref = _make_unit(np.random.default_rng(4).normal(size=(3, 16)))
    cfg = ClusterVerificationConfig(enabled=True)
    keep_mask, method_label = cluster_verify_candidates(
        np.zeros((0, 16)), ref, cfg,
    )
    assert keep_mask.shape == (0,)
    assert method_label == "cluster_no_candidates"


def test_single_exemplar_degenerate_case():
    """use_multi_ref=False callers pass a single fused prototype vector as
    ref_feats (k=1) -- still a valid, if less powerful, exemplar set (the
    DAVE paper's own one-shot variant still beats baselines)."""
    rng = np.random.default_rng(5)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)
    single_ref = ref.mean(axis=0, keepdims=True)
    single_ref = single_ref / np.linalg.norm(single_ref)

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", min_cluster_size=2)
    keep_mask, _ = cluster_verify_candidates(cand, single_ref, cfg)

    assert keep_mask[:n_tp].all()
    assert not keep_mask[n_tp:].any()


def test_all_exemplars_labelled_noise_keeps_nothing():
    """If every exemplar itself lands in HDBSCAN's noise cluster (no known-
    real neighborhood to verify against), trust nothing rather than guess."""
    rng = np.random.default_rng(6)
    d = 16
    # Candidates form one tight, unrelated cluster; the "exemplars" are 3
    # mutually-DISSIMILAR isolated points that won't cluster with anything
    # (or each other) at min_cluster_size=2.
    cand = _make_unit(np.eye(d)[2][None, :] + 0.02 * rng.normal(size=(6, d)))
    ref = _make_unit(rng.normal(size=(3, d)) * 5.0)  # spread far apart, near-orthogonal

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", min_cluster_size=2)
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == "cluster_hdbscan"
    assert not keep_mask.any()


def test_all_points_labelled_noise_falls_back_instead_of_rejecting():
    """Regression test for a real bug found while building
    cluster_secondary_filter: a SINGLE homogeneous, tightly-clustered group
    with NO second population to contrast against (confirmed empirically --
    even a very tight blob of candidates + exemplars with NO outlier
    present at all) can make HDBSCAN label EVERY point noise, since there
    is no density variation to anchor a resolvable cluster on. This is
    fundamentally different from "exemplars are noise but candidates DID
    form a real cluster" (test_all_exemplars_labelled_noise_keeps_nothing
    above, where rejecting is correct) -- here, clustering found ZERO
    structure at all, which is not evidence of an outlier and must not be
    read as one. Must fall back rather than confidently reject everyone."""
    rng = np.random.default_rng(7)
    d = 16
    view = np.zeros(d)
    view[0] = 1.0
    # A single tight, homogeneous population -- candidates AND exemplars
    # all genuinely alike, no confuser anywhere in the mix.
    cand = _make_unit(view[None, :] + 0.02 * rng.normal(size=(6, d)))
    ref = _make_unit(view[None, :] + 0.02 * rng.normal(size=(3, d)))

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", min_cluster_size=2)
    calls = []

    def fallback(cand_feats, ref_feats):
        calls.append((cand_feats.shape, ref_feats.shape))
        return np.ones(cand_feats.shape[0], dtype=bool)

    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg, fallback_keep_mask_fn=fallback)

    assert method_label == "cluster_hdbscan_fallback_inconclusive"
    assert len(calls) == 1, "fallback must have been invoked instead of confidently rejecting"
    assert keep_mask.all()


def test_all_points_labelled_noise_without_fallback_still_rejects():
    """No fallback provided -- can't safely resolve the inconclusive case
    either way, so it falls back to the same conservative reject as
    before (no regression for callers that don't supply one)."""
    rng = np.random.default_rng(8)
    d = 16
    view = np.zeros(d)
    view[0] = 1.0
    cand = _make_unit(view[None, :] + 0.02 * rng.normal(size=(6, d)))
    ref = _make_unit(view[None, :] + 0.02 * rng.normal(size=(3, d)))

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", min_cluster_size=2)
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == "cluster_hdbscan"
    assert not keep_mask.any()


# ---------------------------------------------------------------------------
# _self_tuning_n_clusters -- ported from DAVE's own reference implementation
# (models/dave.py::COTR.eigenDecomposition in the cloned DAVE repo), not
# just the paper text. See the field's own docstring (ClusterVerificationConfig
# .spectral_egv_threshold) for why this replaced an earlier fixed
# spectral_n_clusters=2.
# ---------------------------------------------------------------------------

def test_self_tuning_n_clusters_detects_two_well_separated_clusters():
    rng = np.random.default_rng(10)
    d = 16
    center_a = np.zeros(d)
    center_a[0] = 1.0
    center_b = np.zeros(d)
    center_b[1] = 1.0
    group_a = _make_unit(center_a[None, :] + 0.02 * rng.normal(size=(5, d)))
    group_b = _make_unit(center_b[None, :] + 0.02 * rng.normal(size=(5, d)))
    combined = np.concatenate([group_a, group_b], axis=0)
    affinity = np.clip(combined @ combined.T, 0.0, None)

    assert _self_tuning_n_clusters(affinity, egv_threshold=0.132) == 2


def test_self_tuning_n_clusters_returns_one_for_homogeneous_affinity():
    """No prominent eigengap (a single, undifferentiated cluster) -- DAVE's
    own code skips clustering entirely rather than forcing a split; this
    helper's return value of 1 achieves the same effect (SpectralClustering
    (n_clusters=1) trivially puts everyone in one cluster -- i.e. every
    candidate verifies, matching "keep everything, don't reject")."""
    rng = np.random.default_rng(11)
    d = 16
    center = np.zeros(d)
    center[0] = 1.0
    homogeneous = _make_unit(center[None, :] + 0.05 * rng.normal(size=(10, d)))
    affinity = np.clip(homogeneous @ homogeneous.T, 0.0, None)

    assert _self_tuning_n_clusters(affinity, egv_threshold=0.132) == 1


def test_self_tuning_n_clusters_higher_threshold_is_stricter():
    """A stricter (higher) egv_threshold demands a more prominent gap before
    trusting a split -- the same 2-cluster affinity that clears a lenient
    threshold can fall back to 1 (no split) under a strict enough one."""
    rng = np.random.default_rng(12)
    d = 16
    center_a = np.zeros(d)
    center_a[0] = 1.0
    center_b = np.zeros(d)
    center_b[1] = 1.0
    # Only WEAKLY separated (large noise relative to center separation) --
    # a real but modest eigengap, not the crisp one in the "well separated"
    # test above.
    group_a = _make_unit(center_a[None, :] + 0.35 * rng.normal(size=(5, d)))
    group_b = _make_unit(center_b[None, :] + 0.35 * rng.normal(size=(5, d)))
    combined = np.concatenate([group_a, group_b], axis=0)
    affinity = np.clip(combined @ combined.T, 0.0, None)

    assert _self_tuning_n_clusters(affinity, egv_threshold=0.01) >= 2
    assert _self_tuning_n_clusters(affinity, egv_threshold=0.9) == 1


def test_spectral_backend_uses_eigengap_not_fixed_two():
    """End-to-end: a truly homogeneous keyframe (candidates and exemplars
    all genuinely alike -- no real outlier present) must NOT have a
    spurious split forced onto it. A fixed spectral_n_clusters=2 (the
    earlier, pre-fix version of this module) would have rejected roughly
    half of these candidates for no reason; the eigengap-based estimate
    must keep all of them."""
    rng = np.random.default_rng(13)
    d = 16
    center = np.zeros(d)
    center[0] = 1.0
    cand = _make_unit(center[None, :] + 0.05 * rng.normal(size=(8, d)))
    ref = _make_unit(center[None, :] + 0.03 * rng.normal(size=(3, d)))

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="spectral")
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == "cluster_spectral"
    assert keep_mask.all(), "a homogeneous keyframe (no real outlier) must not have a spurious split forced onto it"


def test_max_candidates_for_cluster_skips_verification_entirely():
    """Matches DAVE's own performance safeguard (models/dave.py::forward:
    "if len(feat_pairs) > 500: return ... generated_bboxes") -- above the
    cap, keep every candidate unchanged rather than attempting (possibly
    very expensive) clustering."""
    rng = np.random.default_rng(14)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)  # would normally reject the FP half

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", max_candidates_for_cluster=5)
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == "cluster_skipped_too_many_candidates"
    assert keep_mask.all()


def test_max_candidates_for_cluster_none_means_no_cap():
    rng = np.random.default_rng(15)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", max_candidates_for_cluster=None)
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == "cluster_hdbscan"
    assert keep_mask[:n_tp].all()
    assert not keep_mask[n_tp:].any()


# ---------------------------------------------------------------------------
# pairwise_metric = "l1" / "mahalanobis" (cluster_verification.pairwise_metric)
# ---------------------------------------------------------------------------

def test_distance_and_affinity_cosine_unchanged():
    """pairwise_metric="cosine" (default) must produce EXACTLY the same
    distance/affinity this project's own cosine path always has -- the
    refactor that introduced _distance_and_affinity must not have changed
    cosine's own numbers even slightly."""
    from aero_eyes.utils.cluster_verify import _distance_and_affinity

    rng = np.random.default_rng(0)
    combined = _make_unit(rng.normal(size=(6, 8)))
    distance, affinity, suffix = _distance_and_affinity(combined, "cosine", None)

    expected_similarity = combined @ combined.T
    expected_distance = np.clip(1.0 - expected_similarity, 0.0, None)
    np.fill_diagonal(expected_distance, 0.0)
    expected_affinity = np.clip(expected_similarity, 0.0, None)

    assert np.allclose(distance, expected_distance)
    assert np.allclose(affinity, expected_affinity)
    assert suffix == ""


def test_distance_and_affinity_l1_differs_from_cosine():
    from aero_eyes.utils.cluster_verify import _distance_and_affinity

    rng = np.random.default_rng(1)
    combined = _make_unit(rng.normal(size=(6, 8)))
    cosine_distance, _, cosine_suffix = _distance_and_affinity(combined, "cosine", None)
    l1_distance, l1_affinity, l1_suffix = _distance_and_affinity(combined, "l1", None)

    assert l1_suffix == "_l1"
    assert not np.allclose(l1_distance, cosine_distance), "L1 must give a genuinely different distance structure"
    assert np.all(l1_affinity >= 0), "spectral needs a non-negative affinity"
    assert np.allclose(np.diag(l1_distance), 0.0), "a point's distance to itself must be 0"


def test_distance_and_affinity_mahalanobis_requires_precision_matrix():
    from aero_eyes.utils.cluster_verify import _distance_and_affinity

    rng = np.random.default_rng(2)
    combined = _make_unit(rng.normal(size=(6, 8)))
    with pytest.raises(ValueError, match="needs a shared precision matrix"):
        _distance_and_affinity(combined, "mahalanobis", None)


def test_distance_and_affinity_mahalanobis_identity_precision_matches_euclidean():
    """Mahalanobis distance with precision_matrix = Identity reduces
    EXACTLY to plain Euclidean (L2) distance -- a clean mathematical
    sanity check that the wiring (scipy's VI= parameter) is correct."""
    from scipy.spatial.distance import pdist, squareform

    from aero_eyes.utils.cluster_verify import _distance_and_affinity

    rng = np.random.default_rng(3)
    combined = rng.normal(size=(6, 8))  # no need to L2-normalize for this check
    identity = np.eye(8)

    maha_distance, _, suffix = _distance_and_affinity(combined, "mahalanobis", identity)
    euclidean_distance = squareform(pdist(combined, metric="euclidean"))

    assert suffix == "_mahalanobis"
    assert np.allclose(maha_distance, euclidean_distance, atol=1e-6)


def test_distance_and_affinity_unknown_pairwise_metric_raises():
    from aero_eyes.utils.cluster_verify import _distance_and_affinity

    combined = _make_unit(np.random.default_rng(4).normal(size=(4, 8)))
    with pytest.raises(ValueError, match="Unknown cluster_verification.pairwise_metric"):
        _distance_and_affinity(combined, "bogus", None)


@pytest.mark.parametrize("cluster_method", ["hdbscan", "spectral"])
def test_cluster_verify_candidates_l1_separates_tp_from_fp(cluster_method):
    """pairwise_metric="l1" end-to-end through cluster_verify_candidates --
    on the SAME easy, well-separated scene as the cosine baseline test
    above, l1 should also cleanly separate TP from FP (not claiming l1 is
    BETTER than cosine, just that it's wired correctly and functional)."""
    rng = np.random.default_rng(5)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)

    cfg = ClusterVerificationConfig(
        enabled=True, cluster_method=cluster_method, pairwise_metric="l1", min_cluster_size=2,
    )
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg)

    assert method_label == f"cluster_{cluster_method}_l1"
    assert keep_mask[:n_tp].all(), "every TP candidate should cluster with an exemplar"
    assert not keep_mask[n_tp:].any(), "every FP candidate should be rejected"


@pytest.mark.parametrize("cluster_method", ["hdbscan", "spectral"])
def test_cluster_verify_candidates_mahalanobis_separates_tp_from_fp(cluster_method):
    """pairwise_metric="mahalanobis" end-to-end, with an identity precision
    matrix (equivalent to Euclidean distance -- see the identity-matches-
    euclidean unit test above) on the same easy scene."""
    rng = np.random.default_rng(6)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)
    d = cand.shape[1]

    cfg = ClusterVerificationConfig(
        enabled=True, cluster_method=cluster_method, pairwise_metric="mahalanobis", min_cluster_size=2,
    )
    keep_mask, method_label = cluster_verify_candidates(cand, ref, cfg, precision_matrix=np.eye(d))

    assert method_label == f"cluster_{cluster_method}_mahalanobis"
    assert keep_mask[:n_tp].all(), "every TP candidate should cluster with an exemplar"
    assert not keep_mask[n_tp:].any(), "every FP candidate should be rejected"


def test_cluster_verify_candidates_mahalanobis_without_precision_matrix_raises():
    rng = np.random.default_rng(7)
    cand, ref, n_tp, n_fp = _two_cluster_scene(rng)

    cfg = ClusterVerificationConfig(enabled=True, cluster_method="hdbscan", pairwise_metric="mahalanobis")
    with pytest.raises(ValueError, match="needs a shared precision matrix"):
        cluster_verify_candidates(cand, ref, cfg)  # precision_matrix defaults to None

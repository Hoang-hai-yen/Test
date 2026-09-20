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
from aero_eyes.utils.cluster_verify import cluster_verify_candidates


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
        enabled=True, cluster_method=cluster_method, min_cluster_size=2, spectral_n_clusters=2,
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

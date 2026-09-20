"""DAVE (arXiv:2404.16622) module (ii)-style candidate verification.

Shared primitive for stage3.verification_method="cluster"
(aero_eyes/stages/stage3.py) and
stage123_geco2.dynamic_prototype.cluster_verification
(aero_eyes/models/geco2_detector.py's GeCo2DynamicPrototypeTracker.offer_topk)
-- see ClusterVerificationConfig's own docstring in aero_eyes/config.py for
the full rationale versus a global scalar threshold (adaptive_threshold's
z_score/otsu/gmm, or topk_fusion's Z-score fusion).

Given a keyframe's own candidate appearance features and the current
exemplar features (per_ref_features), clusters them TOGETHER by pairwise
cosine similarity and keeps a candidate iff it shares a cluster with at
least one exemplar -- the direct analog of DAVE's "cluster contains an
exemplar -> true positive; otherwise -> outlier" rule. No scalar threshold,
no distributional-shape assumption, and the decision only ever reads the
arrays passed in -- callers get causal/online behavior for free by calling
this once per keyframe instead of once per video.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def cluster_verify_candidates(
    cand_feats: np.ndarray,
    ref_feats: np.ndarray,
    cfg,
    fallback_keep_mask_fn=None,
) -> tuple[np.ndarray, str]:
    """Returns (keep_mask [N] bool, method_label).

    cand_feats: [N, D] L2-normalized candidate appearance features (this
        keyframe's own candidates only -- callers must group by keyframe
        themselves, this function has no notion of "video" at all).
    ref_feats: [k, D] L2-normalized exemplar features (per_ref_features),
        or [1, D] as a degenerate single-fused-prototype fallback.
    cfg: ClusterVerificationConfig (aero_eyes.config).
    fallback_keep_mask_fn: callable(cand_feats, ref_feats) -> np.ndarray[bool],
        used when there are fewer than cfg.min_candidates_for_cluster
        candidates this keyframe (clustering can't find meaningful
        structure from too few points -- same "too little data for a
        shape method" precedent stage3's own otsu/gmm use for
        adaptive_threshold_min_samples). Required in that case; a missing
        fallback with too few candidates raises rather than silently
        keeping/dropping everything.
    """
    n = cand_feats.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool), "cluster_no_candidates"

    if ref_feats.ndim == 1:
        ref_feats = ref_feats[None, :]

    if n < cfg.min_candidates_for_cluster:
        if fallback_keep_mask_fn is None:
            raise ValueError(
                f"cluster_verify_candidates: only {n} candidate(s) this keyframe "
                f"(< cluster_verification.min_candidates_for_cluster={cfg.min_candidates_for_cluster}) "
                "but no fallback_keep_mask_fn was provided."
            )
        keep_mask = fallback_keep_mask_fn(cand_feats, ref_feats)
        return keep_mask, "cluster_fallback_threshold"

    combined = np.concatenate([cand_feats, ref_feats], axis=0)  # [N+k, D]
    similarity = combined @ combined.T  # cosine, since inputs are L2-normalized
    k = ref_feats.shape[0]

    if cfg.cluster_method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        # Cosine distance for a precomputed-metric clusterer; clip the tiny
        # negative values floating-point roundoff can produce at distance 0.
        distance = np.clip(1.0 - similarity, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        labels = HDBSCAN(
            metric="precomputed",
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            copy=False,  # `distance` above is a fresh local array, never reused after this call
        ).fit_predict(distance)
        method_label = "cluster_hdbscan"
        noise_label = -1
    elif cfg.cluster_method == "spectral":
        from sklearn.cluster import SpectralClustering

        # Spectral affinity must be a non-negative similarity, not a distance.
        affinity = np.clip(similarity, 0.0, None)
        n_clusters = min(cfg.spectral_n_clusters, combined.shape[0])
        labels = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=0,
        ).fit_predict(affinity)
        method_label = "cluster_spectral"
        noise_label = None  # spectral clustering never labels a point as noise
    else:
        raise ValueError(f"Unknown cluster_verification.cluster_method '{cfg.cluster_method}'")

    cand_labels = labels[:n]
    ref_labels = set(labels[n:].tolist())
    if noise_label is not None:
        ref_labels.discard(noise_label)

    if not ref_labels:
        # Every exemplar itself landed in the noise cluster (hdbscan only) --
        # no known-real neighborhood to verify against this keyframe. Trust
        # nothing rather than guess, consistent with this mechanism's
        # "verified or absent" philosophy.
        log.debug(
            "[cluster_verify] all %d exemplar(s) labelled noise by %s -- "
            "0/%d candidates verified this keyframe",
            k, method_label, n,
        )
        return np.zeros(n, dtype=bool), method_label

    keep_mask = np.array([
        (lbl in ref_labels) and (noise_label is None or lbl != noise_label)
        for lbl in cand_labels
    ], dtype=bool)
    return keep_mask, method_label

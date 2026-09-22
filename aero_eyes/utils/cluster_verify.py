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

Checked directly against a cloned copy of DAVE's own reference
implementation (models/dave.py in the DAVE repo, not just the paper text)
to confirm fidelity:
  - affinity = cosine similarity clipped to >= 0 (matches dst_mtx's
    `dst_mtx[dst_mtx < 0] = 0`).
  - "spectral" cluster count is estimated PER KEYFRAME via the self-tuning
    eigengap heuristic on the affinity's normalized graph Laplacian
    (_self_tuning_n_clusters below), matching eigenDecomposition -- NOT a
    fixed hand-set n_clusters (an earlier version of this module used a
    fixed spectral_n_clusters=2, confirmed against the real code to be a
    deviation that could inject a spurious split on an actually-homogeneous
    keyframe).
  - a very large candidate set skips verification entirely rather than
    paying for clustering at that scale (max_candidates_for_cluster,
    matching DAVE's own `if len(feat_pairs) > 500: return ... unchanged`).
DAVE's own IoU-based extra-inclusion rule (a candidate spatially
overlapping an exemplar's OWN box location in the same image is always
kept) is NOT ported -- it only makes sense when exemplars are annotated
instances living INSIDE the image being counted (FSC147's setup); this
project's exemplars are separate close-up reference photos with no spatial
correspondence to any candidate box in a video frame.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _self_tuning_n_clusters(affinity: np.ndarray, egv_threshold: float) -> int:
    """Self-tuning spectral clustering cluster-count estimate (Zelnik-Manor
    & Perona, "Self-Tuning Spectral Clustering", NeurIPS 2004) via the
    eigengap heuristic on the affinity matrix's normalized graph Laplacian
    -- ported to match DAVE's OWN reference implementation exactly
    (models/dave.py::COTR.eigenDecomposition in the cloned DAVE repo, NOT
    just the paper text): eigenvalues of the normalized Laplacian are
    (near-)zero for each well-separated cluster, so the largest gap(s) in
    the SORTED eigenvalue sequence indicate the natural cluster count.

    One deliberate deviation from DAVE's own code: uses np.linalg.eigvalsh
    (for real symmetric matrices -- the normalized Laplacian of a symmetric
    cosine-similarity affinity always is) instead of DAVE's plain
    np.linalg.eig, which does not guarantee real or sorted eigenvalues and
    can pick up spurious tiny imaginary parts from floating-point
    asymmetry. eigvalsh guarantees real, ascending-sorted eigenvalues,
    which the eigengap heuristic's own math assumes -- this is a
    correctness fix over the literal reference code, not a behavior change
    to the algorithm it implements.

    Returns 1 (no split -- every point ends up in the same cluster once fed
    to SpectralClustering(n_clusters=1), i.e. every candidate verifies)
    when no gap exceeds egv_threshold -- matching DAVE's own
    `if len(k) > 1 or k[0] > 1` skip-clustering-entirely behavior for a
    keyframe whose affinity structure looks homogeneous rather than
    forcing a split onto it.
    """
    from scipy.sparse import csgraph

    laplacian = csgraph.laplacian(affinity, normed=True)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    diffs = np.diff(eigenvalues)
    if diffs.size == 0:
        return 1

    # DAVE's own heuristic: look at the (up to) 5 largest gaps, keep only
    # those exceeding the threshold (in gap-size-descending order), then
    # use the LARGER cluster-count suggested by the top 2 surviving gaps.
    top_gap_indices = np.argsort(diffs)[::-1][:5]
    candidate_counts = [int(i) + 1 for i in top_gap_indices if diffs[i] > egv_threshold]
    if not candidate_counts:
        return 1
    return max(candidate_counts[:2])


def _rbf_affinity(distance: np.ndarray) -> np.ndarray:
    """Converts an arbitrary distance matrix into a non-negative affinity
    matrix for spectral clustering, via a Gaussian/RBF kernel with sigma
    set to the median off-diagonal distance (a standard, scale-adaptive
    default). ONLY used for pairwise_metric in ("l1", "mahalanobis") --
    cosine keeps its own original, DAVE-fidelity-matched affinity
    (clip(similarity, 0, None)) untouched, computed directly rather than
    going through this generic conversion, so this function's introduction
    cannot change cosine's existing behavior at all.

    NOT from DAVE's own reference code (which only ever used cosine
    similarity as-is for its affinity) -- this conversion is new, needed
    only to make l1/mahalanobis distances usable by spectral clustering
    (which requires a similarity/affinity, not a distance), and is NOT YET
    VALIDATED.
    """
    n = distance.shape[0]
    off_diag = distance[~np.eye(n, dtype=bool)]
    sigma = float(np.median(off_diag)) if off_diag.size > 0 else 1.0
    sigma = max(sigma, 1e-8)
    return np.exp(-(distance ** 2) / (2.0 * sigma ** 2))


def _distance_and_affinity(
    combined: np.ndarray, pairwise_metric: str, precision_matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (distance_for_hdbscan, affinity_for_spectral, method_label_suffix)
    for the given cluster_verification.pairwise_metric -- see that field's
    own docstring (aero_eyes/config.py) for the full rationale of each
    option. combined is [N+k, D], candidates then refs, same as the
    caller's own `combined` array.
    """
    if pairwise_metric == "cosine":
        similarity = combined @ combined.T  # cosine, since inputs are L2-normalized
        # Clip the tiny negative values floating-point roundoff can produce
        # at distance 0.
        distance = np.clip(1.0 - similarity, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        affinity = np.clip(similarity, 0.0, None)
        return distance, affinity, ""

    if pairwise_metric == "l1":
        from scipy.spatial.distance import pdist, squareform
        distance = squareform(pdist(combined, metric="cityblock"))
        return distance, _rbf_affinity(distance), "_l1"

    if pairwise_metric == "mahalanobis":
        if precision_matrix is None:
            raise ValueError(
                "cluster_verification.pairwise_metric='mahalanobis' needs a shared precision "
                "matrix (the inverse covariance from RMD's own background fit -- "
                "aero_eyes.stages.stage3._fit_rmd_background) passed through as "
                "cluster_verify_candidates(precision_matrix=...). Currently only wired at "
                "stage3.py's verification_method='cluster' and cluster_secondary_filter call "
                "sites -- stage123_geco2.dynamic_prototype.cluster_verification (GeCo2's own "
                "online/causal path) has no equivalent whole-video background fit yet and will "
                "always hit this error if pairwise_metric='mahalanobis' is set there."
            )
        from scipy.spatial.distance import pdist, squareform
        distance = squareform(pdist(combined, metric="mahalanobis", VI=precision_matrix))
        return distance, _rbf_affinity(distance), "_mahalanobis"

    raise ValueError(
        f"Unknown cluster_verification.pairwise_metric '{pairwise_metric}'. "
        "Must be 'cosine', 'l1', or 'mahalanobis'."
    )


def cluster_verify_candidates(
    cand_feats: np.ndarray,
    ref_feats: np.ndarray,
    cfg,
    fallback_keep_mask_fn=None,
    precision_matrix: np.ndarray | None = None,
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
    precision_matrix: [D, D] shared inverse-covariance -- REQUIRED when
        cfg.pairwise_metric == "mahalanobis" (raises otherwise), ignored
        for every other pairwise_metric. See _distance_and_affinity's own
        docstring for where this comes from and which callers wire it
        through.
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

    k = ref_feats.shape[0]
    if cfg.max_candidates_for_cluster is not None and (n + k) > cfg.max_candidates_for_cluster:
        # DAVE's own performance safeguard (models/dave.py::forward:
        # "if len(feat_pairs) > 500: return ... generated_bboxes") -- a
        # very large affinity matrix makes clustering (especially
        # spectral's O(N^3) eigendecomposition) expensive; skip verification
        # ENTIRELY and keep every candidate, rather than silently paying an
        # unbounded cost or applying a shape-based method with no evidence
        # it stays reliable at this scale. Not a quality decision -- purely
        # "too expensive to even attempt this keyframe."
        return np.ones(n, dtype=bool), "cluster_skipped_too_many_candidates"

    combined = np.concatenate([cand_feats, ref_feats], axis=0)  # [N+k, D]
    distance, affinity, metric_suffix = _distance_and_affinity(
        combined, cfg.pairwise_metric, precision_matrix,
    )

    if cfg.cluster_method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(
            metric="precomputed",
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            copy=False,  # `distance` above is a fresh local array, never reused after this call
        ).fit_predict(distance)
        method_label = f"cluster_hdbscan{metric_suffix}"
        noise_label = -1
    elif cfg.cluster_method == "spectral":
        from sklearn.cluster import SpectralClustering

        n_clusters = min(_self_tuning_n_clusters(affinity, cfg.spectral_egv_threshold), combined.shape[0])
        labels = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=0,
        ).fit_predict(affinity)
        method_label = f"cluster_spectral{metric_suffix}"
        noise_label = None  # spectral clustering never labels a point as noise
    else:
        raise ValueError(f"Unknown cluster_verification.cluster_method '{cfg.cluster_method}'")

    cand_labels = labels[:n]
    ref_labels = set(labels[n:].tolist())
    if noise_label is not None:
        ref_labels.discard(noise_label)

    if noise_label is not None and bool(np.all(labels == noise_label)):
        # EVERY point -- candidates AND exemplars alike -- landed in the
        # noise cluster. Confirmed empirically (not just theoretical): a
        # SINGLE homogeneous, tightly-clustered group with NO second
        # population to contrast against (e.g. a keyframe where every
        # candidate genuinely matches, no confuser present at all) can make
        # HDBSCAN label everything noise even at min_cluster_size=2 -- it
        # has no density VARIATION to anchor a resolvable cluster on, which
        # is different from "the exemplars are outliers relative to a real
        # candidate cluster" (handled below). Treating this the same way
        # as that case would be exactly backwards: "no evidence of any
        # outlier structure at all" should NOT be read as "reject
        # everyone" -- it means clustering was INCONCLUSIVE here, so fall
        # back to whatever policy the caller supplies for "too little
        # evidence to cluster meaningfully" (same fallback_keep_mask_fn
        # used below min_candidates_for_cluster), not a confident reject.
        log.debug(
            "[cluster_verify] %s labelled ALL %d point(s) (candidates + exemplars) as noise -- "
            "inconclusive, not a genuine outlier signal -- falling back",
            method_label, n + k,
        )
        if fallback_keep_mask_fn is None:
            return np.zeros(n, dtype=bool), method_label
        keep_mask = fallback_keep_mask_fn(cand_feats, ref_feats)
        return keep_mask, f"{method_label}_fallback_inconclusive"

    if not ref_labels:
        # Exemplars themselves landed in the noise cluster while some
        # CANDIDATES got a real (non-noise) cluster label -- unlike the
        # all-noise case above, this means clustering DID find resolvable
        # structure, just not one the exemplars belong to. No known-real
        # neighborhood to verify against this keyframe -- trust nothing
        # rather than guess, consistent with this mechanism's "verified or
        # absent" philosophy.
        log.debug(
            "[cluster_verify] all %d exemplar(s) labelled noise by %s (candidates DID form a "
            "real cluster) -- 0/%d candidates verified this keyframe",
            k, method_label, n,
        )
        return np.zeros(n, dtype=bool), method_label

    keep_mask = np.array([
        (lbl in ref_labels) and (noise_label is None or lbl != noise_label)
        for lbl in cand_labels
    ], dtype=bool)
    return keep_mask, method_label

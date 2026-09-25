"""Stage 3 — Cross-domain matching.

Flow:  candidates.json + prototype.npz
       -> cosine similarity
       -> threshold filter
       -> NMS across tiles
       -> top-K per keyframe
       -> detections.json

Reads:  prototype.npz, candidates.json (+.feats.npz)
Writes: detections.json
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from aero_eyes.types import Detection

log = logging.getLogger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (both assumed L2-normalized)."""
    return float(np.dot(a, b))


def apply_whitening(cfg_w, all_feats: np.ndarray, prototype: np.ndarray, per_ref_features: list):
    """stage3.whitening: map candidate features [N,D], the fused prototype [D]
    and per-ref vectors (list of [D]) through the offline-fitted PCA whitening
    at cfg_w.weights_path (see aero_eyes.models.whitening). Returns the
    transformed (all_feats, prototype, per_ref_features), all L2-normalised in
    the whitened space."""
    from aero_eyes.models.whitening import Whitener

    if not cfg_w.weights_path:
        raise ValueError("stage3.whitening.enabled=true needs stage3.whitening.weights_path "
                         "(fit one with scripts/fit_pca_whitening.py).")
    path = Path(cfg_w.weights_path)
    if not path.exists():
        raise FileNotFoundError(f"stage3.whitening.weights_path {path} not found.")
    w = Whitener.load(path)
    if all_feats.shape[1] != w.in_dim:
        raise ValueError(
            f"stage3.whitening: weights expect {w.in_dim}-d embeddings but candidates have "
            f"{all_feats.shape[1]}-d -- they were fitted with a different extractor "
            "(model/variant/pooling); refit with scripts/fit_pca_whitening.py under the same settings."
        )
    return (
        w.transform(all_feats),
        w.transform(prototype),
        [w.transform(f) for f in per_ref_features],
    )


def _fit_rmd_background(all_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fits a shared, regularized covariance (Ledoit-Wolf shrinkage --
    candidate count can be less than embedding dimensionality, where a
    plain sample covariance would be singular/unstable) from ALL candidate
    features this video produced -- the background/generic distribution,
    since most candidates are background/FP by construction. Returns
    (mu_background, precision_background); the precision matrix (inverse
    covariance) is what Mahalanobis distance actually needs, computed once
    here rather than re-inverting per call. See Stage3Config.similarity's
    "rmd" docstring for the full rationale."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(all_feats)
    return lw.location_, lw.precision_


def _mahalanobis_sq(feats: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every row of feats [N,D] to mu [D],
    under the given precision (inverse covariance) matrix [D,D]. Squared
    (not sqrt) since only relative comparisons matter downstream and every
    consumer here (threshold, cluster, margin) is monotonic in the score."""
    diff = feats - mu[None, :]
    return np.einsum("ni,ij,nj->n", diff, precision, diff)


def _score_against_ref(
    feats: np.ndarray, ref: np.ndarray, metric: str,
    background: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Score every row of feats [N,D] against a single reference vector [D].

    Higher score always means "more similar" regardless of metric, so the
    rest of Stage 3 (threshold filtering, adaptive threshold, ranking) works
    unchanged no matter which metric is selected. For distance metrics
    (l1/l2) this means returning the negated distance.

    background: (mu_background, precision_background) from
    _fit_rmd_background -- REQUIRED when metric == "rmd", ignored otherwise.
    """
    if metric == "cosine":
        return feats @ ref
    if metric == "l2":
        return -np.linalg.norm(feats - ref[None, :], axis=1)
    if metric == "l1":
        return -np.sum(np.abs(feats - ref[None, :]), axis=1)
    if metric == "rmd":
        if background is None:
            raise ValueError(
                "stage3.similarity='rmd' requires background stats (mu_background, "
                "precision_background) -- see _fit_rmd_background; none were provided."
            )
        mu_bg, precision_bg = background
        # "how much closer to the exemplar than to generic background, in
        # whitened space" -- higher = more similar, consistent with every
        # other metric here.
        return _mahalanobis_sq(feats, mu_bg, precision_bg) - _mahalanobis_sq(feats, ref, precision_bg)
    raise ValueError(f"Unknown stage3.similarity metric '{metric}'. Must be 'cosine', 'l1', 'l2', or 'rmd'.")


def _pool_sims(
    sims_per_ref: list, pooling: str, ref_feats: np.ndarray | list | None = None,
    agreement_epsilon: float = 10.0,
) -> np.ndarray:
    """Combine per-reference-image similarity arrays into one, per
    accuracy.cheap_boosters.multi_ref_pooling (see that field's own
    docstring in aero_eyes/config.py for the full tradeoff writeup):
      mean -- average across refs (original behavior).
      max -- the single best-matching ref's score per candidate ("OR" over
        refs -- favors recall, structurally risks a confuser leaking
        through via just one coincidentally-permissive reference).
      min -- the single WORST-matching ref's score per candidate ("AND"
        over refs -- favors precision at some recall cost).
      agreement_weighted -- NOT YET VALIDATED -- weighted average using
        aero_eyes.utils.ref_agreement.agreement_weights(ref_feats,
        agreement_epsilon); REQUIRES ref_feats (raises otherwise).
    """
    if pooling == "max":
        return np.max(sims_per_ref, axis=0)
    if pooling == "min":
        return np.min(sims_per_ref, axis=0)
    if pooling == "agreement_weighted":
        if ref_feats is None:
            raise ValueError("_pool_sims(pooling='agreement_weighted') requires ref_feats.")
        from aero_eyes.utils.ref_agreement import agreement_weights
        weights = agreement_weights(np.asarray(ref_feats), agreement_epsilon)
        return np.average(np.stack(sims_per_ref, axis=0), axis=0, weights=weights)
    return np.mean(sims_per_ref, axis=0)


def _otsu_threshold(sims: np.ndarray, num_bins: int) -> float:
    """Otsu's method on a real-valued 1-D array: histogram into num_bins,
    then pick the bin-edge split maximizing the between-class variance of
    the "below" vs. "at-or-above" partitions -- the standard 2-class Otsu
    algorithm (cv2's THRESH_OTSU only accepts 8-bit input, so this is a
    direct numpy implementation instead of reusing it). No z multiplier:
    the split is wherever the data's OWN two implied classes separate
    best, so it self-adapts to how much of `sims` is background vs.
    signal instead of assuming a fixed offset from the center.
    """
    lo, hi = float(sims.min()), float(sims.max())
    if hi <= lo:
        return lo  # degenerate: every value identical, no split possible
    hist, edges = np.histogram(sims, bins=num_bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    weight_below = np.cumsum(hist)
    weight_above = hist.sum() - weight_below
    cum_sum = np.cumsum(hist * bin_centers)
    total_sum = cum_sum[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_below = cum_sum / weight_below
        mean_above = (total_sum - cum_sum) / weight_above
        between_class_var = weight_below * weight_above * (mean_below - mean_above) ** 2
    # Splits with an empty class (all-below or all-above the candidate bin)
    # produce NaN from the 0/0 division above -- not a valid 2-class split.
    between_class_var = np.nan_to_num(between_class_var, nan=-1.0)
    return float(bin_centers[int(np.argmax(between_class_var))])


def _gaussian_intersection(mean1: float, std1: float, w1: float, mean2: float, std2: float, w2: float) -> float:
    """x where w1*N(x; mean1, std1) == w2*N(x; mean2, std2) -- the natural
    decision boundary between two 1-D Gaussian mixture components. Solves
    the log-likelihood-ratio equation analytically (quadratic in the
    general case, linear when std1==std2); falls back to the plain
    midpoint if the quadratic has no real root between the two means (can
    happen when the components nearly coincide)."""
    if abs(std1 - std2) < 1e-9:
        denom = mean2 - mean1
        if abs(denom) < 1e-12:
            return float((mean1 + mean2) / 2.0)
        return float((mean1 + mean2) / 2.0 + (std1 ** 2) * np.log(w2 / w1) / denom)

    a = 1.0 / (2 * std1 ** 2) - 1.0 / (2 * std2 ** 2)
    b = mean2 / (std2 ** 2) - mean1 / (std1 ** 2)
    c = (mean1 ** 2) / (2 * std1 ** 2) - (mean2 ** 2) / (2 * std2 ** 2) - np.log((std2 * w1) / (std1 * w2))
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return float((mean1 + mean2) / 2.0)
    sqrt_disc = np.sqrt(disc)
    roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
    lo, hi = sorted((mean1, mean2))
    in_range = [r for r in roots if lo <= r <= hi]
    return float(in_range[0]) if in_range else float((mean1 + mean2) / 2.0)


def _gmm_threshold(sims: np.ndarray, min_separation_std: float, fallback_percentile: float) -> tuple[float, str]:
    """Fits 1- and 2-component 1-D Gaussian mixtures to `sims`. Bimodal
    (better BIC AND components separated by >= min_separation_std pooled
    std) -> threshold at the analytic crossing point between the two
    fitted Gaussians. Not bimodal (e.g. a low-FP sample where `sims` is
    really one cluster of mostly true positives, nothing resembling a
    second background cluster) -> forcing a 2-cluster split onto it is
    meaningless, so falls back to a permissive percentile cut instead.
    Returns (threshold, stat_label) with stat_label recording which path
    was taken, for observability (matches "mean/std"/"median/MAD" style).
    """
    from sklearn.mixture import GaussianMixture

    x = sims.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(x)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(x)

    means = gmm2.means_.flatten()
    stds = np.sqrt(gmm2.covariances_.flatten())
    weights = gmm2.weights_
    lo_idx, hi_idx = (0, 1) if means[0] <= means[1] else (1, 0)
    pooled_std = float(np.sqrt((stds[lo_idx] ** 2 + stds[hi_idx] ** 2) / 2.0)) + 1e-8
    separation = (means[hi_idx] - means[lo_idx]) / pooled_std

    if gmm2.bic(x) < gmm1.bic(x) and separation >= min_separation_std:
        threshold = _gaussian_intersection(
            float(means[lo_idx]), float(stds[lo_idx]), float(weights[lo_idx]),
            float(means[hi_idx]), float(stds[hi_idx]), float(weights[hi_idx]),
        )
        return threshold, "gmm_bimodal"
    return float(np.percentile(sims, fallback_percentile)), "gmm_unimodal_fallback"


def compute_adaptive_threshold(
    all_sims: np.ndarray,
    all_sims_original_refs: np.ndarray,
    similarity_metric: str,
    s3,
) -> tuple[float, float, float, str]:
    """stage3.adaptive_threshold's actual threshold computation, factored
    out of run_stage3 for direct unit testing. Returns (effective_threshold,
    center, spread, stat_label).

    stats_sims (which distribution gets SUMMARIZED into center/spread) is
    `all_sims_original_refs` when s3.adaptive_threshold_anchor_to_original_refs
    is set, else `all_sims` -- independent of which distribution decides
    ACCEPTANCE (always the caller's own `all_sims`, via its own keep_mask).
    See Stage3Config.adaptive_threshold_anchor_to_original_refs's docstring
    for why these are deliberately different arrays: dynamic_prototype can
    inflate a handful of OTHER candidates' scores via max-pooling once it
    appends narrow, self-selected reference vectors, dragging mean/std (and
    so the threshold) up for everyone -- anchoring keeps the threshold
    computation stable regardless.

    stat_label is "mean/std"/"median/MAD" (s3.adaptive_threshold_robust,
    method="z_score"), "otsu" (method="otsu"), or "gmm_bimodal"/
    "gmm_unimodal_fallback" (method="gmm") -- see
    Stage3Config.adaptive_threshold_method's own docstring for the full
    rationale behind offering "otsu"/"gmm" as alternatives to a fixed z
    multiplier. Both fall back to the z_score path (label suffixed
    "_min_samples_fallback") when stats_sims has fewer than
    s3.adaptive_threshold_min_samples points -- too little data for a
    distribution-SHAPE method to be reliable.
    """
    stats_sims = all_sims_original_refs if s3.adaptive_threshold_anchor_to_original_refs else all_sims

    method = s3.adaptive_threshold_method
    if method in ("otsu", "gmm") and len(stats_sims) < s3.adaptive_threshold_min_samples:
        log.warning(
            "stage3.adaptive_threshold_method=%r but only %d similarity samples on hand "
            "(< adaptive_threshold_min_samples=%d) -- falling back to z_score for this sample.",
            method, len(stats_sims), s3.adaptive_threshold_min_samples,
        )
        method = "z_score"

    if method == "otsu":
        center, spread = float(stats_sims.mean()), float(stats_sims.std())
        raw_threshold = _otsu_threshold(stats_sims, s3.adaptive_otsu_bins)
        stat_label = "otsu"
    elif method == "gmm":
        center, spread = float(stats_sims.mean()), float(stats_sims.std())
        raw_threshold, stat_label = _gmm_threshold(
            stats_sims, s3.adaptive_gmm_min_separation_std, s3.adaptive_gmm_fallback_percentile,
        )
    else:
        if s3.adaptive_threshold_robust:
            center = float(np.median(stats_sims))
            # 1.4826 = consistency constant that makes MAD comparable to std
            # under a roughly-normal distribution, so adaptive_z_score means
            # roughly the same thing in either mode.
            spread = float(1.4826 * np.median(np.abs(stats_sims - center)))
            stat_label = "median/MAD"
        else:
            center = float(stats_sims.mean())
            spread = float(stats_sims.std())
            stat_label = "mean/std"
        raw_threshold = center + s3.adaptive_z_score * spread

    # adaptive_min_floor is calibrated for cosine's roughly [-1,1] range.
    # l1/l2 scores are negated distances (unbounded, typically negative),
    # so the floor has no meaningful interpretation there -- skip it.
    if similarity_metric == "cosine":
        effective_threshold = max(s3.adaptive_min_floor, raw_threshold)
    else:
        effective_threshold = raw_threshold

    return effective_threshold, center, spread, stat_label


class OnlineAdaptiveThreshold:
    """Causal, streaming-compatible replacement for compute_adaptive_
    threshold's whole-video computation -- see Stage3Config.
    adaptive_threshold_online's own docstring for the real-time deployment
    rationale (a live drone feed can't wait for the whole video before
    deciding what to do with keyframe 1).

    threshold_for_next_frame() decides the upcoming keyframe's threshold
    using ONLY similarity scores observed at STRICTLY EARLIER keyframes (a
    running window, oldest evicted once adaptive_threshold_online_window
    is exceeded); observe() then feeds that keyframe's OWN scores in,
    called AFTER its own accept/reject decision was already made against
    the pre-update window -- a keyframe's own candidates can never
    influence their own threshold, only future keyframes'. Reuses
    compute_adaptive_threshold's z_score/otsu/gmm dispatch unchanged,
    just fed the running buffer instead of the whole video's all_sims.
    """

    def __init__(self, s3):
        from collections import deque
        self.s3 = s3
        self.history: deque = deque(maxlen=s3.adaptive_threshold_online_window)

    def threshold_for_next_frame(self) -> tuple[float, str]:
        """Returns (threshold, stat_label) to apply to the UPCOMING
        keyframe's candidates. Cold start (fewer than
        s3.adaptive_threshold_min_samples observed so far) falls back to
        adaptive_min_floor -- same "not enough history to trust a
        distribution-shape method yet" philosophy compute_adaptive_
        threshold's own otsu/gmm min-samples fallback uses, except here
        there is no whole-video z_score to fall back to either (the whole
        point of this class is that one doesn't exist yet), so the floor
        itself is the fallback.
        """
        if len(self.history) < self.s3.adaptive_threshold_min_samples:
            floor = self.s3.adaptive_min_floor if self.s3.similarity == "cosine" else float("-inf")
            return floor, "online_cold_start"
        sims = np.array(self.history)
        threshold, _, _, stat_label = compute_adaptive_threshold(sims, sims, self.s3.similarity, self.s3)
        return threshold, f"online_{stat_label}"

    def observe(self, frame_sims: np.ndarray) -> None:
        self.history.extend(frame_sims.tolist())


class ACIOnlineThreshold:
    """Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021) --
    single-scalar gradient-step threshold update. The UPDATE MECHANISM
    (gradient step toward a target rate, size proportional to a step_size
    hyperparameter) is HIGH CONFIDENCE, taken directly from the paper's own
    formula given explicitly in research_notes/Precision verification
    online tracking/conformal_online_thresholding.md:

        alpha_{t+1} = alpha_t + step_size * (target_error_rate - err_t)

    Adapted here as a RUNNING PERCENTILE into the window's own similarity
    distribution (so the implied threshold is always well-defined
    regardless of the metric's scale -- cosine, RMD, ...). One deliberate
    SIGN ADAPTATION vs. the paper's own literal formula: classic conformal
    regression accepts points with a NONCONFORMITY score <= threshold
    (higher nonconformity = worse fit, so a HIGHER threshold is MORE
    permissive); this pipeline accepts candidates with a SIMILARITY score
    >= threshold (the opposite polarity -- higher similarity = better
    match, so a HIGHER threshold is STRICTER, not more permissive). Naively
    reusing the paper's own sign under this flipped polarity would push the
    threshold the WRONG direction whenever the accept rate drifts off
    target; observe() below applies (err_t - target_error_rate), the
    correctly-flipped version for our score polarity, not the paper's
    literal (target_error_rate - err_t). This flip, and the err_t proxy
    itself (there is no ground truth at inference time -- see observe()'s
    own docstring), are the parts with lower confidence than the core
    gradient-step mechanism.
    """

    def __init__(self, s3):
        from collections import deque

        self.s3 = s3
        self.history: deque = deque(maxlen=s3.adaptive_threshold_online_window)
        # Percentile (0-100) into the window's own distribution -- starts
        # at a permissive prior (top 2x the target error rate), same
        # "don't over-commit before there's evidence" spirit as
        # adaptive_min_floor's own cold-start role.
        self.percentile = 100.0 * (1.0 - min(0.5, 2.0 * s3.aci_target_error_rate))

    def threshold_for_next_frame(self) -> tuple[float, str]:
        if len(self.history) < self.s3.adaptive_threshold_min_samples:
            floor = self.s3.adaptive_min_floor if self.s3.similarity == "cosine" else float("-inf")
            return floor, "aci_cold_start"
        sims = np.array(self.history)
        threshold = float(np.percentile(sims, np.clip(self.percentile, 0.0, 100.0)))
        return threshold, "aci"

    def observe(self, frame_sims: np.ndarray, accepted_mask: np.ndarray) -> None:
        """accepted_mask: which of frame_sims were accepted at the
        threshold threshold_for_next_frame() just returned -- ACI's own
        update needs to know whether this frame's decision, under the
        target error rate, looks like it was "wrong" (see class docstring)."""
        if frame_sims.size > 0:
            err_t = 1.0 if float(accepted_mask.mean()) > self.s3.aci_target_error_rate else 0.0
            # Sign flipped vs. the paper's own (target - err_t): see class
            # docstring's "SIGN ADAPTATION" note -- our score polarity
            # (accept if similarity >= threshold) is the opposite of
            # classic conformal regression's (accept if nonconformity <=
            # threshold), so a too-permissive frame (err_t=1) must RAISE
            # our percentile/threshold, not lower it.
            self.percentile += 100.0 * self.s3.aci_step_size * (err_t - self.s3.aci_target_error_rate)
            self.percentile = float(np.clip(self.percentile, 0.0, 100.0))
        self.history.extend(frame_sims.tolist())


class SaffronInspiredOnlineFDR:
    """SAFFRON (Ramdas, Zrnic, Wainwright, Jordan, PMLR v80 / ICML 2018,
    arXiv:1802.09098 -- docs/1802.09098v2.pdf, read directly) -- see
    OnlineFDRConfig's own docstring (aero_eyes/config.py) for the mapping
    and its one project-specific adaptation (the p-value heuristic).

    FAITHFUL PORT of Section 2.3's algorithm for constant lambda:
        alpha_t = min{lam, (1-lam) * [
            W0 * gamma(t - C_{0+}(t)) +
            (target_fdr - W0) * gamma(t - tau_1 - C_{1+}(t)) +
            sum_{j>=2} target_fdr * gamma(t - tau_j - C_{j+}(t))
        ]},
    where tau_j is the (1-indexed) time of the j-th ACCEPTANCE (the
    paper's own "rejection" -- tau_0 := 0), C_{j+}(t) is the number of
    SAFFRON "candidates" (p-value <= lam) seen strictly between tau_j and
    t, and gamma_j = j^-gamma_exponent, normalized to sum to 1 over
    j=1,2,... via the Riemann zeta function. This single formula also
    reduces exactly to the paper's own separately-stated alpha_1 at t=1
    (only epoch 0 exists then), so no special-cased first step is needed.

    Tests each candidate INDIVIDUALLY (unlike every other online mechanism
    here, which computes one scalar threshold per keyframe) -- SAFFRON's
    own alpha_t is inherently a per-hypothesis significance level, not a
    per-keyframe threshold.
    """

    def __init__(self, cfg, p_value_window: int):
        from collections import deque

        from scipy.special import zeta

        self.cfg = cfg
        self.lam = cfg.lam
        self.W0 = cfg.target_fdr * cfg.initial_wealth_fraction
        self.target_fdr = cfg.target_fdr
        self.gamma_exponent = cfg.gamma_exponent
        self._gamma_normalizer = float(zeta(cfg.gamma_exponent, 1))
        self.history: deque = deque(maxlen=p_value_window)
        self.t = 0  # number of candidates tested so far
        self._cumulative_candidates = 0  # SAFFRON "candidates" (p-value <= lam) seen so far
        self._epoch_taus: list[int] = [0]  # tau_0=0, then tau_1, tau_2, ... appended on each acceptance
        self._epoch_candidate_snapshots: list[int] = [0]  # cumulative-candidate count AT each tau_j
        # Kept for introspection/back-compat with earlier callers that
        # inspected .wealth as a coarse "budget remaining" signal --
        # SAFFRON's real accounting is the multi-epoch sum above, not a
        # single scalar, but W0 - (net spend so far) is a reasonable proxy.
        self.wealth = self.W0

    def _gamma(self, j: int) -> float:
        if j < 1:
            return 0.0
        return (j ** (-self.gamma_exponent)) / self._gamma_normalizer

    def _p_value(self, sim: float) -> float:
        """Project-specific heuristic (NOT from the paper) -- see
        OnlineFDRConfig's own docstring's "ONE PROJECT-SPECIFIC ADAPTATION"."""
        if len(self.history) == 0:
            return 0.5  # cold start: no history yet, no evidence either way
        hist = np.array(self.history)
        return float((hist >= sim).mean())  # fraction of recent history AT LEAST this high

    def test(self, sim: float) -> tuple[bool, float]:
        """Returns (accept, alpha_t_used) for one candidate."""
        t = self.t + 1
        p_value = self._p_value(sim)

        total = 0.0
        for j, (tau_j, snapshot) in enumerate(zip(self._epoch_taus, self._epoch_candidate_snapshots)):
            c_j_plus = self._cumulative_candidates - snapshot
            exponent = t - tau_j - c_j_plus
            coeff = self.W0 if j == 0 else (self.target_fdr - self.W0) if j == 1 else self.target_fdr
            total += coeff * self._gamma(exponent)
        alpha_t = min(self.lam, (1.0 - self.lam) * total)

        accepted = bool(p_value <= alpha_t)
        is_candidate = bool(p_value <= self.lam)

        if accepted:
            self._epoch_taus.append(t)
            self._epoch_candidate_snapshots.append(self._cumulative_candidates)
            self.wealth = max(0.0, self.wealth - alpha_t + self.target_fdr)
        else:
            self.wealth = max(0.0, self.wealth - alpha_t)
        if is_candidate:
            self._cumulative_candidates += 1

        self.t = t
        self.history.append(sim)
        return accepted, alpha_t


class CorruptionCompensatedThreshold:
    """F-ROCP (arXiv:2605.20515, "Online Conformal Prediction with
    Corrupted Feedback", Wang/Zecchin/Simeone -- docs/2605.20515v1.pdf,
    read directly) wrapped around ACIOnlineThreshold's own percentile
    update -- see CorruptionCompensatedThresholdConfig's own docstring
    (aero_eyes/config.py) for the full mapping and HONESTY NOTE on scope
    (only F-ROCP's filtering is ported, not AC-ROCP's active compensation).
    """

    def __init__(self, s3):
        self.aci = ACIOnlineThreshold(s3)

    def threshold_for_next_frame(self) -> tuple[float, str]:
        threshold, label = self.aci.threshold_for_next_frame()
        if label == "aci_cold_start":
            return threshold, "corruption_compensated_cold_start"
        if self.aci.percentile <= 0.0:
            return threshold, "corruption_compensated_permissive_boundary"
        if self.aci.percentile >= 100.0:
            return threshold, "corruption_compensated_strict_boundary"
        return threshold, "corruption_compensated_in_range"

    def observe(self, frame_sims: np.ndarray, was_probe: bool, accepted_mask: np.ndarray) -> None:
        del was_probe  # no active-training probes in this F-ROCP-only port -- see class/config docstring
        if frame_sims.size == 0:
            self.aci.observe(frame_sims, accepted_mask)
            return
        if self.aci.percentile <= 0.0:
            # Boundary certainty (F-ROCP's own core idea, Sec. IV-A): at
            # the maximally permissive extreme, "this frame was too
            # permissive" is a mathematical fact (the accept rate is
            # trivially ~1), not something that needs measuring from this
            # frame's own possibly tiny/noisy sample -- force it.
            self.aci.observe(frame_sims, np.ones_like(accepted_mask))
        elif self.aci.percentile >= 100.0:
            self.aci.observe(frame_sims, np.zeros_like(accepted_mask))
        else:
            self.aci.observe(frame_sims, accepted_mask)


def run_dynamic_prototype_rounds(
    sample_id: str,
    all_feats: np.ndarray,
    all_sims: np.ndarray,
    prototype: np.ndarray,
    per_ref_features: list,
    use_multi_ref: bool,
    multi_ref_pooling: str,
    similarity_metric: str,
    dp,
    on_round=None,
    all_frame_idxs: list[int] | None = None,
    background: tuple[np.ndarray, np.ndarray] | None = None,
    agreement_epsilon: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list]:
    """stage3.dynamic_prototype's iterative refinement loop (opt-in, no-op
    when dp.enabled is False): a fixed high-confidence cutoff only ever
    fires for "easy" targets whose scores are already high; a "hard"
    target's scores may never clear a fixed bar, so the mechanism silently
    never activates for it. Using a percentile of THIS sample's own score
    distribution instead (with an absolute floor so a uniformly-low-scoring
    sample doesn't update from pure noise) makes it fire consistently, then
    blends the resulting high-confidence candidates' mean feature into the
    prototype and re-scores -- repeated for `dp.rounds` passes so the
    prototype drifts toward this specific video's own appearance of the
    target.

    Factored out of run_stage3 so a diagnostic script (see
    scripts/check_dynamic_prototype_purity.py) can replay the exact same
    selection logic against candidates.json + ground truth WITHOUT
    duplicating it -- self-training loops like this can drift toward a
    confuser if a round's "high-confidence" picks are actually wrong, and
    that diagnostic checks the picks' GT IoU per round to catch it.

    on_round: optional callback(round_idx, high_conf_mask, threshold),
    invoked once per round that actually ran (rounds skipped by the
    min_support/require_diverse_picks early-break are NOT reported) --
    called BEFORE that round's prototype update, i.e. high_conf_mask
    indexes all_feats at the state used to SELECT that round's candidates.
    Return value ignored.

    all_frame_idxs: parallel to all_feats/all_sims (all_frame_idxs[i] is
    candidate i's frame index) -- required when dp.require_diverse_picks is
    True (see that field's own docstring); ignored otherwise.

    background: (mu_background, precision_background) from
    _fit_rmd_background -- REQUIRED when similarity_metric == "rmd" (RMD's
    background stats must stay fixed across dynamic_prototype's own
    re-scoring rounds, computed once from the pre-dynamic-prototype
    all_feats); ignored for every other metric.
    """
    if not dp.enabled:
        return prototype, all_sims, per_ref_features

    for round_idx in range(dp.rounds):
        adaptive_high_thresh = max(dp.high_conf_abs_floor, float(np.percentile(all_sims, dp.high_conf_percentile)))
        high_conf_mask = all_sims >= adaptive_high_thresh
        if int(high_conf_mask.sum()) < dp.min_support:
            break

        if dp.require_diverse_picks:
            if all_frame_idxs is None:
                raise ValueError(
                    "stage3.dynamic_prototype.require_diverse_picks=true needs all_frame_idxs "
                    "passed to run_dynamic_prototype_rounds."
                )
            picked_frames = [all_frame_idxs[i] for i in np.where(high_conf_mask)[0]]
            frame_span = max(picked_frames) - min(picked_frames)
            if frame_span < dp.min_frame_span:
                log.info(
                    "[Stage3] %s: dynamic prototype round %d/%d skipped -- %d high-confidence "
                    "candidates span only %d frame(s) (need >= %d), too narrow/clustered to "
                    "trust as representative of the target's full appearance",
                    sample_id, round_idx + 1, dp.rounds, int(high_conf_mask.sum()),
                    frame_span, dp.min_frame_span,
                )
                break

        dynamic_feat = all_feats[high_conf_mask].mean(axis=0)
        dynamic_feat = dynamic_feat / (np.linalg.norm(dynamic_feat) + 1e-8)

        log.info(
            "[Stage3] %s: dynamic prototype update round %d/%d -- adaptive threshold=%.3f "
            "(percentile=%.0f), %d candidates, alpha=%.2f",
            sample_id, round_idx + 1, dp.rounds, adaptive_high_thresh,
            dp.high_conf_percentile, int(high_conf_mask.sum()), dp.alpha,
        )

        if on_round is not None:
            on_round(round_idx, high_conf_mask, adaptive_high_thresh)

        if use_multi_ref:
            per_ref_features.append(dynamic_feat)
            sims_per_ref = [
                _score_against_ref(all_feats, ref_feat, similarity_metric, background=background)
                for ref_feat in per_ref_features
            ]
            all_sims = _pool_sims(sims_per_ref, multi_ref_pooling, per_ref_features, agreement_epsilon)
        else:
            prototype = (1 - dp.alpha) * prototype + dp.alpha * dynamic_feat
            prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
            all_sims = _score_against_ref(all_feats, prototype, similarity_metric, background=background)

    return prototype, all_sims, per_ref_features


def apply_identity_chain_filter(
    all_feats: np.ndarray,
    all_frame_idxs: list[int],
    all_dets: list[Detection],
    all_sims: np.ndarray,
    keep_mask: np.ndarray,
    cfg,
) -> tuple[np.ndarray, int, int]:
    """KeepTrack-style (arXiv:2103.16556) multi-candidate identity tracking
    across keyframes -- see IdentityChainFilterConfig's own docstring
    (aero_eyes/config.py) for the full rationale.

    Groups already-threshold-passing candidates by keyframe (truncated to
    cfg.top_k_per_keyframe per keyframe by similarity), then walks
    keyframes in temporal order building "identity chains": at each step,
    solves a bipartite match (scipy.optimize.linear_sum_assignment, exact
    Hungarian solver) between the CURRENT active chains' tail candidates
    and this keyframe's own candidates, cost = 1 - cosine_similarity (+
    cfg.spatial_weight * normalized center-distance). A chain that finds
    no acceptable match (cost > 1.0, i.e. the two features are more
    dissimilar than similar) ends; an unmatched candidate starts a new
    chain of length 1. After the whole video is processed, a candidate is
    accepted (kept) iff the chain it ultimately belonged to reached
    cfg.min_chain_length at any point.

    NOT causal/online (like stage3.dynamic_prototype, not like
    margin_verification/cluster_secondary_filter's own causal windows): a
    chain's final length -- and therefore whether an EARLY keyframe in it
    gets accepted -- can only be known after seeing LATER keyframes too.
    This is a deliberate simplification for a first implementation (see
    docs/GECO2_precision_improvements_plan.md Phase 3 item 6's own "needs
    more design" framing) -- a genuinely causal variant would need to
    retroactively accept a chain's past members the MOMENT it first
    reaches min_chain_length, never looking further ahead than that.

    Returns (new_keep_mask, n_chains_total, n_chains_kept).
    """
    from collections import defaultdict

    from scipy.optimize import linear_sum_assignment

    idx_by_frame: dict[int, list[int]] = defaultdict(list)
    for i in range(len(all_sims)):
        if keep_mask[i]:
            idx_by_frame[all_frame_idxs[i]].append(i)

    for fi in idx_by_frame:
        idxs = idx_by_frame[fi]
        idxs.sort(key=lambda i: all_sims[i], reverse=True)
        idx_by_frame[fi] = idxs[: cfg.top_k_per_keyframe]

    def _center(i: int) -> np.ndarray:
        b = all_dets[i].box
        return np.array([(b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0])

    # Normalizes the spatial term onto roughly the same [0, ~2] scale as
    # the cosine-distance appearance term, using this video's own median
    # candidate box diagonal as the reference scale.
    all_diagonals = [
        ((all_dets[i].box.x2 - all_dets[i].box.x1) ** 2 + (all_dets[i].box.y2 - all_dets[i].box.y1) ** 2) ** 0.5
        for idxs in idx_by_frame.values() for i in idxs
    ]
    scene_scale = max(float(np.median(all_diagonals)), 1e-6) if all_diagonals else 1.0

    active_chains: list[dict] = []
    finished_chains: list[dict] = []

    for fi in sorted(idx_by_frame):
        idxs = idx_by_frame[fi]
        if not active_chains:
            active_chains = [{"members": [i]} for i in idxs]
            continue
        if not idxs:
            finished_chains.extend(active_chains)
            active_chains = []
            continue

        cost = np.zeros((len(active_chains), len(idxs)))
        for ci, chain in enumerate(active_chains):
            tail_idx = chain["members"][-1]
            tail_feat = all_feats[tail_idx]
            tail_center = _center(tail_idx)
            for cj, cand_i in enumerate(idxs):
                appearance_cost = 1.0 - float(tail_feat @ all_feats[cand_i])
                spatial_cost = (
                    float(np.linalg.norm(tail_center - _center(cand_i))) / scene_scale
                    if cfg.spatial_weight > 0 else 0.0
                )
                cost[ci, cj] = appearance_cost + cfg.spatial_weight * spatial_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_chains, matched_cands = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= cfg.max_match_cost:
                active_chains[r]["members"].append(idxs[c])
                matched_chains.add(r)
                matched_cands.add(c)

        new_active = [chain for ci, chain in enumerate(active_chains) if ci in matched_chains]
        finished_chains.extend(chain for ci, chain in enumerate(active_chains) if ci not in matched_chains)
        new_active.extend({"members": [cand_i]} for cj, cand_i in enumerate(idxs) if cj not in matched_cands)
        active_chains = new_active

    finished_chains.extend(active_chains)

    new_keep_mask = np.zeros(len(keep_mask), dtype=bool)
    n_chains_kept = 0
    for chain in finished_chains:
        if len(chain["members"]) >= cfg.min_chain_length:
            n_chains_kept += 1
            for i in chain["members"]:
                new_keep_mask[i] = True

    return new_keep_mask, len(finished_chains), n_chains_kept


class IsolatedKeyframeGate:
    """Causal (streaming) form of IsolatedDetectionFilterConfig: delayed
    decision. Feed detection-bearing keyframes in increasing frame order via
    push(); each call returns the (frame, keep) decisions that just became
    final. A keyframe f is decided when the NEXT detection g arrives
    (near_next = g - f <= max_gap) or when advance(t) reports t - f > max_gap
    with no detection since (call it for every processed keyframe, detection
    or not, so a lone keyframe is released after max_gap frames instead of
    waiting for the next hit), or at flush() (end of stream). Decisions come
    out in frame order; latency is at most max_gap frames.
    """

    def __init__(self, keyframe_interval: int, cfg):
        self.max_gap = cfg.max_gap_intervals * keyframe_interval
        self.keep_conf = cfg.keep_conf_threshold
        self._last: int | None = None          # last pushed detection frame
        self._pending: tuple[int, float, bool] | None = None  # (frame, score, near_prev)

    def _decide(self, near_next: bool) -> tuple[int, bool]:
        fi, score, near_prev = self._pending
        self._pending = None
        keep = near_prev or near_next or (self.keep_conf is not None and score >= self.keep_conf)
        return fi, keep

    def push(self, frame: int, score: float) -> list[tuple[int, bool]]:
        out = []
        near_prev = self._last is not None and frame - self._last <= self.max_gap
        if self._pending is not None:
            out.append(self._decide(near_next=near_prev))
        self._pending = (frame, score, near_prev)
        self._last = frame
        return out

    def advance(self, current_frame: int) -> list[tuple[int, bool]]:
        if self._pending is not None and current_frame - self._pending[0] > self.max_gap:
            return [self._decide(near_next=False)]
        return []

    def flush(self) -> list[tuple[int, bool]]:
        return [self._decide(near_next=False)] if self._pending is not None else []


def find_isolated_keyframes(
    frame_scores: dict[int, float],
    keyframe_interval: int,
    cfg,
) -> set[int]:
    """Frames (keys of frame_scores) to drop under IsolatedDetectionFilterConfig.

    frame_scores maps each detection-bearing keyframe to its best score. A
    keyframe is supported iff another one lies within
    cfg.max_gap_intervals * keyframe_interval frames; an unsupported
    keyframe is still kept if its score >= cfg.keep_conf_threshold. Support
    is judged against the ORIGINAL set (not iteratively), so two mutually
    supporting keyframes never knock each other out.
    """
    max_gap = cfg.max_gap_intervals * keyframe_interval
    frames = sorted(frame_scores)
    if getattr(cfg, "mode", "offline") == "online":
        gate = IsolatedKeyframeGate(keyframe_interval, cfg)
        decisions = []
        for fi in frames:
            decisions.extend(gate.push(fi, frame_scores[fi]))
        decisions.extend(gate.flush())
        return {fi for fi, keep in decisions if not keep}
    isolated: set[int] = set()
    for k, fi in enumerate(frames):
        near_prev = k > 0 and fi - frames[k - 1] <= max_gap
        near_next = k + 1 < len(frames) and frames[k + 1] - fi <= max_gap
        if near_prev or near_next:
            continue
        if cfg.keep_conf_threshold is not None and frame_scores[fi] >= cfg.keep_conf_threshold:
            continue
        isolated.add(fi)
    return isolated


def run_stage3(cfg, sample_id: str) -> Path:
    """Run Stage 3 for the given sample. Returns path to detections.json."""
    from aero_eyes.stages.stage2 import read_candidates_with_features
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.geometry import nms
    from aero_eyes.utils.io import read_prototype, write_detections, write_prototype
    from aero_eyes.utils.video import read_frame, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage3] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    br_cfg = cfg.box_refine
    box_refine_segmenter = None
    geco2_refine_detector = None
    geco2_refine_prototype = None
    if br_cfg.enabled and (br_cfg.apply_in_stage3 or br_cfg.apply_before_stage3_filtering):
        if br_cfg.method in ("sam", "sam_dense"):
            from aero_eyes.models.segmentation import MobileSAMSegmenter
            box_refine_segmenter = MobileSAMSegmenter(weights_path=cfg.stage1.segmentation.weights)
        elif br_cfg.method == "fastsam_dense":
            from aero_eyes.models.segmentation import FastSAMSegmenter
            fs_cfg = cfg.stage2.fastsam_s
            box_refine_segmenter = FastSAMSegmenter(
                weights=fs_cfg.weights, conf=fs_cfg.conf, iou=fs_cfg.iou, imgsz=fs_cfg.imgsz,
            )
        elif br_cfg.method == "sam2_dense":
            from aero_eyes.models.geco2_detector import load_geco2_detector_and_prototype
            geco2_refine_detector, geco2_refine_prototype = load_geco2_detector_and_prototype(cfg, work_dir)
            if geco2_refine_detector is None:
                log.warning(
                    "[Stage3] %s: box_refine.method=sam2_dense but no %s found -- "
                    "refinement disabled this run (boxes left unchanged).",
                    sample_id, cfg.stage123_geco2.prototype_cache_name,
                )
        elif br_cfg.method == "sam2_native":
            from aero_eyes.models.segmentation import SAM2Segmenter
            box_refine_segmenter = SAM2Segmenter(cfg.stage123_geco2.repo_path)

    # ---- Load prototype ----
    proto_path = work_dir / cfg.stage1.prototype.cache_name
    if not proto_path.exists():
        raise FileNotFoundError(
            f"prototype.npz not found at {proto_path}. Run Stage 1 first."
        )
    prototype, meta, per_ref_features = read_prototype(proto_path)

    # ---- Load candidates ----
    cand_path = work_dir / "candidates.json"
    if not cand_path.exists():
        raise FileNotFoundError(
            f"candidates.json not found at {cand_path}. Run Stage 2 first."
        )
    candidates, feat_matrix = read_candidates_with_features(cand_path)

    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    video_path = video_files[0] if video_files else None

    s3 = cfg.stage3
    if s3.recompute_candidate_features and video_path is not None:
        # stage3.recompute_candidate_features: re-extract features for the
        # EXISTING candidate boxes with the CURRENTLY configured
        # feature_extractor, instead of trusting whatever candidates.feats.npz
        # already holds -- see the field's own docstring for why this exists
        # (Stage 2's cache check can't tell its cached features were built
        # with a now-stale extractor). Box geometry itself is untouched, only
        # each Detection's _feature gets overwritten; no SAHI/proposal-model
        # re-run needed.
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.stages.stage2 import _write_candidates_with_features

        recompute_extractor = build_feature_extractor(cfg)

        # box_refine.apply_before_stage3_filtering: tighten EVERY candidate
        # box (not just the per-keyframe winner apply_in_stage3 refines)
        # BEFORE the crop below is taken -- so the re-extracted embedding
        # reflects the tightened box, not the original loose one. See that
        # field's own docstring (aero_eyes/config.py) for the full
        # rationale. No "selected"/threshold-passing population exists yet
        # at this point in the pipeline for adaptive_context_margin's own
        # relative_to_sample_median sizing to compare against -- approximate
        # it from every RAW candidate box's own size instead (still a
        # reasonable "typical box size in this video" proxy).
        apply_prefilter_refine = br_cfg.enabled and br_cfg.apply_before_stage3_filtering
        n_boxes_refined = 0
        if apply_prefilter_refine:
            from aero_eyes.utils.box_refine import apply_iou_gate, refine_box, refine_boxes_dense

            all_cand_boxes = [d.box for dets in candidates.values() for d in dets]
            prefilter_reference_size = None
            if all_cand_boxes:
                prefilter_reference_size = float(np.median([
                    ((b.x2 - b.x1) * (b.y2 - b.y1)) ** 0.5 for b in all_cand_boxes
                ]))

        n_recomputed = 0
        for frame_idx, cand_dets in candidates.items():
            if not cand_dets:
                continue
            try:
                frame_bgr = read_frame(video_path, frame_idx)
            except Exception:
                continue

            if apply_prefilter_refine:
                boxes = [d.box for d in cand_dets]
                if br_cfg.method in ("sam_dense", "fastsam_dense", "sam2_native"):
                    refined_boxes = refine_boxes_dense(
                        box_refine_segmenter, frame_bgr, boxes,
                        min_iou_with_original=br_cfg.min_iou_with_original,
                        context_margin=br_cfg.context_margin,
                        adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                        sample_reference_size=prefilter_reference_size,
                        use_center_point=br_cfg.use_center_point_prompt,
                    )
                elif br_cfg.method == "sam2_dense":
                    if geco2_refine_detector is not None:
                        refined_boxes = geco2_refine_detector.sam2_refine_boxes(
                            frame_bgr, geco2_refine_prototype, boxes,
                            context_margin=br_cfg.context_margin,
                            adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                            sample_reference_size=prefilter_reference_size,
                            use_center_point=br_cfg.use_center_point_prompt,
                            select_best_mask=br_cfg.sam2_dense_select_best_mask,
                        )
                        refined_boxes = apply_iou_gate(refined_boxes, boxes, br_cfg.min_iou_with_original)
                    else:
                        refined_boxes = boxes
                else:
                    refined_boxes = [
                        refine_box(
                            br_cfg.method, frame_bgr, b, br_cfg.context_margin,
                            segmenter=box_refine_segmenter, min_iou_with_original=br_cfg.min_iou_with_original,
                            adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                            sample_reference_size=prefilter_reference_size,
                            use_center_point=br_cfg.use_center_point_prompt,
                        )
                        for b in boxes
                    ]
                for det, rb in zip(cand_dets, refined_boxes):
                    det.box = rb
                n_boxes_refined += len(cand_dets)

            feats = recompute_extractor.extract_crops(
                frame_bgr, [d.box for d in cand_dets],
                pad_ratio=cfg.stage2.candidate.feature_crop_pad,
                batch_size=cfg.runtime.batch_size,
            )
            for det, feat in zip(cand_dets, feats):
                det._feature = feat
            n_recomputed += len(cand_dets)
        _write_candidates_with_features(candidates, cand_path)
        if apply_prefilter_refine:
            log.info(
                "[Stage3] %s: box_refine.apply_before_stage3_filtering -- refined %d candidate "
                "box(es) (method=%s) before re-extracting their features.",
                sample_id, n_boxes_refined, br_cfg.method,
            )
        log.info(
            "[Stage3] %s: recompute_candidate_features -- re-extracted %d candidate "
            "feature(s) across %d keyframe(s), rewrote %s",
            sample_id, n_recomputed, len(candidates), cand_path,
        )
        # Re-read rather than hand-assemble feat_matrix here -- keeps this
        # path exercising the exact same load code every other run takes.
        candidates, feat_matrix = read_candidates_with_features(cand_path)
    elif br_cfg.enabled and br_cfg.apply_before_stage3_filtering:
        log.info(
            "[Stage3] %s: box_refine.apply_before_stage3_filtering=true but "
            "stage3.recompute_candidate_features=false (or no video found) -- no-op. "
            "Refining box geometry without re-extracting its feature would leave "
            "candidates.json's cached embedding mismatched with its own box.",
            sample_id,
        )

    if feat_matrix is None or feat_matrix.shape[0] == 0:
        log.warning("[Stage3] No candidate features found — writing empty detections.")
        # Still record every keyframe Stage2 scanned (with an empty box
        # list) rather than an empty dict -- see the frame_groups fix below
        # for why stage4.py needs this to tell "keyframe, zero detections"
        # apart from "not a keyframe at all".
        write_detections({fi: [] for fi in candidates}, det_path)
        return det_path

    threshold = s3.match_threshold
    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )
    multi_ref_pooling = cfg.accuracy.cheap_boosters.multi_ref_pooling
    agreement_epsilon = cfg.accuracy.cheap_boosters.agreement_weighted_epsilon

    # ---- Match: global top-K or per-keyframe threshold ----
    detections: dict[int, list[Detection]] = {}
    viz_dir = work_dir / "viz" / "stage3"

    # Build flat list of (frame_idx, det, feat) for all candidates
    all_entries: list[tuple[int, Detection, np.ndarray]] = []
    n_dropped_min_area = 0
    for frame_idx, cand_dets in candidates.items():
        for det in cand_dets:
            # stage3.min_box_area_enabled: reject a degenerate, near-zero-
            # area candidate box BEFORE it can occupy one of this stage's
            # own topk_per_keyframe slots (applied after NMS/topk below) --
            # same AREA-not-min-side-length rationale as
            # stage123_geco2.min_box_area_enabled (see that field's own
            # docstring: this project's own GT survey found a real object's
            # thinnest side can legitimately be ~2px at the frame edge, but
            # no real GT box has area <= 16px^2). Reads candidates.json
            # already on disk -- no candidate-generation stage needs
            # rerunning to retune this threshold.
            #
            # Deliberately does NOT replace stage123_geco2's own
            # min_box_area_enabled (geco2_detector.py): that one runs
            # BEFORE stage123_geco2.cosine_rescore.candidate_topk_per_keyframe
            # caps the raw candidate pool -- a degenerate box surviving
            # that cap crowds out a real candidate PERMANENTLY (it never
            # reaches candidates.json at all), which this later filter
            # cannot recover. Running both is the safe choice; this one
            # alone only protects THIS stage's own topk_per_keyframe cap.
            if s3.min_box_area_enabled and det.box.area() < s3.min_box_area:
                n_dropped_min_area += 1
                continue
            feat = getattr(det, "_feature", None)
            if feat is not None:
                all_entries.append((frame_idx, det, feat))
    if n_dropped_min_area > 0:
        log.info(
            "[Stage3] %s: min_box_area=%d dropped %d degenerate candidate(s) before matching",
            sample_id, s3.min_box_area, n_dropped_min_area,
        )

    if not all_entries:
        write_detections({fi: [] for fi in candidates}, det_path)
        log.warning("[Stage3] %s: no candidate features found", sample_id)
        return det_path

    all_frame_idxs = [e[0] for e in all_entries]
    all_dets = [e[1] for e in all_entries]
    all_feats = np.stack([e[2] for e in all_entries], axis=0)  # [N, D]

    # ---- PCA whitening (stage3.whitening, opt-in) ----
    # Transforms candidates, the fused prototype and every per-ref vector into
    # the same whitened space, so everything below (scoring, rmd,
    # dynamic_prototype, cluster verification) is unchanged apart from dim.
    if s3.whitening.enabled:
        all_feats, prototype, per_ref_features = apply_whitening(
            s3.whitening, all_feats, prototype, per_ref_features,
        )
        log.info(
            "[Stage3] %s: whitening applied (%s) -> feature dim %d",
            sample_id, s3.whitening.weights_path, all_feats.shape[1],
        )

    # RMD (Relative Mahalanobis Distance, opt-in via s3.similarity="rmd")
    # background stats -- fit ONCE from this video's own full candidate
    # pool (mostly background/FP by construction) before any per-ref
    # scoring, so every consumer (main scoring, dynamic_prototype's own
    # re-scoring rounds, the cluster-mode fallback) whitens against the
    # SAME distribution. None for every other metric (ignored there).
    #
    # Also fit (same call, same background) whenever either cluster
    # verification path below needs a precision_matrix for
    # cluster_verification.pairwise_metric="mahalanobis" -- one shared fit
    # regardless of which consumer(s) actually need it, same as the
    # s3.similarity="rmd" case.
    needs_rmd_background = (
        s3.similarity == "rmd"
        or (s3.verification_method == "cluster" and s3.cluster_verification.pairwise_metric == "mahalanobis")
        or (
            s3.cluster_secondary_filter.enabled
            and s3.cluster_secondary_filter.cluster_verification.pairwise_metric == "mahalanobis"
        )
    )
    background = _fit_rmd_background(all_feats) if needs_rmd_background else None
    precision_matrix = background[1] if background is not None else None

    # Compute similarity for every candidate at once (higher = more similar,
    # regardless of metric -- see _score_against_ref).
    if use_multi_ref:
        sims_per_ref = [
            _score_against_ref(all_feats, ref_feat, s3.similarity, background=background)
            for ref_feat in per_ref_features
        ]
        all_sims = _pool_sims(sims_per_ref, multi_ref_pooling, per_ref_features, agreement_epsilon)
    else:
        all_sims = _score_against_ref(all_feats, prototype, s3.similarity, background=background)  # [N]

    # ---- Patch-token re-scoring (stage3.patch_matching, opt-in) ----
    pm = s3.patch_matching
    if pm.enabled:
        if s3.similarity != "cosine":
            raise ValueError("stage3.patch_matching is only implemented for stage3.similarity='cosine'.")
        if video_path is None:
            raise FileNotFoundError(
                f"stage3.patch_matching needs the sample's video to crop candidates but none was "
                f"found for {sample_id!r} matching {cfg.data.video_glob!r}."
            )
        if s3.dynamic_prototype.enabled:
            log.warning(
                "[Stage3] %s: stage3.dynamic_prototype re-scores from CLS features and will "
                "discard patch_matching scores once a round fires.", sample_id,
            )
        from aero_eyes.models.patch_match import score_candidates

        patch_matrix = score_candidates(cfg, sample_id, video_path, all_frame_idxs, all_dets)
        n_refs = patch_matrix.shape[1]
        patch_pooling = multi_ref_pooling if use_multi_ref else "mean"
        if patch_pooling == "agreement_weighted" and n_refs != len(per_ref_features):
            patch_pooling = "mean"
        patch_sims = _pool_sims(
            [patch_matrix[:, r] for r in range(n_refs)], patch_pooling, per_ref_features, agreement_epsilon,
        )
        all_sims = pm.cls_weight * all_sims + (1.0 - pm.cls_weight) * patch_sims

    # Snapshot BEFORE dynamic_prototype runs -- the similarity distribution
    # against only the original reference photo(s), untouched by whatever
    # dynamic_prototype appends/blends later. Used by
    # adaptive_threshold_anchor_to_original_refs below to keep the
    # THRESHOLD stable even when dynamic_prototype's own additions skew the
    # (still used for ACCEPTANCE) all_sims distribution -- see that config
    # field's own docstring. Identical to all_sims when dynamic_prototype is
    # disabled, so this is a no-op then.
    all_sims_original_refs = all_sims.copy()

    # ---- Dynamic prototype update (stage3.dynamic_prototype, opt-in) ----
    # Also a whole-video, 2-pass batch mechanism -- its own rounds need the
    # ENTIRE video's candidates just as much as a batch threshold would, so
    # it's skipped entirely (not just its role in the threshold below) when
    # adaptive_threshold_online is active, to keep all_sims itself causal.
    # verification_method="cluster" is causal for the same reason (each
    # keyframe's decision only ever reads that keyframe's own candidates +
    # per_ref_features as currently populated) -- skipped there too.
    _dp_incompatible_reason = None
    if s3.verification_method == "cluster":
        _dp_incompatible_reason = "verification_method='cluster'"
    elif s3.adaptive_threshold and s3.adaptive_threshold_online:
        _dp_incompatible_reason = "adaptive_threshold_online=true"
    if _dp_incompatible_reason and s3.dynamic_prototype.enabled:
        log.warning(
            "[Stage3] %s: %s is incompatible with stage3.dynamic_prototype "
            "(also a whole-video batch mechanism) -- skipping dynamic_prototype "
            "rounds entirely this run.", sample_id, _dp_incompatible_reason,
        )
    else:
        prototype, all_sims, per_ref_features = run_dynamic_prototype_rounds(
            sample_id, all_feats, all_sims, prototype, per_ref_features,
            use_multi_ref, multi_ref_pooling, s3.similarity, s3.dynamic_prototype,
            all_frame_idxs=all_frame_idxs, background=background,
            agreement_epsilon=agreement_epsilon,
        )

    # Persist the dynamic_prototype-adapted state SEPARATELY from
    # prototype.npz (Stage 1's own, never touched here) -- lets
    # stage4.backward_tracking.validate_against_boundary.cosine_arbitration
    # opt into scoring against this adapted state (original refs PLUS
    # whatever dynamic_prototype appended -- per_ref_features only ever
    # grows via .append, never loses the original 3) instead of only the
    # original references, via cosine_arbitration.use_adaptive_prototype.
    # Written whenever dynamic_prototype is enabled (harmless no-op
    # duplicate of prototype.npz on the rare run where 0 rounds actually
    # fired -- e.g. min_support never met).
    if s3.dynamic_prototype.enabled and s3.whitening.enabled:
        log.warning(
            "[Stage3] %s: stage3.whitening is on -- prototype_adapted.npz NOT written (its whitened "
            "vectors would not match Stage 4's CLS-space candidates); "
            "cosine_arbitration.use_adaptive_prototype will not see this run's adapted prototype.",
            sample_id,
        )
    elif s3.dynamic_prototype.enabled:
        write_prototype(prototype, meta, per_ref_features if use_multi_ref else None, work_dir / "prototype_adapted.npz")

    # CD-ViTO domain prompter (max_accuracy) -- only implemented for cosine;
    # already shown to hurt results (see docs/COLAB_KAGGLE_GUIDE.md), kept
    # off by default and not extended to l1/l2.
    if (cfg.accuracy.mode == "max_accuracy"
            and cfg.accuracy.max_accuracy.domain_prompter.enabled):
        if s3.similarity != "cosine":
            raise ValueError(
                "accuracy.max_accuracy.domain_prompter is only implemented for "
                "stage3.similarity='cosine'. Disable domain_prompter or switch back to cosine."
            )
        all_sims = _apply_domain_prompter(all_feats, prototype, all_sims, cfg)

    # Always log the raw similarity distribution — the ground-to-aerial domain
    # gap means a fixed match_threshold tuned on one dataset can silently pass
    # zero candidates on another; this makes that visible instead of a mute
    # "0 detection frames" result.
    log.info(
        "[Stage3] %s: candidate score stats (metric=%s, higher=more similar) — "
        "min=%.3f p50=%.3f mean=%.3f std=%.3f p95=%.3f max=%.3f (n=%d)",
        sample_id, s3.similarity, float(all_sims.min()), float(np.percentile(all_sims, 50)),
        float(all_sims.mean()), float(all_sims.std()),
        float(np.percentile(all_sims, 95)), float(all_sims.max()), len(all_sims),
    )

    # ---- Compute effective threshold + filter ----
    if s3.verification_method == "cluster":
        # DAVE (arXiv:2404.16622) module (ii)-style per-keyframe exemplar-
        # cluster verification, in place of a global scalar threshold -- see
        # ClusterVerificationConfig's own docstring (aero_eyes/config.py)
        # for the full rationale. Naturally causal: each keyframe's decision
        # below only ever reads that keyframe's own candidates (all_feats
        # sliced by frame_idx) + per_ref_features as currently populated --
        # no dependency on any other keyframe's similarity scores, unlike
        # every branch of compute_adaptive_threshold/OnlineAdaptiveThreshold.
        from collections import defaultdict as _defaultdict

        from aero_eyes.utils.cluster_verify import cluster_verify_candidates

        frame_to_indices: dict[int, list[int]] = _defaultdict(list)
        for i, fi in enumerate(all_frame_idxs):
            frame_to_indices[fi].append(i)

        ref_feats_for_cluster = (
            np.stack(per_ref_features, axis=0) if use_multi_ref else prototype[None, :]
        )

        # embedding_source="dave_verification": swap ONLY this affinity
        # matrix's embedding to DAVE's own verify-stage encoder -- see
        # ClusterVerificationConfig.embedding_source's own docstring
        # (aero_eyes/config.py). all_feats/per_ref_features/all_sims (used
        # everywhere else in this function -- matching/threshold scoring,
        # logging, etc.) stay in stage1.feature_extractor's own space,
        # completely unchanged; only cand_feats_frame/ref_feats_for_cluster
        # fed into cluster_verify_candidates below are re-embedded.
        dave_extractor = None
        if s3.cluster_verification.embedding_source == "dave_verification":
            if s3.cluster_verification.pairwise_metric == "mahalanobis":
                raise ValueError(
                    "stage3.cluster_verification.embedding_source='dave_verification' is not "
                    "compatible with pairwise_metric='mahalanobis' -- precision_matrix is fit "
                    "in stage1.feature_extractor's own embedding space (_fit_rmd_background on "
                    "all_feats), not DAVE's verify-stage embedding space. Use pairwise_metric="
                    "'cosine' (or 'l1') with embedding_source='dave_verification'."
                )
            if video_path is None:
                raise FileNotFoundError(
                    "stage3.cluster_verification.embedding_source='dave_verification' needs the "
                    f"sample's own video (to RoI-Align candidate boxes out of DAVE's backbone "
                    f"feature map) but no video file was found for {sample_id!r} matching "
                    f"{cfg.data.video_glob!r}."
                )
            import cv2

            from aero_eyes.models.dave_verification import build_dave_verification_extractor

            dave_extractor = build_dave_verification_extractor(cfg)

            refs_dir = Path(cfg.data.data_root) / sample_id / cfg.data.refs_subdir
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ref_paths = sorted(
                p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
                if p.suffix.lower() in exts
            )[: cfg.data.num_references]
            ref_imgs = [cv2.imread(str(p)) for p in ref_paths]
            ref_feats_for_cluster = dave_extractor.extract(ref_imgs)
            log.info(
                "[Stage3] %s: cluster_verification.embedding_source=dave_verification -- "
                "re-encoded %d reference image(s) with DAVE's own verify-stage embedding "
                "(dim=%d)", sample_id, len(ref_imgs), ref_feats_for_cluster.shape[-1],
            )

        def _fallback_keep_mask(cand_feats_frame: np.ndarray, ref_feats_frame: np.ndarray) -> np.ndarray:
            # Too few candidates this keyframe for clustering to find
            # meaningful structure -- fall back to a threshold RELATIVE to
            # this keyframe's own top similarity (fallback_relative_ratio),
            # never a hand-set absolute cosine number like s3.match_threshold:
            # on a video with a severe domain gap, EVERY candidate's raw
            # cosine similarity (even the genuine match) can sit well below
            # any plausible absolute cutoff -- confirmed in practice on this
            # project's own footage, where a whole video's max candidate
            # similarity (0.335) stayed under match_threshold's default
            # (0.55), silently zeroing out every fallback-path keyframe.
            if dave_extractor is not None:
                # cand_feats_frame/ref_feats_frame here are in DAVE's own
                # verify-stage embedding space (see embedding_source=
                # "dave_verification" above), not stage1.feature_extractor's
                # -- _score_against_ref's s3.similarity metrics (rmd/l1/l2/
                # etc.) are defined/calibrated for that OTHER space, so they
                # do not apply here. Both extractors' outputs are
                # L2-normalized (see every *FeatureExtractor.extract's own
                # contract), so plain cosine (dot product) is always valid,
                # regardless of s3.similarity.
                sims_frame = np.max(cand_feats_frame @ ref_feats_frame.T, axis=1)
                if sims_frame.size == 0:
                    return sims_frame.astype(bool)
                relative_floor = float(sims_frame.max()) * s3.cluster_verification.fallback_relative_ratio
                return sims_frame >= relative_floor
            sims_per_ref_frame = [
                _score_against_ref(cand_feats_frame, rf, s3.similarity, background=background)
                for rf in ref_feats_frame
            ]
            sims_frame = _pool_sims(sims_per_ref_frame, multi_ref_pooling, ref_feats_frame, agreement_epsilon)
            if sims_frame.size == 0:
                return sims_frame.astype(bool)
            relative_floor = float(sims_frame.max()) * s3.cluster_verification.fallback_relative_ratio
            if s3.similarity == "cosine":
                relative_floor = max(relative_floor, s3.adaptive_min_floor)
            return sims_frame >= relative_floor

        keep_mask = np.zeros(len(all_sims), dtype=bool)
        method_counts: dict[str, int] = _defaultdict(int)
        for fi in sorted(frame_to_indices):
            idxs = frame_to_indices[fi]
            if dave_extractor is not None:
                # Re-embed THIS keyframe's own candidates with DAVE's verify-
                # stage encoder -- one backbone forward pass over the frame,
                # shared across every box in it (see DaveVerificationExtractor.
                # extract_crops's own docstring for why this differs from every
                # other extractor's independent-crop approach).
                frame_bgr = read_frame(video_path, fi)
                cand_feats_frame = dave_extractor.extract_crops(
                    frame_bgr, [all_dets[i].box for i in idxs],
                    pad_ratio=cfg.stage2.candidate.feature_crop_pad,
                    batch_size=cfg.runtime.batch_size,
                )
            else:
                cand_feats_frame = all_feats[idxs]
            frame_keep, method_label = cluster_verify_candidates(
                cand_feats_frame, ref_feats_for_cluster, s3.cluster_verification,
                fallback_keep_mask_fn=_fallback_keep_mask, precision_matrix=precision_matrix,
            )
            method_counts[method_label] += 1
            for local_i, global_i in enumerate(idxs):
                keep_mask[global_i] = frame_keep[local_i]

        effective_threshold = None
        selected = [
            (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
            for i in range(len(all_sims)) if keep_mask[i]
        ]
        log.info(
            "[Stage3] %s: verification_method=cluster (cluster_method=%s) -> %d / %d "
            "candidates verified across %d keyframe(s) (method breakdown: %s)",
            sample_id, s3.cluster_verification.cluster_method, len(selected), len(all_sims),
            len(frame_to_indices), dict(method_counts),
        )
    elif s3.adaptive_threshold and s3.adaptive_threshold_online:
        # Real-time-deployment-compatible path: decide each keyframe's
        # candidates using ONLY strictly-earlier keyframes' own similarity
        # scores -- no single scalar threshold describes the whole video,
        # so effective_threshold is left as None (write_detections accepts
        # that) and only the LAST window's threshold is logged, for a
        # rough sense of where it ended up.
        # adaptive_threshold_online_method picks WHICH online mechanism
        # computes/applies that causal decision -- see each class's own
        # docstring (OnlineAdaptiveThreshold / ACIOnlineThreshold /
        # SaffronInspiredOnlineFDR / CorruptionCompensatedThreshold) for
        # exactly what it does. "aci", "saffron" and "corruption_compensated"
        # are all FAITHFUL ports of their cited papers' own algorithms; the
        # lower-confidence parts are each one's own project-specific
        # adaptation (documented in that class/config's own docstring), not
        # the ported algorithm itself.
        from collections import defaultdict as _defaultdict
        frame_to_indices: dict[int, list[int]] = _defaultdict(list)
        for i, fi in enumerate(all_frame_idxs):
            frame_to_indices[fi].append(i)

        keep_mask = np.zeros(len(all_sims), dtype=bool)
        last_threshold, last_stat_label = None, None
        method = s3.adaptive_threshold_online_method

        if method == "saffron":
            saffron = SaffronInspiredOnlineFDR(s3.online_fdr, s3.online_fdr.p_value_window)
            for fi in sorted(frame_to_indices):
                idxs = frame_to_indices[fi]
                for i in idxs:
                    accepted, alpha_t = saffron.test(float(all_sims[i]))
                    keep_mask[i] = accepted
                    last_threshold, last_stat_label = alpha_t, "saffron"
        else:
            if method == "aci":
                online = ACIOnlineThreshold(s3)
            elif method == "corruption_compensated":
                online = CorruptionCompensatedThreshold(s3)
            else:
                online = OnlineAdaptiveThreshold(s3)

            for fi in sorted(frame_to_indices):
                idxs = frame_to_indices[fi]
                last_threshold, last_stat_label = online.threshold_for_next_frame()
                frame_sims = all_sims[idxs]
                frame_keep = frame_sims >= last_threshold
                keep_mask[idxs] = frame_keep
                if method == "aci":
                    online.observe(frame_sims, frame_keep)
                elif method == "corruption_compensated":
                    online.observe(frame_sims, False, frame_keep)  # 2nd arg unused -- see CorruptionCompensatedThreshold.observe's own docstring
                else:
                    online.observe(frame_sims)

        effective_threshold = None
        selected = [
            (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
            for i in range(len(all_sims)) if keep_mask[i]
        ]
        log.info(
            "[Stage3] %s: adaptive_threshold_online (method=%s, window=%d, min_samples=%d, "
            "final threshold/alpha=%.4f, stat=%s) -> %d / %d candidates pass",
            sample_id, method, s3.adaptive_threshold_online_window, s3.adaptive_threshold_min_samples,
            last_threshold if last_threshold is not None else float("nan"), last_stat_label,
            len(selected), len(all_sims),
        )
    else:
        if s3.adaptive_threshold:
            effective_threshold, center, spread, stat_label = compute_adaptive_threshold(
                all_sims, all_sims_original_refs, s3.similarity, s3,
            )
            anchor_note = ", anchored to original refs" if s3.adaptive_threshold_anchor_to_original_refs else ""
            floor_note = f" (floor={s3.adaptive_min_floor:.3f})" if s3.similarity == "cosine" else ""
            if stat_label in ("mean/std", "median/MAD"):
                # z_score method (s3.adaptive_threshold_method == "z_score"):
                # effective_threshold IS literally center + adaptive_z_score*spread.
                log.info(
                    "[Stage3] %s: adaptive threshold (metric=%s, stat=%s%s) = %.3f + %.1f*%.3f = %.3f%s",
                    sample_id, s3.similarity, stat_label, anchor_note,
                    center, s3.adaptive_z_score, spread, effective_threshold, floor_note,
                )
            else:
                # otsu / gmm_bimodal / gmm_unimodal_fallback: adaptive_z_score
                # plays NO role in how effective_threshold was derived -- center/
                # spread here are just the distribution's own mean/std, reported
                # for reference only, not inputs to a formula that produced
                # effective_threshold (unlike the z_score branch above).
                log.info(
                    "[Stage3] %s: adaptive threshold (metric=%s, method=%s%s) = %.3f%s "
                    "(distribution mean=%.3f, std=%.3f -- adaptive_z_score not used by this method)",
                    sample_id, s3.similarity, stat_label, anchor_note,
                    effective_threshold, floor_note, center, spread,
                )
        else:
            effective_threshold = threshold

        keep_mask = all_sims >= effective_threshold
        selected = [
            (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
            for i in range(len(all_sims)) if keep_mask[i]
        ]
        log.info("[Stage3] %s: threshold=%.3f → %d / %d candidates pass",
                 sample_id, effective_threshold, len(selected), len(all_sims))

    # ---- Optional: hard-negative "negative prototype" SECONDARY filter ----
    # See NegativePrototypeFilterConfig's own docstring (config.py) for the
    # full rationale. Placed BEFORE cluster_secondary_filter/identity_chain_
    # filter below so its causal negative window gets the largest possible
    # reject population as early as possible (most rejection happens at the
    # primary threshold/cluster decision, not in later secondary filters).
    npf_cfg = s3.negative_prototype_filter
    if npf_cfg.enabled:
        from collections import defaultdict as _defaultdict
        from collections import deque as _deque

        ref_feats_base_np = (
            np.stack(per_ref_features, axis=0) if use_multi_ref else prototype[None, :]
        )
        idx_by_frame_np: dict[int, list[int]] = _defaultdict(list)
        for i in range(len(all_sims)):
            idx_by_frame_np[all_frame_idxs[i]].append(i)

        negative_window: _deque = _deque(maxlen=npf_cfg.window_size)
        n_before_np = int(keep_mask.sum())
        n_rejected_np = 0
        for fi in sorted(idx_by_frame_np):
            for i in idx_by_frame_np[fi]:
                if keep_mask[i] and len(negative_window) >= npf_cfg.min_window_for_check:
                    # Own similarity computed fresh from raw cosine, NOT
                    # all_sims[i] -- keeps both sides of this margin
                    # comparable regardless of stage3.similarity (e.g. rmd
                    # has a different scale entirely) -- see the config
                    # docstring's own note on this.
                    own_cosine = float(np.max(all_feats[i] @ ref_feats_base_np.T))
                    neg_stack = np.stack(negative_window, axis=0)
                    max_neg_cosine = float(np.max(all_feats[i] @ neg_stack.T))
                    if max_neg_cosine - own_cosine >= npf_cfg.tau_negative_margin:
                        keep_mask[i] = False
                        n_rejected_np += 1
                if not keep_mask[i]:
                    negative_window.append(all_feats[i])

        selected = [
            (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
            for i in range(len(all_sims)) if keep_mask[i]
        ]
        log.info(
            "[Stage3] %s: negative_prototype_filter (window_size=%d, tau_negative_margin=%.3f) "
            "rejected %d / %d threshold-passing candidate(s) -> %d remain (%d negative anchor(s) "
            "accumulated by the end)",
            sample_id, npf_cfg.window_size, npf_cfg.tau_negative_margin, n_rejected_np, n_before_np,
            len(selected), len(negative_window),
        )

    # ---- Optional: cluster-based SECONDARY precision filter ----
    # Only ever narrows an already-threshold-passing set (verification_method
    # must be "threshold", never "cluster" -- that path already IS the
    # primary decision). See ClusterSecondaryFilterConfig's own docstring
    # for the full empirical rationale.
    csf_cfg = s3.cluster_secondary_filter
    if csf_cfg.enabled and s3.verification_method == "threshold":
        from collections import defaultdict as _defaultdict
        from collections import deque as _deque

        from aero_eyes.utils.cluster_verify import cluster_verify_candidates
        from aero_eyes.utils.detection_confirm import DetectionConfirmer

        idx_by_frame: dict[int, list[int]] = _defaultdict(list)
        for i in range(len(all_sims)):
            if keep_mask[i]:
                idx_by_frame[all_frame_idxs[i]].append(i)

        ref_feats_base = (
            np.stack(per_ref_features, axis=0) if use_multi_ref else prototype[None, :]
        )

        # embedding_source="dave_verification": same swap as
        # stage3.verification_method="cluster" above (see that branch's own
        # comment) -- this filter's own separate dave_extractor/ref_feats_base
        # (a DIFFERENT instance, since this branch only ever runs when
        # verification_method="threshold", i.e. the cluster branch above
        # never executes in the same run). all_sims/all_feats (threshold's own
        # decision + the ranking used to pick each keyframe's window-admission
        # candidate below) stay in stage1.feature_extractor's space,
        # unchanged; only the accept/reject clustering itself is re-embedded.
        csf_dave_extractor = None
        if csf_cfg.cluster_verification.embedding_source == "dave_verification":
            if csf_cfg.cluster_verification.pairwise_metric == "mahalanobis":
                raise ValueError(
                    "stage3.cluster_secondary_filter.cluster_verification.embedding_source="
                    "'dave_verification' is not compatible with pairwise_metric='mahalanobis' "
                    "-- precision_matrix is fit in stage1.feature_extractor's own embedding "
                    "space, not DAVE's verify-stage embedding space. Use pairwise_metric="
                    "'cosine' (or 'l1') with embedding_source='dave_verification'."
                )
            if video_path is None:
                raise FileNotFoundError(
                    "stage3.cluster_secondary_filter.cluster_verification.embedding_source="
                    f"'dave_verification' needs the sample's own video but no video file was "
                    f"found for {sample_id!r} matching {cfg.data.video_glob!r}."
                )
            import cv2

            from aero_eyes.models.dave_verification import build_dave_verification_extractor

            csf_dave_extractor = build_dave_verification_extractor(cfg)

            refs_dir = Path(cfg.data.data_root) / sample_id / cfg.data.refs_subdir
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ref_paths = sorted(
                p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
                if p.suffix.lower() in exts
            )[: cfg.data.num_references]
            ref_imgs = [cv2.imread(str(p)) for p in ref_paths]
            ref_feats_base = csf_dave_extractor.extract(ref_imgs)
            log.info(
                "[Stage3] %s: cluster_secondary_filter.cluster_verification."
                "embedding_source=dave_verification -- re-encoded %d reference image(s) "
                "with DAVE's own verify-stage embedding (dim=%d)",
                sample_id, len(ref_imgs), ref_feats_base.shape[-1],
            )

        def _keep_everyone(cand_feats_frame: np.ndarray, ref_feats_frame: np.ndarray) -> np.ndarray:
            # Too few threshold-survivors this keyframe to add meaningful
            # cluster evidence either way -- trust the threshold's own
            # decision unchanged rather than guessing.
            return np.ones(cand_feats_frame.shape[0], dtype=bool)

        trusted_window: _deque = _deque(maxlen=csf_cfg.window_size)
        # Corroboration gate on WINDOW ADMISSION ONLY (does not affect the
        # keep_mask accept/reject decision below) -- see
        # ClusterSecondaryFilterConfig.window_admission_min_consecutive_hits'
        # own docstring for why: an unconditionally-admitted borderline FP
        # was the suspected reason this filter didn't meaningfully help in
        # real-footage testing. min_consecutive_hits<=1 makes this a no-op
        # (every offer confirms immediately), reproducing today's
        # unconditional-admission behavior unchanged.
        window_confirmer = DetectionConfirmer(
            csf_cfg.window_admission_min_consecutive_hits, csf_cfg.window_admission_iou_threshold,
        )
        n_before = sum(len(v) for v in idx_by_frame.values())
        n_rejected = 0
        n_window_admitted = 0
        for fi in sorted(idx_by_frame):
            idxs = idx_by_frame[fi]
            if csf_dave_extractor is not None:
                # Re-embed THIS keyframe's own threshold-survivors with
                # DAVE's verify-stage encoder -- same one-forward-pass-per-
                # frame approach as verification_method="cluster" above (see
                # DaveVerificationExtractor.extract_crops's own docstring).
                frame_bgr = read_frame(video_path, fi)
                cand_feats_frame = csf_dave_extractor.extract_crops(
                    frame_bgr, [all_dets[i].box for i in idxs],
                    pad_ratio=cfg.stage2.candidate.feature_crop_pad,
                    batch_size=cfg.runtime.batch_size,
                )
            else:
                cand_feats_frame = all_feats[idxs]
            ref_feats_for_frame = (
                np.concatenate([ref_feats_base, np.stack(trusted_window, axis=0)], axis=0)
                if trusted_window else ref_feats_base
            )
            frame_keep, _ = cluster_verify_candidates(
                cand_feats_frame, ref_feats_for_frame, csf_cfg.cluster_verification,
                fallback_keep_mask_fn=_keep_everyone, precision_matrix=precision_matrix,
            )
            verified_local_idxs = []
            verified_global_idxs = []
            for local_i, global_i in enumerate(idxs):
                if frame_keep[local_i]:
                    verified_local_idxs.append(local_i)
                    verified_global_idxs.append(global_i)
                else:
                    keep_mask[global_i] = False
                    n_rejected += 1
            if verified_global_idxs and csf_cfg.accumulate_new_anchors:
                # This keyframe's single best-scoring verified candidate is
                # what's "offered" to the confirmer -- same convention
                # GeCo2DynamicPrototypeTracker.offer() already uses (one
                # representative box per keyframe, not every survivor).
                # Skipped entirely (not just the append below) when
                # accumulate_new_anchors=False -- trusted_window then stays
                # permanently empty, so every keyframe clusters against
                # ONLY the 3 original exemplars for the whole video.
                # Ranking is ALWAYS by all_sims (stage1.feature_extractor's
                # own score) regardless of embedding_source -- clustering
                # only ever decides accept/reject here, never ranking (same
                # split as verification_method="cluster" + NMS/topk_per_
                # keyframe downstream).
                best_i = max(verified_global_idxs, key=lambda i: all_sims[i])
                confirmed_box = window_confirmer.offer(all_dets[best_i].box)
                if confirmed_box is not None:
                    # The window must stay in the SAME embedding space this
                    # keyframe clustered against -- cand_feats_frame[local_i]
                    # (not all_feats[best_i]) when using DAVE, since a
                    # window mixing DINOv2-space and DAVE-space vectors
                    # would silently corrupt every later keyframe's affinity
                    # matrix.
                    best_local_i = verified_local_idxs[verified_global_idxs.index(best_i)]
                    trusted_window.append(cand_feats_frame[best_local_i])
                    n_window_admitted += 1
            # No verified candidate this keyframe: the confirmer is simply
            # not offered anything (same "gap keyframes don't reset the
            # streak" convention GeCo2DynamicPrototypeTracker's own
            # offer()-only-when-boxes-exist call site already uses).

        selected = [
            (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
            for i in range(len(all_sims)) if keep_mask[i]
        ]
        log.info(
            "[Stage3] %s: cluster_secondary_filter (cluster_method=%s, window_size=%d, "
            "window_admission_min_consecutive_hits=%d) rejected %d / %d threshold-passing "
            "candidate(s) -> %d remain (%d admitted into the trusted window)",
            sample_id, csf_cfg.cluster_verification.cluster_method, csf_cfg.window_size,
            csf_cfg.window_admission_min_consecutive_hits, n_rejected, n_before, len(selected),
            n_window_admitted,
        )

    # ---- Identity-chain filter (KeepTrack-style, opt-in) ----
    icf_cfg = s3.identity_chain_filter
    if icf_cfg.enabled:
        n_before_chain = int(keep_mask.sum())
        keep_mask, n_chains_total, n_chains_kept = apply_identity_chain_filter(
            all_feats, all_frame_idxs, all_dets, all_sims, keep_mask, icf_cfg,
        )
        selected = [
            (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
            for i in range(len(all_sims)) if keep_mask[i]
        ]
        log.info(
            "[Stage3] %s: identity_chain_filter (min_chain_length=%d, top_k_per_keyframe=%d) "
            "-- %d/%d identity chain(s) reached min_chain_length -> %d / %d candidates remain",
            sample_id, icf_cfg.min_chain_length, icf_cfg.top_k_per_keyframe,
            n_chains_kept, n_chains_total, len(selected), n_before_chain,
        )

    # ---- Apply global_topk cap (after threshold, not instead of it) ----
    global_topk = s3.global_topk
    if global_topk is not None and len(selected) > global_topk:
        selected.sort(key=lambda x: x[2], reverse=True)
        selected = selected[:global_topk]
        log.info("[Stage3] %s: capped to global_topk=%d", sample_id, global_topk)

    # Group by frame, apply NMS + topk_per_keyframe
    from collections import defaultdict
    frame_groups: dict[int, list[tuple[Detection, float]]] = defaultdict(list)
    for fi, det, sim in selected:
        frame_groups[fi].append((det, sim))

    # ---- Margin-over-runner-up (WildFusion-style, opt-in) ----
    # Only meaningful for a keyframe with >=2 survivors -- a single
    # survivor has no runner-up to compare against and is left untouched.
    mv_cfg = s3.margin_verification
    if mv_cfg.enabled:
        n_ambiguous = 0
        for fi in list(frame_groups.keys()):
            pairs = frame_groups[fi]
            if len(pairs) < 2:
                continue
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            margin = pairs_sorted[0][1] - pairs_sorted[1][1]
            if margin < mv_cfg.tau_margin:
                del frame_groups[fi]
                n_ambiguous += 1
            else:
                # Margin clears the bar -- trust the winner ALONE (the
                # runner-up, however close it individually cleared
                # match_threshold, is not the accepted candidate here).
                frame_groups[fi] = [pairs_sorted[0]]
        if n_ambiguous:
            log.info(
                "[Stage3] %s: margin_verification (tau_margin=%.3f) dropped %d ambiguous "
                "keyframe(s) (top candidate's margin over the runner-up too small to trust)",
                sample_id, mv_cfg.tau_margin, n_ambiguous,
            )
        # Keep `selected` consistent with the pruned frame_groups -- used
        # below for sample_reference_size.
        selected = [(fi, d, s) for fi, pairs in frame_groups.items() for d, s in pairs]

    # ---- Isolated-detection filter (opt-in) ----
    # Drops keyframes with no other detection-bearing keyframe within
    # max_gap_intervals * keyframe_interval frames (unless confident enough).
    idf_cfg = s3.isolated_detection_filter
    if idf_cfg.enabled and frame_groups:
        kf_interval = (
            cfg.stage123_geco2.keyframe_interval
            if cfg.pipeline.detector == "geco2" else cfg.stage2.keyframe_interval
        )
        isolated = find_isolated_keyframes(
            {fi: max(s for _, s in pairs) for fi, pairs in frame_groups.items()},
            kf_interval, idf_cfg,
        )
        for fi in isolated:
            del frame_groups[fi]
        if isolated:
            log.info(
                "[Stage3] %s: isolated_detection_filter (max_gap=%d x %d frames, "
                "keep_conf_threshold=%s) dropped %d isolated keyframe(s): %s",
                sample_id, idf_cfg.max_gap_intervals, kf_interval,
                idf_cfg.keep_conf_threshold, len(isolated), sorted(isolated),
            )
            selected = [(fi, d, s) for fi, pairs in frame_groups.items() for d, s in pairs]

    # box_refine.adaptive_context_margin.relative_to_sample_median: this
    # SAME object's own typical box size across every OTHER threshold-
    # passing detection in this video -- the reference a box needs to be
    # compared against to tell "genuinely tiny" apart from "badly
    # undersized this one time" (see scale_context_margin's own docstring).
    # Computed once here (before any refinement) so every keyframe's refine
    # call below can be judged against the SAME, unrefined baseline.
    sample_reference_size = None
    if selected:
        sample_reference_size = float(np.median([
            ((det.box.x2 - det.box.x1) * (det.box.y2 - det.box.y1)) ** 0.5
            for _, det, _ in selected
        ]))

    pre_refine_detections: dict[int, list[Detection]] = {}

    for frame_idx, det_sim_pairs in frame_groups.items():
        det_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        dets_f = [d for d, _ in det_sim_pairs]
        sims_f = [s for _, s in det_sim_pairs]

        # NMS
        keep_idx = nms(
            [d.box.__class__(d.box.x1, d.box.y1, d.box.x2, d.box.y2, score=s)
             for d, s in zip(dets_f, sims_f)],
            iou_threshold=s3.nms_iou,
        )
        post_nms = [(dets_f[i], sims_f[i]) for i in keep_idx]

        # Top-K per keyframe
        post_nms = post_nms[: s3.topk_per_keyframe]

        result_dets = [
            Detection(frame_idx=frame_idx, box=det.box, similarity=sim, source="detect")
            for det, sim in post_nms
        ]

        needs_frame = (br_cfg.enabled and br_cfg.apply_in_stage3) or cfg.runtime.save_visualizations
        frame_bgr = None
        if needs_frame and video_path:
            try:
                frame_bgr = read_frame(video_path, frame_idx)
            except Exception:
                frame_bgr = None

        if br_cfg.enabled and br_cfg.apply_in_stage3 and frame_bgr is not None:
            # Snapshot the PRE-refine boxes before box_refine mutates
            # result_dets -- written out below as detections_prerefine.json
            # whenever box_refine actually ran, so a diagnostic script (see
            # scripts/check_box_refine_effect.py) always has a guaranteed-
            # clean "before" baseline to compare against, no matter what
            # box_refine.* setting was active on THIS run -- without this,
            # re-running Stage 3 with box_refine enabled overwrites
            # detections.json with already-refined boxes, so a later
            # diagnostic run would silently refine an already-refined box
            # a second time instead of comparing against the true original.
            pre_refine_detections[frame_idx] = result_dets
            if br_cfg.method in ("sam_dense", "fastsam_dense", "sam2_native"):
                # One shared frame "encode" (MobileSAM's own embedding,
                # FastSAM's segment-everything pass, or a standalone SAM2's
                # own encoder) for every surviving box on this keyframe,
                # instead of a crop+re-encode per box -- see
                # refine_boxes_dense's docstring. Same dispatch for all
                # three methods: box_refine_segmenter (built above) already
                # implements the set_frame()/segment_box_cached() interface
                # this function drives generically, regardless of which
                # concrete segmenter it is.
                from aero_eyes.utils.box_refine import refine_boxes_dense
                refined_boxes = refine_boxes_dense(
                    box_refine_segmenter, frame_bgr, [d.box for d in result_dets],
                    min_iou_with_original=br_cfg.min_iou_with_original,
                    context_margin=br_cfg.context_margin,
                    adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                    sample_reference_size=sample_reference_size,
                    use_center_point=br_cfg.use_center_point_prompt,
                )
                result_dets = [
                    Detection(frame_idx=d.frame_idx, box=rb, similarity=d.similarity, source=d.source)
                    for d, rb in zip(result_dets, refined_boxes)
                ]
            elif br_cfg.method == "sam2_dense":
                # GeCo2's own dense Hiera features refine every surviving
                # box on this keyframe in one extra backbone pass -- see
                # GeCo2Detector.sam2_refine_boxes's docstring.
                from aero_eyes.utils.box_refine import apply_iou_gate
                original_boxes = [d.box for d in result_dets]
                if geco2_refine_detector is not None:
                    refined_boxes = geco2_refine_detector.sam2_refine_boxes(
                        frame_bgr, geco2_refine_prototype, original_boxes,
                        context_margin=br_cfg.context_margin,
                        adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                        sample_reference_size=sample_reference_size,
                        use_center_point=br_cfg.use_center_point_prompt,
                        select_best_mask=br_cfg.sam2_dense_select_best_mask,
                    )
                    refined_boxes = apply_iou_gate(refined_boxes, original_boxes, br_cfg.min_iou_with_original)
                else:
                    refined_boxes = original_boxes
                result_dets = [
                    Detection(frame_idx=d.frame_idx, box=rb, similarity=d.similarity, source=d.source)
                    for d, rb in zip(result_dets, refined_boxes)
                ]
            else:
                from aero_eyes.utils.box_refine import refine_box
                result_dets = [
                    Detection(
                        frame_idx=d.frame_idx,
                        box=refine_box(
                            br_cfg.method, frame_bgr, d.box, br_cfg.context_margin,
                            segmenter=box_refine_segmenter, min_iou_with_original=br_cfg.min_iou_with_original,
                            adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                            sample_reference_size=sample_reference_size,
                            use_center_point=br_cfg.use_center_point_prompt,
                        ),
                        similarity=d.similarity, source=d.source,
                    )
                    for d in result_dets
                ]

        detections[frame_idx] = result_dets

        if cfg.runtime.save_visualizations and frame_bgr is not None:
            vizmod.save_stage3_detections(
                frame_bgr, [d.box for d in result_dets],
                [d.similarity for d in result_dets],
                frame_idx, viz_dir,
            )

    n_frames_with_detection = len(detections)

    # frame_groups (built from `selected`, i.e. threshold-passing candidates
    # only) never gets a key for a keyframe that Stage2/candidate-gen scanned
    # but where NOTHING passed the threshold -- so without this, such a
    # keyframe would be entirely absent from detections.json instead of
    # present with an empty box list. Stage4 tells "keyframe with zero
    # surviving detections" (stage4.keep_tracking_on_missed_keyframe's own
    # trigger condition) apart from "not a keyframe at all" purely by key
    # membership in this dict, so silently omitting these erases that
    # distinction -- keep_tracking_on_missed_keyframe then never fires for
    # them; they instead coast through the tracking loop's generic
    # non-keyframe path with none of its retroactive motion-plausibility
    # validation applied.
    for frame_idx in candidates:
        detections.setdefault(frame_idx, [])

    write_detections(detections, det_path, threshold=effective_threshold)
    if pre_refine_detections:
        prerefine_path = work_dir / "detections_prerefine.json"
        write_detections(pre_refine_detections, prerefine_path, threshold=effective_threshold)
        log.info("[Stage3] %s: box_refine was applied -- pre-refine boxes also saved to %s "
                 "(see scripts/check_box_refine_effect.py)", sample_id, prerefine_path)
    elapsed = time.time() - t0
    log.info("[Stage3] %s done in %.1fs -> %s (%d / %d keyframes with a detection)",
             sample_id, elapsed, det_path, n_frames_with_detection, len(detections))
    return det_path


def _apply_domain_prompter(
    feats: np.ndarray,
    prototype: np.ndarray,
    sims: np.ndarray,
    cfg,
) -> np.ndarray:
    """CD-ViTO-style domain feature alignment (simplified).

    Synthesizes 'imaginary domain' feature shifts by interpolating between
    the candidate feature distribution and the prototype direction,
    then re-scores using the shifted features.
    """
    dp = cfg.accuracy.max_accuracy.domain_prompter
    strength = dp.strength

    # Compute the mean domain gap: shift candidate features toward prototype style
    # by blending them with the prototype direction
    proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    shifted = feats + strength * proto_norm[None]
    # Re-normalize
    norms = np.linalg.norm(shifted, axis=-1, keepdims=True).clip(min=1e-8)
    shifted = shifted / norms
    new_sims = shifted @ prototype
    # Blend original and new scores
    return 0.5 * sims + 0.5 * new_sims


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 3 — cross-domain matching")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage3(cfg, args.sample)


if __name__ == "__main__":
    main()

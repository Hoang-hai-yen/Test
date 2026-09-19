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


def _score_against_ref(feats: np.ndarray, ref: np.ndarray, metric: str) -> np.ndarray:
    """Score every row of feats [N,D] against a single reference vector [D].

    Higher score always means "more similar" regardless of metric, so the
    rest of Stage 3 (threshold filtering, adaptive threshold, ranking) works
    unchanged no matter which metric is selected. For distance metrics
    (l1/l2) this means returning the negated distance.
    """
    if metric == "cosine":
        return feats @ ref
    if metric == "l2":
        return -np.linalg.norm(feats - ref[None, :], axis=1)
    if metric == "l1":
        return -np.sum(np.abs(feats - ref[None, :]), axis=1)
    raise ValueError(f"Unknown stage3.similarity metric '{metric}'. Must be 'cosine', 'l1', or 'l2'.")


def _pool_sims(sims_per_ref: list, pooling: str) -> np.ndarray:
    """Combine per-reference-image similarity arrays into one, per
    accuracy.cheap_boosters.multi_ref_pooling:
      mean -- average across refs (original behavior). A candidate that
        matches one ref very well but the other two poorly (e.g. the object
        was photographed from 3 different angles and this candidate's own
        viewing angle only resembles one of them) has that good score
        diluted by the two weak ones.
      max -- the single best-matching ref's score per candidate, so a
        genuinely good match from one well-aligned reference view isn't
        dragged down by refs shot from a different angle/lighting.
    """
    if pooling == "max":
        return np.max(sims_per_ref, axis=0)
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
            sims_per_ref = [_score_against_ref(all_feats, ref_feat, similarity_metric) for ref_feat in per_ref_features]
            all_sims = _pool_sims(sims_per_ref, multi_ref_pooling)
        else:
            prototype = (1 - dp.alpha) * prototype + dp.alpha * dynamic_feat
            prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
            all_sims = _score_against_ref(all_feats, prototype, similarity_metric)

    return prototype, all_sims, per_ref_features


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
    if br_cfg.enabled and br_cfg.apply_in_stage3:
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
        n_recomputed = 0
        for frame_idx, cand_dets in candidates.items():
            if not cand_dets:
                continue
            try:
                frame_bgr = read_frame(video_path, frame_idx)
            except Exception:
                continue
            feats = recompute_extractor.extract_crops(
                frame_bgr, [d.box for d in cand_dets],
                pad_ratio=cfg.stage2.candidate.feature_crop_pad,
                batch_size=cfg.runtime.batch_size,
            )
            for det, feat in zip(cand_dets, feats):
                det._feature = feat
            n_recomputed += len(cand_dets)
        _write_candidates_with_features(candidates, cand_path)
        log.info(
            "[Stage3] %s: recompute_candidate_features -- re-extracted %d candidate "
            "feature(s) across %d keyframe(s), rewrote %s",
            sample_id, n_recomputed, len(candidates), cand_path,
        )
        # Re-read rather than hand-assemble feat_matrix here -- keeps this
        # path exercising the exact same load code every other run takes.
        candidates, feat_matrix = read_candidates_with_features(cand_path)

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

    # Compute similarity for every candidate at once (higher = more similar,
    # regardless of metric -- see _score_against_ref).
    if use_multi_ref:
        sims_per_ref = [_score_against_ref(all_feats, ref_feat, s3.similarity) for ref_feat in per_ref_features]
        all_sims = _pool_sims(sims_per_ref, multi_ref_pooling)
    else:
        all_sims = _score_against_ref(all_feats, prototype, s3.similarity)  # [N]

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
    prototype, all_sims, per_ref_features = run_dynamic_prototype_rounds(
        sample_id, all_feats, all_sims, prototype, per_ref_features,
        use_multi_ref, multi_ref_pooling, s3.similarity, s3.dynamic_prototype,
        all_frame_idxs=all_frame_idxs,
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
    if s3.dynamic_prototype.enabled:
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

    # ---- Compute effective threshold ----
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

    # ---- Filter by threshold ----
    keep_mask = all_sims >= effective_threshold
    selected = [
        (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
        for i in range(len(all_sims)) if keep_mask[i]
    ]
    log.info("[Stage3] %s: threshold=%.3f → %d / %d candidates pass",
             sample_id, effective_threshold, len(selected), len(all_sims))

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

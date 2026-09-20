"""Track A: per-sample automatic scale/detail calibration for GeCo2
exemplars (aero_eyes.config.AutoScaleCalibrationConfig).

Replaces hand-tuning stage123_geco2.ref_downscale_factor (blur/detail) and
crop_context_margin (canvas-relative size) with an automatic per-sample
search: build one candidate exemplar prototype per (crop_margin,
downscale_factor) pair, score each against a handful of frames sampled
from THIS sample's own video, then quality-weighted-blend the appearance
tokens (or hard-select the best one) -- see
aero_eyes.config.AutoScaleCalibrationConfig's docstring for the full
rationale (why both axes matter, why this needs no train/inference
mismatch unlike ref_downscale_levels).

Called from aero_eyes.stages.stage123_geco2.build_exemplar_prototype.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou, crop_to_object
from aero_eyes.utils.video import read_frame, video_info

log = logging.getLogger(__name__)


def sample_uniform_frame_indices(total_frames: int, n: int) -> list[int]:
    """Deterministic, evenly-spaced frame indices via np.linspace (NOT
    random) -- so re-running build_exemplar_prototype with an unchanged
    config/video reproduces an identical calibration, keeping
    project.use_cache's existing cache contract safe without any new
    cache-invalidation machinery."""
    if total_frames <= 0 or n <= 0:
        return []
    n = min(n, total_frames)
    return sorted(set(np.linspace(0, total_frames - 1, num=n).astype(int).tolist()))


def build_candidate_grid(
    crop_margins: list[float], downscale_factors: list[float], max_candidates: int,
) -> list[tuple[float, float]]:
    """Full (crop_margin, downscale_factor) grid, evenly downsampled (not
    truncated from one end) to at most max_candidates if the product of
    the two candidate lists exceeds it."""
    grid = [(m, f) for m in crop_margins for f in downscale_factors]
    if len(grid) <= max_candidates:
        return grid
    idx = sorted(set(np.linspace(0, len(grid) - 1, num=max_candidates).round().astype(int).tolist()))
    log.warning(
        "[auto_scale_calibration] candidate grid %d combos > max_candidates %d -- "
        "sampled down to %d evenly-spaced candidates",
        len(grid), max_candidates, len(idx),
    )
    return [grid[i] for i in idx]


def top1_box_and_score(detector, frame_bgr: np.ndarray, prototype: dict) -> tuple[Box | None, float]:
    """The single highest-scoring box for one frame under one candidate
    prototype -- thresholds at box_v.max()-eps via filter_boxes_by_threshold
    so exactly the argmax survives NMS/top-k trivially. (None, -inf) if the
    frame produced no boxes at all."""
    pred_boxes, box_v, scale = detector.forward_scores(frame_bgr, prototype)
    if pred_boxes.numel() == 0:
        return None, float("-inf")
    threshold = float(box_v.max().item()) - 1e-6
    top = detector.filter_boxes_by_threshold(pred_boxes, box_v, scale, frame_bgr, threshold)
    if not top:
        return None, float("-inf")
    best = max(top, key=lambda b: b.score)
    return best, float(best.score)


def score_candidate_gt_iou(
    detector, prototype: dict, video_path: Path, gt: dict[int, Box], present_frame_idxs: list[int],
) -> float:
    """Mean IoU between the top-1 predicted box and GT, over this sample's
    own GT-present frames -- the reliable metric whenever GT exists (dev/
    calibration set), see AutoScaleCalibrationConfig.quality_metric."""
    ious = []
    for idx in present_frame_idxs:
        frame_bgr = read_frame(video_path, idx)
        box, _ = top1_box_and_score(detector, frame_bgr, prototype)
        ious.append(box_iou(box, gt[idx]) if box is not None else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def score_candidate_self_supervised(
    detector, prototype: dict, video_path: Path, probe_frame_idxs: list[int], eps: float,
) -> float:
    """Mean over probe frames of (max-mean)/(std+eps) of raw_scores -- a
    peakiness proxy for "this exemplar scale produces one confident,
    spatially localized detection" with no GT needed. KNOWN RISK: a
    confidently-wrong high-scoring background patch can also score high
    here -- see AutoScaleCalibrationConfig.quality_metric's own docstring;
    prefer gt_iou whenever GT exists."""
    margins = []
    for idx in probe_frame_idxs:
        frame_bgr = read_frame(video_path, idx)
        scores = detector.raw_scores(frame_bgr, prototype)
        if scores.size == 0:
            continue
        margins.append((float(scores.max()) - float(scores.mean())) / (float(scores.std()) + eps))
    return float(np.mean(margins)) if margins else float("-inf")


def select_weights(qualities: list[float], mode: str, temperature: float) -> list[float]:
    """mode="hard": one-hot argmax. mode="soft": softmax(z-score(qualities)/temperature)."""
    arr = np.array(qualities, dtype=np.float64)
    if mode == "hard":
        weights = np.zeros_like(arr)
        weights[int(np.argmax(arr))] = 1.0
        return weights.tolist()
    std = float(arr.std())
    z = (arr - arr.mean()) / std if std > 1e-12 else np.zeros_like(arr)
    z = z / max(temperature, 1e-6)
    z = z - z.max()  # numerical stability, softmax is shift-invariant
    w = np.exp(z)
    w = w / w.sum()
    return w.tolist()


def _resolve_gt(cfg, sample_id: str) -> dict[int, Box] | None:
    from aero_eyes.utils.io import load_gt

    try:
        gt = load_gt(cfg.data.gt.global_file, sample_id)
    except (KeyError, FileNotFoundError):
        return None
    return gt or None


def build_auto_scaled_prototype(
    cfg, sample_id: str, detector, ref_imgs: list, raw_boxes: list, video_path: Path,
) -> tuple[dict, dict]:
    """Orchestrates the whole per-sample auto-calibration: for each
    (crop_margin, downscale_factor) candidate, crop+downscale the 3
    reference images, encode them ONCE (reused for both scoring and the
    final blend), score the resulting prototype, then blend across
    candidates by aero_eyes.config.AutoScaleCalibrationConfig.
    selection_mode. Returns (prototype_dict, debug_info) where debug_info
    is JSON-serializable (candidate_factors, qualities, weights,
    metric_used, probe_frame_idxs) for stage123_geco2.py to persist.

    raw_boxes: tight MobileSAM mask bbox per ref image, in that ref
    image's own native (pre-crop) pixel coords -- same as
    build_exemplar_prototype's `raw_boxes` local (segmentation.enabled
    required, mirroring crop_to_object's own requirement).
    """
    from aero_eyes.stages.stage123_geco2 import _apply_ref_downscale

    a_cfg = cfg.stage123_geco2.auto_scale_calibration
    candidates = build_candidate_grid(
        a_cfg.candidate_crop_margins, a_cfg.candidate_downscale_factors, a_cfg.max_candidates,
    )

    gt = _resolve_gt(cfg, sample_id)
    metric_used = a_cfg.quality_metric
    if metric_used == "auto":
        metric_used = "gt_iou" if gt else "self_supervised_margin"
    elif metric_used == "gt_iou" and gt is None:
        log.warning(
            "[auto_scale_calibration] %s: quality_metric='gt_iou' forced but no GT found in %s -- "
            "falling back to self_supervised_margin", sample_id, cfg.data.gt.global_file,
        )
        metric_used = "self_supervised_margin"

    info = video_info(video_path)
    total_frames = info["total_frames"]
    if metric_used == "gt_iou":
        present_frames = sorted(gt.keys())
        probe_frame_idxs = sample_uniform_frame_indices(len(present_frames), a_cfg.num_probe_frames)
        probe_frame_idxs = [present_frames[i] for i in probe_frame_idxs]
    else:
        probe_frame_idxs = sample_uniform_frame_indices(total_frames, a_cfg.num_probe_frames)

    proto_per_candidate: list[dict] = []
    qualities: list[float] = []
    boxes_per_candidate: list[list] = []
    for margin, factor in candidates:
        cand_imgs, cand_boxes = [], []
        for img, box in zip(ref_imgs, raw_boxes):
            if box is None:
                cand_imgs.append(_apply_ref_downscale(img, factor))
                cand_boxes.append(None)
                continue
            cropped, cbox = crop_to_object(img, box, margin)
            cand_imgs.append(_apply_ref_downscale(cropped, factor))
            cand_boxes.append(tuple(c * factor for c in cbox))
        prototype = detector.encode_exemplars(cand_imgs, ref_boxes=cand_boxes)
        proto_per_candidate.append(prototype)
        boxes_per_candidate.append(cand_boxes)

        if metric_used == "gt_iou":
            quality = score_candidate_gt_iou(detector, prototype, video_path, gt, probe_frame_idxs)
        else:
            quality = score_candidate_self_supervised(
                detector, prototype, video_path, probe_frame_idxs, a_cfg.eps,
            )
        qualities.append(quality)

    weights = select_weights(qualities, a_cfg.selection_mode, a_cfg.temperature)
    best_idx = int(np.argmax(qualities))

    blended: dict = {}
    for scale in ("main", "l1", "l2"):
        blended[scale] = sum(w * proto_per_candidate[i][scale] for i, w in enumerate(weights))

    if detector.use_shape_token:
        # Shape tokens encode a REAL (w,h) box size -- crop_context_margin
        # changes the box's own w,h per candidate, so a weighted average
        # across candidates would produce a (w,h) that doesn't correspond
        # to any real candidate's box, confusing the shape signal. Use the
        # best-scoring candidate's own shape tokens unblended instead (see
        # AutoScaleCalibrationConfig / build_auto_scaled_prototype's own
        # docstring). Token layout is [app_1, shape_1, app_2, shape_2, ...]
        # -- see GeCo2Detector.calibrate_prototype's own docstring.
        shape_idx = list(range(1, 2 * len(ref_imgs), 2))
        for scale in ("main", "l1", "l2"):
            blended[scale][:, shape_idx, :] = proto_per_candidate[best_idx][scale][:, shape_idx, :]

    debug_info = {
        "sample_id": sample_id,
        "candidates": [{"crop_margin": m, "downscale_factor": f} for m, f in candidates],
        "qualities": qualities,
        "weights": weights,
        "metric_used": metric_used,
        "selection_mode": a_cfg.selection_mode,
        "best_candidate": {"crop_margin": candidates[best_idx][0], "downscale_factor": candidates[best_idx][1]},
        "probe_frame_idxs": probe_frame_idxs,
    }
    return blended, debug_info

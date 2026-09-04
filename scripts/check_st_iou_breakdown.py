"""Diagnostic: WHERE is ST-IoU being lost -- Stage 4 tracking coverage
gaps, or imprecise localization on frames that DID have a prediction?

ST-IoU (see aero_eyes.evaluate.st_iou) is the mean spatial IoU over the
TEMPORAL UNION of frames present in either the predicted tube or the GT
tube; a frame present in only one contributes 0.0. A Precision/Recall
computed only on Stage 3's own keyframes (see check_cosine_effect.py) can
look strong while ST-IoU stays low, because:
  1. Detection only runs every keyframe_interval-th frame; ST-IoU is
     evaluated over EVERY frame -- Stage 4's tracker has to correctly hold
     the object across the frames in between, or those frames contribute
     0.0 regardless of how good detection was at the keyframes it did run.
  2. Precision/Recall at a fixed IoU threshold is a binary pass/fail; a box
     that JUST clears the threshold still drags the ST-IoU average down
     compared to a tight match -- ST-IoU uses the continuous IoU value.
  3. A GT frame where the prediction is entirely MISSING (tracking lost the
     object, or Stage 5's min_tube_length/fill_short_gaps trimmed a short
     segment) contributes IoU=0.0 just like a badly-placed box would -- this
     script tells the two apart.

Reads tracks.json (raw Stage 4 output, BEFORE Stage 5 smoothing/gap-fill/
min_tube_length) and submission.json (final, AFTER Stage 5) against ground
truth, and classifies every frame in each tube's union with GT into:
  MATCH        -- both present, IoU >= --iou-threshold
  LOOSE        -- both present, IoU <  --iou-threshold (localized SOMEWHERE, just not tightly)
  MISSING_PRED -- GT present, no prediction on this frame (tracking/coverage gap)
  MISSING_GT   -- prediction present, GT absent (false positive / stray track)

Then splits the ST-IoU "deficit" (1 - ST-IoU) into:
  coverage gap   -- from MISSING_PRED/MISSING_GT frames (each is a full 1.0 loss)
  imprecision    -- from MATCH/LOOSE frames whose IoU is < 1.0 (both present, just not exact)

Comparing tracks.json vs submission.json also isolates whether Stage 5's
own post-processing (gap-fill, min_tube_length, EMA smoothing) is helping
or hurting, separately from Stage 4's raw tracking behavior.

Usage:
    python -m scripts.check_st_iou_breakdown --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_st_iou_breakdown --config configs/config.yaml --iou-threshold 0.3
    python -m scripts.check_st_iou_breakdown --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.evaluate import _load_submission
from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_tracks

log = logging.getLogger(__name__)


def classify_frames(
    pred_tube: dict[int, Box],
    gt_tube: dict[int, Box],
    iou_threshold: float,
) -> dict:
    """Per-frame breakdown of one predicted tube against GT -- see module
    docstring for exact MATCH/LOOSE/MISSING_PRED/MISSING_GT semantics and
    how the ST-IoU deficit is split into coverage vs imprecision."""
    union_frames = set(pred_tube.keys()) | set(gt_tube.keys())
    n_union = len(union_frames)
    if n_union == 0:
        return {
            "n_union": 0, "match": 0, "loose": 0, "missing_pred": 0, "missing_gt": 0,
            "mean_iou_match": None, "mean_iou_loose": None, "st_iou": 0.0,
            "deficit_coverage": 0.0, "deficit_imprecision": 0.0,
        }

    match = loose = missing_pred = missing_gt = 0
    iou_sum = 0.0
    iou_sum_match = 0.0
    iou_sum_loose = 0.0
    for fi in union_frames:
        in_pred = fi in pred_tube
        in_gt = fi in gt_tube
        if in_pred and in_gt:
            iou = box_iou(pred_tube[fi], gt_tube[fi])
            iou_sum += iou
            if iou >= iou_threshold:
                match += 1
                iou_sum_match += iou
            else:
                loose += 1
                iou_sum_loose += iou
        elif in_gt and not in_pred:
            missing_pred += 1
        else:  # in_pred and not in_gt
            missing_gt += 1

    st_iou_value = iou_sum / n_union
    deficit_total = 1.0 - st_iou_value
    deficit_coverage = (missing_pred + missing_gt) / n_union
    deficit_imprecision = deficit_total - deficit_coverage

    return {
        "n_union": n_union, "match": match, "loose": loose,
        "missing_pred": missing_pred, "missing_gt": missing_gt,
        "mean_iou_match": (iou_sum_match / match) if match else None,
        "mean_iou_loose": (iou_sum_loose / loose) if loose else None,
        "st_iou": st_iou_value,
        "deficit_coverage": deficit_coverage,
        "deficit_imprecision": deficit_imprecision,
    }


def _print_breakdown(label: str, r: dict) -> None:
    print(f"  {label}: ST-IoU={r['st_iou']:.4f}  ({r['n_union']} union frames)")
    if r["n_union"] == 0:
        return
    match_str = f"mean IoU={r['mean_iou_match']:.3f}" if r["mean_iou_match"] is not None else "n/a"
    loose_str = f"mean IoU={r['mean_iou_loose']:.3f}" if r["mean_iou_loose"] is not None else "n/a"
    print(
        f"    MATCH (IoU>=thr)={r['match']} ({match_str})  "
        f"LOOSE (both present, IoU<thr)={r['loose']} ({loose_str})  "
        f"MISSING_PRED (GT there, no pred)={r['missing_pred']}  "
        f"MISSING_GT (pred there, no GT)={r['missing_gt']}"
    )
    deficit_total = r["deficit_coverage"] + r["deficit_imprecision"]
    if deficit_total > 1e-9:
        cov_pct = 100.0 * r["deficit_coverage"] / deficit_total
        imp_pct = 100.0 * r["deficit_imprecision"] / deficit_total
        print(
            f"    Of the {deficit_total:.4f} ST-IoU deficit: "
            f"{cov_pct:.0f}% from coverage gaps (missing pred/GT) -- likely Stage 4 tracking; "
            f"{imp_pct:.0f}% from imprecision (both present but IoU<1.0) -- localization looseness."
        )


def check_sample(cfg, sample_id: str, iou_threshold: float) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    tracks_path = work_dir / "tracks.json"
    submission_path = work_dir / cfg.data.submission.path_name

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) ===")

    if not tracks_path.exists():
        print(f"  tracks.json (raw Stage 4): not found at {tracks_path} -- run Stage 4 first.")
    else:
        raw_tracks = read_tracks(tracks_path)
        pred_tube = {fi: b for fi, b in raw_tracks.items() if b is not None}
        r = classify_frames(pred_tube, gt, iou_threshold)
        _print_breakdown("tracks.json (raw Stage 4, before Stage 5)", r)

    if not submission_path.exists():
        print(f"  submission.json (final): not found at {submission_path} -- run Stage 5 first.")
    else:
        all_submissions = _load_submission(submission_path)
        pred_tube = all_submissions.get(sample_id, {})
        r = classify_frames(pred_tube, gt, iou_threshold)
        _print_breakdown("submission.json (final, after Stage 5)", r)


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Decompose ST-IoU loss into Stage 4 tracking coverage gaps vs "
        "localization imprecision."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as MATCH rather than LOOSE (default: 0.5)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        check_sample(cfg, sid, args.iou_threshold)


if __name__ == "__main__":
    main()

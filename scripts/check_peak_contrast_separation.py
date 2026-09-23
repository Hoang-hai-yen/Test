"""Diagnostic: does peak_contrast (stage123_geco2.peak_contrast_filter)
actually separate REAL target candidates from CLUTTER candidates on real
footage, or is it just as overlapping as raw cosine similarity (this
project's own research notes measured that ceiling at ~0.335-0.38)? This
decides whether hard_reject=true / dynamic_prototype.topk_fusion.
peakiness_weight are worth trusting, and what min_contrast_z to actually
set -- do NOT guess it from log_peak_contrast_summary()'s aggregate mean/
std alone (that mixes real and clutter candidates together and says
nothing about where the two separate).

Needs candidates.json already built with stage123_geco2.peak_contrast_
filter.enabled=true (annotate-only, hard_reject=false is enough -- nothing
needs to be re-run with hard_reject=true first).

For every keyframe, buckets each of that keyframe's candidates into REAL
(IoU >= --iou-threshold against that frame's GT box) or CLUTTER (everything
else -- including every candidate in a keyframe with no GT box present at
all, same "absent frame" semantics as scripts/check_geco2_score_
separation.py). Prints side-by-side peak_contrast distributions for both
buckets, and a suggested min_contrast_z if they're cleanly separable.

Usage:
    python -m scripts.check_peak_contrast_separation --config configs/config.yaml --sample IDCard_1
    python -m scripts.check_peak_contrast_separation --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _print_stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: (no candidates)")
        return
    arr = np.array(values, dtype=np.float64)
    print(
        f"  {label}: n={len(arr)} min={arr.min():.3f} p25={np.percentile(arr, 25):.3f} "
        f"median={np.percentile(arr, 50):.3f} p75={np.percentile(arr, 75):.3f} max={arr.max():.3f} "
        f"mean={arr.mean():.3f}"
    )


def check_sample(cfg, sample_id: str, iou_threshold: float) -> None:
    from aero_eyes.stages.stage2 import read_candidates_with_features
    from aero_eyes.utils.geometry import box_iou
    from aero_eyes.utils.io import load_gt

    work_dir = Path(cfg.project.work_dir) / sample_id
    cand_path = work_dir / "candidates.json"
    if not cand_path.exists():
        print(f"{sample_id}: no candidates.json at {cand_path}, skipping.")
        return

    try:
        gt = load_gt(cfg.data.gt.global_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {cfg.data.gt.global_file}, skipping.")
        return

    candidates, _ = read_candidates_with_features(cand_path)

    real_vals: list[float] = []
    clutter_vals: list[float] = []
    n_missing = 0

    for frame_idx, dets in candidates.items():
        gt_box = gt.get(frame_idx)
        for det in dets:
            pc = det.box.peak_contrast
            if pc is None:
                n_missing += 1
                continue
            if gt_box is not None and box_iou(det.box, gt_box) >= iou_threshold:
                real_vals.append(pc)
            else:
                clutter_vals.append(pc)

    print(f"\n{sample_id}:")
    if n_missing:
        print(
            f"  WARNING: {n_missing} candidates have no peak_contrast -- was this candidates.json "
            "built with stage123_geco2.peak_contrast_filter.enabled=true? (re-run with "
            "project.use_cache=false if it was built before enabling the flag)"
        )
    _print_stats("REAL    (candidate IoU >= threshold with GT box)", real_vals)
    _print_stats("CLUTTER (everything else, incl. GT-absent frames)", clutter_vals)

    if not (real_vals and clutter_vals):
        return
    real_arr, clutter_arr = np.array(real_vals), np.array(clutter_vals)
    real_min, clutter_max = float(real_arr.min()), float(clutter_arr.max())
    if real_min > clutter_max:
        floor = (real_min + clutter_max) / 2
        print(
            f"  -> SEPARABLE: clutter max ({clutter_max:.3f}) < real min ({real_min:.3f}). "
            f"min_contrast_z around {floor:.3f} would cleanly separate them on this sample."
        )
    else:
        overlap = float((clutter_arr >= real_min).mean())
        clutter_p90 = float(np.percentile(clutter_arr, 90))
        real_lost_at_p90 = float((real_arr < clutter_p90).mean())
        print(
            f"  -> NOT cleanly separable: {overlap:.0%} of clutter candidates score >= the "
            f"lowest real candidate. Using clutter's own p90 ({clutter_p90:.3f}) as "
            f"min_contrast_z would still drop {real_lost_at_p90:.0%} of REAL candidates too -- "
            "a precision/recall tradeoff, not a free lunch. Pick a floor by walking the REAL "
            "percentiles above until an acceptable amount of real recall is lost, not by eye."
        )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Check whether peak_contrast separates real-target candidates from clutter"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--set", action="append", default=[])
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

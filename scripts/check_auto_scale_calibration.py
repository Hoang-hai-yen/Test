"""Diagnostic: does stage123_geco2.auto_scale_calibration's automatic
per-sample (crop_margin, ref_downscale_factor) choice track -- or beat --
a manual per-factor sweep, on samples that already have GT?

Extends scripts/check_geco2_score_separation.py's present/absent score-
separation check with: (1) a sweep over a list of FIXED
ref_downscale_factor values (auto_scale_calibration disabled,
crop_context_margin/crop_to_object left at whatever the config already
has), and (2) one run with auto_scale_calibration.enabled=true -- for each,
reports present/absent raw-score separation AND mean top-1 IoU against GT
on the sampled present frames, so the automatic choice's IoU can be
compared directly against every fixed candidate's own IoU.

This is diagnostic only -- it does NOT run Stage 4/5 tracking or ST-IoU;
see scripts/compare_auto_scale_vs_fixed.py for the full-pipeline, LOOCV-
honest comparison.

Usage:
    python -m scripts.check_auto_scale_calibration --config configs/config.yaml \
        --sample BlackBox_0 --factors 1.0,0.5,0.25,0.125,0.0625,0.03
    python -m scripts.check_auto_scale_calibration --config configs/config.yaml
        # all samples with GT in cfg.data.gt.global_file
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _sample_frames(indices: list[int], n: int) -> list[int]:
    if len(indices) <= n:
        return list(indices)
    step = (len(indices) - 1) / (n - 1)
    return sorted({indices[round(i * step)] for i in range(n)})


def _print_stats(label: str, scores: list[float]) -> None:
    if not scores:
        print(f"  {label}: (no frames sampled)")
        return
    arr = np.array(scores, dtype=np.float64)
    print(
        f"  {label}: n={len(arr)} min={arr.min():.4f} median={np.percentile(arr, 50):.4f} "
        f"max={arr.max():.4f} mean={arr.mean():.4f}"
    )


def _clear_prototype_cache(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _evaluate_prototype(detector, prototype, video_path, present_sample, absent_sample, gt) -> dict:
    from aero_eyes.models.geco2_auto_scale import top1_box_and_score
    from aero_eyes.utils.geometry import box_iou
    from aero_eyes.utils.video import frame_iterator

    wanted = present_sample | absent_sample
    present_scores, absent_scores, ious = [], [], []
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in wanted:
            continue
        scores = detector.raw_scores(frame_bgr, prototype)
        m = float(scores.max()) if scores.size else float("-inf")
        (present_scores if frame_idx in present_sample else absent_scores).append(m)
        if frame_idx in present_sample:
            box, _ = top1_box_and_score(detector, frame_bgr, prototype)
            ious.append(box_iou(box, gt[frame_idx]) if box is not None else 0.0)
        if len(present_scores) >= len(present_sample) and len(absent_scores) >= len(absent_sample):
            break
    return {
        "present_scores": present_scores,
        "absent_scores": absent_scores,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def check_sample(cfg, sample_id: str, detector, num_samples: int, factors: list[float]) -> None:
    from aero_eyes.stages.stage123_geco2 import build_exemplar_prototype, _locate_video
    from aero_eyes.utils.io import load_gt
    from aero_eyes.utils.video import video_info

    try:
        gt = load_gt(cfg.data.gt.global_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {cfg.data.gt.global_file}, skipping.")
        return
    if not gt:
        print(f"{sample_id}: 0 GT frames, skipping.")
        return

    video_path = _locate_video(cfg, sample_id)
    total_frames = video_info(video_path)["total_frames"]

    present_frames = sorted(gt.keys())
    absent_pool = [f for f in range(total_frames) if f not in gt]
    present_sample = set(_sample_frames(present_frames, num_samples))
    absent_sample = set(_sample_frames(absent_pool, num_samples))
    if not absent_sample:
        print(f"{sample_id}: every frame has a GT box -- nothing to compare against, skipping.")
        return

    print(f"\n{'=' * 70}\n{sample_id}\n{'=' * 70}")
    work_dir = Path(cfg.project.work_dir) / sample_id

    cfg.stage123_geco2.auto_scale_calibration.enabled = False
    for factor in factors:
        cfg.stage123_geco2.ref_downscale_factor = factor
        _clear_prototype_cache(work_dir)
        prototype = build_exemplar_prototype(cfg, sample_id, detector, work_dir)
        result = _evaluate_prototype(detector, prototype, video_path, present_sample, absent_sample, gt)
        print(f"\n-- ref_downscale_factor={factor} --")
        _print_stats("present", result["present_scores"])
        _print_stats("absent ", result["absent_scores"])
        print(f"  mean top-1 IoU vs GT (present frames): {result['mean_iou']:.4f}")

    cfg.stage123_geco2.auto_scale_calibration.enabled = True
    _clear_prototype_cache(work_dir)
    prototype = build_exemplar_prototype(cfg, sample_id, detector, work_dir)
    result = _evaluate_prototype(detector, prototype, video_path, present_sample, absent_sample, gt)
    print("\n-- auto_scale_calibration=true --")
    _print_stats("present", result["present_scores"])
    _print_stats("absent ", result["absent_scores"])
    print(f"  mean top-1 IoU vs GT (present frames): {result['mean_iou']:.4f}")
    debug_path = work_dir / "geco2_auto_scale_calibration.json"
    if debug_path.exists():
        print(f"  (auto-calibration decision log: {debug_path})")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Compare auto_scale_calibration against a manual ref_downscale_factor sweep"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples with GT in data_root")
    p.add_argument("--num-samples", type=int, default=20,
                    help="frames to sample from each of the present/absent groups")
    p.add_argument("--factors", default="1.0,0.5,0.25,0.125,0.0625,0.03",
                    help="comma-separated ref_downscale_factor values to sweep as the manual baseline")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    from aero_eyes.config import load_config
    from aero_eyes.models.geco2_detector import GeCo2Detector

    cfg = load_config(args.config, args.set)
    # This script rebuilds the prototype under many different configs per
    # sample -- project.use_cache would silently reuse a stale prototype
    # built under a DIFFERENT ref_downscale_factor/auto_scale_calibration
    # setting, so it's forced off here regardless of the config on disk
    # (each sweep point also explicitly clears work_dir/<sample_id> itself,
    # belt-and-suspenders).
    if cfg.project.use_cache:
        log.warning("project.use_cache=true in config -- forcing false for this sweep script.")
        cfg.project.use_cache = False

    print(f">>> Loading GeCo2 weights from: {cfg.stage123_geco2.weights_path}")
    detector = GeCo2Detector(cfg)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    factors = [float(f) for f in args.factors.split(",")]
    for sid in sample_ids:
        check_sample(cfg, sid, detector, args.num_samples, factors)


if __name__ == "__main__":
    main()

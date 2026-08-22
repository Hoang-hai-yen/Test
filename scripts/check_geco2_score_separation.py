"""Diagnostic: does GeCo2's score actually separate "target present" from
"target absent" keyframes, or does it score confidently regardless?

Why this matters: GeCo2 is trained/evaluated on FSC147, a few-shot
COUNTING benchmark where every image is guaranteed to contain >=1 instance
of the counted class (exemplars are literally cropped from the same
image). It may never have seen a genuine "target absent" image during
training. aero_eyes' stage123_geco2 pipeline uses a threshold that is
RELATIVE to each frame's own max score (score_threshold_ratio, see
GeCo2Detector.detect_frame) -- there is no absolute floor, so it always
keeps at least the single highest-scoring point per frame. If GeCo2's raw
score doesn't separate present/absent frames on your data, this pipeline
will produce a "detection" on essentially every keyframe of every video,
regardless of whether the object is actually there.

This script computes GeCo2's raw per-frame max score (before any
thresholding) for a sample of frames that DO have a GT box vs frames that
DO NOT, and reports both distributions so you can see whether they're
separable -- and if so, roughly what an absolute floor would need to be.

Usage:
    python -m scripts.check_geco2_score_separation --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_geco2_score_separation --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
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
        f"  {label}: n={len(arr)} min={arr.min():.4f} p25={np.percentile(arr, 25):.4f} "
        f"median={np.percentile(arr, 50):.4f} p75={np.percentile(arr, 75):.4f} max={arr.max():.4f} "
        f"mean={arr.mean():.4f}"
    )


def check_sample(cfg, sample_id: str, detector, num_samples: int) -> None:
    from aero_eyes.stages.stage123_geco2 import build_exemplar_prototype
    from aero_eyes.utils.io import load_gt
    from aero_eyes.utils.video import frame_iterator, video_info

    try:
        gt = load_gt(cfg.data.gt.global_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {cfg.data.gt.global_file}, skipping.")
        return
    if not gt:
        print(f"{sample_id}: 0 GT frames, skipping.")
        return

    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        print(f"{sample_id}: no video found under {data_root / sample_id}.")
        return
    video_path = video_files[0]
    total_frames = video_info(video_path)["total_frames"]

    present_frames = sorted(gt.keys())
    absent_pool = [f for f in range(total_frames) if f not in gt]
    present_sample = set(_sample_frames(present_frames, num_samples))
    absent_sample = set(_sample_frames(absent_pool, num_samples))
    wanted = present_sample | absent_sample
    if not absent_sample:
        print(f"{sample_id}: every frame has a GT box -- nothing to compare against, skipping.")
        return

    work_dir = Path(cfg.project.work_dir) / sample_id
    # Same build as the real pipeline (MobileSAM masking + tight exemplar
    # box when enabled) -- using a simpler/different exemplar here would
    # make these scores not representative of what run_stage123_geco2
    # actually sees, defeating the point of calibrating against it.
    prototype = build_exemplar_prototype(cfg, sample_id, detector, work_dir)

    present_scores: list[float] = []
    absent_scores: list[float] = []
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in wanted:
            continue
        scores = detector.raw_scores(frame_bgr, prototype)
        m = float(scores.max()) if scores.size else float("-inf")
        (present_scores if frame_idx in present_sample else absent_scores).append(m)
        if len(present_scores) >= len(present_sample) and len(absent_scores) >= len(absent_sample):
            break

    print(f"\n{sample_id}:")
    _print_stats("present (has GT box)", present_scores)
    _print_stats("absent  (no GT box) ", absent_scores)

    if present_scores and absent_scores:
        present_min = min(present_scores)
        absent_max = max(absent_scores)
        if present_min > absent_max:
            floor = (present_min + absent_max) / 2
            print(
                f"  -> SEPARABLE: absent max ({absent_max:.4f}) < present min ({present_min:.4f}). "
                f"An absolute floor around {floor:.4f} would tell present from absent frames apart."
            )
        else:
            overlap = sum(1 for s in absent_scores if s >= present_min) / len(absent_scores)
            print(
                f"  -> NOT cleanly separable: {overlap:.0%} of 'absent' frames score >= the lowest "
                f"'present' frame's score. A relative-to-max threshold (current design) will very "
                f"likely produce false-positive detections on frames without the target."
            )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Check whether GeCo2's raw score separates target-present from target-absent frames"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--num-samples", type=int, default=30,
                    help="frames to sample from each of the present/absent groups")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    from aero_eyes.config import load_config
    from aero_eyes.models.geco2_detector import GeCo2Detector

    cfg = load_config(args.config, args.set)
    # Plain print (not gated by log level, unlike GeCo2Detector's own
    # "GeCo2 loaded from %s" INFO log, which basicConfig(WARNING) above
    # silently swallows) -- makes it unambiguous which checkpoint file this
    # run actually loaded, so a --set stage123_geco2.weights_path=... typo
    # or a stale/reused shell variable is immediately visible in the output
    # instead of producing silently-identical-looking results across runs.
    print(f">>> Loading GeCo2 weights from: {cfg.stage123_geco2.weights_path}")
    detector = GeCo2Detector(cfg)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        check_sample(cfg, sid, detector, args.num_samples)


if __name__ == "__main__":
    main()

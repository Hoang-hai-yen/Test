"""Diagnostic: what similarity ceiling is actually achievable?

Instead of scoring Stage A's candidate boxes (which may or may not land on
the object), this crops the video frame directly at the CONFIRMED
ground-truth box location and scores that crop against the prototype.

This answers a question raised in review: "if you compare the exemplar to
the real object crop (not a candidate), what similarity do you get?" It
isolates whether the low similarity seen in Stage 3 is a fundamental
domain-gap ceiling (GT crop itself scores low) vs. a candidate-cropping or
matching-pipeline problem (GT crop scores much higher than what candidates
ever achieved).

Reads the video directly + the GT annotations + the cached prototype.npz —
does not touch Stage 2/3 output at all.

Usage:
    python -m scripts.check_gt_similarity --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_gt_similarity --config configs/config.yaml   # all samples
    python -m scripts.check_gt_similarity --config configs/config.yaml --stride 5  # subsample GT frames for speed
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import crop_with_pad
from aero_eyes.utils.io import load_gt, read_prototype
from aero_eyes.utils.video import frame_iterator

log = logging.getLogger(__name__)


def check_sample(cfg, sample_id: str, extractor, stride: int) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    proto_path = work_dir / cfg.stage1.prototype.cache_name
    if not proto_path.exists():
        print(f"{sample_id}: prototype.npz not found at {proto_path} — run Stage 1 first.")
        return

    prototype, _, _ = read_prototype(proto_path)

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    gt_frames = sorted(gt.keys())[::stride]
    if not gt_frames:
        print(f"{sample_id}: no GT frames.")
        return
    gt_frame_set = set(gt_frames)

    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        print(f"{sample_id}: no video found under {data_root / sample_id}.")
        return
    video_path = video_files[0]

    pad_ratio = cfg.stage2.candidate.feature_crop_pad
    batch_size = cfg.runtime.batch_size

    crops: list[np.ndarray] = []
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in gt_frame_set:
            continue
        box: Box = gt[frame_idx]
        crops.append(crop_with_pad(frame_bgr, box, pad_ratio=pad_ratio))
        if len(crops) == len(gt_frame_set):
            break

    if not crops:
        print(f"{sample_id}: could not read any GT frames from video.")
        return

    feats = extractor.extract(crops, batch_size=batch_size)
    sims = feats @ prototype

    print(
        f"{sample_id}: GT-crop similarity — n={len(sims)} (stride={stride}) "
        f"min={sims.min():.3f} p50={float(np.percentile(sims, 50)):.3f} "
        f"mean={sims.mean():.3f} max={sims.max():.3f}"
    )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Score the CONFIRMED ground-truth box crop against the prototype "
        "(similarity ceiling), instead of Stage A's candidate boxes."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--stride", type=int, default=1, help="only score every Nth GT frame (speed)")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    from aero_eyes.config import load_config
    from aero_eyes.models.features import build_feature_extractor

    cfg = load_config(args.config, args.set)
    extractor = build_feature_extractor(cfg)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        check_sample(cfg, sid, extractor, args.stride)


if __name__ == "__main__":
    main()

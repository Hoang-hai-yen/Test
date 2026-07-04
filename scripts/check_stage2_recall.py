"""Diagnostic: does Stage 2 (YOLO proposals) ever produce a candidate box that
overlaps the ground-truth object?

This isolates where a "0 detections" result actually comes from:
  - Stage A/2 recall problem: YOLO never proposed anything near the target
    box (no amount of Stage 3 threshold tuning can fix this).
  - Stage B matching problem: a good candidate box exists, but its
    DINOv2/CLIP similarity to the prototype falls below match_threshold
    (tune stage3.match_threshold / adaptive_threshold instead).

Reads candidates.json (Stage 2 output) + the GT annotations file directly —
does not touch Stage 3 at all.

Usage:
    python -m scripts.check_stage2_recall --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_stage2_recall --config configs/config.yaml   # all samples in data_root
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_candidates

log = logging.getLogger(__name__)


def check_sample(cfg, sample_id: str) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    cand_path = work_dir / "candidates.json"
    if not cand_path.exists():
        print(f"{sample_id}: candidates.json not found at {cand_path} — run Stage 2 first.")
        return

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    candidates = read_candidates(cand_path)

    gt_frames = sorted(gt.keys())
    on_keyframe = 0
    best_ious: list[float] = []
    for fi in gt_frames:
        if fi not in candidates:
            continue
        on_keyframe += 1
        gt_box = gt[fi]
        boxes = [d.box for d in candidates[fi]]
        best = max((box_iou(gt_box, b) for b in boxes), default=0.0)
        best_ious.append(best)

    n_gt = len(gt_frames)
    if on_keyframe == 0:
        print(
            f"{sample_id}: 0/{n_gt} GT frames landed on a sampled keyframe "
            f"(keyframe_interval={cfg.stage2.keyframe_interval}) — cannot evaluate "
            f"candidate recall; object may only appear briefly between keyframes."
        )
        return

    hit_03 = sum(1 for i in best_ious if i >= 0.3)
    hit_05 = sum(1 for i in best_ious if i >= 0.5)
    mean_best = sum(best_ious) / len(best_ious)
    max_best = max(best_ious)

    print(
        f"{sample_id}: {on_keyframe}/{n_gt} GT frames on a keyframe | "
        f"best-IoU mean={mean_best:.3f} max={max_best:.3f} | "
        f"frames with a candidate IoU>=0.3: {hit_03}/{on_keyframe} | "
        f">=0.5: {hit_05}/{on_keyframe}"
    )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Check whether Stage 2 candidates ever overlap ground truth "
        "(isolates Stage A/2 recall vs Stage B matching problems)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
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
        check_sample(cfg, sid)


if __name__ == "__main__":
    main()

"""Diagnostic: WHERE in the pipeline does box undersizing (vs ground
truth) originate -- candidate generation, Stage 3's cosine-based
selection, or Stage 4 tracking?

check_box_refine_effect.py found that predicted boxes are already ~23%
smaller in AREA than their matching GT box even BEFORE box_refine touches
them (and IoU ~= area-ratio, meaning position/centering is fine -- the
gap is purely SIZE). This script measures that same area-ratio at 3
checkpoints so the undersizing can be attributed to a specific stage
instead of guessed at:

  1. candidates.json (before Stage 3) -- for each GT-present frame, the
     BEST-matching (highest-IoU-with-GT) candidate among everything GeCo2/
     Stage 2 proposed. This is the best case ACHIEVABLE from the raw
     candidate pool, independent of which one Stage 3 actually picks. If
     this is already undersized, the regression/proposal boxes themselves
     are the root cause (GeCo2's box head, or YOLO/FastSAM).
  2. detections.json (after Stage 3, before tracking) -- prefers
     detections_prerefine.json when present (see run_stage3) so a
     box_refine-enabled run doesn't contaminate this measurement. The box
     Stage 3's cosine-similarity ranking ACTUALLY selected, per keyframe.
     If (1) is well-sized but (2) is undersized, Stage 3's selection
     criterion (similarity score, not IoU) is biased toward smaller boxes
     -- plausible mechanism: a tighter crop embeds "purer" object features
     with less background dilution, scoring higher cosine similarity even
     when it's cutting off part of the true object.
  3. tracks.json (after Stage 4) -- the tracked box on EVERY frame, not
     just keyframes. If (2) is reasonably sized but (3) trends smaller,
     the tracker (e.g. CSRT, which is known to shrink/not re-fit its
     window to the true object shape over time) is compounding the issue
     during propagation between keyframes.

Usage:
    python -m scripts.check_box_size_bias --config configs/config.yaml --sample BlackBox_1
    python -m scripts.check_box_size_bias --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_candidates, read_detections, read_tracks

log = logging.getLogger(__name__)


def _report(label: str, ratios: list[float]) -> None:
    if not ratios:
        print(f"  {label}: no GT-checkable boxes found.")
        return
    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)
    mean_r = sum(ratios_sorted) / n
    median_r = ratios_sorted[n // 2]
    print(
        f"  {label}: n={n}  mean area-ratio={mean_r:.2f}  median={median_r:.2f}  "
        f"(1.0 = same size as GT; <1 = smaller than GT; >1 = larger than GT)"
    )


def check_sample(cfg, sample_id: str) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    cand_path = work_dir / "candidates.json"
    prerefine_path = work_dir / "detections_prerefine.json"
    det_path = prerefine_path if prerefine_path.exists() else work_dir / "detections.json"
    tracks_path = work_dir / "tracks.json"

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    print(f"\n=== {sample_id} ===")
    means = {}

    if not cand_path.exists():
        print(f"  candidates.json not found at {cand_path} -- skipping candidate-pool check.")
    else:
        candidates = read_candidates(cand_path)
        ratios = []
        for fi, dets in candidates.items():
            if fi not in gt or not dets:
                continue
            gt_box = gt[fi]
            gt_area = gt_box.area()
            if gt_area <= 0:
                continue
            best = max(dets, key=lambda d: box_iou(gt_box, d.box))
            ratios.append(best.box.area() / gt_area)
        _report("1. candidates.json  (best-IoU candidate per frame -- best case achievable)", ratios)
        if ratios:
            means["candidates"] = sum(ratios) / len(ratios)

    if not det_path.exists():
        print(f"  detections.json not found at {det_path} -- skipping Stage 3 selection check.")
    else:
        label_suffix = " [detections_prerefine.json, box_refine-free]" if det_path == prerefine_path else ""
        detections = read_detections(det_path)
        ratios = []
        for fi, dets in detections.items():
            if fi not in gt or not dets:
                continue
            gt_box = gt[fi]
            gt_area = gt_box.area()
            if gt_area <= 0:
                continue
            # Same selection rule Stage 4 uses to seed the tracker from a
            # keyframe: the single highest-similarity detection.
            best = max(dets, key=lambda d: d.similarity)
            ratios.append(best.box.area() / gt_area)
        _report(f"2. detections.json  (Stage 3's actually-selected box per keyframe){label_suffix}", ratios)
        if ratios:
            means["detections"] = sum(ratios) / len(ratios)

    if not tracks_path.exists():
        print(f"  tracks.json not found at {tracks_path} -- skipping Stage 4 tracking check.")
    else:
        tracks = read_tracks(tracks_path)
        ratios = []
        for fi, box in tracks.items():
            if box is None or fi not in gt:
                continue
            gt_area = gt[fi].area()
            if gt_area <= 0:
                continue
            ratios.append(box.area() / gt_area)
        _report("3. tracks.json      (tracked box, EVERY frame, not just keyframes)", ratios)
        if ratios:
            means["tracks"] = sum(ratios) / len(ratios)

    # Attribute WHERE the biggest drop happens, if we have >=2 checkpoints.
    order = [k for k in ("candidates", "detections", "tracks") if k in means]
    if len(order) >= 2:
        drops = []
        for a, b in zip(order, order[1:]):
            drops.append((f"{a} -> {b}", means[a] - means[b]))
        worst_step, worst_drop = max(drops, key=lambda x: x[1])
        if worst_drop > 0.05:
            print(f"  -> largest undersizing drop happens at: {worst_step} (-{worst_drop:.2f} area-ratio)")
        else:
            print("  -> no single step shows a clear drop (>0.05) -- undersizing may already be "
                  "present from the very first checkpoint measured, or spread evenly.")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Locate which pipeline stage introduces box undersizing vs ground truth."
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

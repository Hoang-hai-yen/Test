"""Diagnostic: does box_refine ACTUALLY change detections.json's boxes,
and does it help or hurt against ground truth -- measured DIRECTLY,
bypassing run_all's caching entirely, so a "no visible effect" result from
a full pipeline re-run can never be silently confused with "the feature
does nothing".

Reads detections.json + the video + ground truth, and re-applies
box_refine.* from the CURRENT config -- via the EXACT SAME
aero_eyes.utils.box_refine.refine_box() call Stage 3 itself uses -- to
every box that has a frame with GT available. Does not touch or depend on
any cached detections.json/tracks.json from a previous box_refine setting;
always reflects whatever box_refine.* is set to right now.

Reports, per sample:
  - how many boxes box_refine actually CHANGED vs left as-is (rejected by
    min_iou_with_original, or the segmenter found nothing plausible)
  - among GT-checkable boxes, how many got a BETTER, WORSE, or unchanged
    GT-IoU after refinement
  - mean GT-IoU before vs after

If this script shows real (or predominantly negative) changes while a
full run_all re-run showed IDENTICAL numbers to a previous run, that is
strong evidence the full re-run reused a CACHED detections.json/tracks.json
(project.use_cache=true, or --from-stage skipped Stage 3) rather than
box_refine having no effect.

Usage:
    python -m scripts.check_box_refine_effect --config configs/config.yaml --sample BlackBox_1
    python -m scripts.check_box_refine_effect --config configs/config.yaml --set box_refine.enabled=true --set box_refine.method=sam
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.utils.box_refine import refine_box
from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_detections

log = logging.getLogger(__name__)


def check_sample(cfg, sample_id: str) -> None:
    from aero_eyes.utils.video import read_frame

    work_dir = Path(cfg.project.work_dir) / sample_id
    det_path = work_dir / "detections.json"

    print(f"\n=== {sample_id} ===")

    if not cfg.box_refine.enabled:
        print("  box_refine.enabled is false in this config -- nothing to test "
              "(pass --set box_refine.enabled=true).")
        return
    if not det_path.exists():
        print(f"  detections.json not found at {det_path} -- run Stage 3 first.")
        return

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"  not found in {gt_file}, skipping.")
        return

    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        print(f"  no video found under {data_root / sample_id}.")
        return
    video_path = video_files[0]

    detections = read_detections(det_path)

    br_cfg = cfg.box_refine
    segmenter = None
    if br_cfg.method == "sam":
        from aero_eyes.models.segmentation import MobileSAMSegmenter
        segmenter = MobileSAMSegmenter(weights_path=cfg.stage1.segmentation.weights)
        if not segmenter._available:
            print("  warning: MobileSAM unavailable (weights missing/failed to load) -- "
                  "every box will silently fall back UNCHANGED. This alone can explain "
                  "a full pipeline run showing zero effect.")

    n_total = n_changed = 0
    n_better = n_worse = n_same = 0
    iou_before_sum = 0.0
    iou_after_sum = 0.0

    for frame_idx, dets in detections.items():
        if frame_idx not in gt:
            continue  # can only judge refinement quality where GT exists
        gt_box = gt[frame_idx]
        try:
            frame_bgr = read_frame(video_path, frame_idx)
        except Exception:
            continue

        for det in dets:
            n_total += 1
            iou_before = box_iou(gt_box, det.box)
            refined = refine_box(
                br_cfg.method, frame_bgr, det.box, br_cfg.context_margin,
                segmenter=segmenter, min_iou_with_original=br_cfg.min_iou_with_original,
            )
            changed = (
                refined.x1 != det.box.x1 or refined.y1 != det.box.y1
                or refined.x2 != det.box.x2 or refined.y2 != det.box.y2
            )
            if changed:
                n_changed += 1
            iou_after = box_iou(gt_box, refined)
            iou_before_sum += iou_before
            iou_after_sum += iou_after
            if iou_after > iou_before + 1e-6:
                n_better += 1
            elif iou_after < iou_before - 1e-6:
                n_worse += 1
            else:
                n_same += 1

    if n_total == 0:
        print("  no GT-checkable boxes found (no detections.json frame overlaps GT) -- nothing to report.")
        return

    print(
        f"  {n_total} GT-checkable boxes -- box_refine CHANGED {n_changed} "
        f"({100.0 * n_changed / n_total:.0f}%), left {n_total - n_changed} unchanged "
        f"(rejected by min_iou_with_original={br_cfg.min_iou_with_original}, or no "
        f"plausible mask found)."
    )
    print(
        f"  Mean GT-IoU: before={iou_before_sum / n_total:.3f} -> after={iou_after_sum / n_total:.3f}"
    )
    print(
        f"  Of changed boxes: BETTER={n_better} ({100.0 * n_better / n_total:.0f}%)  "
        f"WORSE={n_worse} ({100.0 * n_worse / n_total:.0f}%)  "
        f"SAME_IOU={n_same - (n_total - n_changed)} (changed but IoU tied)"
    )
    if n_worse > n_better:
        print(
            "  -> box_refine is making localization WORSE on this sample more often than "
            "better. Not a caching artifact -- the refinement itself is the problem here."
        )
    elif n_changed == 0:
        print(
            "  -> box_refine changed NOTHING. If you expected an effect, this points to "
            "min_iou_with_original rejecting every candidate, or the segmenter (SAM "
            "unavailable / GrabCut failing) silently falling back every time -- check the "
            "warning above, or lower min_iou_with_original to see raw (ungated) behavior."
        )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Measure box_refine's real effect on detections.json vs ground truth, "
        "bypassing run_all's caching."
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

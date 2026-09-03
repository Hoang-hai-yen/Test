"""Diagnostic: Precision/Recall/F1 of candidates.json (BEFORE Stage 3 cosine
matching) vs detections.json (AFTER Stage 3 cosine matching), against ground
truth -- shows exactly what cosine matching trades: how much recall it gives
up in exchange for how much precision gained.

Works on candidates.json from EITHER pipeline (legacy Stage 2, or GeCo2's
run_stage12_geco2_candidates when stage123_geco2.cosine_rescore.enabled) --
both write the same schema.

Per-frame confusion matrix, over frames actually PROCESSED (present as a key
in the file being evaluated -- i.e. sampled as a keyframe and given a chance):
  GT present + best candidate IoU >= threshold        -> TP
  GT present + best candidate IoU <  threshold
               (including 0 candidates on that frame) -> FN
  GT absent  + >=1 candidate on that frame             -> FP
  GT absent  + 0 candidates on that frame              -> TN

  Recall    = TP / (TP + FN)
  Precision = TP / (TP + FP)
  F1        = 2 * P * R / (P + R)

GT frames that never landed on a processed keyframe at all (keyframe_interval
sampling gap, not a candidates/cosine decision) are reported separately, not
folded into FN.

Usage:
    python -m scripts.check_cosine_effect --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_cosine_effect --config configs/config.yaml --iou-threshold 0.3
    python -m scripts.check_cosine_effect --config configs/config.yaml   # all samples in data_root
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.types import Box, Detection
from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_candidates, read_detections

log = logging.getLogger(__name__)


def compute_prf1(
    dets_by_frame: dict[int, list[Detection]],
    gt: dict[int, Box],
    iou_threshold: float,
) -> dict:
    """Per-frame confusion matrix + Precision/Recall/F1 for one
    frame_idx -> [Detection] mapping against one frame_idx -> Box ground
    truth. See module docstring for exact TP/FN/FP/TN semantics."""
    tp = fn = fp = tn = 0
    for fi, dets in dets_by_frame.items():
        boxes = [d.box for d in dets]
        has_detection = len(boxes) > 0
        if fi in gt:
            best_iou = max((box_iou(gt[fi], b) for b in boxes), default=0.0)
            if best_iou >= iou_threshold:
                tp += 1
            else:
                fn += 1
        else:
            if has_detection:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    n_processed_gt_frames = tp + fn
    n_gt_total = len(gt)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_processed_gt_frames": n_processed_gt_frames,
        "n_gt_total": n_gt_total,
        "unsampled_gt_frames": n_gt_total - n_processed_gt_frames,
    }


def _print_result(label: str, r: dict) -> None:
    print(
        f"  {label}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
        f"(TP={r['tp']} FN={r['fn']} FP={r['fp']} TN={r['tn']}, "
        f"{r['n_processed_gt_frames']}/{r['n_gt_total']} GT frames processed)"
    )
    if r["unsampled_gt_frames"] > 0:
        print(
            f"    note: {r['unsampled_gt_frames']} GT frame(s) never landed on a "
            "processed keyframe -- not counted above (keyframe_interval sampling gap, "
            "not this stage's decision)."
        )


def check_sample(cfg, sample_id: str, iou_threshold: float) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    cand_path = work_dir / "candidates.json"
    det_path = work_dir / "detections.json"

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) ===")

    if not cand_path.exists():
        print(f"  BEFORE cosine (candidates.json): not found at {cand_path} -- "
              "run Stage 2 (legacy) or the GeCo2 candidate stage first.")
    else:
        r = compute_prf1(read_candidates(cand_path), gt, iou_threshold)
        _print_result("BEFORE cosine (candidates.json)", r)

    if not det_path.exists():
        print(f"  AFTER  cosine (detections.json): not found at {det_path} -- run Stage 3 first.")
    else:
        r = compute_prf1(read_detections(det_path), gt, iou_threshold)
        _print_result("AFTER  cosine (detections.json)", r)


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Precision/Recall/F1 of candidates.json (before Stage 3 cosine "
        "matching) vs detections.json (after) against ground truth."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as a correct localization (default: 0.5)")
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

"""Diagnostic: is Stage 4's tracker itself MISSING coverage it should have
("track bỏ sót" -- MISSING_PRED: GT present, no prediction) or
OVER-tracking past when the object is really there ("track dư" --
MISSING_GT: prediction present, GT absent) -- and is the root cause the
TRACKER's own behavior, or something it simply inherited from Stage 3's
detections?

Reuses classify_frames()/_print_breakdown() from check_st_iou_breakdown.py
(imported, not reimplemented) for the headline MISSING_PRED/MISSING_GT
counts against tracks.json (raw Stage 4, before Stage 5). Then attributes
each one using detections.json (Stage 3's own keyframe boxes) --
tracks.json alone can't do this: it stores only frame_idx -> Box|None, no
"detect" vs "track" source tag (see aero_eyes/utils/io.py::write_tracks),
so the ONLY way to tell "did the tracker lose a lock Stage 3 correctly
gave it" from "Stage 3 never gave it anything good to begin with" is to
cross-reference the keyframe detections directly.

MISSING_PRED (track bỏ sót) attribution -- for each GT-present frame with
no prediction, looks at the most recent PRECEDING keyframe (from
detections.json's own key set) Stage 4 would have (re-)initialized the
tracker from:
  tracker_lost_lock -- that keyframe's own detection box was GOOD
                        (IoU-vs-GT >= --iou-threshold) -- Stage 3 handed
                        the tracker the right box, and it still lost the
                        object before reaching this frame. Stage 4's own
                        fault (drift, occlusion handling, max_track_age,
                        etc).
  detection_gap     -- that keyframe's own detection was missing/bad, or
                        no keyframe has been reached yet -- Stage 3 never
                        gave the tracker anything good to hold here. NOT
                        (primarily) Stage 4's fault.

MISSING_GT (track dư) attribution -- groups consecutive MISSING_GT frames
into runs, and checks whether each run starts within --drift-gap frames
of GT last being present:
  post_departure_drift    -- the run starts right as GT disappears --
                              consistent with the tracker (or a stale
                              re-detect) continuing to follow the object,
                              or its own already-drifted lock, past when
                              it actually left. The exact failure mode
                              stage4.verify_interval was built to catch.
  unrelated_false_positive -- the run appears with no recent GT presence
                              nearby -- a stray re-detect or hallucinated
                              lock unrelated to the object ever having
                              just left (e.g. a confuser object).

Read-only.

Usage:
    python -m scripts.check_tracker_coverage --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_tracker_coverage --config configs/config.yaml --iou-threshold 0.3 --drift-gap 3
    python -m scripts.check_tracker_coverage --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import bisect
import logging
from pathlib import Path

from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_detections, read_tracks
from scripts.check_st_iou_breakdown import _print_breakdown, classify_frames

log = logging.getLogger(__name__)


def _attribute_missing_pred(
    gt: dict, pred_tube: dict, keyframes_sorted: list[int],
    detections: dict, iou_threshold: float,
) -> dict:
    """For every GT-present frame with no prediction, classify it as
    tracker_lost_lock or detection_gap -- see module docstring."""
    n_lost_lock = n_detection_gap = 0
    examples_lost: list[tuple] = []
    examples_gap: list[tuple] = []

    for fi in sorted(gt.keys()):
        if fi in pred_tube:
            continue
        idx = bisect.bisect_right(keyframes_sorted, fi) - 1
        if idx < 0:
            n_detection_gap += 1
            if len(examples_gap) < 5:
                examples_gap.append((fi, None))
            continue
        kf = keyframes_sorted[idx]
        kf_dets = detections.get(kf, [])
        kf_iou = max((box_iou(gt[kf], d.box) for d in kf_dets), default=0.0) if kf in gt else 0.0
        if kf in gt and kf_iou >= iou_threshold:
            n_lost_lock += 1
            if len(examples_lost) < 5:
                examples_lost.append((fi, kf, kf_iou))
        else:
            n_detection_gap += 1
            if len(examples_gap) < 5:
                examples_gap.append((fi, kf))

    return {
        "n_lost_lock": n_lost_lock, "n_detection_gap": n_detection_gap,
        "examples_lost": examples_lost, "examples_gap": examples_gap,
    }


def _find_runs(frames: list[int]) -> list[tuple[int, int]]:
    """Group a sorted list of frame indices into maximal consecutive runs."""
    if not frames:
        return []
    runs = []
    start = prev = frames[0]
    for fi in frames[1:]:
        if fi == prev + 1:
            prev = fi
        else:
            runs.append((start, prev))
            start = prev = fi
    runs.append((start, prev))
    return runs


def _attribute_missing_gt(gt: dict, pred_tube: dict, drift_gap: int) -> dict:
    """Group consecutive MISSING_GT frames into runs and classify each as
    post_departure_drift or unrelated_false_positive -- see module docstring."""
    missing_gt_frames = sorted(fi for fi in pred_tube if fi not in gt)
    runs = _find_runs(missing_gt_frames)
    gt_frames_sorted = sorted(gt.keys())

    n_drift = n_stray = 0
    drift_runs: list[tuple[int, int]] = []
    stray_runs: list[tuple[int, int]] = []
    for start, end in runs:
        idx = bisect.bisect_left(gt_frames_sorted, start) - 1
        if idx >= 0 and start - gt_frames_sorted[idx] <= drift_gap:
            n_drift += 1
            drift_runs.append((start, end))
        else:
            n_stray += 1
            stray_runs.append((start, end))

    return {
        "n_runs": len(runs), "n_frames": len(missing_gt_frames),
        "n_drift": n_drift, "n_stray": n_stray,
        "drift_runs": drift_runs, "stray_runs": stray_runs,
    }


def check_sample(cfg, sample_id: str, iou_threshold: float, drift_gap: int) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    tracks_path = work_dir / "tracks.json"
    det_path = work_dir / "detections.json"

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}, drift_gap={drift_gap}) ===")

    if not tracks_path.exists():
        print(f"  tracks.json not found at {tracks_path} -- run Stage 4 first.")
        return

    raw_tracks = read_tracks(tracks_path)
    pred_tube = {fi: b for fi, b in raw_tracks.items() if b is not None}
    r = classify_frames(pred_tube, gt, iou_threshold)
    _print_breakdown("tracks.json (raw Stage 4)", r)

    if not det_path.exists():
        print("  detections.json not found -- cannot attribute MISSING_PRED/MISSING_GT to "
              "the tracker vs Stage 3's own detections; run Stage 3 first for the full breakdown.")
        return

    detections = read_detections(det_path)
    keyframes_sorted = sorted(detections.keys())

    if r["missing_pred"] > 0:
        mp = _attribute_missing_pred(gt, pred_tube, keyframes_sorted, detections, iou_threshold)
        total = mp["n_lost_lock"] + mp["n_detection_gap"]
        print(f"  MISSING_PRED (track bỏ sót) attribution -- {total} frame(s):")
        print(
            f"    tracker_lost_lock (Stage 3 had a good box, tracker lost it anyway) = "
            f"{mp['n_lost_lock']} ({100.0 * mp['n_lost_lock'] / total:.0f}%)"
        )
        print(
            f"    detection_gap (Stage 3 never had a good box here)                  = "
            f"{mp['n_detection_gap']} ({100.0 * mp['n_detection_gap'] / total:.0f}%)"
        )
        for fi, kf, kf_iou in mp["examples_lost"]:
            print(f"      e.g. frame {fi}: last keyframe {kf} had detection IoU={kf_iou:.3f} "
                  f"(>= threshold) yet the tracker lost it by frame {fi}")
        if mp["n_lost_lock"] > mp["n_detection_gap"]:
            print("    -> majority is the TRACKER's own fault (lost a good lock) -- "
                  "stage4.verify_interval, tracker_conf_threshold, or max_track_age tuning "
                  "is the relevant lever.")
        elif mp["n_detection_gap"] > 0:
            print("    -> majority is inherited from Stage 3 (never had a good box to give "
                  "the tracker) -- fixing this needs detection-side work (see "
                  "check_cosine_recall_loss.py), not tracker tuning.")

    if r["missing_gt"] > 0:
        mg = _attribute_missing_gt(gt, pred_tube, drift_gap)
        print(f"  MISSING_GT (track dư) attribution -- {mg['n_runs']} contiguous run(s), "
              f"{mg['n_frames']} frame(s) total:")
        print(
            f"    post_departure_drift (starts within {drift_gap} frame(s) of GT last seen)  = "
            f"{mg['n_drift']} run(s)"
        )
        print(
            f"    unrelated_false_positive (no recent GT presence nearby)                    = "
            f"{mg['n_stray']} run(s)"
        )
        longest = sorted(mg["drift_runs"] + mg["stray_runs"], key=lambda run: run[1] - run[0], reverse=True)[:5]
        for start, end in longest:
            kind = "post_departure_drift" if (start, end) in mg["drift_runs"] else "unrelated_false_positive"
            print(f"      e.g. frames {start}-{end} ({end - start + 1} frame(s), {kind})")
        if mg["n_drift"] > mg["n_stray"]:
            print("    -> majority is the tracker CONTINUING past when the object actually "
                  "left -- stage4.verify_interval (if not already enabled) or a stricter "
                  "tracker_conf_threshold is the relevant lever.")
        elif mg["n_stray"] > 0:
            print("    -> majority is unrelated to a real departure -- likely a confuser "
                  "object or a bad re-detect fallback, not ordinary tracking drift; see "
                  "check_dynamic_prototype_purity.py / check_multi_ref_agreement.py.")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Check whether Stage 4's tracker is missing coverage (track bỏ sót) or "
        "over-tracking (track dư), and attribute each to the tracker vs Stage 3's detections."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as MATCH / a good keyframe box (default: 0.5)")
    p.add_argument("--drift-gap", type=int, default=2,
                    help="A MISSING_GT run starting within this many frames of GT last being "
                    "present counts as post_departure_drift rather than unrelated_false_positive "
                    "(default: 2)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        check_sample(cfg, sid, args.iou_threshold, args.drift_gap)


if __name__ == "__main__":
    main()

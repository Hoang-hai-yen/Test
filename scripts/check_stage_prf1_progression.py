"""Diagnostic: Precision/Recall/F1 tracked across EVERY pipeline stage in
sequence -- candidates.json (Stage 1+2, or Stage123-GeCo2's WIDE candidate
generation when stage123_geco2.cosine_rescore.enabled) -> detections.json
(Stage 3 cosine matching, or Stage123-GeCo2 merged output when
cosine_rescore is OFF -- see stage123_geco2.py's own module docstring:
that path writes straight to detections.json, no separate candidate
stage) -> tracks.json (Stage 4) -> submission.json (Stage 5) -- all
against ground truth, at the SAME --iou-threshold, printed as one
progression instead of having to run check_cosine_effect.py and
check_st_iou_breakdown.py separately and mentally line up their numbers.

Reuses compute_prf1()/_print_result() from check_cosine_effect.py
(imported, not reimplemented) for every stage -- same TP/FN/FP/TN
semantics, same survivorship-bias fix (an EXPLICIT processed_frames set,
never a stage's own key set when that stage can silently omit a key for
"nothing survived here" -- see check_cosine_effect.py's own docstring).

Frame domains genuinely differ between stages, and this script keeps them
honest rather than blurring them together:
  candidates.json / detections.json -- KEYFRAME-ONLY (only every
    keyframe_interval-th frame is even attempted). processed_frames =
    candidates.json's own key set (every attempted keyframe, even an
    empty one) when it exists; detections.json alone cannot supply this.
  tracks.json / submission.json -- EVERY frame in the video is attempted
    (Stage 4 iterates frame-by-frame and writes an entry -- possibly None
    -- for each one, so tracks.json's key set is NOT survivorship-biased
    the way detections.json's is). submission.json's OWN key set DOES
    have the same bias detections.json has (a SpatioTemporalTube only
    stores PRESENT frames -- an absent frame is an omitted key, not an
    empty one) -- so submission.json is evaluated against tracks.json's
    processed_frames too, not its own.

Because the keyframe-only stages and the every-frame stages cover
different frame counts by design, don't compare their raw TP/FN counts
directly -- compare each stage's RATES (P/R/F1) and watch which stage's
rate takes the biggest hit. For a continuous (non-binary-threshold) view
of the SAME progression, especially between tracks.json and
submission.json, see check_st_iou_breakdown.py instead.

Usage:
    python -m scripts.check_stage_prf1_progression --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_stage_prf1_progression --config configs/config.yaml --iou-threshold 0.3
    python -m scripts.check_stage_prf1_progression --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.evaluate import _load_submission
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import load_gt, read_candidates, read_detections, read_tracks
from scripts.check_cosine_effect import _print_result, compute_prf1

log = logging.getLogger(__name__)


def _tube_to_dets(tube: dict[int, Box | None]) -> dict[int, list[Detection]]:
    """Adapt a frame_idx -> Box|None tube (tracks.json/submission.json's
    own shape) into compute_prf1's dict[int, list[Detection]] input --
    None/absent frames simply contribute no Detection, i.e. 0 boxes."""
    return {
        fi: [Detection(frame_idx=fi, box=box, similarity=1.0, source="x")]
        for fi, box in tube.items() if box is not None
    }


def check_sample(cfg, sample_id: str, iou_threshold: float) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    cand_path = work_dir / "candidates.json"
    det_path = work_dir / "detections.json"
    tracks_path = work_dir / "tracks.json"
    submission_path = work_dir / cfg.data.submission.path_name

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) ===")

    # ---- keyframe-only stages: candidates.json -> detections.json ----
    keyframe_processed: set[int] | None = None
    if cand_path.exists():
        candidates = read_candidates(cand_path)
        keyframe_processed = set(candidates.keys())
        r = compute_prf1(candidates, gt, iou_threshold, processed_frames=keyframe_processed)
        _print_result("Stage 1+2 candidates.json (before cosine matching)", r)
    else:
        print("  candidates.json: not found -- either not run yet, or "
              "stage123_geco2.cosine_rescore is OFF (GeCo2 merges Stage 1+2+3 with no "
              "separate candidate stage; see detections.json below instead).")

    if not det_path.exists():
        print("  detections.json: not found -- run Stage 3 (or Stage123-GeCo2) first.")
    else:
        detections = read_detections(det_path)
        if keyframe_processed is None:
            keyframe_processed = set(detections.keys())
            print("    warning: candidates.json unavailable -- FN count below may be "
                  "undercounted (see check_cosine_effect.py's own docstring).")
        label = ("Stage 3 detections.json (after cosine matching)" if cand_path.exists()
                 else "Stage123-GeCo2 detections.json (merged Stage 1+2+3, cosine_rescore off)")
        r = compute_prf1(detections, gt, iou_threshold, processed_frames=keyframe_processed)
        _print_result(label, r)

    # ---- every-frame stages: tracks.json -> submission.json ----
    if not tracks_path.exists():
        print("  tracks.json: not found -- run Stage 4 first.")
        return

    raw_tracks = read_tracks(tracks_path)
    # Stage 4 writes an entry (possibly None) for EVERY frame it iterates --
    # unlike detections.json, this key set is not survivorship-biased.
    dense_processed = set(raw_tracks.keys())
    r = compute_prf1(_tube_to_dets(raw_tracks), gt, iou_threshold, processed_frames=dense_processed)
    _print_result("Stage 4 tracks.json (raw tracking, every frame)", r)

    if not submission_path.exists():
        print("  submission.json: not found -- run Stage 5 first.")
        return

    all_submissions = _load_submission(submission_path)
    pred_tube = all_submissions.get(sample_id, {})
    r = compute_prf1(_tube_to_dets(pred_tube), gt, iou_threshold, processed_frames=dense_processed)
    _print_result("Stage 5 submission.json (final, every frame)", r)


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Track Precision/Recall/F1 across every pipeline stage: candidates -> "
        "detections -> tracks -> submission, against ground truth."
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

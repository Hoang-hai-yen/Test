"""Integration test for scripts.check_tracker_coverage -- writes real
tracks.json + detections.json + a GT file to a temp dir, covering all 4
attribution outcomes (tracker_lost_lock, detection_gap,
post_departure_drift, unrelated_false_positive), and checks the printed
counts match what was constructed."""
from __future__ import annotations

import json

from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import write_detections, write_tracks


def _gt_file(tmp_path, sample_id: str, gt_boxes: dict[int, Box]):
    path = tmp_path / "gt.json"
    bboxes = [
        {"frame": fi, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
        for fi, b in gt_boxes.items()
    ]
    data = [{"video_id": sample_id, "annotations": [{"bboxes": bboxes}]}]
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _cfg(tmp_path, gt_path):
    from aero_eyes.config import AeroEyesConfig
    return AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"gt": {"global_file": str(gt_path)}},
    })


def _det(fi: int, box: Box) -> Detection:
    return Detection(frame_idx=fi, box=box, similarity=1.0, source="detect")


def test_attributes_missing_pred_and_missing_gt_correctly(tmp_path, capsys):
    from scripts.check_tracker_coverage import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    bad_box = Box(500, 500, 510, 510)

    # GT: object present on frames 0-1 (then leaves), and again on frames 10-11.
    gt_boxes = {0: gt_box, 1: gt_box, 10: gt_box, 11: gt_box}
    gt_path = _gt_file(tmp_path, sample_id, gt_boxes)
    cfg = _cfg(tmp_path, gt_path)
    work_dir = tmp_path / "runs" / sample_id

    # detections.json (keyframes): keyframe 0 had a GOOD box (sets up
    # "tracker_lost_lock" for frame 1); keyframe 10 had a BAD box (sets up
    # "detection_gap" for frame 11, since Stage 3 never gave a good box there).
    write_detections(
        {0: [_det(0, gt_box)], 10: [_det(10, bad_box)]},
        work_dir / "detections.json", threshold=0.5,
    )

    # tracks.json (every frame 0-19):
    #   frame 0: good track (matches GT) -- MATCH
    #   frame 1: GT present, tracker has NOTHING -> MISSING_PRED. Last
    #            keyframe (0) had a GOOD box -> tracker_lost_lock.
    #   frames 2-3: tracker keeps outputting the OLD box after GT left ->
    #            MISSING_GT, starts right after GT was last seen (frame 1)
    #            -> post_departure_drift.
    #   frames 4-8: no prediction, GT absent -> TN, not interesting.
    #   frame 9: a stray prediction with GT absent and NOT close to any GT
    #            presence (GT frames are 0,1,10,11; frame 9 is 2 away from
    #            frame 11 but that's in the FUTURE, not the PAST -- so this
    #            is classified relative to the most recent PAST GT frame,
    #            frame 1, which is 8 frames back) -> unrelated_false_positive.
    #   frame 10: no prediction at all, GT present, last keyframe (10) had a
    #            BAD box -> detection_gap.
    #   frame 11: GT present, MATCH (say the tracker/re-detect got it right here).
    tracks = {
        0: gt_box, 1: None,
        2: gt_box, 3: gt_box,
        4: None, 5: None, 6: None, 7: None, 8: None,
        9: bad_box,
        10: None,
        11: gt_box,
    }
    for fi in range(12, 20):
        tracks[fi] = None
    write_tracks(tracks, work_dir / "tracks.json")

    check_sample(cfg, sample_id, iou_threshold=0.5, drift_gap=2)

    out = capsys.readouterr().out
    assert "MISSING_PRED (track bỏ sót) attribution -- 2 frame(s):" in out
    assert "tracker_lost_lock (Stage 3 had a good box, tracker lost it anyway) = 1 (50%)" in out
    assert "detection_gap (Stage 3 never had a good box here)                  = 1 (50%)" in out

    assert "MISSING_GT (track dư) attribution -- 2 contiguous run(s), 3 frame(s) total:" in out
    assert "post_departure_drift (starts within 2 frame(s) of GT last seen)  = 1 run(s)" in out
    assert "unrelated_false_positive (no recent GT presence nearby)                    = 1 run(s)" in out

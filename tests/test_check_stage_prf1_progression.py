"""Integration test for scripts.check_stage_prf1_progression -- writes
real candidates.json/detections.json/tracks.json/submission.json + a GT
file to a temp dir, constructed so each stage's recall visibly changes
(candidates/detections miss frame 1 and 2; Stage 4 tracking recovers
frame 1 but loses frame 2; Stage 5 gap-fills frame 2 too), and checks the
printed P/R numbers match what was constructed at each stage."""
from __future__ import annotations

import json

from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import write_candidates, write_detections, write_submission, write_tracks


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


def test_prf1_progression_reflects_each_stages_own_recall(tmp_path, capsys):
    from scripts.check_stage_prf1_progression import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    far_box = Box(500, 500, 510, 510)
    gt_boxes = {0: gt_box, 1: gt_box, 2: gt_box}  # object present on frames 0, 1, 2
    gt_path = _gt_file(tmp_path, sample_id, gt_boxes)
    cfg = _cfg(tmp_path, gt_path)
    work_dir = tmp_path / "runs" / sample_id

    # candidates.json: only frame 0 has a good box; frame 1 empty (processed,
    # nothing survived), frame 2 has a wrong-location box -> recall 1/3.
    write_candidates(
        {0: [_det(0, gt_box)], 1: [], 2: [_det(2, far_box)]},
        work_dir / "candidates.json",
    )
    # detections.json: cosine matching didn't change anything here -- same 1/3.
    write_detections({0: [_det(0, gt_box)]}, work_dir / "detections.json", threshold=0.5)

    # tracks.json: EVERY frame of a 5-frame video. Stage 4 tracking RECOVERS
    # frame 1 (propagated from frame 0's detect) but loses frame 2 -> recall 2/3.
    write_tracks(
        {0: gt_box, 1: gt_box, 2: None, 3: None, 4: None},
        work_dir / "tracks.json",
    )
    # submission.json: Stage 5 gap-fills the 1-frame hole at frame 2 -> recall 3/3.
    write_submission({0: gt_box, 1: gt_box, 2: gt_box}, sample_id, work_dir / "submission.json")

    check_sample(cfg, sample_id, iou_threshold=0.5)

    out = capsys.readouterr().out
    assert "Stage 1+2 candidates.json (before cosine matching): P=1.000 R=0.333" in out
    assert "Stage 3 detections.json (after cosine matching): P=1.000 R=0.333" in out
    assert "Stage 4 tracks.json (raw tracking, every frame): P=1.000 R=0.667" in out
    assert "Stage 5 submission.json (final, every frame): P=1.000 R=1.000" in out


def test_missing_candidates_json_falls_back_to_detections_only(tmp_path, capsys):
    """cosine_rescore is OFF -> stage123_geco2.py writes straight to
    detections.json, no candidates.json at all -- the script must handle
    this by labeling detections.json as the merged Stage123-GeCo2 output
    and using its own key set (with the documented undercount caveat),
    not crash or silently skip it."""
    from scripts.check_stage_prf1_progression import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_path = _gt_file(tmp_path, sample_id, {0: gt_box})
    cfg = _cfg(tmp_path, gt_path)
    work_dir = tmp_path / "runs" / sample_id

    write_detections({0: [_det(0, gt_box)]}, work_dir / "detections.json", threshold=0.5)

    check_sample(cfg, sample_id, iou_threshold=0.5)

    out = capsys.readouterr().out
    assert "candidates.json: not found" in out
    assert "Stage123-GeCo2 detections.json (merged Stage 1+2+3, cosine_rescore off): P=1.000 R=1.000" in out
    assert "tracks.json: not found" in out

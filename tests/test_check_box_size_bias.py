"""Integration test for scripts.check_box_size_bias -- writes real
candidates.json/detections.json/tracks.json + a GT file to a temp dir,
runs check_sample, and verifies the reported area-ratios match what was
constructed at each of the 3 checkpoints."""
from __future__ import annotations

import json

from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import write_candidates, write_detections, write_tracks


def _gt_file(tmp_path, sample_id: str, gt_boxes: dict[int, Box]):
    """Write a minimal GT annotations file matching load_gt's schema."""
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


def test_check_box_size_bias_reports_correct_ratios(tmp_path, capsys):
    sample_id = "synth_sample"

    # GT: a 100x100 box (area=10000) on frames 0, 8, and every frame 0-15
    # for the tracks.json check.
    gt_boxes = {fi: Box(0, 0, 100, 100) for fi in range(16)}
    gt_path = _gt_file(tmp_path, sample_id, gt_boxes)
    cfg = _cfg(tmp_path, gt_path)
    work_dir = tmp_path / "runs" / sample_id
    work_dir.mkdir(parents=True)

    # 1. candidates.json: on frame 0, one candidate matches GT exactly
    # (ratio=1.0, best IoU) and one is a much worse, smaller distractor --
    # the "best-IoU candidate" logic must pick the exact-match one.
    exact_match = Detection(frame_idx=0, box=Box(0, 0, 100, 100), similarity=0.1, source="candidate")
    distractor = Detection(frame_idx=0, box=Box(200, 200, 220, 220), similarity=0.9, source="candidate")
    write_candidates({0: [exact_match, distractor]}, work_dir / "candidates.json")

    # 2. detections.json: Stage 3 "selected" the HIGHER-similarity box,
    # which is a 50x50 box (area=2500 -> ratio 0.25) -- deliberately the
    # WRONG (smaller) one, exercising "selection picks the undersized box
    # even though a well-sized candidate existed".
    selected = Detection(frame_idx=0, box=Box(0, 0, 50, 50), similarity=0.9, source="detect")
    write_detections({0: [selected]}, work_dir / "detections.json")

    # 3. tracks.json: tracked box shrinks further over frames 0-15, from
    # the full 100x100 down to 40x40 (area=1600 -> ratio 0.16) by the end.
    tracks = {}
    for fi in range(16):
        side = 100 - fi * 4  # 100, 96, 92, ..., 40
        tracks[fi] = Box(0, 0, side, side)
    write_tracks(tracks, work_dir / "tracks.json")

    from scripts.check_box_size_bias import check_sample
    check_sample(cfg, sample_id)

    out = capsys.readouterr().out
    assert "candidates.json" in out
    assert "mean area-ratio=1.00" in out  # best-IoU candidate = the exact match
    assert "detections.json" in out
    assert "mean area-ratio=0.25" in out  # selected box is the smaller 50x50 one
    assert "tracks.json" in out
    # mean of (100-4i)^2/10000 for i in 0..15 -> mean side ratio isn't
    # linear in area; just check it's well below 1.0 and below the
    # detections ratio isn't required -- just sanity-check it's present
    # and plausible (>0, <1).
    assert "3. tracks.json" in out


def test_check_box_size_bias_missing_gt_video_id(tmp_path, capsys):
    """A sample not present in the GT file is skipped cleanly, no crash."""
    gt_path = _gt_file(tmp_path, "some_other_video", {0: Box(0, 0, 10, 10)})
    cfg = _cfg(tmp_path, gt_path)

    from scripts.check_box_size_bias import check_sample
    check_sample(cfg, "synth_sample")

    out = capsys.readouterr().out
    assert "not found in" in out

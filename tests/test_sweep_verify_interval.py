"""Integration test for scripts.sweep_verify_interval -- monkeypatches
run_stage4 (so no real video/tracker/model machinery is needed) with a
FAKE implementation whose output deterministically depends on
cfg.stage4.verify_interval, then checks: (1) the scratch copy step ran
BEFORE run_stage4 was called (a marker file from the real sample dir must
already be present), (2) the real work_dir's own tracks.json is NEVER
touched, (3) the printed ST-IoU progression and ranking summary match
what the fake run_stage4 was constructed to produce."""
from __future__ import annotations

import json
from pathlib import Path

from aero_eyes.types import Box
from aero_eyes.utils.io import write_tracks


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


def _config_yaml(tmp_path, gt_path, work_dir, data_root) -> Path:
    import yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "project": {"work_dir": str(work_dir)},
        "data": {"data_root": str(data_root), "gt": {"global_file": str(gt_path)}},
    }))
    return config_path


def test_sweep_reports_progression_and_never_touches_real_tracks_json(tmp_path, monkeypatch):
    from scripts.sweep_verify_interval import sweep_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_boxes = {fi: gt_box for fi in range(20)}  # object present on every frame 0-19
    gt_path = _gt_file(tmp_path, sample_id, gt_boxes)

    work_dir = tmp_path / "runs"
    data_root = tmp_path / "data"
    (data_root / sample_id).mkdir(parents=True)
    config_path = _config_yaml(tmp_path, gt_path, work_dir, data_root)

    real_sample_dir = work_dir / sample_id
    real_sample_dir.mkdir(parents=True)
    (real_sample_dir / "detections.json").write_text("{}")
    (real_sample_dir / "marker.txt").write_text("copy-me")  # proves the scratch copy step ran

    def _fake_run_stage4(run_cfg, sid):
        vi = run_cfg.stage4.verify_interval
        scratch_dir = Path(run_cfg.project.work_dir) / sid
        assert (scratch_dir / "marker.txt").exists(), (
            "scratch dir must be populated from the real sample dir BEFORE run_stage4 runs"
        )
        n_correct = min(vi, 20)  # deterministic: higher verify_interval -> more correctly tracked frames
        tracks = {fi: (gt_box if fi < n_correct else None) for fi in range(20)}
        write_tracks(tracks, scratch_dir / "tracks.json")

    monkeypatch.setattr("aero_eyes.stages.stage4.run_stage4", _fake_run_stage4)

    sweep_sample(
        str(config_path), [], sample_id,
        values=[0, 5, 10, 15, 20], iou_threshold=0.5, include_stage5=False,
    )

    # real work_dir/<sample_id>/tracks.json must NEVER be created -- the
    # sweep only ever writes into its own scratch subdirectories.
    assert not (real_sample_dir / "tracks.json").exists()

    for value in (0, 5, 10, 15, 20):
        scratch_tracks = work_dir / "_verify_interval_sweep" / f"vi_{value}" / sample_id / "tracks.json"
        assert scratch_tracks.exists()


def test_sweep_output_contents(tmp_path, monkeypatch, capsys):
    from scripts.sweep_verify_interval import sweep_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_boxes = {fi: gt_box for fi in range(20)}
    gt_path = _gt_file(tmp_path, sample_id, gt_boxes)

    work_dir = tmp_path / "runs"
    data_root = tmp_path / "data"
    (data_root / sample_id).mkdir(parents=True)
    config_path = _config_yaml(tmp_path, gt_path, work_dir, data_root)

    real_sample_dir = work_dir / sample_id
    real_sample_dir.mkdir(parents=True)
    (real_sample_dir / "detections.json").write_text("{}")

    def _fake_run_stage4(run_cfg, sid):
        vi = run_cfg.stage4.verify_interval
        scratch_dir = Path(run_cfg.project.work_dir) / sid
        n_correct = min(vi, 20)
        tracks = {fi: (gt_box if fi < n_correct else None) for fi in range(20)}
        write_tracks(tracks, scratch_dir / "tracks.json")

    monkeypatch.setattr("aero_eyes.stages.stage4.run_stage4", _fake_run_stage4)

    sweep_sample(
        str(config_path), [], sample_id,
        values=[0, 5, 10, 15, 20], iou_threshold=0.5, include_stage5=False,
    )

    out = capsys.readouterr().out
    assert "verify_interval=0: ST-IoU=0.0000" in out
    assert "verify_interval=20: ST-IoU=1.0000" in out
    assert "Ranked by ST-IoU (best first): 20=1.0000, 15=0.7500, 10=0.5000, 5=0.2500, 0=0.0000" in out
    assert "best so far: verify_interval=20" in out
    assert "LARGEST value tried and still winning" in out

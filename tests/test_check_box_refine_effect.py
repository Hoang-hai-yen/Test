"""Unit test for scripts.check_box_refine_effect's apply_in_stage4 warning
-- regression test for a real question: the script only ever re-refines
detections.json's (Stage 3 keyframe) boxes, completely ignoring
apply_in_stage3/apply_in_stage4, so --set box_refine.apply_in_stage4=true
silently has NO effect on what gets measured. Must warn loudly instead of
letting that pass unnoticed."""
from __future__ import annotations

import json


def _cfg(tmp_path, sample_id: str, apply_in_stage4: bool):
    from aero_eyes.config import AeroEyesConfig

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps([{"video_id": sample_id, "annotations": [{"bboxes": []}]}]))
    (tmp_path / "data" / sample_id).mkdir(parents=True)

    return AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"data_root": str(tmp_path / "data"), "gt": {"global_file": str(gt_path)}},
        "box_refine": {"enabled": True, "apply_in_stage4": apply_in_stage4},
    })


def test_warns_when_apply_in_stage4_is_set(tmp_path, capsys):
    from scripts.check_box_refine_effect import check_sample

    sample_id = "synth_sample"
    cfg = _cfg(tmp_path, sample_id, apply_in_stage4=True)

    check_sample(cfg, sample_id)

    out = capsys.readouterr().out
    assert "apply_in_stage4=true, but this script ONLY evaluates the apply_in_stage3" in out


def test_no_warning_when_apply_in_stage4_is_off(tmp_path, capsys):
    from scripts.check_box_refine_effect import check_sample

    sample_id = "synth_sample"
    cfg = _cfg(tmp_path, sample_id, apply_in_stage4=False)

    check_sample(cfg, sample_id)

    out = capsys.readouterr().out
    assert "apply_in_stage4=true" not in out

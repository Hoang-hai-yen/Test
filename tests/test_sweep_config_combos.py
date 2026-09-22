"""Tests for scripts.sweep_config_combos -- the full Stage1 x Stage3
matrix sweep. Pure unit tests for build_pairs()'s matrix-vs-ofat pairing
logic, plus an integration test that monkeypatches run_stage1/run_stage3
with FAKE implementations (no real models/video needed) to check the new
per-Stage1-combo wiring: (1) a Stage-1 combo's own Stage 1 rerun happens
exactly once even when paired with several Stage-3 combos (reused, not
re-run per pair -- see module docstring's cost-control rationale), (2)
stage3.recompute_candidate_features is forced true on only the FIRST
Stage-3 run under a Stage-1 combo that needs it, staying false for the
rest, (3) prototype.npz/candidates.json/detections.json are restored
byte-for-byte (or removed, if they didn't exist before) after the sweep
finishes, (4) a Stage-1 combo whose own Stage 1 run raises marks every
Stage-3 combo paired with it as failed instead of aborting the whole sweep.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import write_candidates, write_detections

from scripts.sweep_config_combos import Combo, build_pairs, run_sweep

SAMPLE_ID = "sweepfix001"


def _gt_file(tmp_path) -> Path:
    path = tmp_path / "gt.json"
    data = [{
        "video_id": SAMPLE_ID,
        "annotations": [{"bboxes": [{"frame": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}]}],
    }]
    path.write_text(json.dumps(data))
    return path


def _config_yaml(tmp_path, gt_path: Path, work_dir: Path, data_root: Path) -> Path:
    import yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "project": {"work_dir": str(work_dir)},
        "data": {"data_root": str(data_root), "gt": {"global_file": str(gt_path)}},
    }))
    return config_path


def _seed_candidates(sample_dir: Path) -> None:
    det = Detection(frame_idx=0, box=Box(0.0, 0.0, 10.0, 10.0), similarity=0.9, source="detect")
    write_candidates({0: [det]}, sample_dir / "candidates.json")


def test_build_pairs_matrix_is_full_cross_product():
    stage1 = [Combo("stage1_default", "encoder", {}), Combo("E_a", "encoder", {}, needs_stage1=True)]
    stage3 = [Combo("s3a", "threshold", {}), Combo("s3b", "threshold", {}), Combo("s3c", "threshold", {})]
    pairs = build_pairs(stage1, stage3, matrix=True)
    assert len(pairs) == len(stage1) * len(stage3) == 6
    assert {(s1.id, s3.id) for s1, s3 in pairs} == {
        (s1.id, s3.id) for s1 in stage1 for s3 in stage3
    }


def test_build_pairs_ofat_is_row0_plus_column0():
    stage1 = [Combo("stage1_default", "encoder", {}), Combo("E_a", "encoder", {}, needs_stage1=True),
              Combo("E_b", "encoder", {}, needs_stage1=True)]
    stage3 = [Combo("baseline", "threshold", {}), Combo("s3b", "threshold", {}), Combo("s3c", "threshold", {})]
    pairs = build_pairs(stage1, stage3, matrix=False)
    ids = {(s1.id, s3.id) for s1, s3 in pairs}
    # every non-default Stage-1 combo paired ONLY with baseline
    assert ("E_a", "baseline") in ids and ("E_b", "baseline") in ids
    assert ("E_a", "s3b") not in ids and ("E_b", "s3c") not in ids
    # stage1_default paired with EVERY Stage-3 combo
    assert ("stage1_default", "baseline") in ids
    assert ("stage1_default", "s3b") in ids
    assert ("stage1_default", "s3c") in ids
    assert len(pairs) == 2 + 3  # 2 non-default Stage-1 combos + 3 Stage-3 combos


def test_run_sweep_reuses_stage1_and_recomputes_once_then_restores_files(tmp_path):
    work_dir = tmp_path / "runs"
    data_root = tmp_path / "data"
    (data_root / SAMPLE_ID).mkdir(parents=True)
    sample_dir = work_dir / SAMPLE_ID
    sample_dir.mkdir(parents=True)
    _seed_candidates(sample_dir)
    original_candidates_bytes = (sample_dir / "candidates.json").read_bytes()
    # Seed a PRE-EXISTING detections.json + prototype.npz, exactly like a
    # real work_dir a normal pipeline run already populated -- reproduces
    # the real bug found in practice: without forcing project.use_cache off,
    # run_stage1/run_stage3 both short-circuit on an already-on-disk file
    # and every combo silently scores the SAME stale cached result instead
    # of ever actually re-running with its own overrides.
    (sample_dir / "detections.json").write_bytes(b"pre-existing-stale-detections")
    (sample_dir / "prototype.npz").write_bytes(b"pre-existing-stale-prototype")
    original_detections_bytes = (sample_dir / "detections.json").read_bytes()
    original_prototype_bytes = (sample_dir / "prototype.npz").read_bytes()
    config_path = _config_yaml(tmp_path, _gt_file(tmp_path), work_dir, data_root)

    stage1_combos = [
        Combo("stage1_default", "encoder", {}, needs_stage1=False),
        Combo("E_fake", "encoder", {"stage1.feature_extractor.model": "dinov3"},
              needs_stage1=True, recompute=True),
    ]
    stage3_combos = [
        Combo("s3a", "threshold", {}),
        Combo("s3b", "threshold", {"stage3.adaptive_z_score": 1.5}),
    ]

    stage1_calls: list[str] = []
    stage3_recompute_calls: list[bool] = []
    use_cache_calls: list[bool] = []

    def fake_run_stage1(cfg, sample_id):
        use_cache_calls.append(cfg.project.use_cache)
        stage1_calls.append(cfg.stage1.feature_extractor.model)
        (sample_dir / "prototype.npz").write_bytes(b"fake-prototype-bytes")
        return sample_dir / "prototype.npz"

    def fake_run_stage3(cfg, sample_id):
        use_cache_calls.append(cfg.project.use_cache)
        stage3_recompute_calls.append(cfg.stage3.recompute_candidate_features)
        det = Detection(frame_idx=0, box=Box(0.0, 0.0, 10.0, 10.0), similarity=0.9, source="detect")
        write_detections({0: [det]}, sample_dir / "detections.json", threshold=0.5)
        return sample_dir / "detections.json"

    with patch("aero_eyes.stages.stage1.run_stage1", side_effect=fake_run_stage1), \
         patch("aero_eyes.stages.stage3.run_stage3", side_effect=fake_run_stage3):
        results = run_sweep(str(config_path), [], [SAMPLE_ID], stage1_combos, stage3_combos,
                             matrix=True, iou_threshold=0.5)

    # 2 Stage-1 combos x 2 Stage-3 combos = 4 pairs, all succeeded
    assert len(results) == 4
    assert all(r["error"] is None for r in results)
    combo_ids = {r["combo"] for r in results}
    assert combo_ids == {
        "stage1_default__s3a", "stage1_default__s3b",
        "E_fake__s3a", "E_fake__s3b",
    }

    # Stage 1 reran exactly ONCE for E_fake (never for stage1_default, which
    # has needs_stage1=False) -- reused across BOTH its paired Stage-3 combos.
    assert stage1_calls == ["dinov3"]

    # recompute_candidate_features forced true on exactly the FIRST Stage-3
    # run under E_fake (2 total pairs use E_fake -> exactly one True), false
    # for every pair under stage1_default (needs_stage1=False, recompute=False).
    assert stage3_recompute_calls.count(True) == 1
    assert stage3_recompute_calls.count(False) == 3

    # project.use_cache must be forced False on EVERY run_stage1/run_stage3
    # call, regardless of a pre-existing prototype.npz/detections.json --
    # this is the regression test for the real bug found in practice: every
    # combo silently returning the SAME stale cached detections.json because
    # use_cache wasn't forced off.
    assert use_cache_calls and all(v is False for v in use_cache_calls)

    # candidates.json was only ever READ, never mutated.
    assert (sample_dir / "candidates.json").read_bytes() == original_candidates_bytes
    # prototype.npz/detections.json DID pre-exist (seeded above, like a real
    # work_dir a normal pipeline run already populated) -- must be restored
    # to that exact original content afterward, not left at whatever the
    # last combo that ran happened to write.
    assert (sample_dir / "prototype.npz").read_bytes() == original_prototype_bytes
    assert (sample_dir / "detections.json").read_bytes() == original_detections_bytes


def test_run_sweep_stage1_failure_marks_only_its_own_paired_combos_as_failed(tmp_path):
    work_dir = tmp_path / "runs"
    data_root = tmp_path / "data"
    (data_root / SAMPLE_ID).mkdir(parents=True)
    sample_dir = work_dir / SAMPLE_ID
    sample_dir.mkdir(parents=True)
    _seed_candidates(sample_dir)
    config_path = _config_yaml(tmp_path, _gt_file(tmp_path), work_dir, data_root)

    stage1_combos = [
        Combo("stage1_default", "encoder", {}, needs_stage1=False),
        Combo("E_broken", "encoder", {"stage1.feature_extractor.model": "dinov3"},
              needs_stage1=True, recompute=True),
    ]
    stage3_combos = [Combo("s3a", "threshold", {}), Combo("s3b", "threshold", {})]

    def fake_run_stage1(cfg, sample_id):
        if cfg.stage1.feature_extractor.model == "dinov3":
            raise RuntimeError("gated weights not available")
        return sample_dir / "prototype.npz"

    def fake_run_stage3(cfg, sample_id):
        det = Detection(frame_idx=0, box=Box(0.0, 0.0, 10.0, 10.0), similarity=0.9, source="detect")
        write_detections({0: [det]}, sample_dir / "detections.json", threshold=0.5)
        return sample_dir / "detections.json"

    with patch("aero_eyes.stages.stage1.run_stage1", side_effect=fake_run_stage1), \
         patch("aero_eyes.stages.stage3.run_stage3", side_effect=fake_run_stage3):
        results = run_sweep(str(config_path), [], [SAMPLE_ID], stage1_combos, stage3_combos,
                             matrix=True, iou_threshold=0.5)

    by_combo = {r["combo"]: r for r in results}
    assert len(results) == 4
    assert by_combo["E_broken__s3a"]["error"] is not None
    assert "stage1 failed" in by_combo["E_broken__s3a"]["error"]
    assert by_combo["E_broken__s3b"]["error"] is not None
    # stage1_default never calls run_stage1 at all -- unaffected by E_broken's failure.
    assert by_combo["stage1_default__s3a"]["error"] is None
    assert by_combo["stage1_default__s3b"]["error"] is None

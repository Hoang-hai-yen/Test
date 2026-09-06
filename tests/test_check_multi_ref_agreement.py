"""Integration test for scripts.check_multi_ref_agreement -- writes a real
prototype.npz (2 reference features) + candidates.json(+.feats.npz) + GT
file to a temp dir, with one genuine-match candidate that agrees with BOTH
references and one confuser candidate that matches only ONE reference
well, and checks the script correctly reports the confuser's larger
per-ref spread and that the simulated agreement gate demotes it."""
from __future__ import annotations

import json

import numpy as np

from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import write_prototype


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


def _write_candidates_and_feats(candidates: dict[int, list[Detection]], feats: list[np.ndarray], path):
    from aero_eyes.utils.io import SCHEMA_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)
    frames_json: dict[str, list[dict]] = {}
    feat_idx = 0
    for fi, dets in candidates.items():
        frame_dets = []
        for det in dets:
            d = det.to_dict()
            d["feat_idx"] = feat_idx
            feat_idx += 1
            frame_dets.append(d)
        frames_json[str(fi)] = frame_dets
    payload = {"schema_version": SCHEMA_VERSION, "frames": frames_json}
    with open(path, "w") as f:
        json.dump(payload, f)
    np.savez_compressed(str(path.with_suffix(".feats.npz")), features=np.stack(feats, axis=0))


def _cfg(tmp_path, gt_path, pooling="max", match_threshold=0.5):
    from aero_eyes.config import AeroEyesConfig
    return AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"gt": {"global_file": str(gt_path)}},
        "stage3": {"match_threshold": match_threshold},
        "accuracy": {
            "mode": "cheap_boosters",  # required for use_multi_ref -- see stage3.py
            "cheap_boosters": {"multi_ref_pooling": pooling, "multi_reference_embedding": True},
        },
    })


def test_confuser_shows_larger_spread_and_gate_demotes_it(tmp_path, capsys):
    from scripts.check_multi_ref_agreement import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_path = _gt_file(tmp_path, sample_id, {0: gt_box})  # only frame 0 has GT (object present there)
    cfg = _cfg(tmp_path, gt_path, pooling="max")

    work_dir = tmp_path / "runs" / sample_id
    ref_a = np.array([1.0, 0.0], dtype=np.float32)
    ref_b = np.array([0.0, 1.0], dtype=np.float32)
    write_prototype(np.array([0.7, 0.7], dtype=np.float32), {}, [ref_a, ref_b], work_dir / "prototype.npz")

    # frame 0: genuine match on the GT box -- agrees reasonably with BOTH refs (small spread).
    genuine_box = gt_box
    # frame 1: confuser (e.g. a static background object) on a frame with NO GT entry at
    # all (object absent there) -- matches ref_a strongly but ref_b not at all (large spread).
    confuser_box = Box(500, 500, 510, 510)

    candidates = {
        0: [Detection(frame_idx=0, box=genuine_box, similarity=0.0, source="detect")],
        1: [Detection(frame_idx=1, box=confuser_box, similarity=0.0, source="detect")],
    }
    # feat @ ref_a, feat @ ref_b:
    genuine_feat = np.array([0.6, 0.6], dtype=np.float32)   # scores 0.6 vs both refs -> spread 0.0
    confuser_feat = np.array([0.9, 0.05], dtype=np.float32)  # scores 0.9 vs ref_a, 0.05 vs ref_b -> spread 0.85
    _write_candidates_and_feats(candidates, [genuine_feat, confuser_feat], work_dir / "candidates.json")

    check_sample(cfg, sample_id, iou_threshold=0.5, agreement_floors=(0.0, 0.5))

    out = capsys.readouterr().out
    assert "genuine_match: n=1" in out
    assert "unverifiable: n=1" in out
    assert "MUCH larger typical per-ref spread" in out

    # pooled(max) for confuser = 0.9 (passes match_threshold=0.5) at floor=0.0 (gate off,
    # since per_ref_min(0.05) < floor(0.0) is False -- 0.05 is NOT < 0.0).
    assert "floor=0.00: confuser-suspects passing threshold 1 -> 1" in out
    # at floor=0.5: confuser's min score (0.05) < 0.5 -> gated score becomes 0.05 -> fails
    # threshold(0.5) -- demoted. Genuine's min score (0.6) is NOT < 0.5 -> stays at pooled
    # 0.6 -- still passes, unaffected (the gate's "cost" is zero here by construction).
    assert "floor=0.50: confuser-suspects passing threshold 1 -> 0" in out
    assert "genuine matches passing threshold 1 -> 1" in out
    # no detections.json in this scenario -- must fall back to stage3.match_threshold
    # with a loud warning, not silently use a wrong/undefined threshold.
    assert "warning: could not read the real effective threshold" in out


def test_uses_dynamic_prototype_final_reference_set_not_stale_initial_scores(tmp_path, capsys):
    """Regression test for a real bug: dynamic_prototype can APPEND a new
    reference during its rounds, and Stage 3's own effective threshold is
    computed against that FINAL, larger reference set -- scoring only
    against the ORIGINAL references (as an earlier version of this script
    did) badly understates real scores once that has happened, making the
    whole gate simulation meaningless (observed in practice: pooled scores
    for every group, including genuine matches, came out far below the
    real effective_threshold read from detections.json)."""
    from aero_eyes.config import AeroEyesConfig
    from scripts.check_multi_ref_agreement import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_path = _gt_file(tmp_path, sample_id, {0: gt_box})  # only frame 0 has GT

    cfg = AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"gt": {"global_file": str(gt_path)}},
        "stage3": {
            "match_threshold": 0.55,
            "dynamic_prototype": {
                "enabled": True, "rounds": 1, "alpha": 0.3,
                "high_conf_percentile": 0.0, "high_conf_abs_floor": 0.85,
                "min_support": 1,
            },
        },
        "accuracy": {
            "mode": "cheap_boosters",
            "cheap_boosters": {"multi_ref_pooling": "max", "multi_reference_embedding": True},
        },
    })

    work_dir = tmp_path / "runs" / sample_id
    ref_a = np.array([1.0, 0.0], dtype=np.float32)
    ref_b = np.array([0.0, 1.0], dtype=np.float32)
    write_prototype(np.array([0.5, 0.5], dtype=np.float32), {}, [ref_a, ref_b], work_dir / "prototype.npz")

    genuine_box = gt_box
    driver_box = Box(200, 200, 210, 210)   # the round's only high-confidence pick -- becomes the appended ref
    confuser_box = Box(500, 500, 510, 510)

    candidates = {
        0: [Detection(frame_idx=0, box=genuine_box, similarity=0.0, source="detect")],
        1: [Detection(frame_idx=1, box=driver_box, similarity=0.0, source="detect")],
        2: [Detection(frame_idx=2, box=confuser_box, similarity=0.0, source="detect")],
    }
    genuine_feat = np.array([0.6, 0.6], dtype=np.float32)   # initial pooled(max) vs original 2 refs = 0.6
    driver_feat = np.array([0.9, 0.9], dtype=np.float32)    # initial pooled(max) = 0.9 -- only one >= abs_floor(0.85)
    confuser_feat = np.array([0.4, 0.4], dtype=np.float32)  # initial pooled(max) = 0.4
    _write_candidates_and_feats(
        candidates, [genuine_feat, driver_feat, confuser_feat], work_dir / "candidates.json",
    )

    check_sample(cfg, sample_id, iou_threshold=0.5, agreement_floors=(0.0,))

    out = capsys.readouterr().out
    assert "appended 1 dynamically-derived reference(s)" in out
    # genuine's FINAL pooled score must reflect the round-appended reference
    # (normalize([0.9,0.9]) = [0.7071,0.7071]; [0.6,0.6] . that = 0.8485),
    # NOT its stale initial score against the 2 original refs alone (0.600).
    assert "genuine_match: n=1  pooled(mean=0.849" in out


def test_uses_real_effective_threshold_from_detections_json_not_static_config(tmp_path, capsys):
    """Regression test for a real bug: stage3.match_threshold is IRRELEVANT
    when stage3.adaptive_threshold is what Stage 3 actually used (raw
    cosine scores can sit far below match_threshold's static default in
    this project's ground-to-aerial domain gap) -- the gate simulation
    must compare against the REAL threshold Stage 3 recorded in
    detections.json, not silently use the static config value, or every
    number in the simulation is meaningless (observed in practice: an
    all-zeros simulation when match_threshold=0.55 was used against
    scores that never exceed ~0.15)."""
    from aero_eyes.utils.io import write_detections

    from scripts.check_multi_ref_agreement import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_path = _gt_file(tmp_path, sample_id, {0: gt_box})
    # Deliberately wrong/irrelevant static default -- nothing in this
    # scenario would ever pass it.
    cfg = _cfg(tmp_path, gt_path, pooling="max", match_threshold=0.95)

    work_dir = tmp_path / "runs" / sample_id
    ref_a = np.array([1.0, 0.0], dtype=np.float32)
    ref_b = np.array([0.0, 1.0], dtype=np.float32)
    write_prototype(np.array([0.7, 0.7], dtype=np.float32), {}, [ref_a, ref_b], work_dir / "prototype.npz")

    genuine_box = gt_box
    confuser_box = Box(500, 500, 510, 510)
    candidates = {
        0: [Detection(frame_idx=0, box=genuine_box, similarity=0.0, source="detect")],
        1: [Detection(frame_idx=1, box=confuser_box, similarity=0.0, source="detect")],
    }
    genuine_feat = np.array([0.6, 0.6], dtype=np.float32)
    confuser_feat = np.array([0.9, 0.05], dtype=np.float32)
    _write_candidates_and_feats(candidates, [genuine_feat, confuser_feat], work_dir / "candidates.json")

    # detections.json records the REAL effective threshold Stage 3 actually
    # used (e.g. adaptive_threshold's own computed value) -- 0.5, nothing
    # like the static config's 0.95.
    write_detections(
        {0: [Detection(frame_idx=0, box=genuine_box, similarity=0.6, source="detect")]},
        work_dir / "detections.json", threshold=0.5,
    )

    check_sample(cfg, sample_id, iou_threshold=0.5, agreement_floors=(0.0, 0.5))

    out = capsys.readouterr().out
    assert "effective_threshold=0.500" in out
    assert "warning: could not read the real effective threshold" not in out
    # Same pass/fail pattern as the 0.5-threshold test above -- proves 0.5
    # (from detections.json) was used, NOT the static 0.95 (which would
    # have produced "0 -> 0" for everything, the exact bug observed).
    assert "floor=0.00: confuser-suspects passing threshold 1 -> 1" in out
    assert "floor=0.50: confuser-suspects passing threshold 1 -> 0" in out

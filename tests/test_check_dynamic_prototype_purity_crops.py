"""Integration test for scripts.check_dynamic_prototype_purity's
--export-crops option -- writes a real prototype.npz + candidates.json
(+.feats.npz) + GT file + a dummy video file to a temp dir, monkeypatches
read_frame (no real video decoding needed to test the export mechanism
itself), and checks that one crop file per selected candidate is written,
correctly tagged with its GT IoU in the filename."""
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


def test_export_crops_writes_one_file_per_selected_candidate_with_iou_tag(tmp_path, monkeypatch):
    from aero_eyes.config import AeroEyesConfig
    from scripts.check_dynamic_prototype_purity import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    far_gt_box = Box(50, 50, 60, 60)
    gt_path = _gt_file(tmp_path, sample_id, {0: gt_box, 1: far_gt_box})

    data_root = tmp_path / "data"
    (data_root / sample_id).mkdir(parents=True)
    (data_root / sample_id / "video.mp4").write_bytes(b"")  # dummy -- read_frame is monkeypatched below

    cfg = AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"data_root": str(data_root), "gt": {"global_file": str(gt_path)}},
        "stage3": {
            "dynamic_prototype": {
                "enabled": True, "rounds": 1, "alpha": 0.3,
                "high_conf_percentile": 0.0,  # 0th percentile -> selects every candidate
                "high_conf_abs_floor": -1.0,
                "min_support": 1,
            },
        },
    })

    work_dir = tmp_path / "runs" / sample_id
    write_prototype(np.array([1.0, 0.0], dtype=np.float32), {}, [], work_dir / "prototype.npz")

    candidates = {
        # frame 0: candidate box == GT (IoU=1.0)
        0: [Detection(frame_idx=0, box=gt_box, similarity=0.0, source="detect")],
        # frame 1: candidate box does NOT match far_gt_box (IoU=0.0)
        1: [Detection(frame_idx=1, box=Box(0, 0, 10, 10), similarity=0.0, source="detect")],
    }
    feats = [np.array([0.9, 0.0], dtype=np.float32), np.array([0.8, 0.0], dtype=np.float32)]
    _write_candidates_and_feats(candidates, feats, work_dir / "candidates.json")

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr("aero_eyes.utils.video.read_frame", lambda video_path, frame_idx: fake_frame)

    export_dir = tmp_path / "crops"
    check_sample(
        cfg, sample_id, iou_threshold=0.5,
        export_crops=True, export_dir=export_dir, max_crops_per_round=10,
    )

    round_dir = export_dir / "round_1"
    files = sorted(round_dir.glob("*.jpg"))
    assert len(files) == 2
    names = [f.name for f in files]
    assert any(n.startswith("frame0_cand") and "_iou1.00" in n for n in names)
    assert any(n.startswith("frame1_cand") and "_iou0.00" in n for n in names)


def test_export_crops_off_by_default_writes_nothing(tmp_path, monkeypatch):
    """export_crops defaults to False -- existing callers/behavior must be
    unaffected unless explicitly opted in."""
    from aero_eyes.config import AeroEyesConfig
    from scripts.check_dynamic_prototype_purity import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)
    gt_path = _gt_file(tmp_path, sample_id, {0: gt_box})

    cfg = AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"gt": {"global_file": str(gt_path)}},
        "stage3": {
            "dynamic_prototype": {
                "enabled": True, "rounds": 1, "alpha": 0.3,
                "high_conf_percentile": 0.0, "high_conf_abs_floor": -1.0, "min_support": 1,
            },
        },
    })

    work_dir = tmp_path / "runs" / sample_id
    write_prototype(np.array([1.0, 0.0], dtype=np.float32), {}, [], work_dir / "prototype.npz")
    candidates = {0: [Detection(frame_idx=0, box=gt_box, similarity=0.0, source="detect")]}
    feats = [np.array([0.9, 0.0], dtype=np.float32)]
    _write_candidates_and_feats(candidates, feats, work_dir / "candidates.json")

    check_sample(cfg, sample_id, iou_threshold=0.5)  # export_crops not passed -> default False

    assert not (work_dir / "diagnostics").exists()

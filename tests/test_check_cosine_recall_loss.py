"""Integration test for scripts.check_cosine_recall_loss -- writes a real
prototype.npz + candidates.json(+.feats.npz) + GT file to a temp dir,
covering one frame for each of the 4 possible fates the script classifies
(rejected_threshold, cut_by_topk_per_keyframe, suppressed_by_nms, kept),
and checks the reported counts match what was constructed."""
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


def _cfg(tmp_path, gt_path):
    from aero_eyes.config import AeroEyesConfig
    return AeroEyesConfig.model_validate({
        "project": {"work_dir": str(tmp_path / "runs")},
        "data": {"gt": {"global_file": str(gt_path)}},
        "stage3": {
            "match_threshold": 0.5,
            "nms_iou": 0.5,
            "topk_per_keyframe": 1,
        },
    })


def _write_candidates_and_feats(candidates: dict[int, list[Detection]], feats: list[np.ndarray], path):
    """Mirror aero_eyes.stages.stage2._write_candidates_with_features's own
    on-disk schema directly (rather than importing the private function),
    since this test's candidates need specific hand-picked feature vectors
    per Detection, matched up by insertion order."""
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


def test_check_cosine_recall_loss_classifies_every_fate(tmp_path, capsys):
    from scripts.check_cosine_recall_loss import check_sample

    sample_id = "synth_sample"
    gt_box = Box(0, 0, 10, 10)  # same shape reused across frames, translated per-frame is unnecessary here

    gt_boxes = {0: gt_box, 1: gt_box, 2: gt_box, 3: gt_box}
    gt_path = _gt_file(tmp_path, sample_id, gt_boxes)
    cfg = _cfg(tmp_path, gt_path)

    prototype = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    work_dir = tmp_path / "runs" / sample_id
    write_prototype(prototype, {}, [], work_dir / "prototype.npz")

    far_box = Box(1000, 1000, 1010, 1010)          # IoU=0.0 with gt_box
    overlapping_box = Box(0, 0, 10, 15)            # IoU(gt_box, this) = 100/150 = 0.667 > nms_iou(0.5)

    candidates: dict[int, list[Detection]] = {
        # Frame 0: single correct candidate, but its own score (0.3) never
        # clears match_threshold(0.5) -> rejected_threshold.
        0: [Detection(frame_idx=0, box=gt_box, similarity=0.0, source="detect")],
        # Frame 1: correct candidate (score 0.6, passes threshold) + a
        # NON-overlapping confuser scoring higher (0.9) -> both survive NMS
        # (they don't overlap), but topk_per_keyframe=1 keeps only the
        # confuser -> correct one is cut_by_topk_per_keyframe.
        1: [
            Detection(frame_idx=1, box=gt_box, similarity=0.0, source="detect"),
            Detection(frame_idx=1, box=far_box, similarity=0.0, source="detect"),
        ],
        # Frame 2: correct candidate (score 0.6) + an OVERLAPPING confuser
        # scoring higher (0.9, IoU with the correct box > nms_iou) ->
        # confuser survives NMS, correct one is suppressed_by_nms.
        2: [
            Detection(frame_idx=2, box=gt_box, similarity=0.0, source="detect"),
            Detection(frame_idx=2, box=overlapping_box, similarity=0.0, source="detect"),
        ],
        # Frame 3: single correct candidate, passes threshold, alone on its
        # frame -> kept (the "everything worked" baseline case).
        3: [Detection(frame_idx=3, box=gt_box, similarity=0.0, source="detect")],
    }
    # Feature vectors chosen so feat @ prototype ([1,0,0]) gives exactly the
    # cosine scores described above (_score_against_ref does a plain dot
    # product, no re-normalization).
    feats = [
        np.array([0.3, 0.0, 0.0], dtype=np.float32),   # frame 0 correct: 0.3 (rejected)
        np.array([0.6, 0.0, 0.0], dtype=np.float32),   # frame 1 correct: 0.6
        np.array([0.9, 0.0, 0.0], dtype=np.float32),   # frame 1 confuser: 0.9
        np.array([0.6, 0.0, 0.0], dtype=np.float32),   # frame 2 correct: 0.6
        np.array([0.9, 0.0, 0.0], dtype=np.float32),   # frame 2 confuser: 0.9
        np.array([0.8, 0.0, 0.0], dtype=np.float32),   # frame 3 correct: 0.8
    ]

    _write_candidates_and_feats(candidates, feats, work_dir / "candidates.json")

    check_sample(cfg, sample_id, iou_threshold=0.5)

    out = capsys.readouterr().out
    assert "4 GT frame(s) have a candidate reaching IoU>=0.5" in out
    assert "rejected_threshold: 1" in out
    assert "cut_by_topk_per_keyframe: 1" in out
    assert "suppressed_by_nms: 1" in out
    assert "kept: 1" in out

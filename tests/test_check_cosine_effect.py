"""Unit tests for scripts.check_cosine_effect.compute_prf1 -- known
per-frame candidates + GT -> known confusion matrix / Precision/Recall/F1."""
from __future__ import annotations

from aero_eyes.types import Box, Detection
from scripts.check_cosine_effect import compute_prf1


def _det(box: Box) -> Detection:
    return Detection(frame_idx=0, box=box, similarity=1.0, source="detect")


def test_all_true_positives():
    """Every GT frame has a matching candidate -> perfect P/R/F1."""
    box = Box(0, 0, 10, 10)
    dets_by_frame = {0: [_det(box)], 1: [_det(box)], 2: [_det(box)]}
    gt = {0: box, 1: box, 2: box}

    r = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert r["tp"] == 3 and r["fn"] == 0 and r["fp"] == 0 and r["tn"] == 0
    assert abs(r["precision"] - 1.0) < 1e-9
    assert abs(r["recall"] - 1.0) < 1e-9
    assert abs(r["f1"] - 1.0) < 1e-9
    assert r["unsampled_gt_frames"] == 0


def test_false_negative_wrong_location():
    """Candidate present but IoU below threshold -> FN, not TP."""
    gt_box = Box(0, 0, 10, 10)
    far_box = Box(100, 100, 110, 110)   # IoU = 0.0 with gt_box
    dets_by_frame = {0: [_det(far_box)]}
    gt = {0: gt_box}

    r = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert r["tp"] == 0 and r["fn"] == 1
    assert r["recall"] == 0.0


def test_false_negative_zero_candidates():
    """GT present, zero candidates on that frame -> FN."""
    gt_box = Box(0, 0, 10, 10)
    dets_by_frame = {0: []}
    gt = {0: gt_box}

    r = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert r["tp"] == 0 and r["fn"] == 1


def test_false_positive_on_absent_frame():
    """GT absent on this frame, but a candidate fired -> FP."""
    box = Box(0, 0, 10, 10)
    dets_by_frame = {5: [_det(box)]}  # frame 5 not in gt
    gt = {}

    r = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert r["fp"] == 1 and r["tn"] == 0
    assert r["precision"] == 0.0  # tp=0, fp=1 -> 0/(0+1)


def test_true_negative_on_absent_frame():
    """GT absent on this frame, zero candidates -> TN, no penalty."""
    dets_by_frame = {5: []}
    gt = {}

    r = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert r["tn"] == 1 and r["fp"] == 0
    # No TP/FP/FN at all -> precision/recall/f1 all default to 0.0
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_unsampled_gt_frames_reported_separately():
    """A GT frame that never appears as a key in dets_by_frame is NOT
    counted as FN -- it's reported via unsampled_gt_frames instead."""
    box = Box(0, 0, 10, 10)
    dets_by_frame = {0: [_det(box)]}          # only frame 0 was processed
    gt = {0: box, 1: box, 2: box}              # GT also has frames 1, 2

    r = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert r["tp"] == 1 and r["fn"] == 0       # frames 1,2 never touched the loop
    assert r["n_processed_gt_frames"] == 1
    assert r["n_gt_total"] == 3
    assert r["unsampled_gt_frames"] == 2


def test_iou_threshold_is_configurable():
    """A partial-overlap candidate passes at a loose threshold but fails
    at a strict one -- exactly what --iou-threshold controls."""
    gt_box = Box(0, 0, 10, 10)
    partial_box = Box(5, 5, 15, 15)  # IoU = 25/175 ≈ 0.143 with gt_box
    dets_by_frame = {0: [_det(partial_box)]}
    gt = {0: gt_box}

    loose = compute_prf1(dets_by_frame, gt, iou_threshold=0.1)
    strict = compute_prf1(dets_by_frame, gt, iou_threshold=0.5)
    assert loose["tp"] == 1 and loose["fn"] == 0
    assert strict["tp"] == 0 and strict["fn"] == 1

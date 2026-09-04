"""Unit tests for scripts.check_st_iou_breakdown.classify_frames -- known
pred/GT tubes -> known MATCH/LOOSE/MISSING_PRED/MISSING_GT breakdown and
ST-IoU deficit split (coverage vs imprecision)."""
from __future__ import annotations

from aero_eyes.types import Box
from scripts.check_st_iou_breakdown import classify_frames


def test_perfect_match_no_deficit():
    """Identical pred and GT everywhere -> ST-IoU=1.0, zero deficit."""
    box = Box(0, 0, 10, 10)
    pred = {0: box, 1: box, 2: box}
    gt = {0: box, 1: box, 2: box}

    r = classify_frames(pred, gt, iou_threshold=0.5)
    assert r["match"] == 3 and r["loose"] == 0
    assert r["missing_pred"] == 0 and r["missing_gt"] == 0
    assert abs(r["st_iou"] - 1.0) < 1e-9
    assert abs(r["deficit_coverage"]) < 1e-9
    assert abs(r["deficit_imprecision"]) < 1e-9


def test_pure_coverage_gap():
    """GT present but prediction entirely missing on some frames -> all of
    the deficit attributed to coverage, none to imprecision."""
    box = Box(0, 0, 10, 10)
    pred = {0: box}                 # only frame 0 predicted
    gt = {0: box, 1: box, 2: box}   # GT present on 0, 1, 2

    r = classify_frames(pred, gt, iou_threshold=0.5)
    assert r["match"] == 1
    assert r["missing_pred"] == 2   # frames 1, 2: GT there, no prediction
    assert r["missing_gt"] == 0
    # ST-IoU = (1.0 + 0 + 0) / 3 = 1/3
    assert abs(r["st_iou"] - 1.0 / 3.0) < 1e-9
    deficit_total = r["deficit_coverage"] + r["deficit_imprecision"]
    assert abs(deficit_total - (1.0 - r["st_iou"])) < 1e-9
    # All the loss comes from the 2 missing-pred frames, none from
    # imprecision (the one frame that DID have both was a perfect match).
    assert abs(r["deficit_coverage"] - 2.0 / 3.0) < 1e-9
    assert abs(r["deficit_imprecision"]) < 1e-9


def test_pure_imprecision_no_coverage_gap():
    """Prediction present on every GT frame, but never a tight match ->
    all of the deficit attributed to imprecision, none to coverage."""
    gt_box = Box(0, 0, 10, 10)
    loose_box = Box(5, 5, 15, 15)  # IoU = 25/175 with gt_box, always < 0.5
    pred = {0: loose_box, 1: loose_box}
    gt = {0: gt_box, 1: gt_box}

    r = classify_frames(pred, gt, iou_threshold=0.5)
    assert r["match"] == 0 and r["loose"] == 2
    assert r["missing_pred"] == 0 and r["missing_gt"] == 0
    deficit_total = r["deficit_coverage"] + r["deficit_imprecision"]
    assert abs(r["deficit_coverage"]) < 1e-9
    assert abs(deficit_total - (1.0 - r["st_iou"])) < 1e-9


def test_missing_gt_is_a_false_positive():
    """Prediction fires on a frame where GT has nothing -> MISSING_GT,
    contributes to coverage deficit just like MISSING_PRED does."""
    box = Box(0, 0, 10, 10)
    pred = {5: box}  # frame 5 not in GT at all
    gt = {}

    r = classify_frames(pred, gt, iou_threshold=0.5)
    assert r["missing_gt"] == 1 and r["missing_pred"] == 0
    assert r["st_iou"] == 0.0
    assert abs(r["deficit_coverage"] - 1.0) < 1e-9


def test_empty_tubes():
    """Both empty -> 0 union frames, no crash, zero-valued result."""
    r = classify_frames({}, {}, iou_threshold=0.5)
    assert r["n_union"] == 0
    assert r["st_iou"] == 0.0


def test_loose_vs_match_threshold_boundary():
    """A partial-overlap box counted as LOOSE at a strict threshold but
    MATCH at a loose one -- exactly what --iou-threshold controls."""
    gt_box = Box(0, 0, 10, 10)
    partial_box = Box(5, 5, 15, 15)  # IoU = 25/175 ~= 0.143
    pred = {0: partial_box}
    gt = {0: gt_box}

    loose_thr = classify_frames(pred, gt, iou_threshold=0.1)
    strict_thr = classify_frames(pred, gt, iou_threshold=0.5)
    assert loose_thr["match"] == 1 and loose_thr["loose"] == 0
    assert strict_thr["match"] == 0 and strict_thr["loose"] == 1
    # ST-IoU itself doesn't depend on the threshold, only the MATCH/LOOSE label does.
    assert abs(loose_thr["st_iou"] - strict_thr["st_iou"]) < 1e-9

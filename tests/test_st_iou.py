"""Unit tests for the ST-IoU metric — known input -> known score."""
from __future__ import annotations

import pytest

from aero_eyes.evaluate import st_iou
from aero_eyes.types import Box


def _box(x1, y1, x2, y2) -> Box:
    return Box(float(x1), float(y1), float(x2), float(y2))


def test_perfect_overlap():
    """Identical prediction and GT -> ST-IoU = 1.0."""
    box = _box(10, 10, 50, 50)
    pred = {0: box, 1: box, 2: box}
    gt = {0: box, 1: box, 2: box}
    assert abs(st_iou(pred, gt) - 1.0) < 1e-9


def test_temporally_disjoint():
    """Pred and GT in completely different frame ranges -> ST-IoU = 0.0.

    Union = {0,1,2,3,4,5}; pred only in 0,1,2, GT only in 3,4,5.
    Every frame contributes 0.0 -> mean = 0.0.
    """
    box = _box(0, 0, 10, 10)
    pred = {0: box, 1: box, 2: box}
    gt = {3: box, 4: box, 5: box}
    assert abs(st_iou(pred, gt) - 0.0) < 1e-9


def test_partial_temporal_overlap():
    """Pred covers frames 0-4, GT covers frames 3-7.

    Union = {0,1,2,3,4,5,6,7} (8 frames).
    Frames 0,1,2: only in pred -> IoU = 0.
    Frames 5,6,7: only in GT -> IoU = 0.
    Frames 3,4: both present with identical boxes -> IoU = 1.
    Expected: (0+0+0+1+1+0+0+0) / 8 = 2/8 = 0.25
    """
    box = _box(0, 0, 10, 10)
    pred = {i: box for i in range(5)}   # frames 0-4
    gt = {i: box for i in range(3, 8)}  # frames 3-7
    expected = 2.0 / 8.0
    assert abs(st_iou(pred, gt) - expected) < 1e-9


def test_spatial_partial_overlap():
    """Both tubes cover the same frames but boxes only partially overlap.

    Box A: (0,0,10,10)  area=100
    Box B: (5,5,15,15)  area=100
    Intersection: (5,5,10,10) area=25
    Union: 200-25=175  IoU = 25/175 = 1/7 ≈ 0.142857
    """
    box_a = _box(0, 0, 10, 10)
    box_b = _box(5, 5, 15, 15)
    pred = {0: box_a, 1: box_a}
    gt = {0: box_b, 1: box_b}
    expected_per_frame = 25.0 / 175.0
    assert abs(st_iou(pred, gt) - expected_per_frame) < 1e-6


def test_absent_frames_contribute_zero():
    """Frames present in only one tube contribute 0.0 to the mean.

    GT has frames 0,1,2. Pred has frames 1,2.
    Frame 0: only in GT -> 0.
    Frames 1,2: both present, identical boxes -> 1.
    Union size = 3.  Expected = (0+1+1)/3 = 2/3.
    """
    box = _box(20, 20, 40, 40)
    pred = {1: box, 2: box}
    gt = {0: box, 1: box, 2: box}
    expected = 2.0 / 3.0
    assert abs(st_iou(pred, gt) - expected) < 1e-9


def test_both_empty():
    """Both tubes empty -> 0.0 (documented convention)."""
    assert st_iou({}, {}) == 0.0

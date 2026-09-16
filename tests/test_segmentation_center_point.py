"""Unit tests for MobileSAMSegmenter's box_refine.use_center_point_prompt
support (segment_box_cached / segment_box) -- passing the original box's
own center as a positive point prompt alongside the box prompt, and
isolating the mask to the connected component the point actually anchors.
Exercises the pure logic with a faked SamPredictor, without needing real
MobileSAM weights or GPU."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.models.segmentation import MobileSAMSegmenter
from aero_eyes.types import Box


def _make_segmenter() -> MobileSAMSegmenter:
    """Build a MobileSAMSegmenter instance without running __init__ (which
    needs real MobileSAM weights) -- only the attributes the methods under
    test touch are set."""
    seg = object.__new__(MobileSAMSegmenter)
    seg._available = True
    return seg


def test_segment_box_cached_passes_no_point_by_default():
    """use_center_point defaults to False -- point_coords/point_labels
    stay None, unchanged from before this option existed."""
    seg = _make_segmenter()
    captured = {}

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            captured["point_coords"] = point_coords
            captured["point_labels"] = point_labels
            mask = np.zeros((50, 50), dtype=bool)
            mask[10:20, 10:20] = True
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    result = seg.segment_box_cached(Box(10, 10, 20, 20))

    assert captured["point_coords"] is None
    assert captured["point_labels"] is None
    assert result is not None and result.any()


def test_segment_box_cached_use_center_point_passes_box_center():
    """use_center_point=True passes the ORIGINAL (pre-margin) box's own
    center as a single positive point, even when `margin` expands the box
    prompt itself."""
    seg = _make_segmenter()
    captured = {}

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            captured["point_coords"] = point_coords
            captured["point_labels"] = point_labels
            captured["box"] = box
            mask = np.ones((100, 100), dtype=bool)
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    box = Box(10, 20, 30, 40)  # center = (20, 30)
    seg.segment_box_cached(box, margin=0.5, use_center_point=True)

    assert captured["point_coords"] == pytest.approx(np.array([[20.0, 30.0]]))
    assert list(captured["point_labels"]) == [1]
    # box prompt itself IS expanded by margin, independent of the point.
    bw, bh = 20, 20
    assert captured["box"] == pytest.approx(
        np.array([10 - bw * 0.5, 20 - bh * 0.5, 30 + bw * 0.5, 40 + bh * 0.5])
    )


def test_segment_box_cached_use_center_point_isolates_component_at_point():
    """With use_center_point=True, a mask containing TWO disconnected blobs
    (the real object at the point, plus an unrelated same-colored blob
    elsewhere in the full-frame mask) must be trimmed down to only the
    component the point actually touches."""
    seg = _make_segmenter()

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True   # real object, contains the point (30,30)
    mask[70:90, 70:90] = True   # unrelated confuser blob, disconnected

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    box = Box(20, 20, 40, 40)  # center = (30, 30), inside the real object
    result = seg.segment_box_cached(box, use_center_point=True)

    assert result[25, 25] == True   # real object kept
    assert result[75, 75] == False  # confuser blob dropped


def test_segment_box_use_center_point_isolates_component_at_point_in_crop_local_coords():
    """segment_box() crops the frame first -- the point prompt and the
    component-isolation lookup must both be in CROP-LOCAL coordinates, not
    full-frame ones."""
    seg = _make_segmenter()
    seg.min_area_frac = 0.0
    seg.max_area_frac = 1.0
    seg.score_ratio_floor = 0.0
    seg.max_border_touch_frac = 1.0

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    box = Box(50, 50, 70, 70, score=0.5)  # center = (60, 60) in frame coords

    captured = {}

    class _FakePredictor:
        def set_image(self, img):
            pass

        def predict(self, point_coords, point_labels, box, multimask_output):
            captured["point_coords"] = point_coords
            # Crop spans roughly (40,40)-(80,80) given context_margin=0.2 on
            # a 20x20 box -- box center (60,60) in frame coords maps to
            # (20,20) in crop-local coords (60-40=20).
            mask = np.zeros((40, 40), dtype=bool)
            mask[15:25, 15:25] = True   # contains crop-local point (20,20)
            mask[0:5, 0:5] = True        # unrelated blob, disconnected
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    mask, offset = seg.segment_box(frame, box, context_margin=0.2, use_center_point=True)

    assert offset is not None
    assert captured["point_coords"] is not None
    # point must be crop-local, not the raw frame-space center (60, 60).
    assert not np.allclose(captured["point_coords"][0], [60.0, 60.0])
    assert mask[20, 20] == True     # real component (containing the point) kept
    assert mask[2, 2] == False      # unrelated blob dropped

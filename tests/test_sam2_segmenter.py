"""Unit tests for SAM2Segmenter (box_refine.method="sam2_native") --
exercises set_frame()/segment_box_cached()'s pure logic with a faked
SAM2ImagePredictor, without needing real SAM2 weights, hydra, or GPU. Same
pattern as test_segmentation_center_point.py (MobileSAMSegmenter) since
SAM2Segmenter mirrors that class's segment_box_cached shape exactly --
both wrap a SAM-family box-prompted predict(point_coords, point_labels,
box, multimask_output) -> (masks, scores, logits)."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.models.segmentation import SAM2Segmenter
from aero_eyes.types import Box


def _make_segmenter() -> SAM2Segmenter:
    """Build a SAM2Segmenter instance without running __init__ (which needs
    hydra/omegaconf + a real SAM2 checkpoint download) -- only the
    attributes the methods under test touch are set."""
    seg = object.__new__(SAM2Segmenter)
    seg._available = True
    seg._predictor = None
    seg.min_area_frac = 0.05
    seg.max_area_frac = 0.95
    seg.score_ratio_floor = 0.85
    seg.max_border_touch_frac = 0.02
    seg.use_point_prompt = True
    return seg


def test_segment_box_cached_passes_no_point_by_default():
    seg = _make_segmenter()
    captured = {}

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            captured["point_coords"] = point_coords
            captured["point_labels"] = point_labels
            captured["box"] = box
            captured["multimask_output"] = multimask_output
            mask = np.zeros((50, 50), dtype=bool)
            mask[10:20, 10:20] = True
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    result = seg.segment_box_cached(Box(10, 10, 20, 20))

    assert captured["point_coords"] is None
    assert captured["point_labels"] is None
    assert captured["multimask_output"] is True
    assert result is not None and result.any()


def test_segment_box_cached_use_center_point_passes_box_center():
    seg = _make_segmenter()
    captured = {}

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            captured["point_coords"] = point_coords
            captured["point_labels"] = point_labels
            mask = np.zeros((50, 50), dtype=bool)
            mask[5:25, 5:25] = True
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    box = Box(10, 10, 20, 20)
    seg.segment_box_cached(box, margin=0.5, use_center_point=True)

    assert captured["point_coords"].tolist() == [[15.0, 15.0]]  # original box's own center
    assert captured["point_labels"].tolist() == [1]


def test_segment_box_cached_use_center_point_isolates_component_at_point():
    seg = _make_segmenter()
    box = Box(10, 10, 20, 20)  # center = (15, 15)

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            mask = np.zeros((50, 50), dtype=bool)
            mask[5:25, 5:25] = True       # component containing the point (15,15)
            mask[30:40, 30:40] = True     # unrelated second component elsewhere
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    result = seg.segment_box_cached(box, use_center_point=True)

    assert result is not None
    assert result[15, 15]
    assert not result[35, 35], "unrelated component elsewhere must be dropped"


def test_segment_box_cached_picks_highest_scoring_mask():
    seg = _make_segmenter()

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            m0 = np.zeros((50, 50), dtype=bool)
            m0[0:5, 0:5] = True
            m1 = np.zeros((50, 50), dtype=bool)
            m1[10:20, 10:20] = True
            return np.array([m0, m1]), np.array([0.3, 0.95]), None

    seg._predictor = _FakePredictor()
    result = seg.segment_box_cached(Box(10, 10, 20, 20))

    assert result[15, 15] and not result[2, 2], "must pick mask index 1 (higher predicted score)"


def test_segment_box_cached_returns_none_for_empty_mask():
    seg = _make_segmenter()

    class _FakePredictor:
        def predict(self, point_coords, point_labels, box, multimask_output):
            return np.array([np.zeros((50, 50), dtype=bool)]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    assert seg.segment_box_cached(Box(10, 10, 20, 20)) is None


def test_segment_box_cached_returns_none_when_unavailable():
    seg = object.__new__(SAM2Segmenter)
    seg._available = False
    seg._predictor = None
    assert seg.segment_box_cached(Box(10, 10, 20, 20)) is None


def test_segment_box_cached_swallows_predictor_exception():
    seg = _make_segmenter()

    class _FakePredictor:
        def predict(self, **kwargs):
            raise RuntimeError("boom")

    seg._predictor = _FakePredictor()
    assert seg.segment_box_cached(Box(10, 10, 20, 20)) is None


def test_set_frame_false_when_unavailable():
    seg = object.__new__(SAM2Segmenter)
    seg._available = False
    seg._predictor = None
    assert seg.set_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is False


def test_set_frame_true_on_success():
    seg = _make_segmenter()

    class _FakePredictor:
        def set_image(self, frame_rgb):
            pass

    seg._predictor = _FakePredictor()
    assert seg.set_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is True


def test_set_frame_false_on_predictor_exception():
    seg = _make_segmenter()

    class _FakePredictor:
        def set_image(self, frame_rgb):
            raise RuntimeError("boom")

    seg._predictor = _FakePredictor()
    assert seg.set_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is False


def test_segment_returns_passthrough_when_unavailable():
    seg = object.__new__(SAM2Segmenter)
    seg._available = False
    seg._predictor = None
    result = seg.segment(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result.all()
    assert result.shape == (50, 50)


def test_segment_picks_plausible_mask():
    seg = _make_segmenter()

    class _FakePredictor:
        def set_image(self, frame_rgb):
            pass

        def predict(self, point_coords, point_labels, box, multimask_output):
            mask = np.zeros((100, 100), dtype=bool)
            mask[10:90, 10:90] = True
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    result = seg.segment(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result[50, 50] == True
    assert not result.all()


def test_segment_falls_back_to_passthrough_when_area_implausible():
    seg = _make_segmenter()

    class _FakePredictor:
        def set_image(self, frame_rgb):
            pass

        def predict(self, point_coords, point_labels, box, multimask_output):
            mask = np.zeros((100, 100), dtype=bool)
            mask[45:55, 45:55] = True  # 1% of frame -- below min_area_frac=0.05
            return np.array([mask]), np.array([0.9]), None

    seg._predictor = _FakePredictor()
    result = seg.segment(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.all()


def test_segment_falls_back_to_passthrough_on_predict_exception():
    seg = _make_segmenter()

    class _FakePredictor:
        def set_image(self, frame_rgb):
            pass

        def predict(self, **kwargs):
            raise RuntimeError("boom")

    seg._predictor = _FakePredictor()
    result = seg.segment(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result.all()

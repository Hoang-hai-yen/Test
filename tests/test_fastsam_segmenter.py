"""Unit tests for FastSAMSegmenter -- box_refine.method="fastsam_dense".
FastSAM has no prompt-conditioned mask generation, so segment_box_cached
just MATCHES a given box against whatever instances set_frame()'s
segment-everything pass already cached -- these tests exercise that
matching logic directly (bypassing __init__, which needs the real
ultralytics FastSAM model) and the set_frame() caching itself with a faked
ultralytics-style results object."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.models.segmentation import FastSAMSegmenter
from aero_eyes.types import Box


def _make_segmenter() -> FastSAMSegmenter:
    """Build a FastSAMSegmenter instance without running __init__ (which
    needs the real ultralytics FastSAM model)."""
    seg = object.__new__(FastSAMSegmenter)
    seg._model = None
    seg._cached_masks = []
    seg._cached_boxes = []
    seg.min_area_frac = 0.05
    seg.max_area_frac = 0.95
    seg.max_border_touch_frac = 0.02
    seg.use_point_prompt = True
    return seg


def _rig_cache(seg, boxes_and_masks):
    """boxes_and_masks: list of (Box, mask_slice) where mask_slice is a
    (y1,y2,x1,y2) tuple defining a True rectangle on a shared 100x100 canvas."""
    seg._cached_boxes = []
    seg._cached_masks = []
    for box, (y1, y2, x1, x2) in boxes_and_masks:
        mask = np.zeros((100, 100), dtype=bool)
        mask[y1:y2, x1:x2] = True
        seg._cached_boxes.append(box)
        seg._cached_masks.append(mask)


def test_segment_box_cached_returns_none_when_nothing_cached():
    seg = _make_segmenter()
    result = seg.segment_box_cached(Box(10, 10, 20, 20))
    assert result is None


def test_segment_box_cached_picks_best_iou_match():
    seg = _make_segmenter()
    _rig_cache(seg, [
        (Box(10, 10, 20, 20), (10, 20, 10, 20)),   # good match
        (Box(60, 60, 70, 70), (60, 70, 60, 70)),   # unrelated, far away
    ])
    result = seg.segment_box_cached(Box(11, 11, 19, 19))
    assert result is not None
    assert result[15, 15] == True    # matched the first (nearby) instance
    assert result[65, 65] == False   # not the unrelated one


def test_segment_box_cached_returns_none_when_no_overlap_at_all():
    seg = _make_segmenter()
    _rig_cache(seg, [(Box(60, 60, 70, 70), (60, 70, 60, 70))])
    result = seg.segment_box_cached(Box(0, 0, 5, 5))
    assert result is None


def test_segment_box_cached_applies_margin_before_matching():
    """A box whose UNEXPANDED self barely/doesn't overlap a cached
    instance, but DOES after margin expansion, should still match."""
    seg = _make_segmenter()
    _rig_cache(seg, [(Box(10, 10, 30, 30), (10, 30, 10, 30))])
    tiny_box = Box(28, 28, 32, 32)  # only a sliver overlaps the cached instance's box unexpanded
    no_margin = seg.segment_box_cached(tiny_box, margin=0.0)
    with_margin = seg.segment_box_cached(tiny_box, margin=2.0)  # expands a lot
    assert with_margin is not None
    assert with_margin[15, 15] == True


def test_segment_box_cached_use_center_point_prefers_containment_over_iou():
    """A cached instance with WORSE box-IoU but whose mask actually
    contains the prompt box's center must win over one with better IoU
    that doesn't contain it (e.g. a larger neighboring object)."""
    seg = _make_segmenter()
    box = Box(10, 10, 20, 20)  # center = (15, 15)
    _rig_cache(seg, [
        (Box(8, 8, 22, 22), (40, 60, 40, 60)),   # better box-IoU, but mask elsewhere -- doesn't contain (15,15)
        (Box(9, 9, 19, 19), (10, 20, 10, 20)),   # slightly worse box-IoU, but mask DOES contain (15,15)
    ])
    result = seg.segment_box_cached(box, use_center_point=True)
    assert result is not None
    assert result[15, 15] == True
    assert result[50, 50] == False  # not the non-containing "better IoU" one


def test_segment_box_cached_use_center_point_falls_back_to_iou_if_none_contain():
    """If NO cached instance's mask contains the point, fall back to plain
    box-IoU matching instead of returning None outright."""
    seg = _make_segmenter()
    box = Box(10, 10, 20, 20)  # center = (15, 15), not inside either mask below
    _rig_cache(seg, [
        (Box(10, 10, 20, 20), (70, 80, 70, 80)),  # best box-IoU, mask elsewhere
        (Box(60, 60, 70, 70), (60, 70, 60, 70)),  # unrelated
    ])
    result = seg.segment_box_cached(box, use_center_point=True)
    assert result is not None
    assert result[75, 75] == True


class _FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = xyxy

    def cpu(self):
        return self


class _FakeMasks:
    def __init__(self, data):
        self.data = data


class _FakeResult:
    def __init__(self, masks, boxes):
        self.masks = masks
        self.boxes = boxes


def test_set_frame_caches_instances_from_fastsam_result():
    seg = _make_segmenter()
    mask_data = np.zeros((2, 20, 20), dtype=np.float32)
    mask_data[0, 2:6, 2:6] = 1.0
    mask_data[1, 10:15, 10:15] = 1.0
    boxes_xyxy = np.array([[2, 2, 6, 6], [10, 10, 15, 15]], dtype=np.float32)

    seg._model = lambda frame_bgr, conf, iou, imgsz, verbose, retina_masks: [
        _FakeResult(_FakeMasks(_TorchLike(mask_data)), _FakeBoxes(_TorchLike(boxes_xyxy)))
    ]
    seg.conf, seg.iou, seg.imgsz = 0.2, 0.7, 640

    ok = seg.set_frame(np.zeros((20, 20, 3), dtype=np.uint8))

    assert ok is True
    assert len(seg._cached_masks) == 2
    assert len(seg._cached_boxes) == 2
    assert seg._cached_masks[0][3, 3] == True
    assert (seg._cached_boxes[1].x1, seg._cached_boxes[1].y1) == (10.0, 10.0)


def test_set_frame_returns_false_when_model_unavailable():
    seg = _make_segmenter()
    seg._model = None
    assert seg.set_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is False
    assert seg._cached_masks == []


class _TorchLike:
    """Minimal stand-in for a torch.Tensor -- only .cpu().numpy() is used
    by set_frame(), avoids a real torch dependency in this test."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_segment_returns_passthrough_when_model_unavailable():
    seg = _make_segmenter()
    result = seg.segment(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.all()
    assert result.shape == (100, 100)


def test_segment_picks_mask_matching_synthetic_box():
    """A cached instance covering most of the frame (like a real subject)
    must be picked and returned as-is (not the passthrough fallback)."""
    seg = _make_segmenter()
    h, w = 100, 100
    mask_data = np.zeros((1, h, w), dtype=np.float32)
    mask_data[0, 10:90, 10:90] = 1.0
    boxes_xyxy = np.array([[10, 10, 90, 90]], dtype=np.float32)
    seg._model = lambda frame_bgr, conf, iou, imgsz, verbose, retina_masks: [
        _FakeResult(_FakeMasks(_TorchLike(mask_data)), _FakeBoxes(_TorchLike(boxes_xyxy)))
    ]
    seg.conf, seg.iou, seg.imgsz = 0.2, 0.7, 640

    result = seg.segment(np.zeros((h, w, 3), dtype=np.uint8))
    assert result[50, 50] == True
    assert not result.all()


def test_segment_falls_back_to_passthrough_when_area_implausible():
    seg = _make_segmenter()
    h, w = 100, 100
    mask_data = np.zeros((1, h, w), dtype=np.float32)
    mask_data[0, 45:55, 45:55] = 1.0  # 1% of frame -- below min_area_frac=0.05
    boxes_xyxy = np.array([[45, 45, 55, 55]], dtype=np.float32)
    seg._model = lambda frame_bgr, conf, iou, imgsz, verbose, retina_masks: [
        _FakeResult(_FakeMasks(_TorchLike(mask_data)), _FakeBoxes(_TorchLike(boxes_xyxy)))
    ]
    seg.conf, seg.iou, seg.imgsz = 0.2, 0.7, 640

    result = seg.segment(np.zeros((h, w, 3), dtype=np.uint8))
    assert result.all()


def test_segment_falls_back_to_passthrough_when_touches_border():
    seg = _make_segmenter()
    h, w = 100, 100
    mask_data = np.zeros((1, h, w), dtype=np.float32)
    mask_data[0, 0:60, 0:60] = 1.0  # touches the top-left image border
    boxes_xyxy = np.array([[0, 0, 60, 60]], dtype=np.float32)
    seg._model = lambda frame_bgr, conf, iou, imgsz, verbose, retina_masks: [
        _FakeResult(_FakeMasks(_TorchLike(mask_data)), _FakeBoxes(_TorchLike(boxes_xyxy)))
    ]
    seg.conf, seg.iou, seg.imgsz = 0.2, 0.7, 640

    result = seg.segment(np.zeros((h, w, 3), dtype=np.uint8))
    assert result.all()

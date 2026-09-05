"""Unit tests for GeCo2Detector.sam2_refine_boxes -- box_refine.method=
"sam2_dense", GeCo2's own SAM2-based mask refinement. Exercises the
frame-pixel <-> normalized-padded-canvas coordinate conversion (the most
bug-prone part) and the fallback paths, without needing the real GECO2
repo, SAM2 checkpoint, or GPU -- MaskProcessor and _forward_scores are
faked."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.models.geco2_detector import GeCo2Detector
from aero_eyes.types import Box


def _make_detector(image_size: float = 200.0) -> GeCo2Detector:
    """Build a GeCo2Detector instance without running __init__ (which
    needs real GECO2 weights) -- only the attributes sam2_refine_boxes
    itself touches are set."""
    det = object.__new__(GeCo2Detector)
    det.image_size = image_size
    det.device = torch.device("cpu")
    det._mask_processor = None
    return det


def test_sam2_refine_boxes_converts_coordinates_both_directions():
    """frame-pixel box -> normalized canvas prompt -> canvas-pixel
    corrected box -> frame-pixel result must round-trip through `scale`
    consistently with filter_boxes_by_threshold's own inverse transform
    (px_boxes = cand_boxes_normalized / scale * image_size)."""
    det = _make_detector(image_size=200.0)
    scale = 2.0  # canvas_px = frame_px * scale
    frame_bgr = np.zeros((50, 50, 3), dtype=np.uint8)

    det._forward_scores = lambda frame_bgr, prototype: (None, None, scale, {"sentinel": True})

    captured = {}

    class _FakeMaskProcessor:
        def __call__(self, feats, outputs_wrapped):
            captured["feats"] = feats
            captured["outputs_wrapped"] = outputs_wrapped
            corrected = torch.tensor([
                [20.0, 20.0, 60.0, 60.0],  # a real refinement
                [0.0, 0.0, 0.0, 0.0],      # empty mask -> fall back
            ])
            return None, None, [corrected]

    det._get_mask_processor = lambda: _FakeMaskProcessor()

    boxes = [Box(5, 5, 15, 15, score=0.9), Box(1, 1, 2, 2, score=0.5)]
    refined = det.sam2_refine_boxes(frame_bgr, prototype={}, boxes=boxes)

    # corrected canvas-px (20,20,60,60) / scale(2.0) -> frame-px (10,10,30,30)
    assert (refined[0].x1, refined[0].y1, refined[0].x2, refined[0].y2) == (10.0, 10.0, 30.0, 30.0)
    assert refined[0].score == 0.9

    # all-zero corrected box -> unchanged (same object)
    assert refined[1] is boxes[1]

    # input prompt: frame-px (5,5,15,15) * scale(2.0) / image_size(200.0)
    norm = captured["outputs_wrapped"][0]["pred_boxes"][0][0].tolist()
    assert norm == pytest.approx([0.05, 0.05, 0.15, 0.15])
    assert captured["feats"] == {"sentinel": True}


def test_sam2_refine_boxes_empty_input_returns_empty():
    det = _make_detector()
    assert det.sam2_refine_boxes(np.zeros((10, 10, 3), dtype=np.uint8), {}, []) == []


def test_sam2_refine_boxes_falls_back_when_mask_processor_unavailable():
    det = _make_detector()
    det._get_mask_processor = lambda: None
    boxes = [Box(1, 1, 5, 5, score=0.7)]
    refined = det.sam2_refine_boxes(np.zeros((10, 10, 3), dtype=np.uint8), {}, boxes)
    assert refined is boxes


def test_sam2_refine_boxes_falls_back_when_forward_pass_raises():
    det = _make_detector(image_size=100.0)
    det._forward_scores = lambda frame_bgr, prototype: (None, None, 1.0, {})

    class _RaisingMaskProcessor:
        def __call__(self, feats, outputs_wrapped):
            raise RuntimeError("boom")

    det._get_mask_processor = lambda: _RaisingMaskProcessor()
    boxes = [Box(1, 1, 5, 5, score=0.7)]
    refined = det.sam2_refine_boxes(np.zeros((10, 10, 3), dtype=np.uint8), {}, boxes)
    assert refined is boxes

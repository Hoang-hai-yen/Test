"""Unit tests for GeCo2Detector.filter_boxes_by_threshold's
stage123_geco2.min_box_area_enabled/min_box_area gate -- rejects a
degenerate, near-zero-AREA box the regression head produced, without
needing the real GECO2 repo/weights or GPU."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.models.geco2_detector import GeCo2Detector


def _make_detector(
    image_size: float = 100.0,
    nms_iou: float = 0.99,
    topk_per_keyframe: int = 10,
    min_box_area_enabled: bool = False,
    min_box_area: int = 24,
) -> GeCo2Detector:
    """Build a GeCo2Detector instance without running __init__ (which needs
    real GECO2 weights) -- only the attributes filter_boxes_by_threshold
    itself touches are set."""
    det = object.__new__(GeCo2Detector)
    det.image_size = image_size
    det.nms_iou = nms_iou
    det.topk_per_keyframe = topk_per_keyframe
    det.min_box_area_enabled = min_box_area_enabled
    det.min_box_area = min_box_area
    det.peak_contrast_filter_enabled = False
    det.peak_contrast_radius = 4
    det.peak_contrast_hard_reject = False
    det.peak_contrast_min_z = 0.5
    return det


def _boxes_scale1_size100():
    """Three well-separated boxes (no NMS overlap) in a 100x100 frame,
    scale=1.0 so px_boxes == normalized_boxes * image_size exactly:
      A: (0,0)-(10,3)     area=30  (thin-but-long: min side 3px, area>=24)
      B: (50,50)-(52,52)  area=4   (degenerate: both sides tiny, area<24)
      C: (80,80)-(85,85)  area=25  (just above the default floor of 24)
    """
    pred_boxes = torch.tensor([
        [0.00, 0.00, 0.10, 0.03],
        [0.50, 0.50, 0.52, 0.52],
        [0.80, 0.80, 0.85, 0.85],
    ])
    box_v = torch.tensor([0.9, 0.9, 0.9])
    frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    return pred_boxes, box_v, frame_bgr


def test_min_box_area_disabled_keeps_degenerate_box():
    """Default (disabled) -- unchanged behavior, a degenerate tiny box is
    NOT rejected."""
    det = _make_detector(min_box_area_enabled=False)
    pred_boxes, box_v, frame_bgr = _boxes_scale1_size100()
    results = det.filter_boxes_by_threshold(pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5)
    areas = sorted(b.area() for b in results)
    assert areas == pytest.approx([4.0, 25.0, 30.0])


def test_min_box_area_enabled_rejects_only_degenerate_box():
    """Enabled with the default floor (24) -- rejects ONLY the box whose
    area falls below it; the thin-but-long box (min side 3px, area 30)
    survives because this is an AREA floor, not a min-side-length one (see
    min_box_area's own docstring: real GT boxes in this project can have a
    2px-thin side but never an area <= 16)."""
    det = _make_detector(min_box_area_enabled=True, min_box_area=24)
    pred_boxes, box_v, frame_bgr = _boxes_scale1_size100()
    results = det.filter_boxes_by_threshold(pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5)
    areas = sorted(b.area() for b in results)
    assert areas == pytest.approx([25.0, 30.0])


def test_min_box_area_boundary_is_inclusive_of_the_floor():
    """A box exactly AT min_box_area survives (only strictly-below is rejected)."""
    det = _make_detector(min_box_area_enabled=True, min_box_area=25)
    pred_boxes, box_v, frame_bgr = _boxes_scale1_size100()
    results = det.filter_boxes_by_threshold(pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5)
    areas = sorted(b.area() for b in results)
    assert areas == pytest.approx([25.0, 30.0])

"""Unit tests for aero_eyes.utils.box_refine -- the box_refine.enabled
feature that sharpens an imprecise detection/tracking box via a per-box
segmentation pass (SAM or GrabCut)."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.types import Box
from aero_eyes.utils.box_refine import refine_box, refine_box_with_grabcut, refine_box_with_sam


def _synthetic_frame(size: int = 120) -> np.ndarray:
    """Black frame with a bright, high-contrast 40x40 square at (40,40)-(80,80)."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[40:80, 40:80] = 255
    return frame


def test_refine_box_with_grabcut_tightens_a_loose_box():
    """A seed box padded well beyond the real object should shrink toward
    the object's actual extent after GrabCut refinement. Needs a nonzero
    context_margin -- GrabCut's GMM init requires SOME background-labeled
    pixels around the seed rect within the cropped region; margin=0.0
    would crop exactly to the rect, leaving none (see
    test_refine_box_with_grabcut_zero_margin_falls_back_gracefully)."""
    frame = _synthetic_frame()
    loose_box = Box(20, 20, 100, 100, score=0.8)  # true square is 40-80, well inside

    refined = refine_box_with_grabcut(frame, loose_box, context_margin=0.15)

    # Refined box should be meaningfully tighter than the loose seed...
    assert (refined.x2 - refined.x1) < (loose_box.x2 - loose_box.x1)
    assert (refined.y2 - refined.y1) < (loose_box.y2 - loose_box.y1)
    # ...and roughly centered on the true square (40-80), not the seed's own center.
    cx = (refined.x1 + refined.x2) / 2
    cy = (refined.y1 + refined.y2) / 2
    assert 30 < cx < 90
    assert 30 < cy < 90
    # score is preserved from the original box, not overwritten.
    assert refined.score == 0.8


def test_refine_box_with_grabcut_zero_margin_falls_back_gracefully():
    """context_margin=0.0 crops EXACTLY to the seed box, leaving no
    background-labeled pixels within the crop for GrabCut's GMM init --
    OpenCV raises cv2.error (initGMMs: bgdSamples empty) in that case.
    refine_box_with_grabcut must catch it and fall back to the original
    box, not propagate the exception."""
    frame = _synthetic_frame()
    box = Box(20, 20, 100, 100, score=0.8)
    refined = refine_box_with_grabcut(frame, box, context_margin=0.0)
    assert refined is box


def test_refine_box_with_grabcut_degenerate_box_returns_unchanged():
    """A zero-area box can't seed GrabCut -- falls back to itself, no crash."""
    frame = _synthetic_frame()
    degenerate = Box(10, 10, 10, 10)
    refined = refine_box_with_grabcut(frame, degenerate, context_margin=0.2)
    assert refined is degenerate


def test_refine_box_with_sam_uses_segmenter_mask_and_offset():
    """refine_box_with_sam should map the mask's tight bbox back into
    frame coordinates using the (x1,y1) crop offset segment_box reports."""
    frame = _synthetic_frame()
    box = Box(35, 35, 85, 85, score=0.6)

    # Fake segmenter: mask is a 10x10 block at (5,5)-(15,15) WITHIN a crop
    # whose top-left corner sits at (30, 30) in frame coordinates.
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:15, 5:15] = True

    class _FakeSegmenter:
        def segment_box(self, frame_bgr, box, context_margin):
            return mask, (30, 30)

    refined = refine_box_with_sam(_FakeSegmenter(), frame, box, context_margin=0.2)
    # mask_bbox of that block is (5,5,15,15); offset by (30,30) -> (35,35,45,45).
    assert (refined.x1, refined.y1, refined.x2, refined.y2) == (35.0, 35.0, 45.0, 45.0)
    assert refined.score == 0.6


def test_refine_box_with_sam_falls_back_when_segmenter_none():
    """No segmenter available (e.g. weights missing) -> box unchanged."""
    frame = _synthetic_frame()
    box = Box(35, 35, 85, 85, score=0.6)
    refined = refine_box_with_sam(None, frame, box, context_margin=0.2)
    assert refined is box


def test_refine_box_with_sam_falls_back_when_mask_is_none():
    """segment_box reporting failure (None, None) -> box unchanged."""
    frame = _synthetic_frame()
    box = Box(35, 35, 85, 85, score=0.6)

    class _FailingSegmenter:
        def segment_box(self, frame_bgr, box, context_margin):
            return None, None

    refined = refine_box_with_sam(_FailingSegmenter(), frame, box, context_margin=0.2)
    assert refined is box


def test_refine_box_dispatcher_routes_by_method():
    frame = _synthetic_frame()
    box = Box(35, 35, 85, 85, score=0.6)

    class _FakeSegmenter:
        called = False

        def segment_box(self, frame_bgr, box, context_margin):
            _FakeSegmenter.called = True
            return None, None  # falls back to unchanged; just checking routing

    refine_box("sam", frame, box, context_margin=0.2, segmenter=_FakeSegmenter())
    assert _FakeSegmenter.called is True

    # grabcut path doesn't need (or use) a segmenter.
    refined = refine_box("grabcut", frame, box, context_margin=0.0, segmenter=None)
    assert isinstance(refined, Box)


def test_refine_box_unknown_method_raises():
    frame = _synthetic_frame()
    box = Box(35, 35, 85, 85)
    with pytest.raises(ValueError, match="Unknown box_refine.method"):
        refine_box("nearest-neighbor", frame, box, context_margin=0.2)


def test_refine_box_min_iou_gate_rejects_wildly_different_region():
    """Regression test for a real failure mode: on a real run, SAM latched
    onto an entirely different region than the original box, TANKING
    ST-IoU instead of improving it. min_iou_with_original must reject a
    "refined" box that barely overlaps the original and fall back to it."""
    frame = _synthetic_frame()
    original = Box(35, 35, 85, 85, score=0.6)

    class _WildSegmenter:
        """Always returns a mask far away from the original box -- e.g. a
        confuser or background clutter elsewhere in the padded crop."""
        def segment_box(self, frame_bgr, box, context_margin):
            mask = np.zeros((100, 100), dtype=bool)
            mask[0:5, 0:5] = True  # tiny region nowhere near the original box
            return mask, (0, 0)

    # No gate (0.0, default in refine_box's own signature) -- accepts the
    # wild region verbatim.
    ungated = refine_box("sam", frame, original, context_margin=0.2, segmenter=_WildSegmenter())
    assert (ungated.x1, ungated.y1, ungated.x2, ungated.y2) == (0.0, 0.0, 5.0, 5.0)

    # With the gate on (matches BoxRefineConfig's own default of 0.3) --
    # the wild region's IoU with `original` is 0.0, well below 0.3, so the
    # original box must be returned unchanged instead.
    gated = refine_box(
        "sam", frame, original, context_margin=0.2,
        segmenter=_WildSegmenter(), min_iou_with_original=0.3,
    )
    assert gated is original


def test_refine_box_min_iou_gate_accepts_a_genuine_tightening():
    """The gate must NOT reject a legitimate refinement that stays well
    within/near the original box -- only wildly different regions."""
    frame = _synthetic_frame()
    original = Box(20, 20, 100, 100, score=0.6)

    class _TighteningSegmenter:
        def segment_box(self, frame_bgr, box, context_margin):
            # Mask covers (40,40)-(80,80) within a crop offset at (0, 0) --
            # a real tightening toward the frame's own true square, IoU
            # with `original` is (40*40)/(80*80) = 0.25... let's make it
            # comfortably overlapping instead so this exercises "accepted".
            mask = np.zeros((120, 120), dtype=bool)
            mask[25:95, 25:95] = True
            return mask, (0, 0)

    refined = refine_box(
        "sam", frame, original, context_margin=0.2,
        segmenter=_TighteningSegmenter(), min_iou_with_original=0.3,
    )
    assert (refined.x1, refined.y1, refined.x2, refined.y2) == (25.0, 25.0, 95.0, 95.0)

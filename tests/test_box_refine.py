"""Unit tests for aero_eyes.utils.box_refine -- the box_refine.enabled
feature that sharpens an imprecise detection/tracking box via a per-box
segmentation pass (SAM or GrabCut)."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.config import AdaptiveContextMarginConfig
from aero_eyes.types import Box
from aero_eyes.utils.box_refine import (
    apply_iou_gate, refine_box, refine_box_with_grabcut, refine_box_with_sam, refine_boxes_dense,
    scale_context_margin,
)


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
        def segment_box(self, frame_bgr, box, context_margin, use_center_point=False):
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
        def segment_box(self, frame_bgr, box, context_margin, use_center_point=False):
            return None, None

    refined = refine_box_with_sam(_FailingSegmenter(), frame, box, context_margin=0.2)
    assert refined is box


def test_refine_box_dispatcher_routes_by_method():
    frame = _synthetic_frame()
    box = Box(35, 35, 85, 85, score=0.6)

    class _FakeSegmenter:
        called = False

        def segment_box(self, frame_bgr, box, context_margin, use_center_point=False):
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
    with pytest.raises(ValueError, match="Unknown or unsupported box_refine.method"):
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
        def segment_box(self, frame_bgr, box, context_margin, use_center_point=False):
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
        def segment_box(self, frame_bgr, box, context_margin, use_center_point=False):
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


class _DenseSegmenter:
    """Fake segmenter for refine_boxes_dense: set_frame() must be called
    exactly ONCE (records call count), segment_box_cached() returns a
    fixed full-frame mask per box (independent of which box, to isolate
    testing "one encode, N box calls" from mask-selection logic)."""

    def __init__(self, mask=None, set_frame_ok: bool = True):
        self.set_frame_calls = 0
        self.mask = mask
        self._set_frame_ok = set_frame_ok

    def set_frame(self, frame_bgr):
        self.set_frame_calls += 1
        return self._set_frame_ok

    def segment_box_cached(self, box, margin=0.0, use_center_point=False):
        return self.mask


def test_refine_boxes_dense_encodes_frame_once_for_all_boxes():
    frame = _synthetic_frame()
    boxes = [Box(30, 30, 90, 90), Box(10, 10, 50, 50), Box(60, 60, 100, 100)]

    mask = np.zeros((120, 120), dtype=bool)
    mask[40:80, 40:80] = True  # tight bbox = (40,40,80,80) for every box
    segmenter = _DenseSegmenter(mask=mask)

    refined = refine_boxes_dense(segmenter, frame, boxes, min_iou_with_original=0.0)

    assert segmenter.set_frame_calls == 1  # ONE shared encode, not one per box
    assert len(refined) == 3
    for r in refined:
        assert (r.x1, r.y1, r.x2, r.y2) == (40.0, 40.0, 80.0, 80.0)


def test_refine_boxes_dense_falls_back_when_set_frame_fails():
    frame = _synthetic_frame()
    boxes = [Box(30, 30, 90, 90), Box(10, 10, 50, 50)]
    segmenter = _DenseSegmenter(set_frame_ok=False)

    refined = refine_boxes_dense(segmenter, frame, boxes, min_iou_with_original=0.0)
    assert refined == boxes  # unchanged, in original order


def test_refine_boxes_dense_none_segmenter_falls_back():
    frame = _synthetic_frame()
    boxes = [Box(30, 30, 90, 90)]
    refined = refine_boxes_dense(None, frame, boxes, min_iou_with_original=0.0)
    assert refined == boxes


def test_refine_boxes_dense_min_iou_gate_applies_per_box():
    """The gate is checked independently for each box -- a box whose
    shared mask overlaps it well is accepted, one whose mask barely
    overlaps it is rejected and kept unchanged."""
    frame = _synthetic_frame()
    near_box = Box(35, 35, 85, 85)     # overlaps the mask's bbox (40,40,80,80) well
    far_box = Box(0, 0, 5, 5)          # nowhere near the mask's bbox

    mask = np.zeros((120, 120), dtype=bool)
    mask[40:80, 40:80] = True
    segmenter = _DenseSegmenter(mask=mask)

    refined = refine_boxes_dense(segmenter, frame, [near_box, far_box], min_iou_with_original=0.3)

    assert (refined[0].x1, refined[0].y1, refined[0].x2, refined[0].y2) == (40.0, 40.0, 80.0, 80.0)
    assert refined[1] is far_box  # rejected by the gate, unchanged


def test_apply_iou_gate_keeps_good_overlap_rejects_poor_overlap():
    """apply_iou_gate is box_refine.method="sam2_dense"'s equivalent of the
    gate baked into refine_box/refine_boxes_dense -- since GeCo2Detector.
    sam2_refine_boxes can't route through this module's segmenter-based
    dispatch (it needs live GeCo2 model state), it applies this shared
    helper itself afterward."""
    original = [Box(0, 0, 10, 10), Box(0, 0, 10, 10)]
    refined = [Box(1, 1, 9, 9), Box(50, 50, 60, 60)]  # good overlap, no overlap

    gated = apply_iou_gate(refined, original, min_iou_with_original=0.3)

    assert gated[0] is refined[0]
    assert gated[1] is original[1]


def test_apply_iou_gate_noop_when_threshold_is_zero():
    refined = [Box(50, 50, 60, 60)]
    original = [Box(0, 0, 10, 10)]
    assert apply_iou_gate(refined, original, min_iou_with_original=0.0) is refined


# ---------------------------------------------------------------------------
# scale_context_margin / box_refine.adaptive_context_margin
# ---------------------------------------------------------------------------

def test_scale_context_margin_disabled_or_none_is_noop():
    tiny_box = Box(0, 0, 15, 12)  # geometric-mean side ~13.4px
    assert scale_context_margin(tiny_box, 0.5, None) == 0.5
    off_cfg = AdaptiveContextMarginConfig(enabled=False)
    assert scale_context_margin(tiny_box, 0.5, off_cfg) == 0.5


def test_scale_context_margin_tiny_box_gets_min_ratio():
    """A box at/below min_size_px gets margin scaled to exactly min_ratio
    (0.0 default = no margin at all) -- the helmet-sized case."""
    ac_cfg = AdaptiveContextMarginConfig(enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0)
    tiny_box = Box(0, 0, 15, 12)  # side ~13.4px < min_size_px=20
    assert scale_context_margin(tiny_box, 0.5, ac_cfg) == pytest.approx(0.0)


def test_scale_context_margin_large_box_gets_full_margin():
    """A box at/above max_size_px gets the FULL configured margin,
    unscaled -- the motorbike-sized case."""
    ac_cfg = AdaptiveContextMarginConfig(enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0)
    large_box = Box(0, 0, 200, 200)  # side 200px > max_size_px=100
    assert scale_context_margin(large_box, 0.5, ac_cfg) == pytest.approx(0.5)


def test_scale_context_margin_interpolates_linearly_between():
    ac_cfg = AdaptiveContextMarginConfig(enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0)
    mid_box = Box(0, 0, 60, 60)  # side=60 -> t=(60-20)/(100-20)=0.5
    assert scale_context_margin(mid_box, 0.5, ac_cfg) == pytest.approx(0.25)


def test_scale_context_margin_min_ratio_floor():
    """A nonzero min_ratio keeps SOME margin even for the smallest boxes,
    instead of dropping to exactly 0."""
    ac_cfg = AdaptiveContextMarginConfig(enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.2)
    tiny_box = Box(0, 0, 5, 5)
    assert scale_context_margin(tiny_box, 0.5, ac_cfg) == pytest.approx(0.1)  # 0.5 * 0.2


def test_scale_context_margin_relative_undersized_box_gets_full_margin():
    """A box that's tiny in ABSOLUTE terms (would normally get min_ratio)
    but is anomalously small relative to this SAME sample's own reference
    size (e.g. a motorbike video where this one box badly undersizes the
    object) should get the FULL, unscaled margin instead -- it needs MORE
    room for SAM to reach the true boundary, not less."""
    ac_cfg = AdaptiveContextMarginConfig(
        enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0,
        relative_to_sample_median=True, relative_undersize_ratio=0.5,
    )
    undersized_box = Box(0, 0, 15, 12)  # side ~13.4px -- would get min_ratio=0.0 on its own
    # Sample's own reference size is 150px -- this box is ~9% of that, well under 0.5.
    assert scale_context_margin(undersized_box, 0.5, ac_cfg, sample_reference_size=150.0) == pytest.approx(0.5)


def test_scale_context_margin_relative_check_leaves_genuinely_small_sample_alone():
    """A sample whose OWN reference size is itself small (e.g. a helmet
    video) must be unaffected by the relative check -- its boxes stay
    close to their own reference size, so the ratio never trips, and the
    normal absolute min_ratio shrink still applies."""
    ac_cfg = AdaptiveContextMarginConfig(
        enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0,
        relative_to_sample_median=True, relative_undersize_ratio=0.5,
    )
    tiny_box = Box(0, 0, 15, 12)  # side ~13.4px
    # Sample's own reference size is also ~14px -- ratio ~0.96, well above 0.5.
    assert scale_context_margin(tiny_box, 0.5, ac_cfg, sample_reference_size=14.0) == pytest.approx(0.0)


def test_scale_context_margin_relative_check_disabled_by_default():
    """relative_to_sample_median defaults to False -- passing
    sample_reference_size has no effect unless explicitly enabled."""
    ac_cfg = AdaptiveContextMarginConfig(enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0)
    undersized_box = Box(0, 0, 15, 12)
    assert scale_context_margin(undersized_box, 0.5, ac_cfg, sample_reference_size=150.0) == pytest.approx(0.0)


def test_scale_context_margin_relative_check_no_reference_falls_back_to_absolute():
    """relative_to_sample_median=True but no sample_reference_size given
    (e.g. no confident detections yet) -- falls back to the absolute
    min/max_size_px curve instead of erroring."""
    ac_cfg = AdaptiveContextMarginConfig(
        enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0,
        relative_to_sample_median=True,
    )
    undersized_box = Box(0, 0, 15, 12)
    assert scale_context_margin(undersized_box, 0.5, ac_cfg, sample_reference_size=None) == pytest.approx(0.0)


def test_refine_boxes_dense_uses_per_box_scaled_margin():
    """The margin passed to segment_box_cached must be computed PER BOX
    from its own size, not one shared value for the whole frame -- a tiny
    box and a large box on the SAME frame get different margins."""
    frame = _synthetic_frame()
    tiny_box = Box(0, 0, 15, 12)
    large_box = Box(0, 0, 200, 200)

    seen_margins = []

    class _RecordingSegmenter:
        def set_frame(self, frame_bgr):
            return True

        def segment_box_cached(self, box, margin=0.0, use_center_point=False):
            seen_margins.append(margin)
            return None  # falls back to the original box -- only margin tracking matters here

    ac_cfg = AdaptiveContextMarginConfig(enabled=True, min_size_px=20.0, max_size_px=100.0, min_ratio=0.0)
    refine_boxes_dense(
        _RecordingSegmenter(), frame, [tiny_box, large_box],
        context_margin=0.5, adaptive_context_margin_cfg=ac_cfg,
    )

    assert seen_margins[0] == pytest.approx(0.0)   # tiny box -> min_ratio (0.0)
    assert seen_margins[1] == pytest.approx(0.5)   # large box -> full context_margin

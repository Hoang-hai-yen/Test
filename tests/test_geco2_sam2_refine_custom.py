"""Unit tests for GeCo2Detector.sam2_refine_boxes's CUSTOM path (any of
context_margin/adaptive_context_margin_cfg/use_center_point/
select_best_mask actually requested) -- reimplements GECO2's own
MaskProcessor encode/decode flow using only its PUBLIC submodules
(forward_feats, prompt_encoder_sam, mask_decoder), without touching a
single line inside GECO2/. Exercises the pure logic with a faked
MaskProcessor, without needing the real GECO2 repo, SAM2 checkpoint, or GPU."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.models.geco2_detector import GeCo2Detector
from aero_eyes.types import Box

IMAGE_SIZE = 40  # small but big enough to place well-separated blobs


def _make_detector(image_size: float = IMAGE_SIZE) -> GeCo2Detector:
    """Build a GeCo2Detector instance without running __init__ (which needs
    real GECO2 weights) -- only the attributes sam2_refine_boxes touches
    are set."""
    det = object.__new__(GeCo2Detector)
    det.image_size = image_size
    det.device = torch.device("cpu")
    det._mask_processor = None
    return det


class _FakePromptEncoder:
    def __init__(self, captured: dict):
        self._captured = captured

    def __call__(self, points, boxes, masks):
        self._captured["point_coords"] = points[0]
        self._captured["point_labels"] = points[1]
        return "sparse", "dense"

    def get_dense_pe(self):
        return "pe"


class _FakeMaskDecoder:
    """Returns a fixed low_res_masks tensor [N, 4, IMAGE_SIZE, IMAGE_SIZE]
    with 2 disconnected blobs on channel 2 (legacy's hard-coded index) and
    a DIFFERENT single blob on channel 3, plus iou_predictions favoring
    channel 3 -- lets tests distinguish "always index 2" from
    "argmax-selected" behavior."""

    def __init__(self, n_boxes: int, iou_predictions: "torch.Tensor"):
        self.n_boxes = n_boxes
        self._iou = iou_predictions

    def __call__(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                 dense_prompt_embeddings, multimask_output, repeat_image, high_res_features):
        masks = torch.full((self.n_boxes, 4, IMAGE_SIZE, IMAGE_SIZE), -10.0)
        for i in range(self.n_boxes):
            # channel 2: two disconnected blobs (real object + confuser).
            masks[i, 2, 5:10, 5:10] = 10.0
            masks[i, 2, 30:35, 30:35] = 10.0
            # channel 3: one blob, in a DIFFERENT region from channel 2's
            # "real object" blob -- distinguishes which channel got picked.
            masks[i, 3, 15:25, 15:25] = 10.0
        return masks, self._iou, None, None


class _FakeMaskProcessor:
    def __init__(self, iou_predictions: "torch.Tensor", n_boxes: int):
        self.captured: dict = {}
        self.prompt_encoder_sam = _FakePromptEncoder(self.captured)
        self.mask_decoder = _FakeMaskDecoder(n_boxes, iou_predictions)

    def forward_feats(self, feats):
        return [feats, feats, feats]  # only [-1] / [:-1] slicing matters, content unused


def _rig(det, mask_processor):
    det._get_mask_processor = lambda: mask_processor
    det._forward_scores = lambda frame_bgr, prototype: (None, None, 1.0, {"sentinel": True}, None, None)


def test_legacy_path_used_when_no_new_options_requested():
    """Defaults (all off) must route to the ORIGINAL forward()-based path,
    not the custom one -- exact backward compatibility."""
    det = _make_detector()
    called = {"legacy": False, "custom": False}
    det._legacy = lambda *a, **k: (called.__setitem__("legacy", True), [])[1]
    det._sam2_refine_boxes_legacy = lambda *a, **k: (called.__setitem__("legacy", True), [])[1]
    det._sam2_refine_boxes_custom = lambda *a, **k: (called.__setitem__("custom", True), [])[1]
    det._get_mask_processor = lambda: object()  # non-None, non-callable-checked here
    det._forward_scores = lambda frame_bgr, prototype: (None, None, 1.0, {}, None, None)

    det.sam2_refine_boxes(np.zeros((10, 10, 3), dtype=np.uint8), {}, [Box(1, 1, 5, 5, score=0.5)])

    assert called["legacy"] is True
    assert called["custom"] is False


def test_custom_path_used_when_context_margin_requested():
    det = _make_detector()
    called = {"legacy": False, "custom": False}
    det._sam2_refine_boxes_legacy = lambda *a, **k: (called.__setitem__("legacy", True), [])[1]
    det._sam2_refine_boxes_custom = lambda *a, **k: (called.__setitem__("custom", True), [])[1]
    det._get_mask_processor = lambda: object()
    det._forward_scores = lambda frame_bgr, prototype: (None, None, 1.0, {}, None, None)

    det.sam2_refine_boxes(
        np.zeros((10, 10, 3), dtype=np.uint8), {}, [Box(1, 1, 5, 5, score=0.5)], context_margin=0.2,
    )

    assert called["custom"] is True
    assert called["legacy"] is False


def test_custom_path_expands_box_prompt_by_context_margin():
    """The box prompt itself (points labeled 2/3) must reflect the
    margin-expanded box, in canvas-pixel coords (frame-px * scale)."""
    det = _make_detector()
    iou = torch.zeros((1, 4))
    mp = _FakeMaskProcessor(iou_predictions=iou, n_boxes=1)
    _rig(det, mp)

    box = Box(10, 10, 20, 20, score=0.9)  # w=h=10
    frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    det.sam2_refine_boxes(frame_bgr, {}, [box], context_margin=0.2)

    coords = mp.captured["point_coords"][0]  # [num_points, 2] for box 0
    labels = mp.captured["point_labels"][0].tolist()
    assert labels == [2, 3]
    # margin=0.2 -> mx=my=2 -> expanded box (8,8)-(22,22), scale=1.0 -> canvas-px identical.
    assert coords[0].tolist() == pytest.approx([8.0, 8.0])
    assert coords[1].tolist() == pytest.approx([22.0, 22.0])


def test_custom_path_adds_center_point_at_original_box_center():
    """use_center_point=True adds a 3rd point (label 1) at the ORIGINAL
    (pre-margin) box's own center, even though the box prompt itself is
    margin-expanded."""
    det = _make_detector()
    iou = torch.zeros((1, 4))
    mp = _FakeMaskProcessor(iou_predictions=iou, n_boxes=1)
    _rig(det, mp)

    box = Box(10, 10, 20, 20, score=0.9)  # center = (15, 15)
    frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    det.sam2_refine_boxes(frame_bgr, {}, [box], context_margin=0.2, use_center_point=True)

    coords = mp.captured["point_coords"][0]
    labels = mp.captured["point_labels"][0].tolist()
    assert labels == [2, 3, 1]
    assert coords[2].tolist() == pytest.approx([15.0, 15.0])


def test_custom_path_default_mask_index_matches_legacy_choice():
    """Without select_best_mask, the custom path must still pick channel 2
    (GECO2's own hard-coded choice) -- margin/point are independent knobs
    from mask SELECTION."""
    det = _make_detector()
    iou = torch.tensor([[0.1, 0.2, 0.3, 0.9]])  # channel 3 scores highest, but should be ignored
    mp = _FakeMaskProcessor(iou_predictions=iou, n_boxes=1)
    _rig(det, mp)

    box = Box(6, 6, 9, 9, score=0.8)  # inside channel 2's "real object" blob (5:10, 5:10)
    frame_bgr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    refined = det.sam2_refine_boxes(frame_bgr, {}, [box], context_margin=0.1, select_best_mask=False)

    # channel 2's combined tight bbox spans BOTH its blobs: (5,5)-(35,35).
    r = refined[0]
    assert (r.x1, r.y1, r.x2, r.y2) == pytest.approx((5.0, 5.0, 35.0, 35.0))


def test_custom_path_select_best_mask_picks_highest_scoring_channel():
    """select_best_mask=True picks argmax(iou_predictions[1:4]) instead of
    the hard-coded index 2 -- here channel 3 scores highest."""
    det = _make_detector()
    iou = torch.tensor([[0.1, 0.2, 0.3, 0.9]])  # index 3 (of 0..3) wins among [1,2,3]
    mp = _FakeMaskProcessor(iou_predictions=iou, n_boxes=1)
    _rig(det, mp)

    box = Box(16, 16, 24, 24, score=0.8)
    frame_bgr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    refined = det.sam2_refine_boxes(frame_bgr, {}, [box], context_margin=0.1, select_best_mask=True)

    # channel 3's single blob is exactly (15,15)-(25,25).
    r = refined[0]
    assert (r.x1, r.y1, r.x2, r.y2) == pytest.approx((15.0, 15.0, 25.0, 25.0))


def test_custom_path_center_point_isolates_component_and_drops_confuser():
    """use_center_point=True must trim channel 2's TWO disconnected blobs
    down to only the one containing the point -- not the combined bbox of
    both (see test_custom_path_default_mask_index_matches_legacy_choice
    for the no-point behavior, which spans both)."""
    det = _make_detector()
    iou = torch.zeros((1, 4))
    mp = _FakeMaskProcessor(iou_predictions=iou, n_boxes=1)
    _rig(det, mp)

    box = Box(6, 6, 9, 9, score=0.8)  # center (7.5, 7.5) -> inside the (5:10, 5:10) blob only
    frame_bgr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    refined = det.sam2_refine_boxes(
        frame_bgr, {}, [box], context_margin=0.1, use_center_point=True, select_best_mask=False,
    )

    r = refined[0]
    # Only the (5:10, 5:10) blob survives -- NOT the (30:35, 30:35) confuser,
    # and NOT the combined (5,5)-(35,35) bbox from the un-pointed test above.
    assert (r.x1, r.y1, r.x2, r.y2) == pytest.approx((5.0, 5.0, 10.0, 10.0))

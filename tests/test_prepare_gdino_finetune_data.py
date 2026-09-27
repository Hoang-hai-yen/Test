"""Unit tests for scripts/prepare_gdino_finetune_data.py's pure logic:
frame-diversity selection, rotation box math, hard-negative harvesting, and
copy-paste placement/box correctness -- none of this needs a real video,
GT file, or MobileSAM/transformers model."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gdino_finetune_data.py"
_spec = importlib.util.spec_from_file_location("prepare_gdino_finetune_data", _MODULE_PATH)
pg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pg
_spec.loader.exec_module(pg)

from aero_eyes.types import Box, Detection


# ---------------------------------------------------------------------------
# select_diverse_frames
# ---------------------------------------------------------------------------

def test_select_diverse_frames_drops_near_duplicate_states():
    w = h = 1000
    gt = {
        0: Box(100, 100, 150, 150),    # small box, top-left
        1: Box(102, 101, 152, 151),    # near-identical to frame 0 -- should be dropped
        2: Box(800, 800, 900, 900),    # different position AND scale -- kept
    }
    selected = pg.select_diverse_frames(gt, w, h, target_count=10, min_state_dist=0.03)
    assert 0 in selected
    assert 2 in selected
    assert 1 not in selected


def test_select_diverse_frames_respects_target_count():
    w = h = 1000
    gt = {i: Box(i * 10, i * 10, i * 10 + 20, i * 10 + 20) for i in range(20)}
    selected = pg.select_diverse_frames(gt, w, h, target_count=5, min_state_dist=0.0)
    assert len(selected) == 5


def test_select_diverse_frames_empty_gt():
    assert pg.select_diverse_frames({}, 1000, 1000, target_count=10) == []


def test_select_diverse_frames_never_picks_absent_frames():
    # only frames 0 and 5 have GT; frame 3 (absent) must never appear.
    gt = {0: Box(0, 0, 10, 10), 5: Box(500, 500, 600, 600)}
    selected = pg.select_diverse_frames(gt, 1000, 1000, target_count=10, min_state_dist=0.0)
    assert set(selected) <= {0, 5}


# ---------------------------------------------------------------------------
# rotate_image_and_box / _rotate_box
# ---------------------------------------------------------------------------

def test_rotate_box_180_degrees_maps_correctly():
    import cv2
    w, h = 100, 100
    box = Box(x1=10, y1=10, x2=30, y2=20)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 180, 1.0)
    rotated = pg._rotate_box(box, matrix, w, h)
    assert rotated is not None
    # 180-degree rotation around center: (x,y) -> (w-x, h-y)
    assert rotated.x1 == pytest.approx(w - 30, abs=1e-6)
    assert rotated.x2 == pytest.approx(w - 10, abs=1e-6)
    assert rotated.y1 == pytest.approx(h - 20, abs=1e-6)
    assert rotated.y2 == pytest.approx(h - 10, abs=1e-6)


def test_rotate_box_90_degrees_preserves_area_shape():
    import cv2
    w, h = 200, 200
    box = Box(x1=90, y1=80, x2=110, y2=140)  # 20 x 60
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
    rotated = pg._rotate_box(box, matrix, w, h)
    assert rotated is not None
    # A 90-degree rotation of an axis-aligned box swaps width/height exactly
    # (no hull looseness at multiples of 90 degrees).
    assert (rotated.x2 - rotated.x1) == pytest.approx(60, abs=1.0)
    assert (rotated.y2 - rotated.y1) == pytest.approx(20, abs=1.0)


def test_rotate_box_arbitrary_angle_hull_is_never_smaller_than_original():
    import cv2
    w, h = 300, 300
    box = Box(x1=100, y1=100, x2=160, y2=140)
    for angle in (15, 37, 73, 200):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = pg._rotate_box(box, matrix, w, h)
        assert rotated is not None
        # Axis-aligned hull of a rotated rectangle is always >= original area
        # (equality only at multiples of 90 degrees) -- this script's own
        # docstring calls this out as expected, not a bug.
        assert rotated.area() >= box.area() - 1e-6


def test_rotate_image_and_box_skips_when_box_rotates_out_of_frame():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # A box hugging the corner (rotated 45 degrees around the canvas center,
    # its hull lands entirely outside [0,100]x[0,100] -- confirmed via
    # _rotate_box directly: this exact (canvas, box, angle) combo clips to
    # zero area, not just "small").
    box = Box(x1=0, y1=0, x2=10, y2=10)
    result = pg.rotate_image_and_box(img, box, angle_deg=45, min_area_frac_of_original=0.3)
    assert result is None


def test_rotate_image_and_box_skips_when_hull_is_mostly_clipped():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Same corner box at 15 degrees: hull area drops to ~12.4 (vs. original
    # 100) -- survives raw clipping (nonzero) but must be rejected by a
    # reasonable min_area_frac_of_original floor.
    box = Box(x1=0, y1=0, x2=10, y2=10)
    result = pg.rotate_image_and_box(img, box, angle_deg=15, min_area_frac_of_original=0.3)
    assert result is None


def test_rotate_image_and_box_returns_valid_pair_for_small_centered_box():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    box = Box(x1=90, y1=90, x2=110, y2=110)  # small, centered -- safe at any angle
    result = pg.rotate_image_and_box(img, box, angle_deg=33, min_area_frac_of_original=0.3)
    assert result is not None
    rimg, rbox = result
    assert rimg.shape == img.shape
    assert rbox.area() > 0


# ---------------------------------------------------------------------------
# scale_jitter
# ---------------------------------------------------------------------------

def test_scale_jitter_scales_box_and_image_consistently():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    box = Box(x1=20, y1=10, x2=60, y2=30)
    simg, sbox = pg.scale_jitter(img, box, scale=2.0)
    assert simg.shape[:2] == (200, 400)
    assert sbox.x1 == pytest.approx(40)
    assert sbox.x2 == pytest.approx(120)
    assert sbox.y1 == pytest.approx(20)
    assert sbox.y2 == pytest.approx(60)


# ---------------------------------------------------------------------------
# find_hard_negative_frames
# ---------------------------------------------------------------------------

def _det(frame_idx, box, sim=0.5):
    return Detection(frame_idx=frame_idx, box=box, similarity=sim, source="detect")


def test_find_hard_negative_frames_flags_low_iou_detection():
    gt = {0: Box(0, 0, 10, 10)}
    detections = {0: [_det(0, Box(500, 500, 510, 510))]}  # nowhere near GT
    assert pg.find_hard_negative_frames(detections, gt, iou_thresh=0.1) == [0]


def test_find_hard_negative_frames_skips_true_positive():
    gt = {0: Box(0, 0, 10, 10)}
    detections = {0: [_det(0, Box(0, 0, 10, 10))]}  # perfect match
    assert pg.find_hard_negative_frames(detections, gt, iou_thresh=0.1) == []


def test_find_hard_negative_frames_flags_detection_on_gt_absent_frame():
    gt = {}  # object absent every frame in this fixture
    detections = {7: [_det(7, Box(1, 1, 5, 5))]}
    assert pg.find_hard_negative_frames(detections, gt, iou_thresh=0.1) == [7]


def test_find_hard_negative_frames_ignores_empty_detection_list():
    gt = {0: Box(0, 0, 10, 10)}
    detections = {0: []}
    assert pg.find_hard_negative_frames(detections, gt, iou_thresh=0.1) == []


# ---------------------------------------------------------------------------
# copy-paste: sample_paste_location / feathered_paste / build_copy_paste_sample
# ---------------------------------------------------------------------------

def test_sample_paste_location_avoids_exclude_box():
    rng = np.random.default_rng(0)
    canvas_w = canvas_h = 100
    exclude = Box(x1=0, y1=0, x2=100, y2=50)  # top half excluded
    for _ in range(20):
        loc = pg.sample_paste_location(canvas_w, canvas_h, obj_w=10, obj_h=10, exclude_box=exclude, rng=rng)
        assert loc is not None
        x0, y0 = loc
        candidate = Box(x1=x0, y1=y0, x2=x0 + 10, y2=y0 + 10)
        from aero_eyes.utils.geometry import box_iou
        assert box_iou(candidate, exclude) == 0.0


def test_sample_paste_location_returns_none_when_object_too_big():
    rng = np.random.default_rng(0)
    assert pg.sample_paste_location(50, 50, obj_w=100, obj_h=100, exclude_box=None, rng=rng) is None


def test_feathered_paste_output_shape_and_dtype():
    bg = np.full((50, 50, 3), 10, dtype=np.uint8)
    obj = np.full((10, 10, 3), 200, dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    out = pg.feathered_paste(bg, obj, mask, top_left=(5, 5))
    assert out.shape == bg.shape
    assert out.dtype == np.uint8
    # Center of the fully-masked object should be much closer to the object's
    # value than the background's after compositing.
    assert out[10, 10, 0] > 100


def test_feathered_paste_rejects_out_of_bounds_placement():
    bg = np.zeros((20, 20, 3), dtype=np.uint8)
    obj = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        pg.feathered_paste(bg, obj, mask, top_left=(15, 15))  # 15+10=25 > 20


def test_build_copy_paste_sample_returns_correct_box():
    rng = np.random.default_rng(1)
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    ref_crop = np.full((20, 20, 3), 255, dtype=np.uint8)
    ref_mask = np.ones((20, 20), dtype=bool)
    result = pg.build_copy_paste_sample(bg, ref_crop, ref_mask, target_size=(15, 25), rng=rng)
    assert result is not None
    img, box = result
    assert img.shape == bg.shape
    assert (box.x2 - box.x1) == pytest.approx(15)
    assert (box.y2 - box.y1) == pytest.approx(25)


# ---------------------------------------------------------------------------
# tight_crop_from_mask
# ---------------------------------------------------------------------------

def test_tight_crop_from_mask_tightens_to_mask_bbox():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 15:25] = True
    result = pg.tight_crop_from_mask(img, mask)
    assert result is not None
    crop, crop_mask = result
    assert crop.shape[:2] == (10, 10)
    assert crop_mask.shape == (10, 10)


def test_tight_crop_from_mask_none_for_empty_mask():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    assert pg.tight_crop_from_mask(img, mask) is None

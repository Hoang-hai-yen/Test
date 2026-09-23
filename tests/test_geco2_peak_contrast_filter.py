"""Unit tests for stage123_geco2.peak_contrast_filter --
GeCo2Detector._peak_contrast_scores (pure primitive) and its wiring into
filter_boxes_by_threshold (annotate-only vs hard_reject modes). Runs
without the real GECO2 repo, weights, or GPU -- see
Geco2PeakContrastFilterConfig's own docstring (aero_eyes/config.py) for the
full rationale (a repetitive-texture confuser tends to produce several
closely-spaced, comparably-high local maxima -- low CONTRAST against its
own neighborhood -- even when its raw peak VALUE clears
score_threshold_ratio; a real, isolated object's peak falls off cleanly on
all sides -- high contrast)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.models.geco2_detector import GeCo2Detector, _peak_contrast_scores


# ---------------------------------------------------------------------------
# _peak_contrast_scores: pure primitive
# ---------------------------------------------------------------------------

def _expected_contrast(grid: np.ndarray, y: int, x: int, radius: int) -> float:
    """Independent numpy re-implementation of the same formula, used to
    check the tensor implementation rather than hand-typing magic numbers."""
    h, w = grid.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = grid[y0:y1, x0:x1]
    return (grid[y, x] - patch.mean()) / (patch.std() + 1e-6)


def test_isolated_peak_has_high_contrast():
    """A single sharp peak on an otherwise-flat (zero) background stands
    out strongly from its own neighborhood."""
    grid = np.zeros((20, 20), dtype=np.float32)
    grid[5, 5] = 10.0
    centerness = torch.tensor(grid).view(1, 1, 20, 20)
    ref_points = torch.tensor([[5, 5]])

    out = _peak_contrast_scores(centerness, ref_points, radius=2)

    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(_expected_contrast(grid, 5, 5, 2), rel=1e-4)
    assert float(out[0]) > 3.0, "an isolated peak on a flat background should stand out sharply"


def test_texture_like_plateau_has_low_contrast():
    """Several nearby cells comparably high (simulating a repetitive-texture
    confuser, e.g. a leaf cluster) -- the SAME peak value as the isolated
    case above, but far less contrast against its own neighborhood."""
    grid = np.full((20, 20), 9.0, dtype=np.float32)
    # 6 of the 25 cells in the radius=2 window around (15,15), including the
    # peak itself, sit at 10.0 -- "several comparable local maxima nearby".
    grid[15, 15] = 10.0
    grid[14, 15] = 10.0
    grid[16, 15] = 10.0
    grid[15, 14] = 10.0
    grid[15, 16] = 10.0
    grid[14, 14] = 10.0
    centerness = torch.tensor(grid).view(1, 1, 20, 20)
    ref_points = torch.tensor([[15, 15]])

    out = _peak_contrast_scores(centerness, ref_points, radius=2)

    assert float(out[0]) == pytest.approx(_expected_contrast(grid, 15, 15, 2), rel=1e-4)


def test_isolated_peak_contrast_exceeds_texture_plateau_contrast():
    """The core discriminative claim: same peak VALUE (10.0) in both cases,
    but the isolated peak's contrast is decisively higher than the
    texture-plateau peak's -- this ordering, not the raw peak value, is
    what peak_contrast_filter uses to tell them apart."""
    isolated = np.zeros((20, 20), dtype=np.float32)
    isolated[5, 5] = 10.0

    plateau = np.full((20, 20), 9.0, dtype=np.float32)
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)]:
        plateau[15 + dy, 15 + dx] = 10.0

    out_isolated = _peak_contrast_scores(
        torch.tensor(isolated).view(1, 1, 20, 20), torch.tensor([[5, 5]]), radius=2,
    )
    out_plateau = _peak_contrast_scores(
        torch.tensor(plateau).view(1, 1, 20, 20), torch.tensor([[15, 15]]), radius=2,
    )

    assert float(out_isolated[0]) > float(out_plateau[0])


def test_empty_ref_points_returns_empty_tensor():
    centerness = torch.zeros(1, 1, 10, 10)
    ref_points = torch.zeros(0, 2, dtype=torch.long)
    out = _peak_contrast_scores(centerness, ref_points, radius=2)
    assert out.shape == (0,)


def test_radius_clips_at_grid_edges():
    """A peak near the grid boundary must not index out of bounds -- the
    window is simply clipped, not wrapped or padded."""
    grid = np.zeros((10, 10), dtype=np.float32)
    grid[0, 0] = 5.0
    centerness = torch.tensor(grid).view(1, 1, 10, 10)
    ref_points = torch.tensor([[0, 0]])

    out = _peak_contrast_scores(centerness, ref_points, radius=3)

    assert float(out[0]) == pytest.approx(_expected_contrast(grid, 0, 0, 3), rel=1e-4)


# ---------------------------------------------------------------------------
# filter_boxes_by_threshold wiring (annotate-only vs hard_reject)
# ---------------------------------------------------------------------------

def _make_detector(
    peak_contrast_filter_enabled: bool = False,
    peak_contrast_radius: int = 2,
    peak_contrast_hard_reject: bool = False,
    peak_contrast_min_z: float = 3.0,
) -> GeCo2Detector:
    """Same object.__new__ scaffolding as tests/test_geco2_min_box_area.py
    -- only the attributes filter_boxes_by_threshold itself touches."""
    det = object.__new__(GeCo2Detector)
    det.image_size = 20.0
    det.nms_iou = 0.99
    det.topk_per_keyframe = 10
    det.min_box_area_enabled = False
    det.min_box_area = 24
    det.peak_contrast_filter_enabled = peak_contrast_filter_enabled
    det.peak_contrast_radius = peak_contrast_radius
    det.peak_contrast_hard_reject = peak_contrast_hard_reject
    det.peak_contrast_min_z = peak_contrast_min_z
    return det


def _scene():
    """2 well-separated boxes (no NMS overlap), scale=1.0, image_size=20 so
    px_boxes == normalized_boxes * image_size exactly:
      A: isolated sharp peak at grid (5,5)  -> high contrast
      B: texture-plateau peak at grid (15,15) -> low contrast
    Both share the SAME raw score (0.9) -- score/threshold alone cannot
    tell them apart, only peak_contrast can.
    """
    pred_boxes = torch.tensor([
        [0.10, 0.10, 0.40, 0.40],   # -> px (2,2)-(8,8) in a 20x20 frame
        [0.60, 0.60, 0.90, 0.90],   # -> px (12,12)-(18,18)
    ])
    box_v = torch.tensor([0.9, 0.9])
    frame_bgr = np.zeros((20, 20, 3), dtype=np.uint8)

    grid = np.full((20, 20), 9.0, dtype=np.float32)
    grid[5, 5] = 10.0  # isolated
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)]:
        grid[15 + dy, 15 + dx] = 10.0  # texture plateau
    # background around the isolated peak must actually be low, not 9.0
    # everywhere -- overwrite a local patch around (5,5) with near-zero.
    grid[3:8, 3:8] = 0.0
    grid[5, 5] = 10.0
    centerness = torch.tensor(grid).view(1, 1, 20, 20)
    ref_points = torch.tensor([[5, 5], [15, 15]])

    return pred_boxes, box_v, frame_bgr, centerness, ref_points


def test_disabled_by_default_ignores_ref_points_and_centerness():
    """peak_contrast_filter.enabled=false (default) -- passing ref_points/
    centerness has NO effect at all: same boxes as without them, and
    peak_contrast stays None on every Box (old behavior unchanged)."""
    det = _make_detector(peak_contrast_filter_enabled=False)
    pred_boxes, box_v, frame_bgr, centerness, ref_points = _scene()

    results = det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    assert len(results) == 2
    assert all(b.peak_contrast is None for b in results)


def test_annotate_only_mode_attaches_peak_contrast_without_dropping():
    """enabled=true, hard_reject=false -- both boxes survive, but each now
    carries its own peak_contrast, isolated clearly higher than plateau."""
    det = _make_detector(peak_contrast_filter_enabled=True, peak_contrast_hard_reject=False)
    pred_boxes, box_v, frame_bgr, centerness, ref_points = _scene()

    results = det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    assert len(results) == 2
    contrasts = {round(b.x1): b.peak_contrast for b in results}
    assert all(v is not None for v in contrasts.values())
    isolated_contrast = contrasts[2]   # box A starts at px x1=2
    plateau_contrast = contrasts[12]   # box B starts at px x1=12
    assert isolated_contrast > plateau_contrast


def test_hard_reject_drops_only_the_low_contrast_candidate():
    """enabled=true, hard_reject=true with a threshold between the two
    candidates' contrast values -- the texture-plateau box (B) is dropped,
    the isolated box (A) survives."""
    det = _make_detector(
        peak_contrast_filter_enabled=True, peak_contrast_hard_reject=True, peak_contrast_min_z=3.0,
    )
    pred_boxes, box_v, frame_bgr, centerness, ref_points = _scene()

    results = det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    assert len(results) == 1
    assert round(results[0].x1) == 2, "only the isolated (high-contrast) box should survive"
    assert results[0].peak_contrast > 3.0


def test_hard_reject_with_permissive_threshold_keeps_both():
    det = _make_detector(
        peak_contrast_filter_enabled=True, peak_contrast_hard_reject=True, peak_contrast_min_z=-100.0,
    )
    pred_boxes, box_v, frame_bgr, centerness, ref_points = _scene()

    results = det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    assert len(results) == 2


def test_missing_ref_points_or_centerness_is_a_silent_noop():
    """A caller that doesn't supply ref_points/centerness (e.g.
    stage123_geco2.global_adaptive_threshold's two-pass path today) must
    not crash even with peak_contrast_filter.enabled=true -- filter is
    skipped for that call, exactly like disabled."""
    det = _make_detector(peak_contrast_filter_enabled=True, peak_contrast_hard_reject=True, peak_contrast_min_z=3.0)
    pred_boxes, box_v, frame_bgr, _, _ = _scene()

    results = det.filter_boxes_by_threshold(pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5)

    assert len(results) == 2
    assert all(b.peak_contrast is None for b in results)

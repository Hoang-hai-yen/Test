"""Unit tests for stage123_geco2.peak_contrast_filter --
GeCo2Detector._peak_contrast_scores (pure primitive) and its wiring into
filter_boxes_by_threshold (annotate-only vs hard_reject modes, fixed vs
adaptive_radius). Runs without the real GECO2 repo, weights, or GPU -- see
Geco2PeakContrastFilterConfig's own docstring (aero_eyes/config.py) for the
full rationale AND the real-footage finding that the original fixed-radius
hypothesis (repetitive-texture clutter = low contrast, isolated real
object = high contrast) came out INVERTED on 2 real samples, because a
fixed radius is confounded by candidate box SIZE -- adaptive_radius (tested
below) fixes this by scaling the window to each candidate's own footprint."""
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


def test_per_candidate_radius_overrides_fixed_radius_per_point():
    """adaptive_radius's mechanism at the primitive level: when
    per_candidate_radius is given, EACH point uses its OWN radius instead
    of the shared `radius` argument (which becomes dead/ignored)."""
    rng = np.random.default_rng(0)
    grid = rng.normal(loc=5.0, scale=1.0, size=(30, 30)).astype(np.float32)
    grid[10, 10] = 20.0
    grid[20, 20] = 20.0
    centerness = torch.tensor(grid).view(1, 1, 30, 30)
    ref_points = torch.tensor([[10, 10], [20, 20]])

    out = _peak_contrast_scores(
        centerness, ref_points, radius=2, per_candidate_radius=torch.tensor([2, 9]),
    )

    assert float(out[0]) == pytest.approx(_expected_contrast(grid, 10, 10, 2), rel=1e-4)
    assert float(out[1]) == pytest.approx(_expected_contrast(grid, 20, 20, 9), rel=1e-4)
    # Different radii on an irregular (random) background must not
    # coincidentally agree -- confirms the override actually took effect,
    # not silently ignored in favor of the shared `radius=2`.
    assert float(out[1]) != pytest.approx(_expected_contrast(grid, 20, 20, 2), rel=1e-3)


# ---------------------------------------------------------------------------
# filter_boxes_by_threshold wiring (annotate-only vs hard_reject)
# ---------------------------------------------------------------------------

def _make_detector(
    peak_contrast_filter_enabled: bool = False,
    peak_contrast_radius: int = 2,
    peak_contrast_hard_reject: bool = False,
    peak_contrast_min_z: float = 3.0,
    peak_contrast_adaptive_radius: bool = False,
    peak_contrast_radius_scale: float = 1.0,
    peak_contrast_min_radius: int = 2,
    peak_contrast_max_radius: int = 12,
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
    det.peak_contrast_adaptive_radius = peak_contrast_adaptive_radius
    det.peak_contrast_radius_scale = peak_contrast_radius_scale
    det.peak_contrast_min_radius = peak_contrast_min_radius
    det.peak_contrast_max_radius = peak_contrast_max_radius
    det._n_peak_contrast_seen = 0
    det._n_peak_contrast_hard_rejected = 0
    det._peak_contrast_sum = 0.0
    det._peak_contrast_sumsq = 0.0
    det._peak_contrast_min = None
    det._peak_contrast_max = None
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


# ---------------------------------------------------------------------------
# adaptive_radius (the box-size-confound fix)
# ---------------------------------------------------------------------------

def test_adaptive_radius_derives_window_from_box_footprint():
    """adaptive_radius=true -- the window half-size actually used is
    derived from THIS candidate's own box footprint on the centerness grid
    (radius_scale * max(box_w, box_h)/2 grid cells), not the fixed `radius`
    config value (which must be ignored here). Mechanical check: compare
    filter_boxes_by_threshold's output against an independent numpy
    computation using the radius this formula predicts."""
    det = _make_detector(
        peak_contrast_filter_enabled=True, peak_contrast_hard_reject=False,
        peak_contrast_radius=2,  # fixed radius -- must be IGNORED when adaptive_radius=true
        peak_contrast_adaptive_radius=True, peak_contrast_radius_scale=1.0,
        peak_contrast_min_radius=1, peak_contrast_max_radius=20,
    )
    grid_size = 30
    rng = np.random.default_rng(0)
    grid = rng.normal(loc=5.0, scale=1.0, size=(grid_size, grid_size)).astype(np.float32)
    grid[15, 15] = 20.0
    centerness = torch.tensor(grid).view(1, 1, grid_size, grid_size)
    ref_points = torch.tensor([[15, 15]])

    # Box footprint: 16 grid cells wide/tall -> half-extent = 8 ->
    # radius_scale=1.0 -> predicted radius = 8.
    frac = 16.0 / grid_size
    pred_boxes = torch.tensor([[0.5 - frac / 2, 0.5 - frac / 2, 0.5 + frac / 2, 0.5 + frac / 2]])
    box_v = torch.tensor([0.9])
    frame_bgr = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    det.image_size = float(grid_size)

    results = det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    assert len(results) == 1
    assert results[0].peak_contrast == pytest.approx(_expected_contrast(grid, 15, 15, radius=8), rel=1e-4)
    # Confirms the fixed radius=2 was NOT what got used (otherwise this
    # test wouldn't actually be exercising adaptive_radius).
    assert results[0].peak_contrast != pytest.approx(_expected_contrast(grid, 15, 15, radius=2), rel=1e-3)


def test_adaptive_radius_is_clamped_to_max_radius():
    det = _make_detector(
        peak_contrast_filter_enabled=True, peak_contrast_hard_reject=False,
        peak_contrast_adaptive_radius=True, peak_contrast_radius_scale=1.0,
        peak_contrast_min_radius=1, peak_contrast_max_radius=3,  # smaller than the natural half-extent (8)
    )
    grid_size = 30
    rng = np.random.default_rng(1)
    grid = rng.normal(loc=5.0, scale=1.0, size=(grid_size, grid_size)).astype(np.float32)
    grid[15, 15] = 20.0
    centerness = torch.tensor(grid).view(1, 1, grid_size, grid_size)
    ref_points = torch.tensor([[15, 15]])

    frac = 16.0 / grid_size  # half-extent 8 -- would predict radius=8 without clamping
    pred_boxes = torch.tensor([[0.5 - frac / 2, 0.5 - frac / 2, 0.5 + frac / 2, 0.5 + frac / 2]])
    box_v = torch.tensor([0.9])
    frame_bgr = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    det.image_size = float(grid_size)

    results = det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    assert results[0].peak_contrast == pytest.approx(_expected_contrast(grid, 15, 15, radius=3), rel=1e-4)


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


# ---------------------------------------------------------------------------
# log_peak_contrast_summary
# ---------------------------------------------------------------------------

def test_log_summary_reports_stats_and_reject_count(caplog):
    import logging

    det = _make_detector(peak_contrast_filter_enabled=True, peak_contrast_hard_reject=True, peak_contrast_min_z=3.0)
    pred_boxes, box_v, frame_bgr, centerness, ref_points = _scene()
    det.filter_boxes_by_threshold(
        pred_boxes, box_v, scale=1.0, frame_bgr=frame_bgr, threshold=0.5,
        ref_points=ref_points, centerness=centerness,
    )

    with caplog.at_level(logging.INFO):
        det.log_peak_contrast_summary("sample_x")

    assert "peak_contrast_filter summary" in caplog.text
    assert "n=2" in caplog.text
    assert "1/2 hard-rejected" in caplog.text


def test_log_summary_is_a_noop_when_disabled():
    det = _make_detector(peak_contrast_filter_enabled=False)
    det.log_peak_contrast_summary("sample_x")  # must not raise even with no data at all


def test_log_summary_is_a_noop_when_nothing_was_ever_scored():
    """enabled=true but filter_boxes_by_threshold was never called with
    ref_points/centerness (e.g. only the global_adaptive_threshold path ran
    this whole video) -- n_seen stays 0, must not raise or log."""
    det = _make_detector(peak_contrast_filter_enabled=True)
    det.log_peak_contrast_summary("sample_x")

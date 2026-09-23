"""Unit tests for aero_eyes.stages.stage1.apply_ref_degradation --
stage1.ref_degradation_ensemble's per-level transform (downscale + blur +
JPEG-compression round-trip). Pure cv2/numpy, no torch import needed."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from aero_eyes.stages.stage1 import apply_ref_degradation


def _checkerboard(size: int = 64, square: int = 4) -> np.ndarray:
    """High-frequency-content BGR test image -- a real photo's fine detail
    stand-in, so blur/downscale actually changes something measurable."""
    rng = np.random.default_rng(0)
    base = np.indices((size, size)).sum(axis=0) % (2 * square) < square
    img = np.where(base[..., None], 220, 30).astype(np.uint8)
    img = np.repeat(img, 3, axis=2)
    noise = rng.integers(-10, 10, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_all_defaults_are_a_true_noop():
    img = _checkerboard()
    out = apply_ref_degradation(img, downscale_factor=1.0, blur_ksize=0, jpeg_quality=100)
    assert np.array_equal(out, img)


def test_downscale_changes_the_image():
    img = _checkerboard()
    out = apply_ref_degradation(img, downscale_factor=0.25, blur_ksize=0, jpeg_quality=100)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_blur_reduces_high_frequency_variance():
    """A checkerboard's local (pixel-to-pixel) variance should drop sharply
    after Gaussian blur -- a direct check that blur_ksize actually smooths
    detail, not just that the array changed."""
    img = _checkerboard()
    out = apply_ref_degradation(img, downscale_factor=1.0, blur_ksize=9, jpeg_quality=100)
    assert out.shape == img.shape

    def _local_variance(a: np.ndarray) -> float:
        gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))

    assert _local_variance(out) < _local_variance(img) * 0.5


def test_jpeg_quality_introduces_compression_artifacts():
    img = _checkerboard()
    out = apply_ref_degradation(img, downscale_factor=1.0, blur_ksize=0, jpeg_quality=15)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert not np.array_equal(out, img)


def test_combined_degradation_preserves_shape_and_dtype():
    img = _checkerboard(size=80)
    out = apply_ref_degradation(img, downscale_factor=0.3, blur_ksize=5, jpeg_quality=25)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_jpeg_quality_below_one_is_clamped_not_crashed():
    """jpeg_quality=0 takes the < 100 branch (real compression requested)
    -- must clamp to cv2's valid [1,100] range internally rather than
    crashing cv2.imencode."""
    img = _checkerboard()
    out = apply_ref_degradation(img, downscale_factor=1.0, blur_ksize=0, jpeg_quality=0)
    assert out.shape == img.shape


def test_jpeg_quality_at_or_above_100_skips_compression_entirely():
    """>= 100 never enters the JPEG branch at all (jpeg_quality < 100 is
    false) -- a plain identity pass-through, same as quality=100 exactly."""
    img = _checkerboard()
    out = apply_ref_degradation(img, downscale_factor=1.0, blur_ksize=0, jpeg_quality=500)
    assert np.array_equal(out, img)


# ---------------------------------------------------------------------------
# build_ref_views: replace-not-append semantics
# ---------------------------------------------------------------------------

from aero_eyes.stages.stage1 import build_ref_views


def _mask_for(img):
    return np.ones(img.shape[:2], dtype=bool)


def test_no_levels_is_the_unchanged_clean_pyramid():
    img = _checkerboard(size=80)
    views = build_ref_views(img, _mask_for(img), levels=None)
    assert len(views) == 3
    assert np.array_equal(views[0], img)                 # 1.0x is the clean image
    assert views[1].shape[:2] == (60, 60)                # 0.75x
    assert views[2].shape[:2] == (40, 40)                # 0.5x


def test_levels_replace_clean_pyramid_instead_of_appending():
    """The core fix: with levels active there are n_levels x 3 views and
    NONE of them is the clean image -- the old append behavior would have
    left the clean 1.0x view (and 0.75x/0.5x) in the average."""
    img = _checkerboard(size=80)
    levels = [(0.2, 0, 100), (0.1, 0, 100)]
    views = build_ref_views(img, _mask_for(img), levels=levels)
    assert len(views) == len(levels) * 3
    assert not any(np.array_equal(v, img) for v in views)
    # No pyramid scale of the CLEAN image sneaks in either.
    clean_half = build_ref_views(img, _mask_for(img), levels=None)[2]
    assert not any(v.shape == clean_half.shape and np.array_equal(v, clean_half) for v in views)


def test_each_levels_first_view_is_that_levels_degraded_base():
    img = _checkerboard(size=80)
    levels = [(0.2, 0, 100), (0.1, 3, 50)]
    views = build_ref_views(img, _mask_for(img), levels=levels)
    for k, lv in enumerate(levels):
        expected = apply_ref_degradation(img, *lv)
        assert np.array_equal(views[k * 3], expected)


def test_pyramid_is_built_from_the_degraded_base_not_the_clean_image():
    img = _checkerboard(size=80)
    lv = (0.1, 0, 100)
    views = build_ref_views(img, _mask_for(img), levels=[lv])
    degraded = apply_ref_degradation(img, *lv)
    half_of_degraded = cv2.resize(degraded, (40, 40))
    assert np.array_equal(views[2], half_of_degraded)


def test_identity_level_lets_a_clean_view_in_deliberately():
    img = _checkerboard(size=80)
    views = build_ref_views(img, _mask_for(img), levels=[(1.0, 0, 100), (0.1, 0, 100)])
    assert len(views) == 6
    assert np.array_equal(views[0], img)


def test_synth_views_fn_runs_once_per_base_on_the_degraded_image():
    img = _checkerboard(size=80)
    calls = []

    def fake_synth(base, mask):
        calls.append(base)
        return [base.copy()]

    levels = [(0.2, 0, 100), (0.1, 0, 100)]
    views = build_ref_views(img, _mask_for(img), levels=levels, synth_views_fn=fake_synth)
    assert len(calls) == 2
    assert len(views) == 2 * (3 + 1)
    assert not any(np.array_equal(c, img) for c in calls)   # never the clean image


def test_empty_levels_list_falls_back_to_clean():
    img = _checkerboard(size=80)
    views = build_ref_views(img, _mask_for(img), levels=[])
    assert len(views) == 3 and np.array_equal(views[0], img)

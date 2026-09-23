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

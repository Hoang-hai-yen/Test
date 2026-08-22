"""Cheap, training-free color similarity utilities.

GeCo2 (like most few-shot COUNTING models) is trained to match shape/texture
against exemplars via a vision backbone -- it has no explicit color signal,
so it commonly mis-detects objects with a similar silhouette but a
different color. This module backs a post-detection filter
(stage123_geco2.color_postfilter): compare each candidate box's color
distribution against the reference object's own color signature and
drop/downweight candidates that don't match. Pure OpenCV, no extra model,
no finetuning -- see aero_eyes/stages/stage123_geco2.py::
build_color_signature / apply_color_postfilter for how this is wired in.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_hs_histogram(
    img_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    hue_bins: int = 30,
    sat_bins: int = 32,
) -> np.ndarray:
    """2D hue-saturation histogram, L1-normalized so histograms from
    crops of different pixel counts are directly comparable.

    Deliberately ignores V (brightness/value) -- the whole point is to
    stay robust to lighting differences between the close-up reference
    photo and the video frame (same real-world color, different exposure,
    would otherwise look like a mismatch).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = (mask.astype(np.uint8) * 255) if mask is not None else None
    hist = cv2.calcHist([hsv], [0, 1], mask_u8, [hue_bins, sat_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def compute_mean_saturation(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean HSV saturation (0-255) over `mask` (or the whole image if
    mask is None). Catches near-white/gray reference objects -- see
    ColorPostfilterConfig.min_ref_saturation.

    NOT sufficient on its own to catch DARK objects: S = (max-min)/max is a
    RATIO, so for small V (dark pixels) a small absolute sensor-noise
    difference between channels gets amplified into a large, spuriously
    HIGH saturation reading -- empirically confirmed (a synthetic near-black
    pixel with only +-4/255 channel noise computed mean saturation ~50, well
    above a naive "low saturation" threshold). Combine with
    compute_mean_value() / min_ref_value to also catch dark/black objects.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    if mask is not None:
        sat = sat[mask]
    return float(sat.mean()) if sat.size else 0.0


def compute_mean_value(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean HSV value/brightness (0-255) over `mask` (or the whole image).
    Catches DARK reference objects (e.g. black boxes), for which Hue AND
    Saturation both become noise-dominated/unreliable regardless of the
    raw saturation reading -- see compute_mean_saturation's docstring and
    ColorPostfilterConfig.min_ref_value.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    val = hsv[..., 2]
    if mask is not None:
        val = val[mask]
    return float(val.mean()) if val.size else 0.0


def compute_value_histogram(img_bgr: np.ndarray, mask: np.ndarray | None = None, val_bins: int = 8) -> np.ndarray:
    """1D histogram of HSV value/brightness, L1-normalized. Unlike Hue and
    Saturation, brightness is exactly the ONE property that reliably tells
    black from white/gray objects apart -- deliberately kept as a SEPARATE
    signal from compute_hs_histogram (which excludes V for lighting
    robustness) so callers can fall back to it for near-achromatic
    reference objects, where Hue+Saturation carries no usable signal but a
    black-vs-white confuser is otherwise indistinguishable. See
    ColorPostfilterConfig / apply_color_postfilter's confidence blend.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = (mask.astype(np.uint8) * 255) if mask is not None else None
    hist = cv2.calcHist([hsv], [2], mask_u8, [val_bins], [0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def saturation_value_confidence(
    mean_saturation: float, mean_value: float,
    sat_low: float, sat_high: float, val_low: float, val_high: float,
) -> float:
    """How much to trust Hue-based color comparison for this reference
    object, in [0,1] -- 0 = fully suppress (near-achromatic, Hue is
    noise), 1 = fully trust. Linearly ramps from `_low` (confidence 0) to
    `_high` (confidence 1) for each of saturation and value/brightness
    independently, then takes the MINIMUM of the two (either weak signal
    is enough reason to distrust the comparison -- a dark AND desaturated
    object is even less trustworthy than either alone).

    A graduated ramp instead of a hard on/off cutoff: a real reference
    object (mean saturation=60.1, value=121.3) sat ABOVE naive hard-cutoff
    floors (40 / 50) yet the color signal still measurably hurt accuracy
    (ST-IoU 0.4264 -> 0.3902) -- there is no single "correct" cutoff value
    that cleanly separates "trustworthy" from "not" across different
    objects/datasets, so this degrades gracefully around the boundary
    instead.
    """
    def ramp(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    return min(ramp(mean_saturation, sat_low, sat_high), ramp(mean_value, val_low, val_high))


def histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray, metric: str = "bhattacharyya") -> float:
    """Similarity in [0,1] (higher = more similar), normalized so callers
    don't need to know each OpenCV metric's own return convention.
    """
    if metric == "bhattacharyya":
        dist = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)  # 0=identical, 1=totally different
        return float(1.0 - dist)
    if metric == "correlation":
        corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)  # 1=identical, -1=opposite
        return float(max(0.0, corr))
    raise ValueError(f"Unknown histogram metric '{metric}'. Must be 'bhattacharyya' or 'correlation'.")

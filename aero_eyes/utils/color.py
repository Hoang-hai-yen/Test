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

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


def _circular_gaussian_kernel(sigma: float) -> np.ndarray:
    """1D Gaussian kernel, L1-normalized, radius = 3*sigma (rounded up,
    minimum 1) -- used to smooth the Hue axis of a histogram WITH
    wrap-around (Hue is circular: bin 0 is adjacent to the last bin, not
    an edge)."""
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_hue_axis(hist: np.ndarray, sigma: float) -> np.ndarray:
    """Circularly smooths a 2D (hue, saturation) histogram along its hue
    axis (axis 0) with a Gaussian of the given sigma (in BINS, not
    degrees) -- makes bin-to-bin histogram comparison (Bhattacharyya/
    correlation) tolerant to a moderate Hue shift, e.g. the SAME real
    object's apparent hue drifting under a different light source's color
    temperature (an orange object reading more yellow in direct sunlight
    than under neutral/shaded light). Hue is circular (OpenCV's 0-179
    range wraps: bin 179 is adjacent to bin 0), so this pads with
    `mode="wrap"`, NOT zero/edge padding -- a plain (non-circular) blur
    would incorrectly treat the hue wheel's seam as a hard edge, leaking
    weight away from bins near 0/179 instead of into each other.

    sigma <= 0 returns hist unchanged (no smoothing -- today's original,
    unmodified behavior). Preserves the L1 sum (a normalized kernel over a
    periodic domain cannot gain or lose mass), but callers should
    re-normalize defensively if chaining further float operations.
    """
    if sigma <= 0:
        return hist
    kernel = _circular_gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.pad(hist, ((radius, radius), (0, 0)), mode="wrap")
    out = np.zeros_like(hist, dtype=np.float64)
    for i, w in enumerate(kernel):
        out += w * padded[i: i + hist.shape[0]]
    return out.astype(hist.dtype)


def compute_hs_histogram(
    img_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    hue_bins: int = 30,
    sat_bins: int = 32,
    hue_smoothing_sigma: float = 0.0,
) -> np.ndarray:
    """2D hue-saturation histogram, L1-normalized so histograms from
    crops of different pixel counts are directly comparable.

    Deliberately ignores V (brightness/value) -- the whole point is to
    stay robust to lighting differences between the close-up reference
    photo and the video frame (same real-world color, different exposure,
    would otherwise look like a mismatch).

    hue_smoothing_sigma > 0: circularly smooths the Hue axis afterward
    (see smooth_hue_axis) -- tolerates a moderate lighting-driven hue
    shift instead of only comparing exact bins. 0.0 (default) = no-op,
    original behavior.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = (mask.astype(np.uint8) * 255) if mask is not None else None
    hist = cv2.calcHist([hsv], [0, 1], mask_u8, [hue_bins, sat_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    hist = smooth_hue_axis(hist, hue_smoothing_sigma)
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


def compute_overexposed_fraction(img_bgr: np.ndarray, mask: np.ndarray | None = None, clip_threshold: int = 250) -> float:
    """Fraction of (masked) pixels where ANY of the 3 raw BGR channels is
    clipped (>= clip_threshold) -- the textbook signature of sensor
    overexposure under harsh direct sunlight: the dominant/near-saturated
    channel clips first while another channel keeps climbing toward its
    own ceiling, so a genuinely orange/red object's surviving channel
    ratios shift toward yellow. Confirmed on this project's own footage
    (docs/attribute_taxonomy_plan.md SS9.10): a LifeJacket crop under harsh
    early-video sun read `yellow` instead of its true `red`/`orange`, with
    R clipped on 29-34% of pixels; a later, correctly-lit frame at the same
    location had only 4% clipped and read correctly. A clipped channel's
    ORIGINAL value is genuinely, information-theoretically lost -- this
    cannot be corrected, only detected, so callers use it to LOWER
    confidence in a Hue-based color reading (see
    hue_confidence_from_overexposure), never to try to recover the true
    color from it."""
    b, g, r = cv2.split(img_bgr)
    clipped = (b >= clip_threshold) | (g >= clip_threshold) | (r >= clip_threshold)
    if mask is not None:
        clipped = clipped[mask]
    return float(clipped.mean()) if clipped.size else 0.0


def hue_confidence_from_overexposure(overexposed_fraction: float, ramp_frac: float = 0.40) -> float:
    """How much to trust a Hue-based color reading for this crop, in
    [0,1] -- 1 = fully trust (not overexposed), 0 = fully distrust
    (overexposed_fraction at or above ramp_frac). Linear ramp, same
    graduated-degradation spirit as saturation_value_confidence's dark-side
    ramp but driven by clipping instead of low brightness -- the
    SYMMETRIC, bright-side counterpart to it. ramp_frac=0.40 (default) is
    calibrated from this project's own measurements
    (docs/attribute_taxonomy_plan.md SS9.10/SS9.11): 31-34% clipped
    reliably meant a wrong Hue reading on real footage, 4% did not.
    ramp_frac<=0 returns 1.0 unconditionally (gating disabled)."""
    if ramp_frac <= 0:
        return 1.0
    return float(np.clip(1.0 - overexposed_fraction / ramp_frac, 0.0, 1.0))


def lit_pixel_mask(
    img_bgr: np.ndarray, mask: np.ndarray | None = None,
    min_saturation: float = 40.0, min_value: float = 60.0,
) -> np.ndarray:
    """Boolean mask of "confidently lit" pixels (S >= min_saturation AND
    V >= min_value), intersected with `mask` (or the whole image if mask
    is None) -- excludes SHADOWED pixels from a Hue-based color reading
    (docs/attribute_taxonomy_plan.md SS4 point 2, distinct from SS4 point 1's
    smooth_hue_axis): shade/shadow is lit by ambient SKYLIGHT instead of
    direct sun, a STRONGER, more systematic hue shift (the classic
    "shadows are blue" effect) than smooth_hue_axis's moderate
    color-temperature tolerance is meant to absorb. Excluding shadowed
    pixels from the vote entirely, rather than averaging through the
    shift, keeps the Hue histogram built only from pixels where the
    reading is trustworthy.

    NOT YET VALIDATED -- min_saturation/min_value have no calibrated
    threshold from this project's own footage (unlike
    hue_confidence_from_overexposure's clip_threshold/ramp_frac, which do).
    Callers should fall back to the UNFILTERED mask if this leaves too few
    pixels (a heavily-shadowed-but-genuine object shouldn't lose its color
    reading entirely just because none of it happens to be "lit")."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lit = (hsv[..., 1] >= min_saturation) & (hsv[..., 2] >= min_value)
    if mask is not None:
        lit = lit & mask
    return lit


def compute_is_high_vis(
    img_bgr: np.ndarray, mask: np.ndarray | None = None,
    percentile: float = 80.0, hue_max: float = 35.0, min_value: float = 140.0,
) -> float:
    """Fraction of the TOP-percentile-saturation (masked) pixels that also
    land in the safety orange/yellow hue band and are bright -- ported
    verbatim from scripts/test_group_a_attributes.py::color_descriptors's
    own is_high_vis computation (docs/attribute_taxonomy_plan.md SS4 point
    4 / SS3.2), ONLY the standalone scalar (this project's real color_
    postfilter compares one candidate against a reference signature, not
    the full multi-region taxonomy that script also computes).

    Using the TOP percentile (not the mean) rather than a hard threshold:
    a safety-colored object partially in shadow or partially occluded
    still registers correctly off its own brightest/most-saturated patch,
    instead of being diluted by the rest of the crop.

    hue_max=35.0 (not the general "orange" hue bucket's narrower 8-20) and
    min_value=140.0 are CALIBRATED, not placeholders -- found empirically
    (docs/attribute_taxonomy_plan.md SS8.1/SS8.4) that real safety-orange
    fabric commonly reads hue 0-20 (closer to red than the general-purpose
    "orange" bucket assumes); widening the band moved LifeJacket's own
    is_high_vis reading from 0.44 to 0.97. Returns 0.0 if `mask` (or the
    whole image) has no pixels."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if mask is not None:
        h_ch, s_ch, v_ch = h_ch[mask], s_ch[mask], v_ch[mask]
    if s_ch.size == 0:
        return 0.0
    thresh = np.percentile(s_ch, percentile)
    bright = s_ch >= thresh
    top_hue, top_val = h_ch[bright], v_ch[bright]
    if top_hue.size == 0:
        return 0.0
    return float(((top_hue >= 0) & (top_hue < hue_max) & (top_val > min_value)).mean())

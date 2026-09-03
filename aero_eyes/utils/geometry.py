"""Box geometry: IoU, NMS, format conversions, SAHI tiling, homography warps."""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, NamedTuple

# cv2 and numpy are only needed for image-processing functions (crop, warp, SAHI).
# Pure-Python functions (box_iou, nms, convert_box, sahi_tiles) work without them,
# allowing unit tests to run without compiled C extensions.
from aero_eyes.types import Box


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two xyxy boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(boxes: list[Box], iou_threshold: float = 0.5) -> list[int]:
    """Greedy NMS. Returns indices of surviving boxes sorted by score desc."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].score, reverse=True)
    keep: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j in suppressed or j == i:
                continue
            if box_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)
    return keep


# ---------------------------------------------------------------------------
# Box format conversions
# ---------------------------------------------------------------------------

def convert_box(
    box: Box,
    from_fmt: str,
    to_fmt: str,
    img_w: float = 1.0,
    img_h: float = 1.0,
    normalized_in: bool = False,
    normalized_out: bool = False,
) -> Box:
    """Convert between xyxy / xywh / cxcywh, optionally (de-)normalizing."""
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

    # De-normalize input
    if normalized_in:
        x1 *= img_w; x2 *= img_w
        y1 *= img_h; y2 *= img_h

    # Convert from source format to xyxy
    if from_fmt == "xyxy":
        pass
    elif from_fmt == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2
    elif from_fmt == "cxcywh":
        half_w = x2 / 2
        half_h = y2 / 2
        x1, y1 = x1 - half_w, y1 - half_h
        x2, y2 = x1 + 2 * half_w, y1 + 2 * half_h
    else:
        raise ValueError(f"Unknown box format '{from_fmt}'")

    # Convert xyxy to target format
    if to_fmt == "xyxy":
        ox1, oy1, ox2, oy2 = x1, y1, x2, y2
    elif to_fmt == "xywh":
        ox1, oy1, ox2, oy2 = x1, y1, x2 - x1, y2 - y1
    elif to_fmt == "cxcywh":
        ox1 = (x1 + x2) / 2
        oy1 = (y1 + y2) / 2
        ox2 = x2 - x1
        oy2 = y2 - y1
    else:
        raise ValueError(f"Unknown box format '{to_fmt}'")

    # Normalize output
    if normalized_out:
        ox1 /= img_w; ox2 /= img_w
        oy1 /= img_h; oy2 /= img_h

    return Box(ox1, oy1, ox2, oy2, score=box.score)


# ---------------------------------------------------------------------------
# SAHI tiling
# ---------------------------------------------------------------------------

class TileRect(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


def sahi_tiles(img_w: int, img_h: int, tile_wh: list[int], overlap: float) -> list[TileRect]:
    """Generate overlapping tile rects covering the full image."""
    tw, th = tile_wh
    stride_x = max(1, int(tw * (1 - overlap)))
    stride_y = max(1, int(th * (1 - overlap)))
    tiles: list[TileRect] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + tw, img_w)
            y2 = min(y + th, img_h)
            # Shift left/up so tile is full-sized at image boundaries
            x1 = max(0, x2 - tw)
            y1 = max(0, y2 - th)
            tiles.append(TileRect(x1, y1, x2, y2))
            if x2 == img_w:
                break
            x += stride_x
        if y2 == img_h:
            break
        y += stride_y
    return tiles


def remap_box_from_tile(box: Box, tile: TileRect) -> Box:
    """Translate a box detected inside a tile to full-image coordinates."""
    return Box(
        x1=box.x1 + tile.x1,
        y1=box.y1 + tile.y1,
        x2=box.x2 + tile.x1,
        y2=box.y2 + tile.y1,
        score=box.score,
    )


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def crop_with_pad(img_bgr, box: Box, pad_ratio: float = 0.1):
    """Crop a padded region around box from img_bgr; returns BGR uint8 ndarray."""
    import numpy as np
    h, w = img_bgr.shape[:2]
    bw = box.x2 - box.x1
    bh = box.y2 - box.y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    x1 = max(0, int(box.x1 - px))
    y1 = max(0, int(box.y1 - py))
    x2 = min(w, int(box.x2 + px))
    y2 = min(h, int(box.y2 + py))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return crop


def mask_bbox(mask) -> tuple[float, float, float, float] | None:
    """Tight bbox around the True region of a boolean mask. None if empty."""
    import numpy as np
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def apply_background_mode(img, mask, mode: str, blur_sigma: float = 25.0):
    """Replace (or keep) the non-mask region of `img` per one of 3 modes:
      mean_fill -- flat mean-color fill. Cheapest, but a large flat,
                   textureless region is far outside what an ImageNet/web-
                   photo-pretrained backbone (DINOv2/CLIP/GeCo2's Hiera)
                   ever saw -- can push the resulting embedding into an
                   unnatural part of feature space.
      keep_real -- leave the photo's real background untouched. Useful
                   when a separate tight-box crop/RoI-Align already
                   controls WHERE the model actually pools its embedding
                   from, so background pixels never enter the embedding
                   directly -- the backbone's self-attention still sees a
                   natural image overall (its own real photo context), not
                   a synthetic flat region.
      blur      -- strong Gaussian blur of the real background: keeps
                   natural color/texture statistics but discards fine
                   detail that could otherwise cause spurious background
                   matches.
    """
    import cv2
    import numpy as np

    if mode == "keep_real":
        return img
    if mode == "blur":
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)
        out = img.copy()
        out[~mask] = blurred[~mask]
        return out
    masked = img.copy()
    bg_color = img.reshape(-1, 3).mean(axis=0)
    masked[~mask] = bg_color
    return masked


def crop_to_object(
    img, tight_box: tuple[float, float, float, float], context_margin: float
) -> tuple:
    """Crop `img` to `tight_box` expanded by `context_margin` (a fraction of
    the box's own width/height), clamped to image bounds. Keeps 100% real
    pixels -- no masking/fill -- just a tighter field of view than the
    whole reference photo.

    Why this helps: whole-image resize/resize_and_pad preprocessing (DINOv2,
    GeCo2) always renormalizes whichever image it's given so its longer
    side fits the model's fixed canvas size. Feeding it the WHOLE reference
    photo means the object occupies only whatever (small) fraction of the
    photo it originally did, so it ends up occupying that same small
    fraction of the canvas -- less detail/resolution for the object.
    Cropping to just the object (plus context) BEFORE that resize means the
    object occupies a much LARGER fraction of the (now smaller) cropped
    image, so after renormalizing it also occupies a larger fraction of the
    final canvas. Unlike a scale-calibrated canvas, this needs no oracle
    estimate of the deployment video's apparent object size -- it's purely
    a function of the reference photo's own (already-computed) object
    bounds.

    Returns (cropped_img, box_in_cropped_image_coords). Falls back to
    (img, tight_box) unchanged if the expanded crop region is degenerate
    (shouldn't happen for a valid tight_box, but cheap to guard).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = tight_box
    mx, my = (x2 - x1) * context_margin, (y2 - y1) * context_margin
    cx1 = max(0.0, x1 - mx)
    cy1 = max(0.0, y1 - my)
    cx2 = min(float(w), x2 + mx)
    cy2 = min(float(h), y2 + my)
    icx1, icy1, icx2, icy2 = int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
    if icx2 <= icx1 or icy2 <= icy1:
        return img, tight_box
    cropped = img[icy1:icy2, icx1:icx2]
    box_in_crop = (x1 - icx1, y1 - icy1, x2 - icx1, y2 - icy1)
    return cropped, box_in_crop


def inset_box(box: Box, ratio: float) -> Box:
    """Shrink `box` inward by `ratio` of its own width/height on EACH side
    (0.15 = 15% off each edge, i.e. the box keeps its middle 70% x 70%).
    A rectangular detector box's edges/corners commonly include background
    the (usually non-rectangular) real object doesn't actually cover --
    sampling color/appearance from the full box dilutes the measurement
    with that background. Insetting first removes exactly the pixels most
    likely to be background, without needing a real per-candidate
    segmentation mask. 0.0 (or a ratio that would collapse/invert the box)
    = no-op, returns `box` unchanged.
    """
    if ratio <= 0.0:
        return box
    w = box.x2 - box.x1
    h = box.y2 - box.y1
    dx, dy = w * ratio, h * ratio
    x1, y1, x2, y2 = box.x1 + dx, box.y1 + dy, box.x2 - dx, box.y2 - dy
    if x2 <= x1 or y2 <= y1:
        return box
    return Box(x1, y1, x2, y2, score=box.score)


# ---------------------------------------------------------------------------
# Synthetic viewpoint augmentation
# ---------------------------------------------------------------------------

def homography_warp(img_bgr, pitch_deg: float, seed: int | None = None,
                    output_size: tuple[int, int] | None = None):
    """Apply a synthetic top-down homography warp to simulate aerial view."""
    import cv2
    import numpy as np

    rng = random.Random(seed)
    h, w = img_bgr.shape[:2]
    out_w, out_h = output_size if output_size else (w, h)

    pitch_rad = math.radians(pitch_deg)
    f = max(w, h) * 1.2

    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]], dtype=np.float64)

    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    yaw_deg = rng.uniform(-15, 15)
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    Rx = np.array([[1, 0, 0],
                   [0, cos_p, -sin_p],
                   [0, sin_p, cos_p]], dtype=np.float64)
    Ry = np.array([[cos_y, 0, sin_y],
                   [0, 1, 0],
                   [-sin_y, 0, cos_y]], dtype=np.float64)
    R = Ry @ Rx

    H = K @ R @ np.linalg.inv(K)
    return cv2.warpPerspective(img_bgr, H, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def generate_synth_views(img_bgr, mask, method: str, num_views: int,
                         pitch_range_deg: list[float], seed: int = 42) -> list:
    """Generate synthetic aerial views of a reference image."""
    import numpy as np
    rng = random.Random(seed)
    views = []
    for i in range(num_views):
        pitch = rng.uniform(*pitch_range_deg)
        view_seed = seed + i
        if method in ("homography", "perspective_warp"):
            warped = homography_warp(img_bgr, pitch_deg=pitch, seed=view_seed)
        else:
            raise ValueError(f"Unknown synth viewpoint method '{method}'")
        if mask is not None:
            # cv2.warpPerspective on a single-channel (H,W) input returns (H,W),
            # not (H,W,1) — squeezing the singleton channel dim. Warp the mask
            # as plain 2D uint8 instead of round-tripping through a 3D shape.
            warped_mask = homography_warp(
                (mask * 255).astype(np.uint8), pitch_deg=pitch, seed=view_seed
            ).astype(bool)
            warped = warped * warped_mask[:, :, None]
        views.append(warped)
    return views


# ---------------------------------------------------------------------------
# Segmentation mask fallback
# ---------------------------------------------------------------------------

def center_box_mask(shape: tuple, ratio: float):
    """A rectangular mask covering the center `ratio` fraction of an image's
    height/width -- a safe fallback when a segmenter's own mask is
    implausible (too small = likely noise, too large = likely background
    bleed/near-passthrough). Reference photos (object_images) are always
    close-up shots of the target centered in frame, so this is a reasonable
    stand-in for "the object", far better than passing the implausible mask
    (or a full-frame passthrough) straight into a prototype/exemplar build.
    """
    import numpy as np
    h, w = shape[:2]
    mh, mw = int(round(h * ratio)), int(round(w * ratio))
    y0, x0 = max(0, (h - mh) // 2), max(0, (w - mw) // 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y0 + mh, x0:x0 + mw] = True
    return mask

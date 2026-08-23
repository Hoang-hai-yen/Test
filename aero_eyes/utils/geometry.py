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


def dense_patches_to_boxes(
    grid,
    prototype,
    sim_threshold: float,
    min_blob_patches: int,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    patch_size: int = 14,
) -> list[Box]:
    """Convert a DINOv2 dense patch-similarity grid into candidate boxes.

    `grid` is the [Hp, Wp, dim] L2-normalized patch-token grid from
    `DINOv2FeatureExtractor.extract_dense_grid`; `prototype` is the Stage-1
    target signature (also L2-normalized). Cosine-scores every patch,
    thresholds, and groups connected above-threshold patches into blobs via
    `cv2.connectedComponentsWithStats` -- one Box per blob, `score` = the
    blob's max patch similarity. `scale_x`/`scale_y` are
    `extract_dense_grid`'s own return values (resized-square -> tile-native
    pixels); `offset_x`/`offset_y` additionally translate into full-frame
    coordinates when the grid came from a tile, mirroring
    `remap_box_from_tile`'s translation-only convention.
    """
    import cv2
    import numpy as np

    sim = grid @ prototype  # [Hp, Wp]
    mask = (sim >= sim_threshold).astype(np.uint8)
    if not mask.any():
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[Box] = []
    for label in range(1, num_labels):  # label 0 = background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_blob_patches:
            continue
        gx = int(stats[label, cv2.CC_STAT_LEFT])
        gy = int(stats[label, cv2.CC_STAT_TOP])
        gw = int(stats[label, cv2.CC_STAT_WIDTH])
        gh = int(stats[label, cv2.CC_STAT_HEIGHT])
        blob_sim = float(sim[labels == label].max())

        px1 = gx * patch_size * scale_x
        py1 = gy * patch_size * scale_y
        px2 = (gx + gw) * patch_size * scale_x
        py2 = (gy + gh) * patch_size * scale_y

        boxes.append(Box(
            x1=px1 + offset_x, y1=py1 + offset_y,
            x2=px2 + offset_x, y2=py2 + offset_y,
            score=blob_sim,
        ))
    return boxes


# ---------------------------------------------------------------------------
# FPN-style multi-scale feature pyramid fusion
# ---------------------------------------------------------------------------

def _bilinear_resize_grid(grid, new_h: int, new_w: int):
    """Resize a [H, W, C] float array to [new_h, new_w, C] via bilinear
    interpolation, pure numpy (no cv2 -- OpenCV's resize doesn't reliably
    support arbitrary channel counts like a 768-dim DINOv2 feature grid)."""
    import numpy as np

    h, w, _ = grid.shape
    if (h, w) == (new_h, new_w):
        return grid

    y = (np.arange(new_h) + 0.5) * h / new_h - 0.5
    x = (np.arange(new_w) + 0.5) * w / new_w - 0.5
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (y - y0)[:, None, None]
    wx = (x - x0)[None, :, None]

    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bot = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def fuse_feature_pyramid(grids: list):
    """FPN-style top-down fusion of multi-scale DINOv2 patch-token grids.

    `grids` must be ordered coarsest -> finest and must all come from the
    SAME crop at different resolutions (identical field of view, only grid
    resolution differs) -- that's what makes plain bilinear upsampling
    spatially valid here; this is not a general-purpose spatial pyramid
    across different crops/tiles.

    Starting from the coarsest grid (large receptive field per token,
    semantically strong, less prone to single-patch noise), each step
    upsamples it to the next-finer grid's resolution and adds it in,
    L2-renormalizing after every step so the fused tokens stay unit vectors
    (comparable to a plain DINOv2 grid for cosine similarity -- callers like
    `dense_patches_to_boxes` need no changes). Returns the finest grid,
    fused, same shape as `grids[-1]`.
    """
    import numpy as np

    fused = grids[0].astype(np.float32)
    for grid in grids[1:]:
        h, w = grid.shape[:2]
        fused = _bilinear_resize_grid(fused, h, w) + grid
        norm = np.linalg.norm(fused, axis=-1, keepdims=True)
        fused = fused / np.clip(norm, 1e-8, None)
    return fused.astype(np.float32)


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

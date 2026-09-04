"""Box refinement: sharpen an imprecise detection/tracking box to tightly
fit the actual object silhouette, via a per-box segmentation pass.

Addresses the "box not tight" (localization imprecision) component of
ST-IoU loss (see scripts/check_st_iou_breakdown.py) -- distinct from the
tracking-COVERAGE problem stage4.verify_interval addresses (this doesn't
help a box that's simply missing; it only tightens a box that IS present
but loosely placed).

Two backends, both taking (frame_bgr, box, context_margin) -> Box:
  refine_box_with_sam      -- MobileSAM prompted directly with the box.
                               Higher quality, needs a loaded
                               MobileSAMSegmenter (extra forward pass per
                               refined box).
  refine_box_with_grabcut  -- classic OpenCV GrabCut seeded from the box.
                               No model/weights, much cheaper, noticeably
                               lower quality than SAM.

Both fall back to returning `box` UNCHANGED (never raise) if refinement
is unavailable, fails, or finds no plausible mask -- callers never need
their own try/except around these.
"""
from __future__ import annotations

import logging

from aero_eyes.types import Box
from aero_eyes.utils.geometry import mask_bbox

log = logging.getLogger(__name__)


def refine_box_with_sam(segmenter, frame_bgr, box: Box, context_margin: float = 0.2) -> Box:
    """Refine `box` via a MobileSAMSegmenter (see
    MobileSAMSegmenter.segment_box). `segmenter` may be None (e.g. weights
    unavailable) -- returns `box` unchanged in that case."""
    if segmenter is None:
        return box
    mask, offset = segmenter.segment_box(frame_bgr, box, context_margin)
    if mask is None or offset is None:
        return box
    tight = mask_bbox(mask)
    if tight is None:
        return box
    ox, oy = offset
    x1, y1, x2, y2 = tight
    return Box(x1 + ox, y1 + oy, x2 + ox, y2 + oy, score=box.score)


def refine_box_with_grabcut(frame_bgr, box: Box, context_margin: float = 0.2) -> Box:
    """Refine `box` via OpenCV GrabCut, seeded with `box` as the initial
    foreground rectangle -- a model-free, much cheaper (but noticeably
    lower-quality) alternative to refine_box_with_sam. Runs entirely on a
    small crop around `box` (+ context_margin), same as the SAM path."""
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    if bw <= 1 or bh <= 1:
        return box
    mx, my = bw * context_margin, bh * context_margin
    cx1 = max(0, int(box.x1 - mx))
    cy1 = max(0, int(box.y1 - my))
    cx2 = min(w, int(box.x2 + mx))
    cy2 = min(h, int(box.y2 + my))
    if cx2 <= cx1 or cy2 <= cy1:
        return box
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return box

    rect = (
        max(0, int(box.x1 - cx1)), max(0, int(box.y1 - cy1)),
        int(box.x2 - box.x1), int(box.y2 - box.y1),
    )
    # GrabCut requires the seed rect to sit strictly inside the image and
    # have positive size -- both should hold by construction (rect is the
    # ORIGINAL box, crop is that box PLUS margin), but crop edge rounding
    # can occasionally clip it by a pixel; guard rather than let cv2 raise.
    if rect[0] + rect[2] > crop.shape[1] or rect[1] + rect[3] > crop.shape[0]:
        return box
    if rect[2] <= 0 or rect[3] <= 0:
        return box

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        log.warning("GrabCut box-refine failed (%s).", e)
        return box

    fg_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if not fg_mask.any():
        return box
    tight = mask_bbox(fg_mask)
    if tight is None:
        return box
    x1, y1, x2, y2 = tight
    return Box(x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1, score=box.score)


def refine_box(method: str, frame_bgr, box: Box, context_margin: float, segmenter=None) -> Box:
    """Dispatch to refine_box_with_sam ('sam') or refine_box_with_grabcut
    ('grabcut') per box_refine.method. `segmenter` is required (and used)
    only for 'sam' -- ignored otherwise."""
    if method == "sam":
        return refine_box_with_sam(segmenter, frame_bgr, box, context_margin)
    if method == "grabcut":
        return refine_box_with_grabcut(frame_bgr, box, context_margin)
    raise ValueError(f"Unknown box_refine.method '{method}'. Must be 'sam' or 'grabcut'.")

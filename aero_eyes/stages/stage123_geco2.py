"""Stage 1+2+3 replacement — GeCo2 single-shot exemplar detector.

Selected via config: pipeline.detector: geco2  (default stays "legacy",
i.e. the original Stage1/2/3 DINOv2+YOLO/FastSAM pipeline).

Flow:  3 reference images (exemplar box = MobileSAM mask bbox if
       stage123_geco2.segmentation.enabled, else whole image)
       -> GeCo2 exemplar tokens                      (replaces Stage 1)
       -> per-keyframe GeCo2 forward pass
          -> dense box map -> per-frame relative threshold -> NMS -> top-K
                                                        (replaces Stage 2+3)
       -> detections.json (same schema Stage 3 writes, so Stage 4/5 need
          no changes to consume it)

Reads:  cfg.data reference images + video
Writes: <work_dir>/<sample_id>/geco2_prototype.pt  (cached exemplar tokens)
        <work_dir>/<sample_id>/detections.json
Viz:    <work_dir>/<sample_id>/viz/stage123_geco2/ (when save_visualizations=true)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box, Detection
from aero_eyes.utils.geometry import apply_background_mode, crop_to_object, mask_bbox

log = logging.getLogger(__name__)


def _apply_ref_downscale(img, downscale_factor: float):
    """Shrink a reference image to narrow the ground-to-aerial domain gap
    (close-up ref photos are otherwise much crisper/larger-looking than how
    the object appears in the drone video). Actually resizes the array down
    (output is smaller than input) -- GeCo2Detector._load_and_pad's
    resize_and_pad() call right after this always upscales whichever size
    it's given back up to fit stage123_geco2.image_size, so this still gets
    seen by the model at full canvas resolution either way.

    Note: unlike a shrink-then-upscale-back-to-original approach, the
    resulting blur amount is NOT a fixed ratio -- it depends on how the
    shrunk size compares to image_size (e.g. downscale_factor=0.125 on a
    4000x3000 photo -> 500x375, mild blur after re-upscaling to 1024; the
    same factor on a 224x224 photo -> 28x28, extreme blur). Tune per your
    actual reference photo resolution if using this.
    No-op at the default 1.0.
    """
    if downscale_factor >= 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * downscale_factor)))
    new_h = max(1, int(round(h * downscale_factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _render_object_canvas(
    img: np.ndarray,
    mask: np.ndarray,
    tight_box: tuple[float, float, float, float],
    resize_ratio: float,
    background_mode: str,
    blur_sigma: float,
    canvas_px: int,
    antialias: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Draw the reference photo scaled by `resize_ratio` onto a `canvas_px`
    x `canvas_px` canvas, with the object's box center at the canvas center
    -- the canvas is already image_size, so GeCo2's resize_and_pad leaves it
    as is. Pixels beyond the photo's own bounds get the photo's mean color.
    antialias: shrink with INTER_AREA first (like _apply_ref_downscale) and
    then only translate, instead of letting warpAffine's bilinear sampling
    skip pixels on a strong shrink. Returns (canvas_bgr, object box in
    canvas px -- may extend past the canvas if the scaled object is larger)."""
    x1, y1, x2, y2 = tight_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    processed = apply_background_mode(img, mask, background_mode, blur_sigma)
    bg_color = img.reshape(-1, 3).mean(axis=0)

    sx = sy = float(resize_ratio)
    affine_scale = float(resize_ratio)
    if antialias and resize_ratio < 1.0:
        h, w = processed.shape[:2]
        nw, nh = max(1, int(round(w * resize_ratio))), max(1, int(round(h * resize_ratio)))
        processed = cv2.resize(processed, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = nw / w, nh / h
        affine_scale = 1.0
    half = canvas_px / 2.0
    affine = np.array([
        [affine_scale, 0.0, half - cx * sx],
        [0.0, affine_scale, half - cy * sy],
    ], dtype=np.float32)
    canvas = cv2.warpAffine(
        processed, affine, (canvas_px, canvas_px),
        borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(c) for c in bg_color),
    )
    box_in_canvas = (
        (x1 - cx) * sx + half, (y1 - cy) * sy + half,
        (x2 - cx) * sx + half, (y2 - cy) * sy + half,
    )
    return canvas, box_in_canvas


def _clip_box_to_canvas(box, canvas_px: int, sample_id: str, ref_idx: int, factor: float):
    """scale_calibration.mode=factor: an object scaled larger than the canvas
    only partly fits on it -- clip its RoI-Align box to the canvas and say so."""
    x1, y1, x2, y2 = box
    clipped = (max(0.0, x1), max(0.0, y1), min(float(canvas_px), x2), min(float(canvas_px), y2))
    if clipped != (x1, y1, x2, y2):
        log.warning(
            "[Stage123-GeCo2] %s: scale_calibration mode=factor, ref %d at factor %g -- the scaled object "
            "(%.0fx%.0f px) does not fit the %dpx canvas; its box was clipped. Use a smaller factor.",
            sample_id, ref_idx, factor, x2 - x1, y2 - y1, canvas_px,
        )
    return clipped


# ImageNet mean in BGR uint8: what GECO2's resize_and_pad zero padding of the
# NORMALIZED tensor looks like once denormalized (GeCo2Detector._load_and_pad
# normalizes before padding).
_IMAGENET_MEAN_BGR = (0.406 * 255, 0.456 * 255, 0.485 * 255)


def _video_frame_canvas(frame_bgr: np.ndarray, canvas_px: int) -> tuple[np.ndarray, tuple[int, int]]:
    """A video frame put through the same geometry resize_and_pad gives every
    query frame: longer side -> canvas_px (bilinear), top-left aligned, the
    rest padded. Returns (canvas, (valid_h, valid_w)) -- the real-frame area."""
    h, w = frame_bgr.shape[:2]
    scale = canvas_px / float(max(h, w))
    vw, vh = min(canvas_px, max(1, int(round(w * scale)))), min(canvas_px, max(1, int(round(h * scale))))
    canvas = np.empty((canvas_px, canvas_px, 3), np.uint8)
    canvas[:] = np.array(_IMAGENET_MEAN_BGR, np.float64).round().astype(np.uint8)
    canvas[:vh, :vw] = cv2.resize(frame_bgr, (vw, vh), interpolation=cv2.INTER_LINEAR)
    return canvas, (vh, vw)


def _flattest_window(frame_canvas: np.ndarray, valid_hw: tuple[int, int], win_w: int, win_h: int,
                     avoid_boxes=()):
    """Top-left (x, y) of the win_w x win_h window inside the real-frame area
    with the lowest mean gradient magnitude (Sobel), via an integral image.
    Clipped pixels (any channel >= 250) count as maximally busy: blown-out
    highlights have no gradient, but they are sunlit objects (e.g. white
    sheets), not empty ground. avoid_boxes (x1,y1,x2,y2) count as maximally
    busy too -- e.g. objects already pasted onto this frame. None if the
    window does not fit."""
    vh, vw = valid_hw
    if win_w > vw or win_h > vh:
        return None
    real = frame_canvas[:vh, :vw]
    gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad[real.max(axis=2) >= 250] = 1000.0
    for bx1, by1, bx2, by2 in avoid_boxes:
        grad[max(0, int(by1)):max(0, int(np.ceil(by2))), max(0, int(bx1)):max(0, int(np.ceil(bx2)))] = 1000.0
    ii = cv2.integral(grad, sdepth=cv2.CV_64F)
    sums = ii[win_h:, win_w:] - ii[:-win_h, win_w:] - ii[win_h:, :-win_w] + ii[:-win_h, :-win_w]
    y, x = np.unravel_index(int(np.argmin(sums)), sums.shape)
    return int(x), int(y)


def _scaled_object_patch(
    img: np.ndarray, mask: np.ndarray, tight_box, resize_ratio: float, feather_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The object's tight box from the reference photo scaled by resize_ratio
    (INTER_AREA when shrinking) plus its mask as a feathered alpha in [0,1]."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(np.floor(tight_box[0]))), max(0, int(np.floor(tight_box[1])))
    x2, y2 = min(w, int(np.ceil(tight_box[2]))), min(h, int(np.ceil(tight_box[3])))
    crop, crop_mask = img[y1:y2, x1:x2], mask[y1:y2, x1:x2].astype(np.float32)
    pw = max(1, int(round((x2 - x1) * resize_ratio)))
    ph = max(1, int(round((y2 - y1) * resize_ratio)))
    interp = cv2.INTER_AREA if resize_ratio < 1.0 else cv2.INTER_LINEAR
    patch = cv2.resize(crop, (pw, ph), interpolation=interp)
    alpha = cv2.resize(crop_mask, (pw, ph), interpolation=cv2.INTER_AREA if resize_ratio < 1.0 else cv2.INTER_LINEAR)
    if feather_px > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_px, borderType=cv2.BORDER_CONSTANT)
    return patch, np.clip(alpha, 0.0, 1.0)


def _paste_object_on_frame(
    frame_canvas: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x: int, y: int, feather_px: float,
    blur_box_surround: bool = True,
) -> np.ndarray:
    """Composite `patch` (alpha) onto a copy of frame_canvas at (x, y). With
    blur_box_surround, the patch's box area of the frame is first swapped for
    a strongly blurred copy (soft-edged, so it does not draw a rectangle) --
    the pixels inside the box but outside the object's mask are pooled by
    RoI-Align directly, so they must not carry sharp frame structure such as
    a confuser; without it the object sits on the sharp frame. Parts of the
    patch falling outside the canvas are dropped."""
    out = frame_canvas.astype(np.float32)
    H, W = out.shape[:2]
    ph, pw = alpha.shape
    cx1, cy1, cx2, cy2 = max(0, x), max(0, y), min(W, x + pw), min(H, y + ph)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame_canvas.copy()
    if blur_box_surround:
        sigma = max(3.0, 0.5 * max(ph, pw))
        pad = int(np.ceil(3 * sigma))
        ox1, oy1, ox2, oy2 = max(0, cx1 - pad), max(0, cy1 - pad), min(W, cx2 + pad), min(H, cy2 + pad)
        region = out[oy1:oy2, ox1:ox2]
        blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=sigma)
        box_w = np.zeros(region.shape[:2], np.float32)
        box_w[cy1 - oy1:cy2 - oy1, cx1 - ox1:cx2 - ox1] = 1.0
        if feather_px > 0:
            box_w = cv2.GaussianBlur(box_w, (0, 0), sigmaX=max(feather_px, 1.0))
        region[:] = box_w[..., None] * blurred + (1.0 - box_w[..., None]) * region

    a = alpha[cy1 - y:cy2 - y, cx1 - x:cx2 - x][..., None]
    p = patch[cy1 - y:cy2 - y, cx1 - x:cx2 - x].astype(np.float32)
    out[cy1:cy2, cx1:cx2] = a * p + (1.0 - a) * out[cy1:cy2, cx1:cx2]
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_on_video_frame(
    img: np.ndarray, mask: np.ndarray, tight_box, resize_ratio: float,
    frame_canvas: np.ndarray, valid_hw: tuple[int, int], context_margin: float, feather_px: float,
    log_ctx: str = "", blur_box_surround: bool = True, avoid_boxes=(),
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """scale_calibration.background_source=video_frame: paste the object
    (scaled by resize_ratio, feathered) at the flattest spot of the real
    frame area, away from avoid_boxes; returns (canvas, object box in
    canvas px)."""
    patch, alpha = _scaled_object_patch(img, mask, tight_box, resize_ratio, feather_px)
    ph, pw = alpha.shape
    win_w = int(round(pw * (1.0 + 2.0 * context_margin)))
    win_h = int(round(ph * (1.0 + 2.0 * context_margin)))
    loc = _flattest_window(frame_canvas, valid_hw, win_w, win_h, avoid_boxes)
    if loc is None:
        loc = _flattest_window(frame_canvas, valid_hw, pw, ph, avoid_boxes)
        if loc is not None:
            x, y = loc
        else:
            vh, vw = valid_hw
            x, y = int(round((vw - pw) / 2.0)), int(round((vh - ph) / 2.0))
            log.warning(
                "%spaste on video frame: scaled object (%dx%d) does not fit the real frame area (%dx%d) "
                "-- centered and clipped. Use a smaller scale.", log_ctx, pw, ph, vw, vh,
            )
    else:
        x, y = loc[0] + (win_w - pw) // 2, loc[1] + (win_h - ph) // 2
    canvas = _paste_object_on_frame(frame_canvas, patch, alpha, x, y, feather_px, blur_box_surround)
    H, W = canvas.shape[:2]
    box = (float(max(0, x)), float(max(0, y)), float(min(W, x + pw)), float(min(H, y + ph)))
    return canvas, box


def _object_px_resize_ratio(tight_box, expected_object_px, video_longer_dim: int,
                            context_margin: float, canvas_px: int) -> float:
    """scale_calibration.mode=object_px: the scale that makes the object (+
    context_margin) occupy the same fraction of a canvas_px canvas as
    expected_object_px does of the raw video frame."""
    x1, y1, x2, y2 = tight_box
    obj_size = max(x2 - x1, y2 - y1) * (1.0 + context_margin)
    target_ratio = max(expected_object_px) / float(video_longer_dim)
    if target_ratio <= 0:
        raise ValueError("scale_calibration: expected_object_px / video frame size must be > 0")
    return canvas_px / (obj_size / target_ratio)


def _build_scale_calibrated_canvas(
    img: np.ndarray,
    mask: np.ndarray,
    tight_box: tuple[float, float, float, float],
    expected_object_px: tuple[float, float],
    video_longer_dim: int,
    context_margin: float,
    background_mode: str,
    blur_sigma: float,
    canvas_px: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Build a `canvas_px` x `canvas_px` canvas where the object occupies
    the SAME fraction of the canvas's side as it's expected to occupy in
    the query video frame after ITS OWN resize_and_pad -- the one thing
    stage123_geco2.ref_downscale_factor cannot do (see
    ScaleCalibrationConfig docstring for why a uniform pre-shrink of the
    whole photo is exactly cancelled out by resize_and_pad; only changing
    how much the object fills a canvas -- i.e. cropping tighter/looser --
    actually changes that ratio).

    Conceptually this crops a square region of the ORIGINAL reference
    photo, centered on the object and sized so the object (plus
    `context_margin` of extra padding) maps to exactly `expected_object_px`
    at native resolution -- but for a small/distant target that "ideal"
    region can be enormous (tens of thousands of px, mostly empty
    background) relative to the object, so it's never materialized at that
    size: cv2.warpAffine renders directly into the final `canvas_px` output
    (matching stage123_geco2.image_size, so the resize_and_pad call right
    after this is a geometric no-op -- our canvas is already the right
    size), scaling and cropping/padding in one pass regardless of how
    large the conceptual source region is.

    Returns (canvas_bgr, object_box_in_canvas_px). Pixels outside the
    photo's own bounds are filled with the photo's mean color (there is no
    real pixel data out there, regardless of background_mode).
    """
    resize_ratio = _object_px_resize_ratio(tight_box, expected_object_px, video_longer_dim, context_margin, canvas_px)
    return _render_object_canvas(img, mask, tight_box, resize_ratio, background_mode, blur_sigma, canvas_px)


def _locate_video(cfg, sample_id: str) -> Path:
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(
            f"No video matching '{cfg.data.video_glob}' found in {video_dir}."
        )
    return video_files[0]


def list_ref_paths(refs_dir: Path, num_references: int) -> list[Path]:
    """The reference-image files of one sample, in the order every consumer
    uses (sorted, first num_references) -- shared with
    scripts/precompute_ref_masks.py so precomputed masks line up by index."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in exts
    )
    if len(ref_paths) < num_references:
        raise FileNotFoundError(
            f"Expected {num_references} reference images in {refs_dir}, "
            f"found {len(ref_paths)}."
        )
    return ref_paths[:num_references]


def _load_ref_images(cfg, sample_id: str) -> list:
    refs_dir = Path(cfg.data.data_root) / sample_id / cfg.data.refs_subdir
    return [cv2.imread(str(p)) for p in list_ref_paths(refs_dir, cfg.data.num_references)]


def _save_mask_box_viz(ref_imgs: list[np.ndarray], raw_boxes: list, out_dir: Path) -> None:
    """Debug viz: draws MobileSAM's tight mask_bbox on top of the ORIGINAL
    (unprocessed, un-background-filled) reference photo -- lets you
    sanity-check that segmentation actually bounds the real object BEFORE
    trusting a downstream crop/scale sweep built from it
    (auto_scale_calibration, learned_scale_fusion, crop_to_object,
    scale_calibration all pool from exactly this box). Drawn on the
    original photo (not background_mode-filled/blurred) so surrounding
    context stays visible for judging whether the box is tight/correct or
    has drifted onto background/a wrong object.
    """
    from aero_eyes.utils.viz import draw_box

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, b) in enumerate(zip(ref_imgs, raw_boxes)):
        annotated = img.copy()
        if b is not None:
            draw_box(annotated, Box(*b), "MobileSAM tight box", (0, 255, 0))
        else:
            cv2.putText(annotated, "no mask/box (empty mask?)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"ref_{i}_mask_box.jpg"), annotated)


def build_exemplar_prototype(cfg, sample_id: str, detector, work_dir: Path):
    """Exemplar-token build shared by run_stage123_geco2() (below) and
    scripts/check_geco2_score_separation.py, so the diagnostic script always
    scores against the SAME exemplar quality (MobileSAM-masked + tight box)
    the real pipeline uses -- building it any simpler here would make that
    script's calibration numbers not representative of a real run.

    Uses/writes the same on-disk cache (work_dir/prototype_cache_name) as
    run_stage123_geco2, respecting cfg.project.use_cache.
    """
    from aero_eyes.models.geco2_detector import GeCo2Detector
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.video import read_frame, video_info

    g = cfg.stage123_geco2
    proto_path = work_dir / g.prototype_cache_name
    if cfg.project.use_cache and proto_path.exists():
        log.info("[Stage123-GeCo2] %s: using cached exemplar tokens at %s", sample_id, proto_path)
        return GeCo2Detector.load_prototype(proto_path)

    ref_imgs = _load_ref_images(cfg, sample_id)

    # Exemplar box passed to encode_exemplars(), in the coord system of
    # whichever ref_imgs actually get passed to it. None = whole image
    # (encode_exemplars' default).
    ref_boxes: list[tuple[float, float, float, float] | None] | None = None

    seg_cfg = g.segmentation
    sc_cfg = g.scale_calibration
    asc_cfg = g.auto_scale_calibration

    if asc_cfg.enabled:
        # Track A (see docs/GECO2_scale_domain_gap_plan.md): per-sample
        # automatic replacement for hand-tuning ref_downscale_factor AND
        # crop_context_margin -- takes over entirely instead of composing
        # with scale_calibration/crop_to_object/ref_downscale_* (it does
        # its own internal crop_to_object + _apply_ref_downscale sweep per
        # candidate), so this is a separate top-level branch rather than
        # threaded through the manual-tuning branches below.
        if not seg_cfg.enabled:
            raise ValueError(
                "stage123_geco2.auto_scale_calibration.enabled requires "
                "segmentation.enabled (needs a tight mask box to crop/scale around)."
            )
        if sc_cfg.enabled or g.crop_to_object or g.ref_downscale_levels or g.ref_downscale_factor != 1.0:
            log.warning(
                "[Stage123-GeCo2] %s: auto_scale_calibration.enabled overrides "
                "scale_calibration/crop_to_object/ref_downscale_factor/ref_downscale_levels -- "
                "their configured values are ignored this run.", sample_id,
            )
        from aero_eyes.models.geco2_auto_scale import build_auto_scaled_prototype
        from aero_eyes.models.segmentation import build_segmenter

        segmenter = build_segmenter(seg_cfg, cfg)
        masks = []
        for img in ref_imgs:
            mask = segmenter.segment(img)
            if seg_cfg.center_crop_fallback:
                mask_ratio = float(mask.sum()) / float(mask.size)
                if mask_ratio < seg_cfg.min_valid_mask_ratio or mask_ratio > seg_cfg.max_valid_mask_ratio:
                    from aero_eyes.utils.geometry import center_box_mask
                    log.warning(
                        "[Stage123-GeCo2] %s: MobileSAM mask area implausible (%.1f%% of frame), "
                        "using center-crop fallback (ratio=%.2f) instead of passthrough.",
                        sample_id, mask_ratio * 100.0, seg_cfg.center_fallback_ratio,
                    )
                    mask = center_box_mask(img.shape, seg_cfg.center_fallback_ratio)
            masks.append(mask)
        raw_boxes = [mask_bbox(m) for m in masks]
        bg_imgs = [apply_background_mode(img, m, seg_cfg.background_mode, seg_cfg.blur_sigma)
                   for img, m in zip(ref_imgs, masks)]
        if cfg.runtime.save_visualizations:
            vizmod.save_stage1_refs(bg_imgs, masks, work_dir / "viz" / "stage123_geco2" / "refs")
            _save_mask_box_viz(ref_imgs, raw_boxes, work_dir / "viz" / "stage123_geco2" / "refs_mask_box")

        video_path = _locate_video(cfg, sample_id)
        prototype, debug_info = build_auto_scaled_prototype(
            cfg, sample_id, detector, bg_imgs, raw_boxes, video_path,
        )
        debug_path = work_dir / "geco2_auto_scale_calibration.json"
        debug_path.write_text(json.dumps(debug_info, indent=2))
        log.info(
            "[Stage123-GeCo2] %s: auto_scale_calibration selected weights=%s over %d "
            "candidate(s) (metric=%s) -- see %s",
            sample_id, [round(w, 3) for w in debug_info["weights"]],
            len(debug_info["candidates"]), debug_info["metric_used"], debug_path,
        )
    elif g.learned_scale_fusion.enabled:
        # Track B (see docs/GECO2_scale_domain_gap_plan.md): inference-side
        # use of a checkpoint finetuned with --num-ref-scale-variants>1
        # (has a trained scale_fusion_gates submodule). Builds one
        # candidate exemplar per (ref image, factor) -- same mechanism as
        # ref_downscale_levels below -- then fuses them with the trained
        # gate instead of flat-concatenating. Also a separate top-level
        # branch (like auto_scale_calibration) for the same reason: it
        # takes over entirely instead of composing with the manual-tuning
        # branches below.
        lsf_cfg = g.learned_scale_fusion
        if not seg_cfg.enabled:
            raise ValueError(
                "stage123_geco2.learned_scale_fusion.enabled requires "
                "segmentation.enabled (needs a tight mask box to build candidates around)."
            )
        if g.use_shape_token:
            raise ValueError(
                "stage123_geco2.learned_scale_fusion.enabled requires use_shape_token=false -- "
                "Track B checkpoints are trained without shape tokens (see "
                "docs/GECO2_FINETUNE_PLAN.md point 1)."
            )
        if sc_cfg.enabled or g.crop_to_object or g.ref_downscale_levels or g.ref_downscale_factor != 1.0:
            log.warning(
                "[Stage123-GeCo2] %s: learned_scale_fusion.enabled overrides "
                "scale_calibration/crop_to_object/ref_downscale_factor/ref_downscale_levels -- "
                "their configured values are ignored this run.", sample_id,
            )
        from aero_eyes.models.segmentation import build_segmenter

        segmenter = build_segmenter(seg_cfg, cfg)
        masks = [segmenter.segment(img) for img in ref_imgs]
        raw_boxes = [mask_bbox(m) for m in masks]
        bg_imgs = [apply_background_mode(img, m, seg_cfg.background_mode, seg_cfg.blur_sigma)
                   for img, m in zip(ref_imgs, masks)]
        if cfg.runtime.save_visualizations:
            vizmod.save_stage1_refs(bg_imgs, masks, work_dir / "viz" / "stage123_geco2" / "refs")
            _save_mask_box_viz(ref_imgs, raw_boxes, work_dir / "viz" / "stage123_geco2" / "refs_mask_box")

        multiscale_imgs: list[np.ndarray] = []
        multiscale_boxes: list[tuple[float, float, float, float] | None] = []
        group_ids: list[int] = []
        for ref_idx, (img, b) in enumerate(zip(bg_imgs, raw_boxes)):
            for f in lsf_cfg.candidate_factors:
                multiscale_boxes.append(tuple(c * f for c in b) if b is not None else None)
                multiscale_imgs.append(_apply_ref_downscale(img, f))
                group_ids.append(ref_idx)

        video_path = _locate_video(cfg, sample_id)
        info = video_info(video_path)
        total_frames = info["total_frames"]
        from aero_eyes.models.geco2_auto_scale import sample_uniform_frame_indices
        context_idxs = sample_uniform_frame_indices(total_frames, lsf_cfg.num_context_frames)
        context_frames = [read_frame(video_path, i) for i in context_idxs]

        prototype = detector.encode_exemplars_fused(
            multiscale_imgs, multiscale_boxes, group_ids, context_frames,
        )
        log.info(
            "[Stage123-GeCo2] %s: learned_scale_fusion -- %d ref image(s) x %d factor(s), "
            "context averaged over %d frame(s)",
            sample_id, len(bg_imgs), len(lsf_cfg.candidate_factors), len(context_frames),
        )
    elif seg_cfg.enabled:
        from aero_eyes.models.segmentation import build_segmenter
        segmenter = build_segmenter(seg_cfg, cfg)
        masks = []
        for img in ref_imgs:
            mask = segmenter.segment(img)
            if seg_cfg.center_crop_fallback:
                mask_ratio = float(mask.sum()) / float(mask.size)
                if mask_ratio < seg_cfg.min_valid_mask_ratio or mask_ratio > seg_cfg.max_valid_mask_ratio:
                    from aero_eyes.utils.geometry import center_box_mask
                    log.warning(
                        "[Stage123-GeCo2] %s: MobileSAM mask area implausible (%.1f%% of frame), "
                        "using center-crop fallback (ratio=%.2f) instead of passthrough.",
                        sample_id, mask_ratio * 100.0, seg_cfg.center_fallback_ratio,
                    )
                    mask = center_box_mask(img.shape, seg_cfg.center_fallback_ratio)
            masks.append(mask)
        # Tight box around the actual segmented object, on the ORIGINAL
        # (pre-downscale/pre-canvas) ref photo -- RoI-align-ing the whole
        # (masked) image instead pools in a lot of background + any
        # resize_and_pad zero-padding, diluting the exemplar token
        # (empirically confirmed to matter).
        raw_boxes = [mask_bbox(m) for m in masks]
        if cfg.runtime.save_visualizations:
            _save_mask_box_viz(ref_imgs, raw_boxes, work_dir / "viz" / "stage123_geco2" / "refs_mask_box")

        if sc_cfg.enabled:
            # scale_calibration: build a canvas per (ref image, scale) pair
            # where the object occupies the same canvas-relative size it's
            # expected to occupy in the query video frame -- see
            # _build_scale_calibrated_canvas for why ref_downscale_factor
            # alone cannot do this. multi_scale_mode=="first" (default) only
            # uses expected_object_px[0], reproducing the original
            # one-canvas-per-ref behavior exactly; "all" builds one canvas
            # per scale for EVERY ref image and feeds all of them into
            # encode_exemplars as independent exemplar entries (each with
            # its own appearance token AND, when use_shape_token, its own
            # shape token derived from that scale's own box size) -- see
            # ScaleCalibrationConfig.multi_scale_mode docstring.
            # scale_calibration.mode picks the scale(s) each ref photo is
            # drawn onto the canvas at -- see ScaleCalibrationConfig:
            #   object_px: from expected_object_px (object size in the video)
            #   factor:    ref_downscale_factor / ref_downscale_levels, as is
            if sc_cfg.mode == "factor":
                factors = list(g.ref_downscale_levels) if g.ref_downscale_levels else [g.ref_downscale_factor]
                variants = [(f"_down_{f:g}" if len(factors) > 1 else "", f) for f in factors]
            else:
                video_path = _locate_video(cfg, sample_id)
                info = video_info(video_path)
                video_longer_dim = max(info["width"], info["height"])
                scales = (
                    sc_cfg.expected_object_px if sc_cfg.multi_scale_mode == "all"
                    else sc_cfg.expected_object_px[:1]
                )
                variants = [(f"_scale_{i}" if len(scales) > 1 else "", tuple(px)) for i, px in enumerate(scales)]
            # scale_calibration.background_source=video_frame: paste onto the
            # sample's video frame (same resize_and_pad geometry as queries)
            # instead of the photo's own surroundings.
            frame_canvas = valid_hw = None
            if sc_cfg.background_source == "video_frame":
                frame = read_frame(_locate_video(cfg, sample_id), sc_cfg.video_frame_index)
                if frame is None:
                    raise ValueError(
                        f"scale_calibration.background_source=video_frame: could not read frame "
                        f"{sc_cfg.video_frame_index} of {sample_id}'s video"
                    )
                frame_canvas, valid_hw = _video_frame_canvas(frame, g.image_size)
            num_orig_refs = len(ref_imgs)
            canvases, canvas_boxes, canvas_labels = [], [], []
            for ref_idx, (img, m, b) in enumerate(zip(ref_imgs, masks, raw_boxes)):
                if b is None:
                    # Empty mask (fallback/edge case) -- nothing to
                    # calibrate against; fall back to whole-image exemplar
                    # (once, regardless of how many scales were requested).
                    canvases.append(img)
                    canvas_boxes.append(None)
                    canvas_labels.append(f"ref_{ref_idx}")
                    continue
                for suffix, value in variants:
                    if frame_canvas is not None:
                        ratio = value if sc_cfg.mode == "factor" else _object_px_resize_ratio(
                            b, value, video_longer_dim, sc_cfg.context_margin, g.image_size,
                        )
                        canvas, box_c = _render_on_video_frame(
                            img, m, b, ratio, frame_canvas, valid_hw, sc_cfg.context_margin, sc_cfg.feather_px,
                            log_ctx=f"[Stage123-GeCo2] {sample_id}: ref {ref_idx}: ",
                            blur_box_surround=sc_cfg.blur_box_surround,
                        )
                    elif sc_cfg.mode == "factor":
                        canvas, box_c = _render_object_canvas(
                            img, m, b, value, seg_cfg.background_mode, seg_cfg.blur_sigma,
                            canvas_px=g.image_size, antialias=True,
                        )
                        box_c = _clip_box_to_canvas(box_c, g.image_size, sample_id, ref_idx, value)
                    else:
                        canvas, box_c = _build_scale_calibrated_canvas(
                            img, m, b, value, video_longer_dim,
                            sc_cfg.context_margin, seg_cfg.background_mode, seg_cfg.blur_sigma,
                            canvas_px=g.image_size,
                        )
                    canvases.append(canvas)
                    canvas_boxes.append(box_c)
                    canvas_labels.append(f"ref_{ref_idx}{suffix}")
            if frame_canvas is not None:
                where = [f"{bx[2] - bx[0]:.0f}x{bx[3] - bx[1]:.0f}@({bx[0]:.0f},{bx[1]:.0f})"
                         for bx in canvas_boxes if bx is not None]
                log.info(
                    "[Stage123-GeCo2] %s: scale_calibration background_source=video_frame (frame %d, "
                    "feather_px=%.1f, blur_box_surround=%s) -- object pasted at: %s", sample_id,
                    sc_cfg.video_frame_index, sc_cfg.feather_px, sc_cfg.blur_box_surround, ", ".join(where),
                )
            if sc_cfg.mode == "factor":
                sizes = [f"{bx[2] - bx[0]:.0f}x{bx[3] - bx[1]:.0f}" for bx in canvas_boxes if bx is not None]
                log.info(
                    "[Stage123-GeCo2] %s: scale_calibration mode=factor -- %d ref image(s) x %d factor(s) %s "
                    "= %d exemplar entries; object box on the %dpx canvas: %s",
                    sample_id, num_orig_refs, len(variants), [v for _, v in variants], len(canvases),
                    g.image_size, ", ".join(sizes),
                )
            elif len(variants) > 1:
                log.info(
                    "[Stage123-GeCo2] %s: scale_calibration multi_scale_mode=all -- "
                    "%d ref image(s) x %d scale(s) = %d exemplar entries",
                    sample_id, num_orig_refs, len(variants), len(canvases),
                )
            if cfg.runtime.save_visualizations:
                out_dir = work_dir / "viz" / "stage123_geco2" / "refs_scale_calibrated"
                out_dir.mkdir(parents=True, exist_ok=True)
                for label, c, box_c in zip(canvas_labels, canvases, canvas_boxes):
                    cv2.imwrite(str(out_dir / f"{label}_calibrated.jpg"), c)
                    # ALSO save a copy with the actual RoI-Align pooling box
                    # drawn on top -- background_mode=keep_real (or a huge
                    # conceptual crop region relative to the reference photo)
                    # can make the canvas visually look "unsegmented" even
                    # when the pooling region itself is correctly tight;
                    # this makes what GeCo2 actually pools from unambiguous,
                    # independent of background_mode.
                    if box_c is not None:
                        annotated = c.copy()
                        from aero_eyes.utils.viz import draw_box
                        draw_box(annotated, Box(*box_c), "RoI-Align region", (0, 255, 0))
                        cv2.imwrite(str(out_dir / f"{label}_calibrated_box.jpg"), annotated)
            ref_imgs = canvases
            ref_boxes = canvas_boxes
        else:
            ref_imgs = [apply_background_mode(img, m, seg_cfg.background_mode, seg_cfg.blur_sigma)
                        for img, m in zip(ref_imgs, masks)]
            if cfg.runtime.save_visualizations:
                vizmod.save_stage1_refs(ref_imgs, masks, work_dir / "viz" / "stage123_geco2" / "refs")

            if g.crop_to_object:
                cropped_imgs, cropped_boxes = [], []
                for img, b in zip(ref_imgs, raw_boxes):
                    if b is None:
                        cropped_imgs.append(img)
                        cropped_boxes.append(None)
                        continue
                    cimg, cbox = crop_to_object(img, b, g.crop_context_margin)
                    cropped_imgs.append(cimg)
                    cropped_boxes.append(cbox)
                if cfg.runtime.save_visualizations:
                    out_dir = work_dir / "viz" / "stage123_geco2" / "refs_cropped"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for i, (c, box_c) in enumerate(zip(cropped_imgs, cropped_boxes)):
                        cv2.imwrite(str(out_dir / f"ref_{i}_cropped.jpg"), c)
                        if box_c is not None:
                            annotated = c.copy()
                            from aero_eyes.utils.viz import draw_box
                            draw_box(annotated, Box(*box_c), "RoI-Align region", (0, 255, 0))
                            cv2.imwrite(str(out_dir / f"ref_{i}_cropped_box.jpg"), annotated)
                ref_imgs = cropped_imgs
                raw_boxes = cropped_boxes

            # Scale into the coord system _apply_ref_downscale below produces
            # (uniform factor in both axes, matching that function). NOTE:
            # this only affects blur/detail -- it does NOT change the
            # object's final size on the model's canvas (resize_and_pad
            # re-normalizes the whole image's longer side regardless; see
            # ScaleCalibrationConfig docstring). Use scale_calibration above
            # to fix apparent-size mismatch. crop_to_object above (if
            # enabled) is the mechanism that DOES change the object's final
            # canvas size, without needing an oracle scale estimate.
            # ref_downscale_levels (opt-in): one exemplar entry per (ref
            # image, factor) instead of one fixed factor -- see
            # Stage123Geco2Config.ref_downscale_levels docstring. Defaults
            # to [ref_downscale_factor], i.e. exactly the old behavior.
            levels = list(g.ref_downscale_levels) if g.ref_downscale_levels else [g.ref_downscale_factor]
            multiscale_imgs: list[np.ndarray] = []
            multiscale_boxes: list[tuple[float, float, float, float] | None] = []
            for img, b in zip(ref_imgs, raw_boxes):
                for f in levels:
                    multiscale_boxes.append(tuple(c * f for c in b) if b is not None else None)
                    multiscale_imgs.append(_apply_ref_downscale(img, f))
            if len(levels) > 1:
                log.info(
                    "[Stage123-GeCo2] %s: ref_downscale_levels multi-scale exemplar -- "
                    "%d ref image(s) x %d level(s) = %d exemplar entries (levels=%s)",
                    sample_id, len(ref_imgs), len(levels), len(multiscale_imgs), levels,
                )
            ref_imgs = multiscale_imgs
            ref_boxes = multiscale_boxes
    else:
        # No mask/box exists without segmentation, so background-fill/
        # crop_to_object above (both need one) are skipped here -- but
        # ref_downscale_factor/ref_downscale_levels is just a resize, no
        # mask required, so it still applies to the RAW reference images.
        # Previously there was no else branch at all here, which silently
        # made both a no-op whenever segmentation.enabled was false,
        # regardless of their own value -- neither field's own docstring
        # documented that dependency. ref_boxes stays None (whole image,
        # its value from above) since there's no box to scale alongside.
        levels = list(g.ref_downscale_levels) if g.ref_downscale_levels else [g.ref_downscale_factor]
        num_orig_refs = len(ref_imgs)
        ref_imgs = [_apply_ref_downscale(img, f) for img in ref_imgs for f in levels]
        if len(levels) > 1:
            log.info(
                "[Stage123-GeCo2] %s: ref_downscale_levels multi-scale exemplar "
                "(segmentation disabled) -- %d ref image(s) x %d level(s) = %d "
                "exemplar entries (levels=%s)",
                sample_id, num_orig_refs, len(levels), len(ref_imgs), levels,
            )

    # Debug viz + the final encode_exemplars call -- skipped when
    # auto_scale_calibration or learned_scale_fusion already built
    # `prototype` itself above (their own per-candidate images/boxes are
    # saved separately -- geco2_auto_scale_calibration.json for the former;
    # ref_imgs/ref_boxes here still hold the ORIGINAL unprocessed refs in
    # both cases, not what was actually encoded).
    if not asc_cfg.enabled and not g.learned_scale_fusion.enabled:
        # Debug viz: the ACTUAL final images/boxes about to be encoded, after
        # every step above (mask, background-fill, crop_to_object,
        # scale_calibration, ref_downscale_factor/levels -- or none of those,
        # if segmentation is disabled) -- every EARLIER viz call above only
        # shows an intermediate stage, none of them show what encode_exemplars
        # itself actually receives. Saved unconditionally right before the
        # call so this always reflects reality regardless of which branch ran.
        if cfg.runtime.save_visualizations:
            from aero_eyes.utils.viz import draw_box
            out_dir = work_dir / "viz" / "stage123_geco2" / "refs_final"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(ref_imgs):
                cv2.imwrite(str(out_dir / f"ref_final_{i:02d}.jpg"), img)
                box_i = ref_boxes[i] if ref_boxes is not None else None
                if box_i is not None:
                    annotated = img.copy()
                    draw_box(annotated, Box(*box_i), "RoI-Align region", (0, 255, 0))
                    cv2.imwrite(str(out_dir / f"ref_final_{i:02d}_box.jpg"), annotated)

        prototype = detector.encode_exemplars(ref_imgs, ref_boxes=ref_boxes)

    dc_cfg = g.domain_calibration
    if dc_cfg.enabled:
        video_path = _locate_video(cfg, sample_id)
        info = video_info(video_path)
        total_frames = info["total_frames"]
        n = max(1, min(dc_cfg.num_sample_frames, total_frames))
        sample_idxs = sorted(set(np.linspace(0, max(total_frames - 1, 0), num=n).astype(int).tolist()))
        sample_frames = [read_frame(video_path, i) for i in sample_idxs]
        video_domain_means = detector.estimate_domain_shift(sample_frames)
        prototype = GeCo2Detector.calibrate_prototype(
            prototype, video_domain_means, num_refs=len(ref_imgs), strength=dc_cfg.strength,
            tokens_per_ref=2 if g.use_shape_token else 1,
        )
        log.info("[Stage123-GeCo2] %s: domain-calibrated exemplar tokens using %d sample frames "
                 "(strength=%.2f)", sample_id, len(sample_frames), dc_cfg.strength)

    GeCo2Detector.save_prototype(prototype, proto_path)
    log.info("[Stage123-GeCo2] %s: encoded %d reference exemplars -> %s",
             sample_id, len(ref_imgs), proto_path)
    return prototype


class ColorSignature:
    """Reference object color signature: TWO independent per-ref-image
    histogram sets (Hue+Saturation, and Value alone) plus the blend weight
    between them -- see ColorPostfilterConfig docstring for why both are
    needed (Hue+Sat is lighting-robust but useless for near-achromatic
    objects; Value is the only reliable signal for exactly those, at the
    cost of being lighting-sensitive)."""

    def __init__(
        self, hs_hists: list[np.ndarray], v_hists: list[np.ndarray], confidence: float,
        ref_agreement: float = 1.0, is_high_vis: float = 0.0,
    ):
        self.hs_hists = hs_hists
        self.v_hists = v_hists
        self.confidence = confidence
        # Mean pairwise Hue+Saturation histogram similarity AMONG the ref
        # photos themselves -- see ColorPostfilterConfig.overexposure_ramp_
        # frac's neighboring docstring (the "ref_color_agreement" field) for
        # the full rationale (docs/attribute_taxonomy_plan.md SS8.3). 1.0 =
        # fully agree (or trivially true with a single ref photo).
        self.ref_agreement = ref_agreement
        # Mean is_high_vis (aero_eyes.utils.color.compute_is_high_vis) across
        # the ref photos -- see ColorPostfilterConfig.is_high_vis_gating_
        # enabled's own docstring.
        self.is_high_vis = is_high_vis


class ClassifierColorSignature:
    """color_postfilter.method == "classifier": the reference object's color
    GROUP according to a learned classifier (aero_eyes.models.
    color_classifier), plus the classifier itself so apply_color_postfilter
    can score candidates with the same model. `active` is False when the
    reference's own group is too uncertain to filter against (see
    ColorPostfilterConfig.classifier_min_ref_confidence) -- every candidate
    is then kept."""

    def __init__(self, classifier, ref_probs: np.ndarray, active: bool):
        self.classifier = classifier
        self.ref_probs = ref_probs
        self.ref_group = int(ref_probs.argmax())
        self.ref_group_name = classifier.group_names[self.ref_group]
        self.ref_confidence = float(ref_probs.max())
        self.active = active


def _build_classifier_signature(cfg, sample_id: str, cpf, seg_cfg, log_prefix: str) -> ClassifierColorSignature:
    """Classify each reference photo cropped to its segmentation mask's bbox
    (+10% pad), then average the softmax over the photos. The background
    inside that crop is kept as-is, NOT blacked out: black fill would pull
    every reference toward the dark group, and the classifier was trained on
    whole photos with natural backgrounds. Not cached to disk -- a few
    MobileSAM + classifier passes over 3 photos are cheap."""
    from aero_eyes.models.color_classifier import load_color_classifier

    if not cpf.classifier_weights_path:
        raise ValueError("color_postfilter.method='classifier' needs color_postfilter.classifier_weights_path")
    clf = load_color_classifier(cpf.classifier_weights_path, cpf.classifier_device)

    ref_imgs = _load_ref_images(cfg, sample_id)
    crops = ref_imgs
    if seg_cfg.enabled:
        from aero_eyes.models.segmentation import build_segmenter
        segmenter = build_segmenter(seg_cfg, cfg)
        crops = []
        for img in ref_imgs:
            mask = segmenter.segment(img)
            if mask is not None and mask.any():
                ys, xs = np.nonzero(mask)
                x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                px, py = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                img = img[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
            crops.append(img)
    else:
        log.warning(
            "[%s] %s: color_postfilter (classifier) with segmentation.enabled=false -- "
            "classifying the WHOLE reference photos, background included.", log_prefix, sample_id,
        )

    per_ref = clf.predict_proba(crops, batch_size=cpf.classifier_batch_size)
    ref_probs = per_ref.mean(axis=0)
    sig = ClassifierColorSignature(clf, ref_probs, active=float(ref_probs.max()) >= cpf.classifier_min_ref_confidence)
    log.info(
        "[%s] %s: color_postfilter (classifier) reference group = %s (p=%.2f) [%s]; per photo: %s",
        log_prefix, sample_id, sig.ref_group_name, sig.ref_confidence,
        ", ".join(f"{g} {v:.2f}" for g, v in zip(clf.group_names, ref_probs)),
        ", ".join(clf.group_names[i] for i in per_ref.argmax(axis=1)),
    )
    if not sig.active:
        log.warning(
            "[%s] %s: color_postfilter (classifier) reference group is ambiguous (p=%.2f < "
            "classifier_min_ref_confidence=%.2f) -- keeping ALL candidates for this sample.",
            log_prefix, sample_id, sig.ref_confidence, cpf.classifier_min_ref_confidence,
        )
    return sig


def _apply_classifier_postfilter(
    frame_bgr: np.ndarray, boxes: list[Box], color_sig: ClassifierColorSignature, cpf_cfg,
    stats_out: list | None, records: list[dict] | None,
) -> list[Box]:
    """Keep a box iff P(box crop, inset by classifier_inset_ratio, is in the reference's group) >=
    classifier_min_prob (every box when color_sig.active is False). All of
    a keyframe's boxes are classified in one batch. stats_out gets
    (nan, nan, p, nan) per box so it stays column-compatible with the
    histogram path's (sim_hs, sim_v, effective_sim, overexposed_fraction)
    -- see log_color_postfilter_stats."""
    from aero_eyes.utils.geometry import crop_with_pad, inset_box

    if not boxes:
        return []
    crops = [crop_with_pad(frame_bgr, inset_box(b, cpf_cfg.classifier_inset_ratio), pad_ratio=0.0) for b in boxes]
    probs = color_sig.classifier.predict_proba(crops, batch_size=cpf_cfg.classifier_batch_size)
    threshold = cpf_cfg.classifier_min_prob if color_sig.active else 0.0
    names = color_sig.classifier.group_names
    kept: list[Box] = []
    for box, pr in zip(boxes, probs):
        p = float(pr[color_sig.ref_group])
        keep = p >= threshold
        if stats_out is not None:
            stats_out.append((float("nan"), float("nan"), p, float("nan")))
        if records is not None:
            records.append({
                "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                "pass1_score": box.score, "method": "classifier",
                "ref_group": color_sig.ref_group_name, "p_ref_group": p,
                "pred_group": names[int(pr.argmax())], "probs": [round(float(v), 4) for v in pr],
                "min_prob_used": threshold, "kept": keep,
            })
        if keep:
            kept.append(Box(box.x1, box.y1, box.x2, box.y2, score=box.score * p) if cpf_cfg.reweight else box)
    if cpf_cfg.reweight:
        kept.sort(key=lambda b: b.score, reverse=True)
    return kept


def log_color_postfilter_stats(log_prefix: str, sample_id: str, color_stats: list, cpf_cfg) -> None:
    """End-of-run summary of every candidate apply_color_postfilter scored
    (stats_out tuples), for either method -- shared by the GeCo2, GDINO and
    PET-DINO stages."""
    if not color_stats:
        return
    arr = np.array(color_stats, dtype=float)  # sim_hs, sim_v, effective_sim (or p), overexposed_fraction
    if cpf_cfg.method == "classifier":
        log.info(
            "[%s] %s: color_postfilter (classifier) over %d candidates -- p(ref group) "
            "p10/p50/p90=%.3f/%.3f/%.3f, %% below classifier_min_prob=%.2f: %.1f%%",
            log_prefix, sample_id, len(arr), *np.percentile(arr[:, 2], [10, 50, 90]),
            cpf_cfg.classifier_min_prob, 100.0 * float((arr[:, 2] < cpf_cfg.classifier_min_prob).mean()),
        )
        return
    log.info(
        "[%s] %s: color_postfilter similarity stats over %d candidates "
        "(min_similarity=%.2f) -- sim_hs p10/p50/p90=%.3f/%.3f/%.3f, "
        "sim_v p10/p50/p90=%.3f/%.3f/%.3f, effective_sim p10/p50/p90=%.3f/%.3f/%.3f, "
        "%% below min_similarity=%.1f%%, %% overexposed(>=%.0f%% clipped)=%.1f%%",
        log_prefix, sample_id, len(arr), cpf_cfg.min_similarity,
        *np.percentile(arr[:, 0], [10, 50, 90]),
        *np.percentile(arr[:, 1], [10, 50, 90]),
        *np.percentile(arr[:, 2], [10, 50, 90]),
        100.0 * float((arr[:, 2] < cpf_cfg.min_similarity).mean()),
        cpf_cfg.overexposure_ramp_frac * 100.0,
        100.0 * float((arr[:, 3] >= cpf_cfg.overexposure_ramp_frac).mean()),
    )


def build_color_signature(
    cfg, sample_id: str, work_dir: Path, cpf_cfg, seg_cfg,
    cache_name: str = "color_signature.npz", log_prefix: str = "Stage123-GeCo2",
) -> ColorSignature:
    """Color histograms of each reference image's masked object region --
    used by apply_color_postfilter() to catch same-shape-different-color
    false positives a detector's own vision backbone cannot distinguish by
    shape/texture alone (see ColorPostfilterConfig's own docstring).
    Detector-agnostic despite living in stage123_geco2.py (kept here to
    avoid duplicating ~90 lines -- aero_eyes.stages.stage123_gdino imports
    it too, passing its OWN cpf_cfg/seg_cfg/cache_name instead of GeCo2's):
    cpf_cfg/seg_cfg are passed in explicitly rather than read from
    cfg.stage123_geco2.* internally, and cache_name lets two different
    callers on the same work_dir avoid silently sharing a cache built under
    different hue_bins/sat_bins/segmentation settings. Cached independently
    of geco2_prototype.pt (this is pure OpenCV, does not need the GeCo2
    model at all) -- so it's computed even on a cache hit for the prototype
    file, and vice versa; the two caches don't need to be in sync.

    .confidence in [0,1] (see saturation_value_confidence) controls how
    apply_color_postfilter() blends the two histogram sets: 1 = trust
    Hue+Saturation fully (colorful reference object), 0 = trust Value
    fully (near-achromatic reference object, where Hue+Saturation is pure
    noise but Value still reliably tells e.g. black from white).

    Note: if seg_cfg.enabled, this runs its OWN MobileSAM pass over the
    reference images -- independent from (and possibly duplicating)
    whatever segmentation build_exemplar_prototype/run_stage1 already ran,
    since that function may have taken its cached-prototype early-return
    path without computing masks at all this run. Kept decoupled for
    simplicity; MobileSAM (ViT-tiny) is cheap relative to either detector's
    own backbone.

    cpf_cfg.method == "classifier" returns a ClassifierColorSignature
    instead (see _build_classifier_signature); cache_name is unused then.
    """
    if cpf_cfg.method == "classifier":
        return _build_classifier_signature(cfg, sample_id, cpf_cfg, seg_cfg, log_prefix)

    from aero_eyes.utils.color import (
        compute_hs_histogram, compute_is_high_vis, compute_mean_saturation, compute_mean_value,
        compute_value_histogram, histogram_similarity, lit_pixel_mask, saturation_value_confidence,
    )

    cpf = cpf_cfg
    sig_path = work_dir / cache_name
    if cfg.project.use_cache and sig_path.exists():
        data = np.load(sig_path)
        hs_hists = [data[k] for k in sorted(data.files) if k.startswith("hshist_")]
        v_hists = [data[k] for k in sorted(data.files) if k.startswith("vhist_")]
        mean_sat = float(data["mean_saturation"])
        mean_val = float(data["mean_value"])
        is_high_vis = float(data["is_high_vis"]) if "is_high_vis" in data else 0.0
    else:
        ref_imgs = _load_ref_images(cfg, sample_id)
        masks: list[np.ndarray | None] = [None] * len(ref_imgs)
        if seg_cfg.enabled:
            from aero_eyes.models.segmentation import build_segmenter
            segmenter = build_segmenter(seg_cfg, cfg)
            masks = [segmenter.segment(img) for img in ref_imgs]
        else:
            log.warning(
                "[%s] %s: color_postfilter.enabled but segmentation.enabled=false -- "
                "color signature built from the WHOLE reference photo (diluted by background), "
                "not just the object.",
                log_prefix, sample_id,
            )

        # Shadow-pixel exclusion (SS4 point 2) -- see ColorPostfilterConfig.
        # shadow_filter_enabled's own docstring. Applied on the REFERENCE
        # side here, and symmetrically on the CANDIDATE side in
        # apply_color_postfilter -- filtering only one side would introduce
        # a new ref/candidate asymmetry instead of fixing one. Falls back
        # to the unfiltered mask if too few pixels survive.
        hue_masks = masks
        if cpf.shadow_filter_enabled:
            hue_masks = []
            for img, mask in zip(ref_imgs, masks):
                lit = lit_pixel_mask(img, mask, cpf.shadow_min_saturation, cpf.shadow_min_value)
                hue_masks.append(lit if lit.sum() >= 10 else mask)

        hs_hists = [
            compute_hs_histogram(img, mask, cpf.hue_bins, cpf.sat_bins, cpf.hue_smoothing_sigma)
            for img, mask in zip(ref_imgs, hue_masks)
        ]
        v_hists = [
            compute_value_histogram(img, mask, cpf.value_bins)
            for img, mask in zip(ref_imgs, masks)
        ]
        mean_sat = float(np.mean([compute_mean_saturation(img, mask) for img, mask in zip(ref_imgs, masks)]))
        mean_val = float(np.mean([compute_mean_value(img, mask) for img, mask in zip(ref_imgs, masks)]))
        is_high_vis = float(np.mean([
            compute_is_high_vis(img, mask, cpf.is_high_vis_percentile, cpf.is_high_vis_hue_max, cpf.is_high_vis_min_value)
            for img, mask in zip(ref_imgs, masks)
        ]))
        work_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs = {f"hshist_{i}": h for i, h in enumerate(hs_hists)}
        save_kwargs.update({f"vhist_{i}": h for i, h in enumerate(v_hists)})
        save_kwargs["mean_saturation"] = np.array(mean_sat)
        save_kwargs["mean_value"] = np.array(mean_val)
        save_kwargs["is_high_vis"] = np.array(is_high_vis)
        np.savez(sig_path, **save_kwargs)

    confidence = saturation_value_confidence(
        mean_sat, mean_val,
        cpf.min_ref_saturation, cpf.saturation_full_confidence,
        cpf.min_ref_value, cpf.value_full_confidence,
    )
    log.info("[%s] %s: reference object mean HSV saturation=%.1f, value=%.1f "
             "[0-255 scale] -> color_confidence=%.2f (1=trust Hue+Sat, 0=trust Value only)",
             log_prefix, sample_id, mean_sat, mean_val, confidence)
    if confidence < 1.0:
        log.warning(
            "[%s] %s: color_postfilter blending %.0f%% Value-based comparison "
            "in (and %.0f%% Hue+Saturation) -- reference object's color (saturation=%.1f, "
            "value=%.1f) is not fully trustworthy for Hue-based comparison alone "
            "(near-achromatic objects give unstable Hue; Value still separates e.g. black "
            "from white).",
            log_prefix, sample_id, (1 - confidence) * 100, confidence * 100, mean_sat, mean_val,
        )

    # Cross-photo consistency (SS8.3) -- see ColorPostfilterConfig's
    # overexposure_ramp_frac-neighboring docstring for the full rationale.
    # Mean pairwise Hue+Saturation similarity among the ref photos
    # themselves; 1.0 (trivially "fully agree") with fewer than 2 refs.
    if len(hs_hists) >= 2:
        pair_sims = [
            histogram_similarity(hs_hists[i], hs_hists[j], cpf.metric)
            for i in range(len(hs_hists)) for j in range(i + 1, len(hs_hists))
        ]
        ref_agreement = float(np.mean(pair_sims))
    else:
        ref_agreement = 1.0
    log.info(
        "[%s] %s: ref_color_agreement=%.2f (mean pairwise Hue+Sat similarity among the "
        "%d reference photos themselves -- low means the refs don't agree on color, so no "
        "candidate comparison against them can be fully trusted)",
        log_prefix, sample_id, ref_agreement, len(hs_hists),
    )
    if cpf.is_high_vis_gating_enabled:
        log.info(
            "[%s] %s: reference is_high_vis=%.2f (SS3.2/SS8.1 safety orange/yellow signal, "
            "mean across ref photos -- 0=not a safety color, 1=strongly is)",
            log_prefix, sample_id, is_high_vis,
        )
    return ColorSignature(hs_hists, v_hists, confidence, ref_agreement, is_high_vis)


def apply_color_postfilter(
    frame_bgr: np.ndarray, boxes: list[Box], color_sig: ColorSignature, cpf_cfg,
    stats_out: list[tuple[float, float, float, float]] | None = None, segmenter=None,
    records: list[dict] | None = None,
) -> list[Box]:
    """Drop/downweight candidate boxes whose color doesn't match the
    reference object's own color signature (best-of-N-refs match per
    signal, so legitimate lighting/angle variation across the 3 reference
    photos isn't penalized). Blends Hue+Saturation similarity and Value
    similarity by color_sig.confidence -- see ColorSignature /
    build_color_signature and ColorPostfilterConfig for why both signals
    exist and how they're weighted (Hue+Sat alone cannot tell e.g. black
    from white; Value alone is more lighting-sensitive).

    Beyond that base comparison, THREE independent signals from
    docs/attribute_taxonomy_plan.md's color investigation (SS4/SS8/SS9)
    further adjust the accept/reject decision -- one is EVIDENCE (acts on
    the SCORE, effective_sim), two are CONFIDENCE (act on the THRESHOLD,
    NOT the score -- see the BUG note further down for why that distinction
    matters):
      1. [evidence, multiplies effective_sim] is_high_vis agreement
         (SS3.2/SS4 point 4/SS8.1, opt-in via cpf_cfg.
         is_high_vis_gating_enabled) -- a ref that IS a safety-colored
         object (LifeJacket/Lifering) but whose candidate ISN'T (or vice
         versa) is real information the Hue+Sat/Value histograms alone
         might miss. A perfect match (agreement=1) leaves effective_sim
         unchanged; a total mismatch (agreement=0) zeroes it. See
         compute_is_high_vis's own docstring.
      2. [confidence, scales down the ACCEPTANCE THRESHOLD] Overexposure
         (SS9.10/SS9.11) -- a clipped color channel reads as a WRONG hue,
         not just a noisier one. See compute_overexposed_fraction/
         hue_confidence_from_overexposure and overexposure_ramp_frac.
      3. [confidence, scales down the ACCEPTANCE THRESHOLD] Cross-photo
         disagreement (SS8.3) -- color_sig.ref_agreement; if the 3 ref
         photos don't even agree on their OWN color, no candidate
         comparison against them can be trusted regardless of how well it
         matches any single one.
    effective_min_similarity = cpf_cfg.min_similarity * hue_conf *
    color_sig.ref_agreement -- low confidence in EITHER signal lowers how
    much similarity a candidate needs to clear, instead of raising the
    candidate's own score. A candidate is kept iff effective_sim >=
    effective_min_similarity.

    BUG FIXED (confirmed on this project's own real footage, not just
    theoretical): an EARLIER version of this function blended the two
    confidence gates INTO effective_sim instead
    (`effective_sim = conf*effective_sim + (1-conf)*1.0`), which creates a
    HARD FLOOR of (1-conf) on every single candidate's score -- e.g.
    ref_agreement=0.57 floored effective_sim at 0.43 UNCONDITIONALLY,
    which exceeds a stricter min_similarity (e.g. the default 0.30) and
    makes min_similarity unable to reject ANYTHING at all, no matter how
    badly a candidate's actual color matches. Observed directly: a plainly
    wrong-colored candidate (a white box, against a reference whose own 3
    photos only agreed with each other at ref_agreement=0.57) still scored
    0.868, and 0 of 1109 real candidates in that run fell below
    min_similarity=0.30. Scaling the threshold down instead of the score
    up preserves the same "don't false-reject when the comparison itself
    is unreliable" intent WITHOUT creating a floor that silently overrides
    a user's own min_similarity, and WITHOUT collapsing every candidate's
    score toward 1.0 (which also destroyed the score's own discriminative
    value for `reweight` and for reading effective_sim back out of
    stats_out/records).

    Also applies shadow-pixel exclusion (SS4 point 2, opt-in via
    cpf_cfg.shadow_filter_enabled -- see lit_pixel_mask's own docstring)
    and, when `segmenter` is given and cpf_cfg.candidate_segmentation_
    enabled, DENSE full-frame-context segmentation for each candidate crop
    (SS9.1/SS9.2/SS9.8 -- background contamination was the single largest
    measured color error source; candidate_inset_ratio's inward shrink is
    a much cruder proxy for the same problem) instead of the plain
    candidate_inset_ratio crop, falling back to it whenever segmentation
    is unavailable or returns too small a mask for this box.

    stats_out: if given, appends (sim_hs, sim_v, effective_sim,
    overexposed_fraction) for EVERY candidate evaluated (before the
    min_similarity cutoff) -- lets a caller collect the REAL distribution
    of similarity scores seen on actual video frames, since a threshold
    picked from a synthetic/clean test image (as this codebase already
    learned the hard way once, with score_threshold_abs) may not reflect
    what real footage produces. See run_stage123_geco2's end-of-run
    summary log. Bare tuples (not per-box dicts) -- kept as-is for
    backward compatibility with existing summary-stat consumers; use
    `records` below for per-box debugging/viz instead of parallel-parsing
    this.

    records: if given (a list the caller owns), appends one dict per INPUT
    box -- {x1,y1,x2,y2,pass1_score,sim_hs,sim_v,effective_sim,
    overexposed_fraction,min_similarity_used,kept} -- for EVERY box, not
    just survivors, same "log/viz everything, not just the decision"
    spirit as aero_eyes.stages.stage123_gdino.cascade_verify_boxes's own
    `records` argument. `pass1_score` is the box's own detector score
    (score*effective_sim if cpf_cfg.reweight, else unchanged) BEFORE this
    filter. `min_similarity_used` is the ACTUAL (confidence-scaled)
    threshold this candidate was compared against -- see the BUG note
    above for why this can differ from cpf_cfg.min_similarity itself, and
    why that distinction is worth logging per-candidate for debugging.
    Lets a caller feed this into aero_eyes.utils.viz.save_color_postfilter
    or a JSONL log the same way cascade_verification.jsonl already works.

    segmenter: an already-constructed MobileSAM/FastSAM/SAM2Segmenter (see
    aero_eyes.models.segmentation.build_segmenter), or None (default) to
    skip dense segmentation entirely and use candidate_inset_ratio as
    before -- even when cpf_cfg.candidate_segmentation_enabled is true,
    passing no segmenter here is a no-op fallback, not an error. Calls
    segmenter.set_frame(frame_bgr) ONCE per call (this function is already
    called once per keyframe), reusing it via segment_box_cached() for
    every box -- same "encode once, reuse per box" pattern box_refine's
    own dense methods use.

    cpf_cfg.method == "classifier" (color_sig is then a
    ClassifierColorSignature) dispatches to _apply_classifier_postfilter;
    `segmenter` is unused there.
    """
    if cpf_cfg.method == "classifier":
        return _apply_classifier_postfilter(frame_bgr, boxes, color_sig, cpf_cfg, stats_out, records)

    from aero_eyes.utils.color import (
        compute_hs_histogram, compute_is_high_vis, compute_overexposed_fraction, compute_value_histogram,
        histogram_similarity, hue_confidence_from_overexposure, lit_pixel_mask,
    )
    from aero_eyes.utils.geometry import crop_with_pad, inset_box

    conf = color_sig.confidence
    h, w = frame_bgr.shape[:2]

    dense_ready = False
    if segmenter is not None and cpf_cfg.candidate_segmentation_enabled:
        dense_ready = segmenter.set_frame(frame_bgr)
        if not dense_ready:
            log.debug(
                "color_postfilter: dense candidate segmentation unavailable for this frame "
                "-- falling back to candidate_inset_ratio for every box this keyframe."
            )

    kept: list[Box] = []
    for box in boxes:
        # Sample color from an INSET box (see ColorPostfilterConfig.
        # candidate_inset_ratio) -- the box KEPT below is still the
        # original, unshrunk one; the inset only narrows what pixels the
        # color histogram is measured from.
        color_box = inset_box(box, cpf_cfg.candidate_inset_ratio)
        crop = crop_with_pad(frame_bgr, color_box, pad_ratio=0.0)

        mask = None
        if dense_ready:
            full_mask = segmenter.segment_box_cached(box, margin=cpf_cfg.candidate_segmentation_margin)
            if full_mask is not None:
                x1 = max(0, int(color_box.x1)); y1 = max(0, int(color_box.y1))
                x2 = min(w, int(color_box.x2)); y2 = min(h, int(color_box.y2))
                cand_mask = full_mask[y1:y2, x1:x2]
                if cand_mask.shape == crop.shape[:2] and cand_mask.sum() >= cpf_cfg.candidate_segmentation_min_mask_px:
                    mask = cand_mask

        hue_mask = mask
        if cpf_cfg.shadow_filter_enabled:
            lit = lit_pixel_mask(crop, mask, cpf_cfg.shadow_min_saturation, cpf_cfg.shadow_min_value)
            hue_mask = lit if lit.sum() >= 10 else mask

        hs_hist = compute_hs_histogram(crop, hue_mask, cpf_cfg.hue_bins, cpf_cfg.sat_bins, cpf_cfg.hue_smoothing_sigma)
        v_hist = compute_value_histogram(crop, mask, cpf_cfg.value_bins)
        sim_hs = max(histogram_similarity(hs_hist, r, cpf_cfg.metric) for r in color_sig.hs_hists)
        sim_v = max(histogram_similarity(v_hist, r, cpf_cfg.metric) for r in color_sig.v_hists)
        effective_sim = conf * sim_hs + (1.0 - conf) * sim_v

        # is_high_vis agreement (SS3.2/SS4 point 4/SS8.1) -- EVIDENCE:
        # a genuine mismatch here (ref IS a safety color, candidate ISN'T,
        # or vice versa) is real information that can REJECT a candidate.
        # Multiplicative -- a perfect match (agreement=1) leaves
        # effective_sim unchanged; a total mismatch (agreement=0) zeroes
        # it. Safe to apply directly to the score (unlike the two
        # CONFIDENCE gates below): it can only ever push effective_sim
        # DOWN, never create a floor that overrides min_similarity.
        if cpf_cfg.is_high_vis_gating_enabled:
            cand_high_vis = compute_is_high_vis(
                crop, mask, cpf_cfg.is_high_vis_percentile, cpf_cfg.is_high_vis_hue_max, cpf_cfg.is_high_vis_min_value,
            )
            hivis_agreement = 1.0 - abs(color_sig.is_high_vis - cand_high_vis)
            effective_sim = effective_sim * hivis_agreement

        # Overexposure (SS9.10/SS9.11) and cross-photo disagreement (SS8.3)
        # are CONFIDENCE gates, not evidence -- they say "don't trust this
        # comparison," not "this candidate matches." EMPIRICALLY CONFIRMED
        # BUG in an earlier version of this function: blending them INTO
        # effective_sim (`effective_sim = conf*effective_sim + (1-conf)`)
        # creates a HARD FLOOR of (1-conf) on every candidate's score --
        # e.g. ref_agreement=0.57 floors effective_sim at 0.43 for EVERY
        # candidate regardless of how badly it actually matches, which
        # exceeds a stricter min_similarity (e.g. 0.30) and makes
        # min_similarity unable to reject ANYTHING at all (observed
        # directly: a plainly wrong-colored candidate -- white box vs. a
        # near-achromatic-but-only-57%-self-consistent reference -- still
        # scored 0.868, and 0% of 1109 real candidates fell below
        # min_similarity=0.30 that run). Fixed by scaling the THRESHOLD
        # down instead of the score up: low confidence lowers the bar a
        # candidate needs to clear (fewer false rejects when the
        # comparison itself is unreliable, same intent as before) WITHOUT
        # creating a floor that silently overrides min_similarity, and
        # WITHOUT collapsing every candidate's score toward 1.0
        # (destroying the score's own discriminative value for `reweight`
        # or for reading these numbers back out of stats_out/records).
        overexposed_frac = compute_overexposed_fraction(crop, mask, cpf_cfg.overexposure_clip_threshold)
        hue_conf = hue_confidence_from_overexposure(overexposed_frac, cpf_cfg.overexposure_ramp_frac)
        effective_min_similarity = cpf_cfg.min_similarity * hue_conf * color_sig.ref_agreement

        if stats_out is not None:
            stats_out.append((sim_hs, sim_v, effective_sim, overexposed_frac))
        keep = effective_sim >= effective_min_similarity
        if records is not None:
            records.append({
                "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                "pass1_score": box.score, "sim_hs": sim_hs, "sim_v": sim_v,
                "effective_sim": effective_sim, "overexposed_fraction": overexposed_frac,
                "min_similarity_used": effective_min_similarity,
                "kept": keep,
            })
        if not keep:
            continue
        kept.append(Box(box.x1, box.y1, box.x2, box.y2, score=box.score * effective_sim) if cpf_cfg.reweight else box)
    if cpf_cfg.reweight:
        kept.sort(key=lambda b: b.score, reverse=True)
    return kept


class Geco2OnlineAdaptiveThreshold:
    """stage123_geco2.online_adaptive_threshold (see
    Geco2OnlineAdaptiveThresholdConfig): stage3.OnlineAdaptiveThreshold's
    contract on GeCo2's own score. threshold_for_next_frame() uses only the
    scores observe()d at STRICTLY EARLIER keyframes; observe() is called with
    a keyframe's candidate scores AFTER its own decision. The statistic itself
    is stage3.compute_adaptive_threshold (z_score/otsu/gmm), fed this
    detector's window through an adapter with stage3's field names."""

    def __init__(self, oat_cfg):
        from collections import deque
        from types import SimpleNamespace

        self.cfg = oat_cfg
        self.history: deque = deque(maxlen=oat_cfg.window)
        self._s3_view = SimpleNamespace(
            adaptive_threshold_anchor_to_original_refs=False,
            adaptive_threshold_method=oat_cfg.method,
            adaptive_threshold_min_samples=oat_cfg.min_samples,
            adaptive_threshold_robust=oat_cfg.robust,
            adaptive_z_score=oat_cfg.z_score,
            adaptive_min_floor=oat_cfg.abs_floor,
            adaptive_otsu_bins=oat_cfg.otsu_bins,
            adaptive_gmm_min_separation_std=oat_cfg.gmm_min_separation_std,
            adaptive_gmm_fallback_percentile=oat_cfg.gmm_fallback_percentile,
        )

    def threshold_for_next_frame(self) -> tuple[float, str]:
        if len(self.history) < self.cfg.min_samples:
            return self.cfg.abs_floor, "online_cold_start"
        from aero_eyes.stages.stage3 import compute_adaptive_threshold

        scores = np.array(self.history, dtype=np.float64)
        # "cosine" only selects compute_adaptive_threshold's floor branch:
        # max(adaptive_min_floor (= abs_floor), raw threshold).
        threshold, _, _, stat_label = compute_adaptive_threshold(scores, scores, "cosine", self._s3_view)
        return threshold, f"online_{stat_label}"

    def observe(self, frame_scores) -> None:
        self.history.extend(np.atleast_1d(frame_scores).tolist())


def _detect_keyframe(detector, frame_bgr, prototype, online, online_stats: dict | None) -> list[Box]:
    """detector.detect_frame, or -- when stage123_geco2.online_adaptive_threshold
    is on -- decide against the running window, then feed it this keyframe's
    candidate scores."""
    if online is None:
        return detector.detect_frame(frame_bgr, prototype)
    threshold, label = online.threshold_for_next_frame()
    boxes, observed = detector.detect_frame_online(frame_bgr, prototype, threshold, online.cfg.combine_with_ratio)
    online.observe(observed)
    if online_stats is not None:
        online_stats["frames"] = online_stats.get("frames", 0) + 1
        online_stats["cold_start"] = online_stats.get("cold_start", 0) + (label == "online_cold_start")
        online_stats["empty"] = online_stats.get("empty", 0) + (len(observed) > 0 and not boxes)
        online_stats["last"] = (threshold, label)
    return boxes


def _log_online_threshold_summary(sample_id: str, oat_cfg, stats: dict) -> None:
    if not stats:
        return
    thr, label = stats.get("last", (float("nan"), "n/a"))
    log.info(
        "[Stage123-GeCo2] %s: online_adaptive_threshold (method=%s, window=%d, min_samples=%d, "
        "combine_with_ratio=%s) -- %d keyframe(s), %d in cold start, %d left EMPTY by the threshold; "
        "final threshold=%.4f (%s)",
        sample_id, oat_cfg.method, oat_cfg.window, oat_cfg.min_samples, oat_cfg.combine_with_ratio,
        stats.get("frames", 0), stats.get("cold_start", 0), stats.get("empty", 0), thr, label,
    )


def _check_fusion_cfg(fusion_cfg, nms_iou: float, sample_id: str, log_prefix: str, rescored: bool) -> None:
    """Warn about stage123_geco2.candidate_fusion settings that can't do
    anything. rescored: True on the cosine_rescore path (Stage 3's cosine
    re-ranks the boxes afterwards), False on the default path."""
    if fusion_cfg is None or not fusion_cfg.enabled:
        return
    if fusion_cfg.mode == "wbf" and fusion_cfg.iou_thresh >= nms_iou:
        log.warning(
            "[%s] %s: candidate_fusion mode=wbf with iou_thresh=%.2f >= nms_iou=%.2f -- "
            "boxes have already been NMS'd at nms_iou, so no pair can exceed iou_thresh and "
            "nothing will be fused. Lower iou_thresh (or raise stage123_geco2.nms_iou).",
            log_prefix, sample_id, fusion_cfg.iou_thresh, nms_iou,
        )
    if not rescored and fusion_cfg.keep_originals:
        log.warning(
            "[%s] %s: candidate_fusion.keep_originals=true without cosine_rescore -- the fused "
            "box only ties its best member's score and nothing re-ranks them, so Stage 4 keeps "
            "picking the original member. Set candidate_fusion.keep_originals=false here to "
            "replace the parts with the fused box.", log_prefix, sample_id,
        )


def _fuse_keyframe_boxes(boxes: list[Box], fusion_cfg, stats: dict | None) -> list[Box]:
    """stage123_geco2.candidate_fusion for one keyframe (no-op when disabled);
    stats accumulates counts for _log_fusion_summary."""
    from aero_eyes.utils.box_fusion import fuse_overlapping_boxes

    if fusion_cfg is None or not fusion_cfg.enabled:
        return boxes
    out = fuse_overlapping_boxes(boxes, fusion_cfg)
    if stats is not None:
        n_new = sum(1 for b in out if getattr(b, "fused", False))
        stats["boxes_in"] = stats.get("boxes_in", 0) + len(boxes)
        stats["fused"] = stats.get("fused", 0) + n_new
        stats["frames_fused"] = stats.get("frames_fused", 0) + (1 if n_new else 0)
        stats["frames"] = stats.get("frames", 0) + 1
    return out


def _log_fusion_summary(log_prefix: str, sample_id: str, fusion_cfg, stats: dict) -> None:
    if fusion_cfg is None or not fusion_cfg.enabled:
        return
    n_fused = stats.get("fused", 0)
    log.info(
        "[%s] %s: candidate_fusion (mode=%s, keep_originals=%s): %d fused box(es) in %d/%d "
        "keyframe(s), from %d detected box(es)%s",
        log_prefix, sample_id, fusion_cfg.mode, fusion_cfg.keep_originals, n_fused, stats.get("frames_fused", 0),
        stats.get("frames", 0), stats.get("boxes_in", 0),
        "" if n_fused else " -- nothing was linked; see CandidateFusionConfig "
        "(containment_thresh / max_union_area_ratio / iou_thresh)",
    )


def _finalize_keyframe_detections(
    frame_idx: int,
    frame_bgr: np.ndarray,
    boxes: list[Box],
    color_sig,
    cpf_cfg,
    color_stats: list | None,
    viz_dir: Path,
    save_viz: bool,
    fusion_cfg=None,
    fusion_stats: dict | None = None,
) -> list[Detection]:
    """Color postfilter + candidate_fusion + Detection wrapping + viz save --
    the per-keyframe tail shared by both the default (per-frame-relative)
    and global_adaptive_threshold detection paths in run_stage123_geco2, so
    the two paths can't silently drift apart on this shared bookkeeping.
    Same order as the cosine_rescore candidate path: color first, then
    fusion (so a wrong-color part can't widen a fused box)."""
    from aero_eyes.utils import viz as vizmod

    if color_sig is not None:
        boxes = apply_color_postfilter(frame_bgr, boxes, color_sig, cpf_cfg, stats_out=color_stats)
    boxes = _fuse_keyframe_boxes(boxes, fusion_cfg, fusion_stats)
    result_dets = [
        Detection(frame_idx=frame_idx, box=b, similarity=b.score, source="detect")
        for b in boxes
    ]
    if save_viz:
        vizmod.save_stage3_detections(
            frame_bgr, [d.box for d in result_dets], [d.similarity for d in result_dets],
            frame_idx, viz_dir,
        )
    return result_dets


def _run_geco2_default_pass(
    detector, video_path: Path, kf_indices: set, get_prototype, color_sig, cpf_cfg,
    color_stats: list | None, viz_dir: Path, save_viz: bool, on_result=None,
    fusion_cfg=None, fusion_stats: dict | None = None,
    online_threshold: Geco2OnlineAdaptiveThreshold | None = None, online_stats: dict | None = None,
) -> dict[int, list[Detection]]:
    """One full sweep over the video's keyframes, detecting against
    whatever get_prototype() currently returns -- shared by
    run_stage123_geco2's pass 1 (online dynamic_prototype, if enabled) and
    optional pass 2 (dynamic_prototype.second_pass: same sweep again with
    pass 1's final prototype frozen) so the two passes can't silently
    diverge in behavior. on_result(frame_idx, frame_bgr, result_dets), when
    given, runs after each keyframe's detections are finalized -- pass 1
    uses it to feed dyn_proto_tracker.offer(); pass 2 passes None since the
    prototype is frozen by then.
    """
    from aero_eyes.utils.video import frame_iterator

    detections: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue
        boxes = _detect_keyframe(detector, frame_bgr, get_prototype(), online_threshold, online_stats)
        result_dets = _finalize_keyframe_detections(
            frame_idx, frame_bgr, boxes, color_sig, cpf_cfg, color_stats, viz_dir, save_viz,
            fusion_cfg=fusion_cfg, fusion_stats=fusion_stats,
        )
        detections[frame_idx] = result_dets
        log.debug("[Stage123-GeCo2] frame %d: %d detections", frame_idx, len(result_dets))
        if on_result is not None:
            on_result(frame_idx, frame_bgr, result_dets)
    return detections


def run_stage123_geco2(cfg, sample_id: str) -> Path:
    """Run the merged GeCo2 stage for one sample. Returns path to detections.json."""
    from aero_eyes.models.geco2_detector import GeCo2Detector
    from aero_eyes.utils.io import write_detections
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage123-GeCo2] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    detector = GeCo2Detector(cfg)
    prototype = build_exemplar_prototype(cfg, sample_id, detector, work_dir)

    # stage123_geco2.dynamic_prototype (opt-in): online/incremental analog
    # of stage3.dynamic_prototype for GeCo2's own exemplar tokens -- see
    # GeCo2DynamicPrototypeTracker's own docstring. Only wired into the
    # "Default" (non-global_adaptive_threshold) branch below -- that path's
    # own 2-pass score-pooling would need the prototype held FIXED across
    # both passes to stay comparable, which this online update can't
    # guarantee.
    dyn_proto_tracker = None
    if cfg.stage123_geco2.dynamic_prototype.enabled:
        if cfg.stage123_geco2.global_adaptive_threshold.enabled:
            log.warning(
                "[Stage123-GeCo2] %s: dynamic_prototype.enabled=true has no effect while "
                "global_adaptive_threshold.enabled=true (that path's own 2-pass score "
                "pooling needs a fixed prototype across both passes) -- disable one of the two.",
                sample_id,
            )
        else:
            from aero_eyes.models.geco2_detector import GeCo2DynamicPrototypeTracker
            dyn_proto_tracker = GeCo2DynamicPrototypeTracker(cfg, detector, prototype, work_dir, sample_id)

    cpf_cfg = cfg.stage123_geco2.color_postfilter
    color_sig = (
        build_color_signature(cfg, sample_id, work_dir, cpf_cfg, cfg.stage123_geco2.segmentation)
        if cpf_cfg.enabled else None
    )

    # ---- Locate video ----
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(
            f"No video matching '{cfg.data.video_glob}' found in {video_dir}."
        )
    video_path = video_files[0]
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[Stage123-GeCo2] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    kf_indices = set(keyframe_indices(total_frames, cfg.stage123_geco2.keyframe_interval))
    viz_dir = work_dir / "viz" / "stage123_geco2"
    save_viz = cfg.runtime.save_visualizations

    color_stats: list[tuple[float, float, float]] | None = [] if color_sig is not None else None

    fusion_cfg = cfg.stage123_geco2.candidate_fusion
    _check_fusion_cfg(fusion_cfg, detector.nms_iou, sample_id, "Stage123-GeCo2", rescored=False)
    fusion_stats: dict = {}

    gat_cfg = cfg.stage123_geco2.global_adaptive_threshold
    detections: dict[int, list[Detection]] = {}
    effective_threshold: float | None = None

    if gat_cfg.enabled:
        # ---- Pass 1: pool RAW (unfiltered) scores across every keyframe ----
        # so Pass 2 can decide what counts as a real detection from the
        # WHOLE video's score distribution instead of each frame's own max
        # (which structurally always keeps >=1 box -- see
        # GlobalAdaptiveThresholdConfig docstring). Raw tensors are moved to
        # CPU and cached per frame so Pass 2 does not re-run the backbone.
        per_frame_raw: dict[int, tuple] = {}
        raw_score_chunks: list[np.ndarray] = []
        for frame_idx, frame_bgr in frame_iterator(video_path):
            if frame_idx not in kf_indices:
                continue
            pred_boxes, box_v, scale = detector.forward_scores(frame_bgr, prototype)
            per_frame_raw[frame_idx] = (pred_boxes.cpu(), box_v.cpu(), scale)
            if box_v.numel() > 0:
                raw_score_chunks.append(box_v.cpu().numpy())

        if raw_score_chunks:
            all_scores = np.concatenate(raw_score_chunks)
            sim_mean = float(all_scores.mean())
            sim_std = float(all_scores.std())
            raw_threshold = sim_mean + gat_cfg.z_score * sim_std
            effective_threshold = max(gat_cfg.abs_floor, raw_threshold)
            sim_max = float(all_scores.max())
            if effective_threshold > sim_max:
                log.info(
                    "[Stage123-GeCo2] %s: global adaptive threshold %.3f exceeds max score %.3f "
                    "-- capping at max so the best candidate isn't dropped.",
                    sample_id, effective_threshold, sim_max,
                )
                effective_threshold = sim_max
            log.info(
                "[Stage123-GeCo2] %s: global adaptive threshold = %.3f (mean=%.3f std=%.3f "
                "z=%.2f, n=%d candidates over %d keyframes)",
                sample_id, effective_threshold, sim_mean, sim_std, gat_cfg.z_score,
                len(all_scores), len(per_frame_raw),
            )
        else:
            effective_threshold = gat_cfg.abs_floor
            log.warning(
                "[Stage123-GeCo2] %s: no raw candidates collected -- defaulting global "
                "adaptive threshold to abs_floor=%.3f", sample_id, effective_threshold,
            )

        # ---- Pass 2: apply the global threshold, then per-frame NMS/top-K + ----
        # color postfilter + viz, reusing each frame's cached raw tensors.
        for frame_idx, frame_bgr in frame_iterator(video_path):
            if frame_idx not in kf_indices:
                continue
            pred_boxes, box_v, scale = per_frame_raw[frame_idx]
            boxes = detector.filter_boxes_by_threshold(
                pred_boxes.to(detector.device), box_v.to(detector.device), scale,
                frame_bgr, effective_threshold,
            )
            result_dets = _finalize_keyframe_detections(
                frame_idx, frame_bgr, boxes, color_sig, cpf_cfg, color_stats, viz_dir, save_viz,
                fusion_cfg=fusion_cfg, fusion_stats=fusion_stats,
            )
            detections[frame_idx] = result_dets
            log.debug("[Stage123-GeCo2] frame %d: %d detections", frame_idx, len(result_dets))
    else:
        # ---- Default: GeCo2's own per-frame-relative decision, unchanged
        # (+ optional dynamic_prototype online update -- see tracker built
        # above). ----
        def _offer_best(frame_idx, frame_bgr, result_dets):
            if dyn_proto_tracker is not None and result_dets:
                # Highest-scoring surviving box this keyframe -- the SAME
                # one _finalize_keyframe_detections already picked as the
                # frame's best (see its own NMS/top-K ordering).
                dyn_proto_tracker.offer(frame_bgr, result_dets[0].box, frame_idx=frame_idx)

        # stage123_geco2.online_adaptive_threshold (opt-in): causal running
        # window on GeCo2's own score -- a fresh window per pass.
        oat_cfg = cfg.stage123_geco2.online_adaptive_threshold
        online_stats: dict = {}

        def _new_online():
            return Geco2OnlineAdaptiveThreshold(oat_cfg) if oat_cfg.enabled else None

        get_prototype = dyn_proto_tracker.effective_prototype if dyn_proto_tracker is not None else (lambda: prototype)
        detections = _run_geco2_default_pass(
            detector, video_path, kf_indices, get_prototype, color_sig, cpf_cfg, color_stats, viz_dir, save_viz,
            on_result=_offer_best, fusion_cfg=fusion_cfg, fusion_stats=fusion_stats,
            online_threshold=_new_online(), online_stats=online_stats,
        )
        if dyn_proto_tracker is not None:
            dyn_proto_tracker.log_summary()

        # dynamic_prototype.second_pass (opt-in): pass 1 above only ever
        # saw a GROWING prototype, so early frames judged against just the
        # 3 original refs may have missed a target whose on-video look
        # differs from them -- re-run the whole video once more with pass
        # 1's FINAL (now frozen) prototype available from frame 1, and use
        # THIS pass's detections/viz as the answer instead. No-op if pass 1
        # accepted no dynamic tokens (would be identical to pass 1).
        dp_cfg = cfg.stage123_geco2.dynamic_prototype
        if dyn_proto_tracker is not None and dp_cfg.second_pass:
            n_dynamic = dyn_proto_tracker.dynamic_token_count()
            if n_dynamic == 0:
                log.info(
                    "[Stage123-GeCo2] %s: dynamic_prototype.second_pass=true but pass 1 "
                    "accepted no dynamic tokens -- skipping (pass 2 would be identical).",
                    sample_id,
                )
            else:
                log.info(
                    "[Stage123-GeCo2] %s: dynamic_prototype.second_pass -- re-running the "
                    "whole video with pass 1's final prototype (%d dynamic token(s)) frozen "
                    "from frame 1, replacing pass 1's detections%s.",
                    sample_id, n_dynamic, " and viz" if save_viz else "",
                )
                frozen_prototype = dyn_proto_tracker.effective_prototype()
                color_stats_pass2 = [] if color_sig is not None else None
                fusion_stats = {}
                online_stats = {}
                detections = _run_geco2_default_pass(
                    detector, video_path, kf_indices, lambda: frozen_prototype,
                    color_sig, cpf_cfg, color_stats_pass2, viz_dir, save_viz,
                    fusion_cfg=fusion_cfg, fusion_stats=fusion_stats,
                    online_threshold=_new_online(), online_stats=online_stats,
                )
                color_stats = color_stats_pass2
        if oat_cfg.enabled:
            _log_online_threshold_summary(sample_id, oat_cfg, online_stats)

    if color_stats:
        log_color_postfilter_stats("Stage123-GeCo2", sample_id, color_stats, cpf_cfg)
    _log_fusion_summary("Stage123-GeCo2", sample_id, fusion_cfg, fusion_stats)

    # No single global threshold applies when global_adaptive_threshold is
    # disabled (GeCo2 thresholds relative to each frame's own max score) --
    # Stage 4's geco2-aware re-detect path reads
    # stage123_geco2.score_threshold_ratio directly instead of this field
    # either way, so recording effective_threshold here is informational
    # only (for inspecting detections.json), not consumed downstream.
    idf_cfg = cfg.stage123_geco2.isolated_detection_filter
    if idf_cfg.enabled:
        from aero_eyes.stages.stage3 import find_isolated_keyframes
        isolated = find_isolated_keyframes(
            {fi: max(d.similarity for d in dets) for fi, dets in detections.items() if dets},
            cfg.stage123_geco2.keyframe_interval, idf_cfg,
        )
        for fi in isolated:
            detections[fi] = []
        if isolated:
            log.info(
                "[Stage123-GeCo2] %s: isolated_detection_filter (max_gap=%d x %d frames, "
                "keep_conf_threshold=%s) dropped %d isolated keyframe(s): %s",
                sample_id, idf_cfg.max_gap_intervals, cfg.stage123_geco2.keyframe_interval,
                idf_cfg.keep_conf_threshold, len(isolated), sorted(isolated),
            )

    write_detections(detections, det_path, threshold=effective_threshold)
    detector.log_peak_contrast_summary(sample_id)

    elapsed = time.time() - t0
    log.info("[Stage123-GeCo2] %s done in %.1fs -> %s (%d detection frames)",
              sample_id, elapsed, det_path, len(detections))
    return det_path


def _run_geco2_candidate_pass(
    detector, extractor, video_path: Path, kf_indices: set, get_prototype, color_sig, cpf_cfg, cfg,
    on_result=None, viz_dir: Path | None = None, fusion_cfg=None, encode: bool = True,
    fusion_stats: dict | None = None,
) -> dict[int, list[Detection]]:
    """One full sweep over the video's keyframes building candidates.json
    entries -- shared by run_stage12_geco2_candidates's pass 1 (online
    dynamic_prototype, if enabled) and optional pass 2
    (dynamic_prototype.second_pass), same rationale as
    _run_geco2_default_pass. on_result(frame_idx, frame_bgr, boxes, feats),
    when given, runs after each keyframe's candidates are built -- pass 1
    uses it to feed dyn_proto_tracker.offer(); pass 2 passes None.

    encode=False (cosine_rescore.skip_candidate_encoding) skips the per-crop
    embedding: extractor may then be None and every candidate carries a 1-d zero
    placeholder feature (stage3.recompute_candidate_features replaces it).

    fusion_cfg (a CandidateFusionConfig, stage123_geco2.candidate_fusion),
    when given and enabled, fuses each keyframe's overlapping boxes before
    their features are extracted (see aero_eyes.utils.box_fusion); counts go
    into fusion_stats for _log_fusion_summary.

    viz_dir, when given, saves every keyframe that has at least one candidate
    (after the color postfilter) as viz_dir/frame_XXXXXX.jpg with each box
    labelled by GeCo2's own score -- the RAW candidates, before Stage 3's
    cosine threshold/NMS/top-K (Stage 3 draws its own, filtered frames).
    """
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.video import frame_iterator

    candidates: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue

        boxes = detector.detect_frame(frame_bgr, get_prototype())
        if color_sig is not None:
            boxes = apply_color_postfilter(frame_bgr, boxes, color_sig, cpf_cfg)
        boxes = _fuse_keyframe_boxes(boxes, fusion_cfg, fusion_stats)

        if not encode:
            feats = np.zeros((len(boxes), 1), dtype=np.float32)
        elif boxes:
            feats = extractor.extract_crops(
                frame_bgr, boxes,
                pad_ratio=cfg.stage2.candidate.feature_crop_pad,
                batch_size=cfg.runtime.batch_size,
            )
        else:
            feats = np.zeros((0, extractor._feature_dim()), dtype=np.float32)

        frame_dets: list[Detection] = []
        for i, box in enumerate(boxes):
            d = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="candidate")
            d._feature = feats[i]  # type: ignore[attr-defined]
            frame_dets.append(d)
        candidates[frame_idx] = frame_dets
        log.debug("[Stage12-GeCo2] frame %d: %d candidates", frame_idx, len(frame_dets))
        if viz_dir is not None and boxes:
            vizmod.save_stage2_keyframe(frame_bgr, boxes, None, frame_idx, viz_dir)
        if on_result is not None:
            on_result(frame_idx, frame_bgr, boxes, feats)
    return candidates


def validate_skip_candidate_encoding(cfg) -> None:
    """cosine_rescore.skip_candidate_encoding needs stage3 to embed the crops
    itself and nothing in this stage to consume the embeddings."""
    if not cfg.stage123_geco2.cosine_rescore.skip_candidate_encoding:
        return
    if not cfg.stage3.recompute_candidate_features:
        raise ValueError(
            "stage123_geco2.cosine_rescore.skip_candidate_encoding=true needs "
            "stage3.recompute_candidate_features=true -- without it Stage 3 would score placeholder "
            "features. Enable recompute or turn skip_candidate_encoding off."
        )
    dp = cfg.stage123_geco2.dynamic_prototype
    if dp.enabled and dp.cross_check_source == "feature_extractor":
        raise ValueError(
            "stage123_geco2.cosine_rescore.skip_candidate_encoding=true is incompatible with "
            "stage123_geco2.dynamic_prototype.cross_check_source='feature_extractor' (that mode scores "
            "each keyframe's candidates with the embeddings this option skips). Use cross_check_source="
            "'hiera' or turn skip_candidate_encoding off."
        )


def run_stage12_geco2_candidates(cfg, sample_id: str) -> Path:
    """Stage 1+2 replacement (cosine_rescore variant) — GeCo2 exemplar
    detection as a CANDIDATE generator instead of the final word.

    Used instead of run_stage123_geco2 when
    stage123_geco2.cosine_rescore.enabled=true. Differs from
    run_stage123_geco2 in exactly one way: GeCo2's own
    score_threshold_ratio/topk_per_keyframe are replaced with the looser
    cosine_rescore.candidate_* values (so real detections aren't filtered
    out before Stage 3 gets to see them), each surviving candidate crop is
    embedded with a SEPARATE DINOv2 prototype (built via stage1.run_stage1
    from the same reference images -- an independent signal from a
    different backbone than GeCo2's own Hiera), and the result is written
    to candidates.json (+ .feats.npz) in the same schema Stage 2 writes,
    instead of straight to detections.json. aero_eyes.stages.stage3.run_stage3
    then does the actual threshold/NMS/top-K filtering that produces
    detections.json for Stage 4/5.

    Reads:  cfg.data reference images + video
    Writes: <work_dir>/<sample_id>/geco2_prototype.pt (cached GeCo2 exemplar tokens)
            <work_dir>/<sample_id>/prototype.npz (cached DINOv2 prototype, via run_stage1)
            <work_dir>/<sample_id>/candidates.json (+ .feats.npz)
            <work_dir>/<sample_id>/viz/stage123_geco2/candidates/frame_XXXXXX.jpg
              (raw candidates + GeCo2 score per box, when runtime.save_visualizations=true)
    """
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.geco2_detector import GeCo2Detector
    from aero_eyes.stages.stage1 import run_stage1
    from aero_eyes.stages.stage2 import _write_candidates_with_features
    from aero_eyes.utils.video import keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    cand_path = work_dir / "candidates.json"
    if cfg.project.use_cache and cand_path.exists():
        log.info("[Stage12-GeCo2] %s: using cached candidates at %s", sample_id, cand_path)
        return cand_path

    validate_skip_candidate_encoding(cfg)    # fail fast, before any model is loaded

    # DINOv2 prototype for Stage 3's cosine matching -- independent of (and
    # cached separately from) GeCo2's own exemplar tokens below.
    run_stage1(cfg, sample_id)

    detector = GeCo2Detector(cfg)
    prototype = build_exemplar_prototype(cfg, sample_id, detector, work_dir)

    # Loosen GeCo2's own cut so real detections survive through to Stage 3's
    # cosine matching -- see Geco2CosineRescoreConfig docstring.
    cr = cfg.stage123_geco2.cosine_rescore
    detector.score_threshold_ratio = cr.candidate_score_threshold_ratio
    detector.topk_per_keyframe = cr.candidate_topk_per_keyframe
    # Per-box absolute floor (applied before the ratio); 0 + ratio 0 = no score
    # filtering. The generic stage123_geco2.score_threshold_abs (a floor on the
    # frame's best score only) is superseded here.
    if detector.score_threshold_abs > 0:
        log.warning(
            "[Stage12-GeCo2] %s: stage123_geco2.score_threshold_abs=%.3f is ignored in cosine_rescore "
            "candidate mode -- use stage123_geco2.cosine_rescore.candidate_score_threshold_abs instead.",
            sample_id, detector.score_threshold_abs,
        )
    detector.score_threshold_abs = 0.0
    detector.score_threshold_box_abs = cr.candidate_score_threshold_abs
    fusion_cfg = cfg.stage123_geco2.candidate_fusion
    _check_fusion_cfg(fusion_cfg, detector.nms_iou, sample_id, "Stage12-GeCo2", rescored=True)
    if cfg.stage123_geco2.online_adaptive_threshold.enabled:
        log.warning(
            "[Stage12-GeCo2] %s: stage123_geco2.online_adaptive_threshold has no effect with "
            "cosine_rescore (candidates are cut by Stage 3's cosine instead) -- use "
            "stage3.adaptive_threshold_online for a causal threshold on this path.", sample_id,
        )

    cpf_cfg = cfg.stage123_geco2.color_postfilter
    color_sig = (
        build_color_signature(cfg, sample_id, work_dir, cpf_cfg, cfg.stage123_geco2.segmentation)
        if cpf_cfg.enabled else None
    )

    skip_encoding = cr.skip_candidate_encoding
    extractor = None if skip_encoding else build_feature_extractor(cfg)
    if skip_encoding:
        log.info(
            "[Stage12-GeCo2] %s: skip_candidate_encoding -- candidate crops are NOT embedded here; "
            "stage3.recompute_candidate_features will embed them.", sample_id,
        )

    # stage123_geco2.dynamic_prototype (opt-in): same online/incremental
    # mechanism as run_stage123_geco2's own wiring -- see
    # GeCo2DynamicPrototypeTracker's docstring. This function already
    # builds `extractor` (stage1.feature_extractor) and just ran run_stage1
    # above (so prototype.npz already exists) for its OWN candidate-
    # embedding purpose -- reuse both here instead of the tracker lazily
    # loading a second, redundant extractor instance when
    # cross_check_source="feature_extractor".
    dyn_proto_tracker = None
    if cfg.stage123_geco2.dynamic_prototype.enabled:
        from aero_eyes.models.geco2_detector import GeCo2DynamicPrototypeTracker

        dp_cfg = cfg.stage123_geco2.dynamic_prototype
        cross_extractor = cross_prototype = cross_per_ref_features = None
        if dp_cfg.cross_check_source == "feature_extractor":
            from aero_eyes.utils.io import read_prototype
            cross_extractor = extractor
            cross_prototype, _, cross_per_ref_features = read_prototype(work_dir / cfg.stage1.prototype.cache_name)
        dyn_proto_tracker = GeCo2DynamicPrototypeTracker(
            cfg, detector, prototype, work_dir, sample_id,
            cross_check_extractor=cross_extractor, cross_check_prototype=cross_prototype,
            cross_check_per_ref_features=cross_per_ref_features,
        )

    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(
            f"No video matching '{cfg.data.video_glob}' found in {video_dir}."
        )
    video_path = video_files[0]
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[Stage12-GeCo2] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    kf_indices = set(keyframe_indices(total_frames, cfg.stage123_geco2.keyframe_interval))

    def _offer_best(frame_idx, frame_bgr, boxes, feats):
        if dyn_proto_tracker is not None and boxes:
            # offer_topk() considers EVERY surviving candidate (not just
            # boxes[0]) when dynamic_prototype.topk_fusion.enabled -- see
            # its own docstring -- and transparently falls back to plain
            # offer(boxes[0], ...) (today's behavior) otherwise. feats[i]
            # is already computed for candidates.json regardless, reused
            # here at no extra cost.
            dyn_proto_tracker.offer_topk(frame_bgr, boxes, feats, frame_idx=frame_idx)

    get_prototype = dyn_proto_tracker.effective_prototype if dyn_proto_tracker is not None else (lambda: prototype)
    # Raw (pre-Stage-3) candidate frames, only when runtime.save_visualizations is on.
    cand_viz_dir = (
        work_dir / "viz" / "stage123_geco2" / "candidates" if cfg.runtime.save_visualizations else None
    )
    fusion_stats: dict = {}
    candidates = _run_geco2_candidate_pass(
        detector, extractor, video_path, kf_indices, get_prototype, color_sig, cpf_cfg, cfg,
        on_result=_offer_best, viz_dir=cand_viz_dir, fusion_cfg=fusion_cfg, encode=not skip_encoding,
        fusion_stats=fusion_stats,
    )
    if dyn_proto_tracker is not None:
        dyn_proto_tracker.log_summary()

    # dynamic_prototype.second_pass (opt-in): see run_stage123_geco2's own
    # wiring for the full rationale -- re-run the whole video once more
    # with pass 1's final (frozen) prototype so Stage 3 gets candidates
    # built from a fuller exemplar set even for early keyframes, instead
    # of just replaying pass 1's growing-prototype candidates. No-op if
    # pass 1 accepted no dynamic tokens.
    dp_cfg = cfg.stage123_geco2.dynamic_prototype
    if dyn_proto_tracker is not None and dp_cfg.second_pass:
        n_dynamic = dyn_proto_tracker.dynamic_token_count()
        if n_dynamic == 0:
            log.info(
                "[Stage12-GeCo2] %s: dynamic_prototype.second_pass=true but pass 1 accepted "
                "no dynamic tokens -- skipping (pass 2 would be identical).", sample_id,
            )
        else:
            log.info(
                "[Stage12-GeCo2] %s: dynamic_prototype.second_pass -- re-running the whole "
                "video with pass 1's final prototype (%d dynamic token(s)) frozen from frame "
                "1, replacing pass 1's candidates.", sample_id, n_dynamic,
            )
            frozen_prototype = dyn_proto_tracker.effective_prototype()
            if cand_viz_dir is not None and cand_viz_dir.exists():
                # pass 2 replaces pass 1's candidates -- drop pass 1's frames so none go stale
                for old in cand_viz_dir.glob("frame_*.jpg"):
                    old.unlink()
            fusion_stats = {}
            candidates = _run_geco2_candidate_pass(
                detector, extractor, video_path, kf_indices, lambda: frozen_prototype,
                color_sig, cpf_cfg, cfg, viz_dir=cand_viz_dir, fusion_cfg=fusion_cfg,
                encode=not skip_encoding, fusion_stats=fusion_stats,
            )

    _log_fusion_summary("Stage12-GeCo2", sample_id, fusion_cfg, fusion_stats)
    _write_candidates_with_features(candidates, cand_path, placeholder_features=skip_encoding)
    detector.log_peak_contrast_summary(sample_id)

    elapsed = time.time() - t0
    log.info("[Stage12-GeCo2] %s done in %.1fs -> %s (%d keyframes)",
              sample_id, elapsed, cand_path, len(candidates))
    if cand_viz_dir is not None:
        log.info("[Stage12-GeCo2] %s: raw candidate frames (%d with >=1 box) saved to %s",
                 sample_id, sum(1 for d in candidates.values() if d), cand_viz_dir)
    return cand_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 1+2+3 — GeCo2 exemplar detector")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    # Off by default (unchanged behavior: writes detections.json via
    # run_stage123_geco2). With this flag, only generates candidates.json
    # via run_stage12_geco2_candidates (the cosine_rescore variant) --
    # useful to build candidates.json for a sample WITHOUT also paying for
    # Stage 4/5 tracking, e.g. to prepare hard-negative mining data for
    # scripts/train_projection_head.py on samples that were never run
    # through the full pipeline. Still requires
    # stage123_geco2.cosine_rescore.enabled=true (same requirement
    # run_stage12_geco2_candidates itself has via run_all.py).
    p.add_argument("--candidates-only", action="store_true",
                    help="write only candidates.json (run_stage12_geco2_candidates) instead of "
                    "running the full GeCo2-only detector to detections.json. Requires "
                    "stage123_geco2.cosine_rescore.enabled=true.")
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    if args.candidates_only:
        if not cfg.stage123_geco2.cosine_rescore.enabled:
            raise ValueError(
                "--candidates-only requires stage123_geco2.cosine_rescore.enabled=true "
                "(pass --set stage123_geco2.cosine_rescore.enabled=true)."
            )
        run_stage12_geco2_candidates(cfg, args.sample)
    else:
        run_stage123_geco2(cfg, args.sample)


if __name__ == "__main__":
    main()

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
    x1, y1, x2, y2 = tight_box
    obj_size = max(x2 - x1, y2 - y1) * (1.0 + context_margin)
    target_ratio = max(expected_object_px) / float(video_longer_dim)
    if target_ratio <= 0:
        raise ValueError("scale_calibration: expected_object_px / video frame size must be > 0")
    ideal_native_size = obj_size / target_ratio   # conceptual source-crop side, native px (can be huge)
    resize_ratio = canvas_px / ideal_native_size    # scale applied while rendering into canvas_px

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    src_x1, src_y1 = cx - ideal_native_size / 2.0, cy - ideal_native_size / 2.0

    processed = apply_background_mode(img, mask, background_mode, blur_sigma)
    bg_color = img.reshape(-1, 3).mean(axis=0)
    affine = np.array([
        [resize_ratio, 0.0, -src_x1 * resize_ratio],
        [0.0, resize_ratio, -src_y1 * resize_ratio],
    ], dtype=np.float32)
    canvas = cv2.warpAffine(
        processed, affine, (canvas_px, canvas_px),
        borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(c) for c in bg_color),
    )

    box_in_canvas = (
        (x1 - src_x1) * resize_ratio, (y1 - src_y1) * resize_ratio,
        (x2 - src_x1) * resize_ratio, (y2 - src_y1) * resize_ratio,
    )
    return canvas, box_in_canvas


def _locate_video(cfg, sample_id: str) -> Path:
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(
            f"No video matching '{cfg.data.video_glob}' found in {video_dir}."
        )
    return video_files[0]


def _load_ref_images(cfg, sample_id: str) -> list:
    data_root = Path(cfg.data.data_root)
    refs_dir = data_root / sample_id / cfg.data.refs_subdir
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in exts
    )
    if len(ref_paths) < cfg.data.num_references:
        raise FileNotFoundError(
            f"Expected {cfg.data.num_references} reference images in {refs_dir}, "
            f"found {len(ref_paths)}."
        )
    ref_paths = ref_paths[: cfg.data.num_references]
    return [cv2.imread(str(p)) for p in ref_paths]


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
    if seg_cfg.enabled:
        from aero_eyes.models.segmentation import MobileSAMSegmenter
        segmenter = MobileSAMSegmenter(
            weights_path=seg_cfg.weights,
            fallback_if_missing=seg_cfg.fallback_if_missing,
            min_area_frac=seg_cfg.min_area_frac,
            max_area_frac=seg_cfg.max_area_frac,
            score_ratio_floor=seg_cfg.score_ratio_floor,
            max_border_touch_frac=seg_cfg.max_border_touch_frac,
            use_point_prompt=seg_cfg.use_point_prompt,
        )
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
            video_path = _locate_video(cfg, sample_id)
            info = video_info(video_path)
            video_longer_dim = max(info["width"], info["height"])
            scales = (
                sc_cfg.expected_object_px if sc_cfg.multi_scale_mode == "all"
                else sc_cfg.expected_object_px[:1]
            )
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
                for scale_idx, expected_px in enumerate(scales):
                    canvas, box_c = _build_scale_calibrated_canvas(
                        img, m, b, tuple(expected_px), video_longer_dim,
                        sc_cfg.context_margin, seg_cfg.background_mode, seg_cfg.blur_sigma,
                        canvas_px=g.image_size,
                    )
                    canvases.append(canvas)
                    canvas_boxes.append(box_c)
                    label = f"ref_{ref_idx}" if len(scales) == 1 else f"ref_{ref_idx}_scale_{scale_idx}"
                    canvas_labels.append(label)
            if len(scales) > 1:
                log.info(
                    "[Stage123-GeCo2] %s: scale_calibration multi_scale_mode=all -- "
                    "%d ref image(s) x %d scale(s) = %d exemplar entries",
                    sample_id, num_orig_refs, len(scales), len(canvases),
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
            f = g.ref_downscale_factor
            ref_boxes = [
                tuple(c * f for c in b) if b is not None else None
                for b in raw_boxes
            ]
            ref_imgs = [_apply_ref_downscale(img, f) for img in ref_imgs]

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

    def __init__(self, hs_hists: list[np.ndarray], v_hists: list[np.ndarray], confidence: float):
        self.hs_hists = hs_hists
        self.v_hists = v_hists
        self.confidence = confidence


def build_color_signature(cfg, sample_id: str, work_dir: Path) -> ColorSignature:
    """Color histograms of each reference image's masked object region --
    used by apply_color_postfilter() to catch same-shape-different-color
    false positives that GeCo2 itself cannot distinguish (it has no color
    signal, see stage123_geco2.color_postfilter). Cached independently of
    geco2_prototype.pt (this is pure OpenCV, does not need the GeCo2 model
    at all) -- so it's computed even on a cache hit for the prototype file,
    and vice versa; the two caches don't need to be in sync.

    .confidence in [0,1] (see saturation_value_confidence) controls how
    apply_color_postfilter() blends the two histogram sets: 1 = trust
    Hue+Saturation fully (colorful reference object), 0 = trust Value
    fully (near-achromatic reference object, where Hue+Saturation is pure
    noise but Value still reliably tells e.g. black from white).

    Note: if segmentation.enabled, this runs its OWN MobileSAM pass over
    the reference images -- independent from (and possibly duplicating)
    the one build_exemplar_prototype already ran, since that function may
    have taken its cached-prototype early-return path without computing
    masks at all this run. Kept decoupled for simplicity; MobileSAM
    (ViT-tiny) is cheap relative to GeCo2's own SAM2-Hiera-base backbone.
    """
    from aero_eyes.utils.color import (
        compute_hs_histogram, compute_mean_saturation, compute_mean_value,
        compute_value_histogram, saturation_value_confidence,
    )

    cpf = cfg.stage123_geco2.color_postfilter
    sig_path = work_dir / "color_signature.npz"
    if cfg.project.use_cache and sig_path.exists():
        data = np.load(sig_path)
        hs_hists = [data[k] for k in sorted(data.files) if k.startswith("hshist_")]
        v_hists = [data[k] for k in sorted(data.files) if k.startswith("vhist_")]
        mean_sat = float(data["mean_saturation"])
        mean_val = float(data["mean_value"])
    else:
        ref_imgs = _load_ref_images(cfg, sample_id)
        seg_cfg = cfg.stage123_geco2.segmentation
        masks: list[np.ndarray | None] = [None] * len(ref_imgs)
        if seg_cfg.enabled:
            from aero_eyes.models.segmentation import MobileSAMSegmenter
            segmenter = MobileSAMSegmenter(
                weights_path=seg_cfg.weights, fallback_if_missing=seg_cfg.fallback_if_missing,
                min_area_frac=seg_cfg.min_area_frac, max_area_frac=seg_cfg.max_area_frac,
                score_ratio_floor=seg_cfg.score_ratio_floor, max_border_touch_frac=seg_cfg.max_border_touch_frac,
                use_point_prompt=seg_cfg.use_point_prompt,
            )
            masks = [segmenter.segment(img) for img in ref_imgs]
        else:
            log.warning(
                "[Stage123-GeCo2] %s: color_postfilter.enabled but segmentation.enabled=false -- "
                "color signature built from the WHOLE reference photo (diluted by background), "
                "not just the object.",
                sample_id,
            )

        hs_hists = [
            compute_hs_histogram(img, mask, cpf.hue_bins, cpf.sat_bins)
            for img, mask in zip(ref_imgs, masks)
        ]
        v_hists = [
            compute_value_histogram(img, mask, cpf.value_bins)
            for img, mask in zip(ref_imgs, masks)
        ]
        mean_sat = float(np.mean([compute_mean_saturation(img, mask) for img, mask in zip(ref_imgs, masks)]))
        mean_val = float(np.mean([compute_mean_value(img, mask) for img, mask in zip(ref_imgs, masks)]))
        work_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs = {f"hshist_{i}": h for i, h in enumerate(hs_hists)}
        save_kwargs.update({f"vhist_{i}": h for i, h in enumerate(v_hists)})
        save_kwargs["mean_saturation"] = np.array(mean_sat)
        save_kwargs["mean_value"] = np.array(mean_val)
        np.savez(sig_path, **save_kwargs)

    confidence = saturation_value_confidence(
        mean_sat, mean_val,
        cpf.min_ref_saturation, cpf.saturation_full_confidence,
        cpf.min_ref_value, cpf.value_full_confidence,
    )
    log.info("[Stage123-GeCo2] %s: reference object mean HSV saturation=%.1f, value=%.1f "
             "[0-255 scale] -> color_confidence=%.2f (1=trust Hue+Sat, 0=trust Value only)",
             sample_id, mean_sat, mean_val, confidence)
    if confidence < 1.0:
        log.warning(
            "[Stage123-GeCo2] %s: color_postfilter blending %.0f%% Value-based comparison "
            "in (and %.0f%% Hue+Saturation) -- reference object's color (saturation=%.1f, "
            "value=%.1f) is not fully trustworthy for Hue-based comparison alone "
            "(near-achromatic objects give unstable Hue; Value still separates e.g. black "
            "from white).",
            sample_id, (1 - confidence) * 100, confidence * 100, mean_sat, mean_val,
        )
    return ColorSignature(hs_hists, v_hists, confidence)


def apply_color_postfilter(
    frame_bgr: np.ndarray, boxes: list[Box], color_sig: ColorSignature, cpf_cfg,
    stats_out: list[tuple[float, float, float]] | None = None,
) -> list[Box]:
    """Drop/downweight candidate boxes whose color doesn't match the
    reference object's own color signature (best-of-N-refs match per
    signal, so legitimate lighting/angle variation across the 3 reference
    photos isn't penalized). Blends Hue+Saturation similarity and Value
    similarity by color_sig.confidence -- see ColorSignature /
    build_color_signature and ColorPostfilterConfig for why both signals
    exist and how they're weighted (Hue+Sat alone cannot tell e.g. black
    from white; Value alone is more lighting-sensitive).

    stats_out: if given, appends (sim_hs, sim_v, effective_sim) for EVERY
    candidate evaluated (before the min_similarity cutoff) -- lets a
    caller collect the REAL distribution of similarity scores seen on
    actual video frames, since a threshold picked from a synthetic/clean
    test image (as this codebase already learned the hard way once, with
    score_threshold_abs) may not reflect what real footage produces. See
    run_stage123_geco2's end-of-run summary log.
    """
    from aero_eyes.utils.color import compute_hs_histogram, compute_value_histogram, histogram_similarity
    from aero_eyes.utils.geometry import crop_with_pad, inset_box

    conf = color_sig.confidence
    kept: list[Box] = []
    for box in boxes:
        # Sample color from an INSET box (see ColorPostfilterConfig.
        # candidate_inset_ratio) -- the box KEPT below is still the
        # original, unshrunk one; the inset only narrows what pixels the
        # color histogram is measured from.
        color_box = inset_box(box, cpf_cfg.candidate_inset_ratio)
        crop = crop_with_pad(frame_bgr, color_box, pad_ratio=0.0)
        hs_hist = compute_hs_histogram(crop, None, cpf_cfg.hue_bins, cpf_cfg.sat_bins)
        v_hist = compute_value_histogram(crop, None, cpf_cfg.value_bins)
        sim_hs = max(histogram_similarity(hs_hist, r, cpf_cfg.metric) for r in color_sig.hs_hists)
        sim_v = max(histogram_similarity(v_hist, r, cpf_cfg.metric) for r in color_sig.v_hists)
        effective_sim = conf * sim_hs + (1.0 - conf) * sim_v
        if stats_out is not None:
            stats_out.append((sim_hs, sim_v, effective_sim))
        if effective_sim < cpf_cfg.min_similarity:
            continue
        kept.append(Box(box.x1, box.y1, box.x2, box.y2, score=box.score * effective_sim) if cpf_cfg.reweight else box)
    if cpf_cfg.reweight:
        kept.sort(key=lambda b: b.score, reverse=True)
    return kept


def _finalize_keyframe_detections(
    frame_idx: int,
    frame_bgr: np.ndarray,
    boxes: list[Box],
    color_sig,
    cpf_cfg,
    color_stats: list | None,
    viz_dir: Path,
    save_viz: bool,
) -> list[Detection]:
    """Color postfilter + Detection wrapping + viz save -- the per-keyframe
    tail shared by both the default (per-frame-relative) and
    global_adaptive_threshold detection paths in run_stage123_geco2, so the
    two paths can't silently drift apart on this shared bookkeeping."""
    from aero_eyes.utils import viz as vizmod

    if color_sig is not None:
        boxes = apply_color_postfilter(frame_bgr, boxes, color_sig, cpf_cfg, stats_out=color_stats)
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

    cpf_cfg = cfg.stage123_geco2.color_postfilter
    color_sig = build_color_signature(cfg, sample_id, work_dir) if cpf_cfg.enabled else None

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
            )
            detections[frame_idx] = result_dets
            log.debug("[Stage123-GeCo2] frame %d: %d detections", frame_idx, len(result_dets))
    else:
        # ---- Default: GeCo2's own per-frame-relative decision, unchanged. ----
        for frame_idx, frame_bgr in frame_iterator(video_path):
            if frame_idx not in kf_indices:
                continue
            boxes = detector.detect_frame(frame_bgr, prototype)
            result_dets = _finalize_keyframe_detections(
                frame_idx, frame_bgr, boxes, color_sig, cpf_cfg, color_stats, viz_dir, save_viz,
            )
            detections[frame_idx] = result_dets
            log.debug("[Stage123-GeCo2] frame %d: %d detections", frame_idx, len(result_dets))

    if color_stats:
        arr = np.array(color_stats)  # columns: sim_hs, sim_v, effective_sim
        log.info(
            "[Stage123-GeCo2] %s: color_postfilter similarity stats over %d candidates "
            "(min_similarity=%.2f) -- sim_hs p10/p50/p90=%.3f/%.3f/%.3f, "
            "sim_v p10/p50/p90=%.3f/%.3f/%.3f, effective_sim p10/p50/p90=%.3f/%.3f/%.3f, "
            "%% below min_similarity=%.1f%%",
            sample_id, len(color_stats), cpf_cfg.min_similarity,
            *np.percentile(arr[:, 0], [10, 50, 90]),
            *np.percentile(arr[:, 1], [10, 50, 90]),
            *np.percentile(arr[:, 2], [10, 50, 90]),
            100.0 * float((arr[:, 2] < cpf_cfg.min_similarity).mean()),
        )

    # No single global threshold applies when global_adaptive_threshold is
    # disabled (GeCo2 thresholds relative to each frame's own max score) --
    # Stage 4's geco2-aware re-detect path reads
    # stage123_geco2.score_threshold_ratio directly instead of this field
    # either way, so recording effective_threshold here is informational
    # only (for inspecting detections.json), not consumed downstream.
    write_detections(detections, det_path, threshold=effective_threshold)

    elapsed = time.time() - t0
    log.info("[Stage123-GeCo2] %s done in %.1fs -> %s (%d detection frames)",
              sample_id, elapsed, det_path, len(detections))
    return det_path


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
    """
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.geco2_detector import GeCo2Detector
    from aero_eyes.stages.stage1 import run_stage1
    from aero_eyes.stages.stage2 import _write_candidates_with_features
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    cand_path = work_dir / "candidates.json"
    if cfg.project.use_cache and cand_path.exists():
        log.info("[Stage12-GeCo2] %s: using cached candidates at %s", sample_id, cand_path)
        return cand_path

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

    cpf_cfg = cfg.stage123_geco2.color_postfilter
    color_sig = build_color_signature(cfg, sample_id, work_dir) if cpf_cfg.enabled else None

    extractor = build_feature_extractor(cfg)

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

    candidates: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue

        boxes = detector.detect_frame(frame_bgr, prototype)
        if color_sig is not None:
            boxes = apply_color_postfilter(frame_bgr, boxes, color_sig, cpf_cfg)

        if boxes:
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

    _write_candidates_with_features(candidates, cand_path)

    elapsed = time.time() - t0
    log.info("[Stage12-GeCo2] %s done in %.1fs -> %s (%d keyframes)",
              sample_id, elapsed, cand_path, len(candidates))
    return cand_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 1+2+3 — GeCo2 exemplar detector")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage123_geco2(cfg, args.sample)


if __name__ == "__main__":
    main()

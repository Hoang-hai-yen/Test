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

from aero_eyes.types import Detection

log = logging.getLogger(__name__)


def _center_box_mask(shape: tuple, ratio: float) -> np.ndarray:
    """CẢI TIẾN (ported from stage1.py): fallback an toàn khi MobileSAM cho ra
    mask diện tích phi lý (quá nhỏ hoặc quá lớn so với khung ảnh). Thay vì
    passthrough toàn khung (kéo theo nhiễu nền/môi trường thẳng vào exemplar
    token -> hỏng matching ở bước detect), ta giả định ảnh reference luôn có
    vật thể nằm ở trung tâm khung hình (đặc thù ảnh chụp cận cảnh tham
    chiếu), nên dùng 1 vùng crop trung tâm với padding hợp lý làm mask thay
    thế.
    """
    h, w = shape[:2]
    mh, mw = int(round(h * ratio)), int(round(w * ratio))
    y0, x0 = max(0, (h - mh) // 2), max(0, (w - mw) // 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y0 + mh, x0:x0 + mw] = True
    return mask


def _apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill everything outside `mask` with the image's own mean color
    (same approach as stage1.py) -- a solid-black background would be
    out-of-distribution for the backbone; the mean color is a milder,
    less distracting stand-in. This is background_mode == "mean_fill".
    """
    masked = img.copy()
    bg_color = img.reshape(-1, 3).mean(axis=0)
    masked[~mask] = bg_color
    return masked


def _apply_background(img: np.ndarray, mask: np.ndarray, mode: str, blur_sigma: float) -> np.ndarray:
    """Replace (or keep) the non-mask region of `img` per
    stage123_geco2.segmentation.background_mode:

      mean_fill -- flat mean-color fill (_apply_mask above, old default).
                   Cheapest, but a large flat, textureless region is far
                   outside what the backbone (pretrained on natural photos)
                   ever saw -- can push the exemplar token into an
                   unnatural part of feature space.
      keep_real -- leave the reference photo's real background untouched.
                   The tight mask bbox still controls WHERE RoI-Align
                   pools from, so background pixels never enter the token
                   directly -- but the backbone's self-attention still
                   sees a natural image overall (its own real photo
                   context), not a synthetic flat region.
      blur      -- strong Gaussian blur of the real background: keeps
                   natural color/texture statistics but discards fine
                   detail that could otherwise cause spurious background
                   matches.
    """
    if mode == "keep_real":
        return img
    if mode == "blur":
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)
        out = img.copy()
        out[~mask] = blurred[~mask]
        return out
    return _apply_mask(img, mask)


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight bbox around the True region of `mask`. None if mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


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

    processed = _apply_background(img, mask, background_mode, blur_sigma)
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
        )
        # CẢI TIẾN (từ Stage 1): nếu MobileSAM cho ra mask diện tích phi lý
        # (quá nhỏ hoặc quá lớn so với khung ảnh reference), dùng center-crop
        # fallback thay vì để nguyên -- một mask gần như trống hoặc gần như
        # phủ hết khung sẽ kéo theo nhiễu nền/môi trường thẳng vào exemplar
        # token và hỏng luôn matching ở bước detect. Có default an toàn nên
        # KHÔNG cần sửa file config để phát huy tác dụng.
        mask_min_ratio = getattr(seg_cfg, "min_valid_mask_ratio", 0.03)
        mask_max_ratio = getattr(seg_cfg, "max_valid_mask_ratio", 0.92)
        center_fallback_ratio = getattr(seg_cfg, "center_fallback_ratio", 0.75)

        masks = []
        for img in ref_imgs:
            mask = segmenter.segment(img)
            mask_ratio = float(mask.sum()) / float(mask.size)
            if mask_ratio < mask_min_ratio or mask_ratio > mask_max_ratio:
                log.warning(
                    "[Stage123-GeCo2] %s: MobileSAM mask area implausible (%.1f%% of frame), "
                    "dùng center-crop fallback (ratio=%.2f) thay vì passthrough toàn khung.",
                    sample_id, mask_ratio * 100.0, center_fallback_ratio,
                )
                mask = _center_box_mask(img.shape, center_fallback_ratio)
            masks.append(mask)
        # Tight box around the actual segmented object, on the ORIGINAL
        # (pre-downscale/pre-canvas) ref photo -- RoI-align-ing the whole
        # (masked) image instead pools in a lot of background + any
        # resize_and_pad zero-padding, diluting the exemplar token
        # (empirically confirmed to matter).
        raw_boxes = [_mask_bbox(m) for m in masks]

        if sc_cfg.enabled:
            # scale_calibration: build a canvas per ref image where the
            # object occupies the same canvas-relative size it's expected
            # to occupy in the query video frame -- see
            # _build_scale_calibrated_canvas for why ref_downscale_factor
            # alone cannot do this.
            video_path = _locate_video(cfg, sample_id)
            info = video_info(video_path)
            video_longer_dim = max(info["width"], info["height"])
            canvases, canvas_boxes = [], []
            for img, m, b in zip(ref_imgs, masks, raw_boxes):
                if b is None:
                    # Empty mask (fallback/edge case) -- nothing to
                    # calibrate against; fall back to whole-image exemplar.
                    canvases.append(img)
                    canvas_boxes.append(None)
                    continue
                canvas, box_c = _build_scale_calibrated_canvas(
                    img, m, b, tuple(sc_cfg.expected_object_px), video_longer_dim,
                    sc_cfg.context_margin, seg_cfg.background_mode, seg_cfg.blur_sigma,
                    canvas_px=g.image_size,
                )
                canvases.append(canvas)
                canvas_boxes.append(box_c)
            if cfg.runtime.save_visualizations:
                out_dir = work_dir / "viz" / "stage123_geco2" / "refs_scale_calibrated"
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, c in enumerate(canvases):
                    cv2.imwrite(str(out_dir / f"ref_{i}_calibrated.jpg"), c)
            ref_imgs = canvases
            ref_boxes = canvas_boxes
        else:
            ref_imgs = [_apply_background(img, m, seg_cfg.background_mode, seg_cfg.blur_sigma)
                        for img, m in zip(ref_imgs, masks)]
            if cfg.runtime.save_visualizations:
                vizmod.save_stage1_refs(ref_imgs, masks, work_dir / "viz" / "stage123_geco2" / "refs")
            # Scale into the coord system _apply_ref_downscale below produces
            # (uniform factor in both axes, matching that function). NOTE:
            # this only affects blur/detail -- it does NOT change the
            # object's final size on the model's canvas (resize_and_pad
            # re-normalizes the whole image's longer side regardless; see
            # ScaleCalibrationConfig docstring). Use scale_calibration above
            # to fix apparent-size mismatch.
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
        )
        log.info("[Stage123-GeCo2] %s: domain-calibrated exemplar tokens using %d sample frames "
                 "(strength=%.2f)", sample_id, len(sample_frames), dc_cfg.strength)

    GeCo2Detector.save_prototype(prototype, proto_path)
    log.info("[Stage123-GeCo2] %s: encoded %d reference exemplars -> %s",
             sample_id, len(ref_imgs), proto_path)
    return prototype


def _detect_frame_raw(detector, frame_bgr: np.ndarray, prototype):
    """Pass-1 proposals for one keyframe, BEFORE any per-frame thresholding.

    detector.detect_frame() normally filters+NMS's internally against
    cfg.stage123_geco2.score_threshold_ratio -- i.e. relative to THAT
    frame's own max score. That is exactly the failure mode CẢI TIẾN 2/3
    below fix: a keyframe with no real target still has a "best" box by
    construction, so a per-frame-relative threshold always keeps something
    (rác boxes on empty frames -- e.g. tracker bám theo bãi cỏ). We need the
    raw, un-thresholded scores here so run_stage123_geco2 can build one
    global score distribution across the whole video first, and only then
    decide what counts as a real match.

    `score_threshold_ratio=0.0` is the override GeCo2Detector exposes for
    this (falls back to the detector's own configured ratio on older
    builds that don't support the kwarg, in which case the global pass
    below just degrades to reusing the per-frame-filtered boxes).
    """
    try:
        return detector.detect_frame(frame_bgr, prototype, score_threshold_ratio=0.0)
    except TypeError:
        return detector.detect_frame(frame_bgr, prototype)


def run_stage123_geco2(cfg, sample_id: str) -> Path:
    """Run the merged GeCo2 stage for one sample. Returns path to detections.json."""
    from aero_eyes.models.geco2_detector import GeCo2Detector
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.io import write_detections
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, read_frame, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage123-GeCo2] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    detector = GeCo2Detector(cfg)
    prototype = build_exemplar_prototype(cfg, sample_id, detector, work_dir)

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

    # CẢI TIẾN 2 (từ Stage 2/3): per-frame relative thresholding filters
    # against each frame's OWN best score, so a keyframe with no real target
    # still keeps its "best" box -- across a whole video that means rác
    # boxes on every empty frame. Instead: collect raw candidate boxes for
    # every keyframe first (Pass 1), then filter using ONE global score
    # distribution over the entire video (Pass 2), matching what stage3.py
    # already did for the legacy pipeline.
    per_frame_raw: dict[int, list] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue
        per_frame_raw[frame_idx] = _detect_frame_raw(detector, frame_bgr, prototype)
        log.debug("[Stage123-GeCo2] frame %d: %d raw candidates", frame_idx, len(per_frame_raw[frame_idx]))

    all_scores = np.array(
        [b.score for boxes in per_frame_raw.values() for b in boxes], dtype=np.float32
    )

    # CẢI TIẾN 3: Adaptive Global Z-Score Threshold — threshold = max(abs
    # floor, mean + z*std) over the pooled score distribution of the whole
    # video, rather than per-frame. Có default an toàn (getattr) nên KHÔNG
    # cần sửa file config để phát huy tác dụng.
    ad_enabled = getattr(cfg.stage123_geco2, "adaptive_threshold", True)
    ad_z = getattr(cfg.stage123_geco2, "adaptive_z_score", 1.0)
    ad_abs_floor = getattr(cfg.stage123_geco2, "adaptive_min_floor", 0.15)
    nms_iou = getattr(cfg.stage123_geco2, "nms_iou", 0.5)
    topk_per_keyframe = getattr(cfg.stage123_geco2, "topk_per_keyframe", None)

    if ad_enabled and all_scores.size > 0:
        sim_mean = float(all_scores.mean())
        sim_std = float(all_scores.std())
        raw_threshold = sim_mean + ad_z * sim_std
        effective_threshold = max(ad_abs_floor, raw_threshold)

        # Cap at the video's own best observed score -- mean + z*std is a
        # statistical estimate, not a hard ceiling; on a high-spread video
        # it can end up above the actual max, rejecting a genuinely good
        # top match and producing zero detections for the whole video even
        # though a strong candidate existed (same failure mode fixed in
        # stage3.py's adaptive threshold).
        sim_max = float(all_scores.max())
        if effective_threshold > sim_max:
            log.info(
                "[Stage123-GeCo2] %s: adaptive threshold %.3f exceeds max score %.3f -- "
                "capping at max so the best candidate isn't dropped.",
                sample_id, effective_threshold, sim_max,
            )
            effective_threshold = sim_max
        log.info(
            "[Stage123-GeCo2] %s: global adaptive threshold = %.3f "
            "(mean=%.3f std=%.3f z=%.2f, n=%d candidates over %d keyframes)",
            sample_id, effective_threshold, sim_mean, sim_std, ad_z,
            all_scores.size, len(per_frame_raw),
        )
    elif all_scores.size > 0:
        effective_threshold = ad_abs_floor
    else:
        effective_threshold = None

    from aero_eyes.utils.geometry import nms as do_nms

    detections: dict[int, list[Detection]] = {}
    for frame_idx, boxes in per_frame_raw.items():
        if effective_threshold is not None:
            boxes = [b for b in boxes if b.score >= effective_threshold]

        if boxes:
            keep = do_nms(boxes, iou_threshold=nms_iou)
            boxes = [boxes[i] for i in keep]
            boxes.sort(key=lambda b: b.score, reverse=True)
            if topk_per_keyframe is not None:
                boxes = boxes[:topk_per_keyframe]

        result_dets = [
            Detection(frame_idx=frame_idx, box=b, similarity=b.score, source="detect")
            for b in boxes
        ]
        detections[frame_idx] = result_dets

        if cfg.runtime.save_visualizations:
            frame_bgr = read_frame(video_path, frame_idx)
            vizmod.save_stage3_detections(
                frame_bgr, [d.box for d in result_dets], [d.similarity for d in result_dets],
                frame_idx, viz_dir,
            )

    write_detections(detections, det_path, threshold=effective_threshold)

    elapsed = time.time() - t0
    log.info("[Stage123-GeCo2] %s done in %.1fs -> %s (%d detection frames)",
              sample_id, elapsed, det_path, len(detections))
    return det_path


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
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


def _apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill everything outside `mask` with the image's own mean color
    (same approach as stage1.py) -- a solid-black background would be
    out-of-distribution for the backbone; the mean color is a milder,
    less distracting stand-in.
    """
    masked = img.copy()
    bg_color = img.reshape(-1, 3).mean(axis=0)
    masked[~mask] = bg_color
    return masked


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


def run_stage123_geco2(cfg, sample_id: str) -> Path:
    """Run the merged GeCo2 stage for one sample. Returns path to detections.json."""
    from aero_eyes.models.geco2_detector import GeCo2Detector
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.io import write_detections
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage123-GeCo2] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    g = cfg.stage123_geco2
    detector = GeCo2Detector(cfg)

    # ---- Exemplar tokens (Stage 1 equivalent), cached like prototype.npz ----
    proto_path = work_dir / g.prototype_cache_name
    if cfg.project.use_cache and proto_path.exists():
        log.info("[Stage123-GeCo2] %s: using cached exemplar tokens at %s", sample_id, proto_path)
        prototype = GeCo2Detector.load_prototype(proto_path)
    else:
        ref_imgs = _load_ref_images(cfg, sample_id)

        # Exemplar box passed to encode_exemplars(), in the coord system of
        # whichever ref_imgs actually get passed to it (i.e. post-downscale
        # below). None = whole image (encode_exemplars' default).
        ref_boxes: list[tuple[float, float, float, float] | None] | None = None

        seg_cfg = g.segmentation
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
            masks = [segmenter.segment(img) for img in ref_imgs]
            # Tight box around the actual segmented object, BEFORE downscale --
            # RoI-align-ing the whole (masked) image instead pools in a lot of
            # mean-color-filled background + any resize_and_pad zero-padding,
            # diluting the exemplar token (empirically confirmed to matter).
            raw_boxes = [_mask_bbox(m) for m in masks]
            ref_imgs = [_apply_mask(img, m) for img, m in zip(ref_imgs, masks)]
            if cfg.runtime.save_visualizations:
                vizmod.save_stage1_refs(ref_imgs, masks, work_dir / "viz" / "stage123_geco2" / "refs")
            # Scale into the coord system _apply_ref_downscale below produces
            # (uniform factor in both axes, matching that function).
            f = g.ref_downscale_factor
            ref_boxes = [
                tuple(c * f for c in b) if b is not None else None
                for b in raw_boxes
            ]

        ref_imgs = [_apply_ref_downscale(img, g.ref_downscale_factor) for img in ref_imgs]
        prototype = detector.encode_exemplars(ref_imgs, ref_boxes=ref_boxes)
        GeCo2Detector.save_prototype(prototype, proto_path)
        log.info("[Stage123-GeCo2] %s: encoded %d reference exemplars -> %s",
                 sample_id, len(ref_imgs), proto_path)

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

    kf_indices = set(keyframe_indices(total_frames, g.keyframe_interval))
    viz_dir = work_dir / "viz" / "stage123_geco2"

    detections: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue

        boxes = detector.detect_frame(frame_bgr, prototype)
        result_dets = [
            Detection(frame_idx=frame_idx, box=b, similarity=b.score, source="detect")
            for b in boxes
        ]
        detections[frame_idx] = result_dets
        log.debug("[Stage123-GeCo2] frame %d: %d detections", frame_idx, len(result_dets))

        if cfg.runtime.save_visualizations:
            vizmod.save_stage3_detections(
                frame_bgr, [d.box for d in result_dets], [d.similarity for d in result_dets],
                frame_idx, viz_dir,
            )

    # No single global threshold applies (GeCo2 thresholds relative to each
    # frame's own max score) -- Stage 4's geco2-aware re-detect path reads
    # stage123_geco2.score_threshold_ratio directly instead of this field.
    write_detections(detections, det_path, threshold=None)

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

"""Data-prep pipeline for a future Grounding DINO deep-fusion fine-tune
(docs/test.txt's Method 1/2/3 -- feature concat, visual-prompt injection, or
decoder cross-attention with DINOv3). This script does NOT touch any model
internals and trains nothing -- it only builds a curated, augmented image
dataset from this project's own labeled videos, so that work is ready
whenever the model-surgery side of that effort actually starts.

Implements the low-data-regime recipe (frame curation, drone-specific
augmentation, hard-negative harvesting) with the specific corrections this
project's own review of that recipe called for:

  1. Frame curation: diversity is measured from the GT box's OWN trajectory
     (center position + sqrt(area), normalized) via greedy farthest-point
     selection, NOT whole-frame SSIM -- SSIM is dominated by the (usually
     much larger) background, not the target object's own pose/scale.

  2. Rotation augmentation: rotates the WHOLE frame (not just a crop) and
     recomputes the GT box as the axis-aligned bounding rectangle of the
     ORIGINAL box's 4 corners rotated by the same affine transform -- the
     standard technique (albumentations/imgaug use the same one for
     axis-aligned boxes under arbitrary-angle rotation). This is provably
     conservative (the new box always fully contains the rotated object)
     but not mask-tight; that's an inherent, accepted property of
     representing an arbitrarily-rotated object with an axis-aligned box,
     not a bug in this implementation.

  3. Copy-paste augmentation: pastes the reference photo's OBJECT MASK
     (MobileSAM, feathered at the boundary), not a hard rectangular crop --
     an unfeathered paste creates a sharp rectangular seam around every
     synthetic positive that a detector can learn as a shortcut ("sharp
     rectangle = object") instead of the object's own appearance, working
     directly against copy-paste's whole purpose. The pasted crop is
     degraded first via aero_eyes.stages.stage1.apply_ref_degradation
     (downscale+blur+JPEG) -- pasting a crisp, well-lit close-up photo onto
     drone footage untouched teaches "detect a sharp cutout", not "detect
     the object as it actually appears from a drone" (this project's own
     stage123_geco2.ref_downscale_factor docstring already documents this
     exact domain gap).

  4. Hard negatives: harvested AUTOMATICALLY by diffing an existing
     detections.json against GT (IoU < a low threshold = a false positive
     the pipeline already produced) -- no manual re-labeling of "clutter"
     needed. Written as the SAME real frame with an EMPTY box list (this
     dataset is one-object-per-video, so "the target is not in this frame"
     is exactly what should be taught).

Output: <out-dir>/images/*.jpg + <out-dir>/manifest.json, a flat list of
{file, video_id, frame_idx, boxes, text_prompt, source} entries. This is
deliberately a generic, custom schema (not COCO) since COCO has no natural
place for a per-image TEXT prompt, which Grounding DINO training needs --
whatever training loop eventually consumes this can convert to its own
loader trivially.

Usage:
    python -m scripts.prepare_gdino_finetune_data --config configs/config.yaml \\
        --out-dir /path/to/gdino_finetune_data
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Frame curation -- GT-box-trajectory diversity, not whole-frame SSIM
# ---------------------------------------------------------------------------

def _box_state(box: Box, w: int, h: int) -> np.ndarray:
    """Normalized (cx, cy, sqrt(area)) descriptor of a box's position+scale,
    invariant to the video's own resolution -- the thing that actually
    varies as the drone changes altitude/angle, unlike whole-frame pixel
    structure (dominated by background, see this script's own docstring)."""
    cx = 0.5 * (box.x1 + box.x2) / w
    cy = 0.5 * (box.y1 + box.y2) / h
    scale = math.sqrt(max(box.area(), 0.0)) / math.sqrt(w * h)
    return np.array([cx, cy, scale], dtype=np.float64)


def select_diverse_frames(
    gt: dict[int, Box], w: int, h: int, target_count: int, min_state_dist: float = 0.03,
) -> list[int]:
    """Greedy farthest-point selection over _box_state vectors: repeatedly
    picks the GT-bearing frame whose state is farthest (min distance) from
    every frame already picked, until target_count frames are picked or no
    remaining frame clears min_state_dist from the current picked set.
    Frames where the object is absent (not a key in `gt`) are never picked
    -- there is no box to supervise a detector with there; those frames are
    a job for hard-negative harvesting instead (harvest_hard_negatives), not
    this function. Deterministic (no RNG): given the same GT, always
    returns the same frames, so re-running this script doesn't silently
    reshuffle an already-reviewed dataset.
    """
    frame_idxs = sorted(gt.keys())
    if not frame_idxs:
        return []
    states = {fi: _box_state(gt[fi], w, h) for fi in frame_idxs}

    picked = [frame_idxs[0]]
    remaining = set(frame_idxs[1:])
    while remaining and len(picked) < target_count:
        best_fi, best_dist = None, -1.0
        for fi in remaining:
            d = min(float(np.linalg.norm(states[fi] - states[p])) for p in picked)
            if d > best_dist:
                best_fi, best_dist = fi, d
        if best_dist < min_state_dist:
            break
        picked.append(best_fi)
        remaining.discard(best_fi)
    return sorted(picked)


# ---------------------------------------------------------------------------
# 2. Rotation augmentation -- corner-rotation + axis-aligned hull
# ---------------------------------------------------------------------------

def _rotate_box(box: Box, matrix: np.ndarray, w: int, h: int) -> Box | None:
    """Rotates box's 4 corners by the same 2x3 affine `matrix` used to
    rotate the image, then takes the axis-aligned bounding rectangle of the
    rotated corners, clipped to the (w, h) canvas -- see this script's own
    docstring (point 2) for why this specific technique, not a mask, is the
    correct/standard one here. Returns None if the rotated+clipped box is
    degenerate (fully rotated out of frame)."""
    corners = np.array([
        [box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2],
    ], dtype=np.float64)
    ones = np.ones((4, 1))
    rotated = (matrix @ np.hstack([corners, ones]).T).T  # [4, 2]
    x1, y1 = rotated[:, 0].min(), rotated[:, 1].min()
    x2, y2 = rotated[:, 0].max(), rotated[:, 1].max()
    result = Box(x1=x1, y1=y1, x2=x2, y2=y2).clip(w, h)
    if result.area() <= 0:
        return None
    return result


def rotate_image_and_box(
    img: np.ndarray, box: Box, angle_deg: float, min_area_frac_of_original: float = 0.3,
) -> tuple[np.ndarray, Box] | None:
    """Rotates the WHOLE frame by angle_deg around its own center
    (BORDER_REFLECT101 fill -- no hard black border for a shortcut-learner
    to latch onto, same reasoning as the mask-feathered copy-paste below),
    and recomputes box via _rotate_box. Returns None (caller should skip
    this sample) if the rotated box's area drops below
    min_area_frac_of_original of the original -- rotation near a frame edge
    can clip most of the object out of frame; better to drop the sample
    than train on a badly-truncated box."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
    rotated_box = _rotate_box(box, matrix, w, h)
    if rotated_box is None or rotated_box.area() < min_area_frac_of_original * box.area():
        return None
    return rotated_img, rotated_box


# ---------------------------------------------------------------------------
# 3. Scale jitter
# ---------------------------------------------------------------------------

def scale_jitter(img: np.ndarray, box: Box, scale: float) -> tuple[np.ndarray, Box]:
    """Resizes the whole frame by `scale` (keeping the same aspect ratio);
    the box's coordinates scale linearly with it. scale > 1.0 simulates the
    drone flying lower (object apparently bigger), < 1.0 higher."""
    h, w = img.shape[:2]
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_box = Box(x1=box.x1 * scale, y1=box.y1 * scale, x2=box.x2 * scale, y2=box.y2 * scale)
    return resized, scaled_box


# ---------------------------------------------------------------------------
# 4. Copy-paste augmentation -- mask-feathered, degraded ref crop
# ---------------------------------------------------------------------------

def tight_crop_from_mask(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(crop, crop_mask) tightened to mask's own bounding box -- None if the
    mask is empty."""
    from aero_eyes.utils.geometry import mask_bbox

    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def feathered_paste(
    bg: np.ndarray, obj: np.ndarray, obj_mask: np.ndarray, top_left: tuple[int, int],
    feather_ksize: int = 7,
) -> np.ndarray:
    """Alpha-composites obj (using obj_mask, feathered at the boundary) onto
    bg at top_left -- see this script's own docstring (point 3) for why
    feathering matters. Returns a NEW array; bg is not modified in place."""
    out = bg.copy()
    x0, y0 = top_left
    h, w = obj.shape[:2]
    y1, x1 = y0 + h, x0 + w
    if y0 < 0 or x0 < 0 or y1 > bg.shape[0] or x1 > bg.shape[1]:
        raise ValueError(f"feathered_paste: placement {top_left} + object size {(w, h)} exceeds bg bounds {bg.shape[:2]}")
    alpha = cv2.GaussianBlur(obj_mask.astype(np.float32), (feather_ksize, feather_ksize), 0)
    alpha = alpha[..., None]
    region = out[y0:y1, x0:x1].astype(np.float32)
    blended = region * (1.0 - alpha) + obj.astype(np.float32) * alpha
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def sample_paste_location(
    canvas_w: int, canvas_h: int, obj_w: int, obj_h: int, exclude_box: Box | None,
    rng: np.random.Generator, max_tries: int = 30,
) -> tuple[int, int] | None:
    """A random top-left placement for an obj_w x obj_h paste inside the
    canvas, retrying up to max_tries times to avoid overlapping
    exclude_box (the current frame's own real GT box, when pasting onto a
    frame that already has one) -- returns None if no non-overlapping
    placement was found (caller should skip pasting on this background)."""
    if obj_w > canvas_w or obj_h > canvas_h:
        return None
    for _ in range(max_tries):
        x0 = int(rng.integers(0, canvas_w - obj_w + 1))
        y0 = int(rng.integers(0, canvas_h - obj_h + 1))
        if exclude_box is None:
            return x0, y0
        candidate = Box(x1=x0, y1=y0, x2=x0 + obj_w, y2=y0 + obj_h)
        if box_iou(candidate, exclude_box) <= 0.0:
            return x0, y0
    return None


def build_copy_paste_sample(
    bg: np.ndarray, ref_crop: np.ndarray, ref_mask: np.ndarray,
    target_size: tuple[int, int], rng: np.random.Generator, exclude_box: Box | None = None,
) -> tuple[np.ndarray, Box] | None:
    """One synthetic copy-paste (image, box) pair, or None if no valid
    placement was found. ref_crop/ref_mask should already be
    degraded/tightened by the caller (see this script's own docstring)."""
    tw, th = target_size
    obj_resized = cv2.resize(ref_crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(ref_mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
    h, w = bg.shape[:2]
    loc = sample_paste_location(w, h, tw, th, exclude_box, rng)
    if loc is None:
        return None
    composited = feathered_paste(bg, obj_resized, mask_resized, loc)
    x0, y0 = loc
    return composited, Box(x1=x0, y1=y0, x2=x0 + tw, y2=y0 + th)


# ---------------------------------------------------------------------------
# 5. Hard-negative harvesting -- automatic, from an existing detections.json
# ---------------------------------------------------------------------------

def find_hard_negative_frames(
    detections: dict[int, list], gt: dict[int, Box], iou_thresh: float = 0.1,
) -> list[int]:
    """Frame indices where an existing pipeline run produced at least one
    detection whose IoU with that frame's GT box is below iou_thresh (or
    the object is GT-absent that frame at all) -- i.e. a confirmed false
    positive/confuser the pipeline already fell for. `detections` is
    whatever aero_eyes.utils.io.read_detections returns (dict[frame_idx ->
    list[Detection]]); only Detection.box is used here."""
    hard_frames = []
    for fi, dets in detections.items():
        if not dets:
            continue
        gt_box = gt.get(fi)
        for d in dets:
            iou = box_iou(d.box, gt_box) if gt_box is not None else 0.0
            if iou < iou_thresh:
                hard_frames.append(fi)
                break
    return sorted(hard_frames)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    file: str
    video_id: str
    frame_idx: int
    boxes: list[Box]
    text_prompt: str
    source: str

    def to_dict(self) -> dict:
        return {
            "file": self.file, "video_id": self.video_id, "frame_idx": self.frame_idx,
            "boxes": [b.to_dict() for b in self.boxes], "text_prompt": self.text_prompt,
            "source": self.source,
        }


@dataclass
class Manifest:
    entries: list[ManifestEntry] = field(default_factory=list)

    def add(self, entry: ManifestEntry) -> None:
        self.entries.append(entry)

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps({"schema_version": "1.0", "images": [e.to_dict() for e in self.entries]}, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_preview(out_dir: Path, name: str, img: np.ndarray, boxes: list[Box]) -> None:
    from aero_eyes.utils.viz import draw_box

    preview = img.copy()
    for b in boxes:
        draw_box(preview, b, "", (0, 255, 0))
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_dir / f"{name}.jpg"), preview)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample", default=None, help="omit to process every sample under data.data_root")
    p.add_argument("--frames-per-video", type=int, default=60,
                   help="target REAL curated frames per video (~500-1000 total over ~14 videos)")
    p.add_argument("--min-state-dist", type=float, default=0.03,
                   help="min normalized (position+scale) distance between two selected frames")
    p.add_argument("--rotations-per-frame", type=int, default=2,
                   help="random-angle rotated variants generated per curated real frame")
    p.add_argument("--scale-jitters-per-frame", type=int, default=1,
                   help="random-scale variants generated per curated real frame")
    p.add_argument("--copy-paste-count", type=int, default=200,
                   help="total synthetic copy-paste images to generate across all samples")
    p.add_argument("--hard-negative-iou-thresh", type=float, default=0.1)
    p.add_argument("--no-hard-negatives", action="store_true",
                   help="skip hard-negative harvesting (needs an existing detections.json per sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview-n", type=int, default=20,
                   help="save this many preview images (with boxes drawn) under <out-dir>/preview for manual QA")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest()
    n_previewed = 0

    def _maybe_preview(name, img, boxes):
        nonlocal n_previewed
        if n_previewed < args.preview_n:
            _save_preview(out_dir, name, img, boxes)
            n_previewed += 1

    def _save(name: str, img: np.ndarray) -> str:
        rel = f"images/{name}.jpg"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), img)
        return rel

    from aero_eyes.stages.stage123_gdino import resolve_text_prompt
    from aero_eyes.stages.stage1 import apply_ref_degradation
    from aero_eyes.utils.io import load_gt, list_video_ids, read_detections
    from aero_eyes.utils.video import read_frame, video_info

    data_root = Path(cfg.data.data_root)
    sample_ids = [args.sample] if args.sample else [
        d.name for d in sorted(data_root.iterdir()) if d.is_dir() and not d.name.startswith(".")
    ]

    # Reusable across samples for copy-paste backgrounds: (video_id, frame_idx, gt_box_or_None).
    background_pool: list[tuple[str, int]] = []
    ref_crops: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}  # sample_id -> (crop, mask, text_prompt)

    for sample_id in sample_ids:
        try:
            video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
            if not video_files:
                log.warning("[prepare-gdino] %s: no video found -- skipped.", sample_id)
                continue
            video_path = video_files[0]
            info = video_info(video_path)
            w, h = info["width"], info["height"]

            gt = load_gt(cfg.data.gt.global_file, sample_id)
        except Exception as e:
            log.warning("[prepare-gdino] %s: could not load GT/video (%s) -- skipped.", sample_id, e)
            continue

        try:
            text_prompt = resolve_text_prompt(cfg, sample_id)
        except ValueError as e:
            log.warning("[prepare-gdino] %s: %s -- skipped.", sample_id, e)
            continue

        # ---- 1. Frame curation ----
        selected = select_diverse_frames(gt, w, h, args.frames_per_video, args.min_state_dist)
        log.info("[prepare-gdino] %s: selected %d/%d GT-bearing frames.", sample_id, len(selected), len(gt))

        for fi in selected:
            box = gt[fi]
            frame = read_frame(video_path, fi)
            name = f"{sample_id}_{fi:06d}_real"
            rel = _save(name, frame)
            manifest.add(ManifestEntry(rel, sample_id, fi, [box], text_prompt, "real"))
            _maybe_preview(name, frame, [box])
            background_pool.append((sample_id, fi))

            # ---- 2. Rotation augmentation ----
            for k in range(args.rotations_per_frame):
                angle = float(rng.uniform(0, 360))
                result = rotate_image_and_box(frame, box, angle)
                if result is None:
                    continue
                rimg, rbox = result
                rname = f"{sample_id}_{fi:06d}_rot{k}"
                rrel = _save(rname, rimg)
                manifest.add(ManifestEntry(rrel, sample_id, fi, [rbox], text_prompt, "aug_rotate"))
                _maybe_preview(rname, rimg, [rbox])

            # ---- 3. Scale jitter ----
            for k in range(args.scale_jitters_per_frame):
                scale = float(rng.uniform(0.6, 1.6))
                simg, sbox = scale_jitter(frame, box, scale)
                sname = f"{sample_id}_{fi:06d}_scale{k}"
                srel = _save(sname, simg)
                manifest.add(ManifestEntry(srel, sample_id, fi, [sbox], text_prompt, "aug_scale"))
                _maybe_preview(sname, simg, [sbox])

        # ---- Prepare this sample's ref crop+mask for copy-paste (once) ----
        if args.copy_paste_count > 0:
            try:
                from aero_eyes.models.segmentation import MobileSAMSegmenter

                refs_dir = data_root / sample_id / cfg.data.refs_subdir
                ref_paths = sorted(
                    q for q in (refs_dir.iterdir() if refs_dir.is_dir() else [])
                    if q.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                )[: cfg.data.num_references]
                if ref_paths:
                    ref_img = cv2.imread(str(ref_paths[0]))
                    segmenter = MobileSAMSegmenter(weights_path=cfg.stage1.segmentation.weights)
                    mask = segmenter.segment(ref_img)
                    tightened = tight_crop_from_mask(ref_img, mask)
                    if tightened is not None:
                        crop, crop_mask = tightened
                        degraded = apply_ref_degradation(crop, downscale_factor=0.3, blur_ksize=3, jpeg_quality=60)
                        ref_crops[sample_id] = (degraded, crop_mask, text_prompt)
            except Exception as e:
                log.warning("[prepare-gdino] %s: copy-paste ref prep failed (%s) -- skipping copy-paste for this sample.", sample_id, e)

        # ---- 4. Hard negatives (needs an existing detections.json) ----
        if not args.no_hard_negatives:
            det_path = Path(cfg.project.work_dir) / sample_id / "detections.json"
            if det_path.exists():
                detections = read_detections(det_path)
                hard_frames = find_hard_negative_frames(detections, gt, args.hard_negative_iou_thresh)
                log.info("[prepare-gdino] %s: %d hard-negative frame(s) from %s.",
                         sample_id, len(hard_frames), det_path)
                for fi in hard_frames:
                    frame = read_frame(video_path, fi)
                    name = f"{sample_id}_{fi:06d}_hardneg"
                    rel = _save(name, frame)
                    manifest.add(ManifestEntry(rel, sample_id, fi, [], text_prompt, "hard_negative"))
                    _maybe_preview(name, frame, [])
            else:
                log.info("[prepare-gdino] %s: no detections.json at %s -- skipping hard-negative harvesting "
                         "for this sample (run the pipeline on it first if you want its false positives here).",
                         sample_id, det_path)

    # ---- 3(cont). Copy-paste: draw random (sample, background) pairs ----
    if args.copy_paste_count > 0 and ref_crops and background_pool:
        for i in range(args.copy_paste_count):
            src_sample = list(ref_crops.keys())[int(rng.integers(0, len(ref_crops)))]
            crop, mask, text_prompt = ref_crops[src_sample]
            bg_sample, bg_fi = background_pool[int(rng.integers(0, len(background_pool)))]
            bg_video = list((data_root / bg_sample).glob(cfg.data.video_glob))[0]
            bg_frame = read_frame(bg_video, bg_fi)
            try:
                bg_gt = load_gt(cfg.data.gt.global_file, bg_sample)
                exclude = bg_gt.get(bg_fi)
            except Exception:
                exclude = None

            ch, cw = crop.shape[:2]
            bh, bw = bg_frame.shape[:2]
            target_scale = float(rng.uniform(0.5, 1.0))
            tw = max(4, min(bw - 1, round(cw * target_scale)))
            th = max(4, min(bh - 1, round(ch * target_scale)))

            result = build_copy_paste_sample(bg_frame, crop, mask, (tw, th), rng, exclude_box=exclude)
            if result is None:
                continue
            comp_img, comp_box = result
            name = f"copypaste_{src_sample}_on_{bg_sample}_{bg_fi:06d}_{i}"
            rel = _save(name, comp_img)
            manifest.add(ManifestEntry(rel, f"{src_sample}_on_{bg_sample}", bg_fi, [comp_box], text_prompt, "aug_copypaste"))
            _maybe_preview(name, comp_img, [comp_box])

    manifest.write(out_dir / "manifest.json")
    by_source: dict[str, int] = {}
    for e in manifest.entries:
        by_source[e.source] = by_source.get(e.source, 0) + 1
    log.info("[prepare-gdino] done -- %d image(s) written to %s (%s). Manifest: %s",
              len(manifest.entries), images_dir, by_source, out_dir / "manifest.json")
    log.info("[prepare-gdino] REVIEW %d preview image(s) under %s before trusting this dataset.",
              n_previewed, out_dir / "preview")


if __name__ == "__main__":
    main()

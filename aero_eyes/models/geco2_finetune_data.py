"""Dataset loader for finetuning GeCo2 on the AERO EYES domain.

See docs/GECO2_FINETUNE_PLAN.md for the full design rationale. This module
is pure Python/OpenCV/torch (no CUDA ops, no GECO2 model construction) --
it can be imported and partially unit-tested on any machine, unlike
aero_eyes/models/geco2_train_wrapper.py which requires a GPU with the
compiled MultiScaleDeformableAttention extension built.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aero_eyes.models.segmentation import MobileSAMSegmenter
from aero_eyes.stages.stage123_geco2 import (
    _apply_background,
    _apply_ref_downscale,
    _crop_to_object,
    _load_ref_images,
    _locate_video,
    _mask_bbox,
)
from aero_eyes.types import Box
from aero_eyes.utils.io import load_gt
from aero_eyes.utils.video import read_frame, video_info

log = logging.getLogger(__name__)

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Hard safety lists -- never allow the 6 held-out test videos into training,
# and hard-fail loudly if the training video-ID set doesn't match exactly
# what docs/GECO2_FINETUNE_PLAN.md specifies. data.gt.global_file's config
# default is "annotations (1).json" -- the 6-video TEST file, not the
# 14-video training file -- an easy footgun to copy-paste past.
HELD_OUT_TEST_VIDEOS = frozenset({
    "BlackBox_0", "BlackBox_1", "CardboardBox_0", "CardboardBox_1",
    "LifeJacket_0", "LifeJacket_1",
})
EXPECTED_TRAIN_VIDEOS = frozenset({
    "Backpack_0", "Backpack_1", "Jacket_0", "Jacket_1", "Laptop_0", "Laptop_1",
    "Lifering_0", "Lifering_1", "MobilePhone_0", "MobilePhone_1",
    "Person1_0", "Person1_1", "WaterBottle_0", "WaterBottle_1",
})
# Held out for internal train/val split (whole categories, not videos --
# see split_train_val's docstring for why). Lifering has the smallest
# present-frame counts in the training set (tests low-data generalization);
# Person1 is a visually distinct object class (tests generalization to
# dissimilar appearance).
DEFAULT_HOLDOUT_CATEGORIES = ("Lifering", "Person1")


def video_category(video_id: str) -> str:
    """'Backpack_0' -> 'Backpack'. Splits on the LAST '_' -- category names
    in this dataset never contain an underscore themselves."""
    return video_id.rsplit("_", 1)[0]


def assert_no_test_leakage(video_ids: list[str]) -> None:
    """Raises if any of the 6 held-out test videos appear in video_ids."""
    leaked = sorted(set(video_ids) & HELD_OUT_TEST_VIDEOS)
    if leaked:
        raise ValueError(
            f"Refusing to train on held-out TEST video(s): {leaked}. These must "
            "never appear in a training video-ID list -- double check that "
            "data.gt.global_file points at the 14-video annotations.json, not "
            "the 6-video annotations (1).json (the config default)."
        )


def validate_training_video_ids(video_ids: list[str]) -> None:
    """Hard guard against the training/test GT-file mix-up (see
    docs/GECO2_FINETUNE_PLAN.md point 12): asserts video_ids is EXACTLY the
    14 expected training videos and contains none of the 6 held-out test
    videos. A real raise, not a warning -- call this before doing anything
    else in a training entrypoint.
    """
    assert_no_test_leakage(video_ids)
    got = set(video_ids)
    if got != EXPECTED_TRAIN_VIDEOS:
        raise ValueError(
            "video_ids does not match the expected 14 training videos.\n"
            f"  missing: {sorted(EXPECTED_TRAIN_VIDEOS - got)}\n"
            f"  unexpected: {sorted(got - EXPECTED_TRAIN_VIDEOS)}\n"
            "Check that data.gt.global_file points at annotations.json."
        )


def split_train_val(
    video_ids: list[str],
    holdout_categories: tuple[str, ...] = DEFAULT_HOLDOUT_CATEGORIES,
) -> tuple[list[str], list[str]]:
    """Category-level train/val split of the 14 training videos.

    Splits by WHOLE OBJECT CATEGORY, not by video: '_0'/'_1' pairs (e.g.
    Backpack_0/Backpack_1) are almost certainly two takes of the SAME
    physical object (same naming convention, reference photos live in
    separate per-video folders). Splitting within a category would leak
    that object's appearance into "validation," making the internal val
    split measure memorization rather than generalization -- undermining
    the overfitting check this split exists for. See
    docs/GECO2_FINETUNE_PLAN.md point 6.

    Raises via validate_training_video_ids if video_ids isn't exactly the
    expected 14-video training set.
    """
    validate_training_video_ids(video_ids)
    holdout = set(holdout_categories)
    val_ids = sorted(v for v in video_ids if video_category(v) in holdout)
    train_ids = sorted(v for v in video_ids if video_category(v) not in holdout)
    if not val_ids:
        raise ValueError(f"holdout_categories {holdout_categories} matched no video in {video_ids}")
    return train_ids, val_ids


def build_present_absent_pools(
    video_path: str | Path, gt: dict[int, Box]
) -> tuple[list[int], list[int], int]:
    """Return (present_frames, absent_frames, total_frames) for one video.

    present_frames = frames with a GT box, absent_frames = frames without.
    Mirrors scripts/check_geco2_score_separation.py's inline logic,
    factored out here for reuse + testability.
    """
    total_frames = video_info(video_path)["total_frames"]
    present_frames = sorted(gt.keys())
    present_set = set(present_frames)
    absent_frames = [f for f in range(total_frames) if f not in present_set]
    return present_frames, absent_frames, total_frames


def sample_ref_downscale_factor(rng: np.random.Generator, lo: float = 0.03, hi: float = 1.0) -> float:
    """Log-uniform sample in [lo, hi] -- the per-reference-image, per-step
    domain-randomization factor (see docs/GECO2_FINETUNE_PLAN.md point 4).
    Log-uniform (not linear-uniform) spends equal sampling density across
    orders of magnitude of detail loss, so the heavily-blurred end of the
    range isn't drastically under-sampled relative to the near-1.0 end.
    """
    if not (0.0 < lo <= hi):
        raise ValueError(f"require 0 < lo <= hi, got lo={lo} hi={hi}")
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_brightness_contrast(
    rng: np.random.Generator,
    brightness_range: tuple[float, float] = (0.0, 0.0),
    contrast_range: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float]:
    """Sample a fresh (brightness_delta, contrast_factor) pair -- same
    per-reference-image, per-step domain-randomization pattern as
    sample_ref_downscale_factor, but for lighting instead of detail level.
    Reference photos are typically taken in controlled/even lighting, while
    the drone video sees natural outdoor light (shifting sun angle, shadows,
    exposure) -- this is a second, independent axis of the same ground-to-
    aerial domain gap ref_downscale_factor already addresses for detail.

    Default range is a no-op ((0.0, 0.0), (1.0, 1.0)) -- this augmentation
    is opt-in; pass a wider range explicitly to enable it. Uses linear
    (not log) uniform sampling for both -- unlike detail level, there's no
    reason to expect brightness/contrast shift to be better modeled on a
    log scale.
    """
    brightness = float(rng.uniform(*brightness_range))
    contrast = float(rng.uniform(*contrast_range))
    return brightness, contrast


def apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """out = clip(img * contrast + brightness, 0, 255), matching OpenCV's
    own standard brightness/contrast convention. No-op fast path when both
    are neutral (the default), to avoid a redundant copy every step when
    this augmentation isn't enabled.
    """
    if brightness == 0.0 and contrast == 1.0:
        return img
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def convert_gt_box_to_canvas(
    frame_bgr: np.ndarray,
    gt_box: Box | None,
    image_size: float = 1024.0,
) -> tuple[torch.Tensor, tuple[float, float, float, float] | None, float]:
    """Convert a GT box from ORIGINAL video-frame pixel coords into the
    padded-canvas pixel coords the loss needs.

    Single shared source of truth, reused by BOTH the training loop's loss
    target AND the visual sanity-check cell -- so "what we visualize" and
    "what we train on" can never silently diverge (see
    docs/GECO2_FINETUNE_PLAN.md point 7 -- coordinate bugs are the most
    likely silent failure mode of this whole plan).

    Mirrors GeCo2Detector._load_and_pad exactly (same ImageNet normalize +
    GECO2/utils/data.py::resize_and_pad call). Requires GECO2/ already on
    sys.path (see aero_eyes.models.geco2_detector._ensure_geco2_on_path).

    Returns (padded_canvas_tensor [3,image_size,image_size], gt_box_on_
    canvas_px or None, scale_factor). The returned box is still in PIXELS
    on the padded canvas (not yet divided by image_size) -- normalization
    to [0,1] happens at loss-computation time, matching GECO2/train.py's
    own `/1024` convention.
    """
    import cv2

    from utils.data import resize_and_pad  # GECO2/utils/data.py

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD

    h, w = frame_bgr.shape[:2]
    whole_box = torch.tensor([[0.0, 0.0, float(w), float(h)]])
    padded, _, scale = resize_and_pad(t, whole_box, size=image_size, zero_shot=True)
    scale = float(scale)

    if gt_box is None:
        return padded, None, scale
    box_canvas = (gt_box.x1 * scale, gt_box.y1 * scale, gt_box.x2 * scale, gt_box.y2 * scale)
    return padded, box_canvas, scale


class RefImageCache:
    """Precomputes, ONCE per video_id (not per training step): the 3
    native-resolution reference BGR images + their MobileSAM tight-box, via
    aero_eyes.models.segmentation.MobileSAMSegmenter + stage123_geco2's
    _mask_bbox/_apply_background (reused unmodified). Only ~14*3=42
    MobileSAM calls total for the whole training run.

    Per-step augmentation (domain randomization) reads from this cache and
    does cheap array ops only -- see docs/GECO2_FINETUNE_PLAN.md point 4.
    """

    def __init__(self, cfg, video_ids: list[str]):
        seg_cfg = cfg.stage123_geco2.segmentation
        # Only construct MobileSAMSegmenter (which may attempt a weights
        # download) when segmentation is actually enabled -- mirrors
        # stage123_geco2.py::build_exemplar_prototype's own gating.
        segmenter = MobileSAMSegmenter(
            weights_path=seg_cfg.weights,
            fallback_if_missing=seg_cfg.fallback_if_missing,
            min_area_frac=seg_cfg.min_area_frac,
            max_area_frac=seg_cfg.max_area_frac,
            score_ratio_floor=seg_cfg.score_ratio_floor,
            max_border_touch_frac=seg_cfg.max_border_touch_frac,
            use_point_prompt=seg_cfg.use_point_prompt,
        ) if seg_cfg.enabled else None
        self._cache: dict[str, tuple[list[np.ndarray], list[tuple | None]]] = {}
        for video_id in video_ids:
            ref_imgs = _load_ref_images(cfg, video_id)
            if seg_cfg.enabled:
                masks = [segmenter.segment(img) for img in ref_imgs]
                boxes = [_mask_bbox(m) for m in masks]
                ref_imgs = [
                    _apply_background(img, m, seg_cfg.background_mode, seg_cfg.blur_sigma)
                    for img, m in zip(ref_imgs, masks)
                ]
                if cfg.stage123_geco2.crop_to_object:
                    # See aero_eyes/stages/stage123_geco2.py::_crop_to_object --
                    # tighter field of view (real pixels, no masking) so the
                    # object occupies more of the 1024 canvas after
                    # resize_and_pad, without needing any oracle scale
                    # estimate. Mirrors build_exemplar_prototype's identical
                    # step so training and inference stay symmetric.
                    cropped_imgs, cropped_boxes = [], []
                    for img, b in zip(ref_imgs, boxes):
                        if b is None:
                            cropped_imgs.append(img)
                            cropped_boxes.append(None)
                            continue
                        cimg, cbox = _crop_to_object(img, b, cfg.stage123_geco2.crop_context_margin)
                        cropped_imgs.append(cimg)
                        cropped_boxes.append(cbox)
                    ref_imgs, boxes = cropped_imgs, cropped_boxes
            else:
                boxes = [None] * len(ref_imgs)
            self._cache[video_id] = (ref_imgs, boxes)
        log.info("RefImageCache: precomputed %d reference set(s)", len(self._cache))

    def get(self, video_id: str) -> tuple[list[np.ndarray], list[tuple[float, float, float, float] | None]]:
        return self._cache[video_id]


@dataclass
class FinetuneSample:
    video_id: str
    frame_idx: int
    is_present: bool
    ref_images: list[np.ndarray]
    ref_boxes: list[tuple[float, float, float, float] | None]
    frame_bgr: np.ndarray
    gt_box: Box | None


class Geco2FinetuneDataset(Dataset):
    """Every __getitem__ call independently samples video -> present/absent
    -> frame -> per-reference-image downscale factor, fresh every time (see
    docs/GECO2_FINETUNE_PLAN.md points 4-5). `idx` is ignored for content --
    it only satisfies torch.utils.data.Dataset's index protocol; every call
    is an independent draw, not a lookup into a fixed sequence.

    Sampling is two-level, both re-drawn per call:
      1. Uniform over `video_ids` (NOT over frames) -- counteracts the
         28x-749x present-frame-count imbalance across the 14 training
         videos; without this, a high-frame-count video would dominate
         gradient updates over a low-frame-count one despite both being one
         object identity, worsening the overfitting risk this dataset's
         small object count already carries.
      2. Bernoulli(p_present) -- explicit, sweepable present/absent mix,
         independent of each video's true (highly variable) present:absent
         frame-count ratio.
    """

    def __init__(
        self,
        cfg,
        video_ids: list[str],
        ref_cache: RefImageCache,
        steps_per_epoch: int,
        p_present: float = 0.5,
        ref_downscale_range: tuple[float, float] = (0.03, 1.0),
        brightness_range: tuple[float, float] = (0.0, 0.0),
        contrast_range: tuple[float, float] = (1.0, 1.0),
        seed: int | None = None,
    ):
        if not video_ids:
            raise ValueError("video_ids must be non-empty")
        self.cfg = cfg
        self.video_ids = list(video_ids)
        self.ref_cache = ref_cache
        self.steps_per_epoch = steps_per_epoch
        self.p_present = p_present
        self.ref_downscale_lo, self.ref_downscale_hi = ref_downscale_range
        # Opt-in lighting augmentation -- default (0,0)/(1,1) is a no-op, see
        # sample_brightness_contrast's docstring for the rationale.
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rng = np.random.default_rng(seed)

        self._gt: dict[str, dict[int, Box]] = {}
        self._video_paths: dict[str, Path] = {}
        self._pools: dict[str, tuple[list[int], list[int], int]] = {}
        for video_id in self.video_ids:
            video_path = _locate_video(cfg, video_id)
            gt = load_gt(cfg.data.gt.global_file, video_id)
            self._gt[video_id] = gt
            self._video_paths[video_id] = video_path
            self._pools[video_id] = build_present_absent_pools(video_path, gt)

        # Realized present/absent step counts -- log these per epoch since
        # steps_per_epoch redefines "epoch" as a fixed step count rather
        # than an exhaustive pass (sampling is video-uniform + Bernoulli,
        # not exhaustive iteration).
        self.present_count = 0
        self.absent_count = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_frame(self, video_id: str) -> tuple[int, bool]:
        present, absent, _total = self._pools[video_id]
        want_present = self.rng.random() < self.p_present
        if want_present and present:
            return int(self.rng.choice(present)), True
        if not want_present and absent:
            return int(self.rng.choice(absent)), False
        # Degrade gracefully if one pool is empty (shouldn't happen for
        # absent frames on a real video, but keep this robust).
        pool = present or absent
        frame_idx = int(self.rng.choice(pool))
        return frame_idx, frame_idx in set(present)

    def __getitem__(self, idx: int) -> FinetuneSample:
        video_id = str(self.rng.choice(self.video_ids))
        frame_idx, is_present = self._sample_frame(video_id)
        if is_present:
            self.present_count += 1
        else:
            self.absent_count += 1

        native_imgs, native_boxes = self.ref_cache.get(video_id)
        ref_images: list[np.ndarray] = []
        ref_boxes: list[tuple[float, float, float, float] | None] = []
        for img, box in zip(native_imgs, native_boxes):
            factor = sample_ref_downscale_factor(self.rng, self.ref_downscale_lo, self.ref_downscale_hi)
            downscaled = _apply_ref_downscale(img, factor)
            # Brightness/contrast only changes pixel VALUES, never geometry --
            # sampled independently per ref image, same as the downscale
            # factor, but applied after it (order doesn't matter for a
            # geometry-vs-pixel-value pair of ops, but keeps downscale's own
            # blur computed from the ORIGINAL pixel values, not re-lit ones).
            brightness, contrast = sample_brightness_contrast(
                self.rng, self.brightness_range, self.contrast_range,
            )
            ref_images.append(apply_brightness_contrast(downscaled, brightness, contrast))
            ref_boxes.append(tuple(c * factor for c in box) if box is not None else None)

        video_path = self._video_paths[video_id]
        frame_bgr = read_frame(video_path, frame_idx)
        gt_box = self._gt[video_id].get(frame_idx)

        return FinetuneSample(
            video_id=video_id, frame_idx=frame_idx, is_present=is_present,
            ref_images=ref_images, ref_boxes=ref_boxes,
            frame_bgr=frame_bgr, gt_box=gt_box,
        )


def finetune_collate(batch: list[FinetuneSample]) -> list[FinetuneSample]:
    """No padding needed (0-or-1 GT box per sample) -- returns the list
    as-is; the training loop iterates it at the Python-list level (mirrors
    GECO2/train.py's per-idx loop), since each sample also carries its own
    variable-shaped, independently-augmented reference images."""
    return list(batch)

"""Dataset loader for finetuning GeCo2 on the AERO EYES domain.

See docs/GECO2_FINETUNE_PLAN.md for the full design rationale. This module
is pure Python/OpenCV/torch (no CUDA ops, no GECO2 model construction) --
it can be imported and partially unit-tested on any machine, unlike
aero_eyes/models/geco2_train_wrapper.py which requires a GPU with the
compiled MultiScaleDeformableAttention extension built.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aero_eyes.models.segmentation import MobileSAMSegmenter
from aero_eyes.stages.stage123_geco2 import (
    _apply_ref_downscale,
    _load_ref_images,
    _locate_video,
)
from aero_eyes.utils.geometry import apply_background_mode, crop_to_object, mask_bbox
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


def jitter_box(
    rng: np.random.Generator, box: tuple[float, float, float, float], jitter: float,
) -> tuple[float, float, float, float]:
    """Randomly shift the box's center by up to `jitter` * its own width/
    height, and rescale it by a factor in [1-jitter, 1+jitter] -- used to
    make a GT-sourced dynamic exemplar (see max_dynamic_exemplars) NOT a
    pixel-perfect crop, approximating the imprecision of a real inference-
    time confirmed detection box (which passed consecutive-hit + cross-
    check, but is a MODEL prediction, not ground truth). jitter=0.0 (the
    default) is a no-op, returning `box` unchanged.
    """
    if jitter <= 0.0:
        return box
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0 + rng.uniform(-jitter, jitter) * w
    cy = (y1 + y2) / 2.0 + rng.uniform(-jitter, jitter) * h
    scale = rng.uniform(1.0 - jitter, 1.0 + jitter)
    hw, hh = (w * scale) / 2.0, (h * scale) / 2.0
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _apply_query_downscale(img: np.ndarray, downscale_factor: float) -> np.ndarray:
    """Shrink-then-upscale-BACK-to-original-size detail-loss degradation
    for the QUERY frame. Deliberately NOT _apply_ref_downscale (which
    leaves the array smaller, relying on GeCo2Detector._load_and_pad's
    LATER resize_and_pad to re-normalize canvas size): convert_gt_box_to_
    canvas computes its scale factor from frame_bgr.shape and multiplies
    gt_box (fixed, ORIGINAL video-frame coordinates) by it, so frame_bgr's
    pixel DIMENSIONS must stay exactly unchanged here, or that scale would
    silently desync from gt_box's true coordinate space and corrupt every
    training target. No-op at the default 1.0 (downscale_factor >= 1.0).
    """
    if downscale_factor >= 1.0:
        return img
    h, w = img.shape[:2]
    small_w = max(1, int(round(w * downscale_factor)))
    small_h = max(1, int(round(h * downscale_factor)))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


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
    aero_eyes.models.segmentation.MobileSAMSegmenter + aero_eyes.utils.
    geometry's mask_bbox/apply_background_mode (reused unmodified). Only
    ~14*3=42 MobileSAM calls total for the whole training run.

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
                boxes = [mask_bbox(m) for m in masks]
                ref_imgs = [
                    apply_background_mode(img, m, seg_cfg.background_mode, seg_cfg.blur_sigma)
                    for img, m in zip(ref_imgs, masks)
                ]
                if cfg.stage123_geco2.crop_to_object:
                    # See aero_eyes/utils/geometry.py::crop_to_object --
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
                        cimg, cbox = crop_to_object(img, b, cfg.stage123_geco2.crop_context_margin)
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
    # How many of ref_images/ref_boxes (beyond the 3 fixed reference
    # photos) are extra dynamic-style exemplars this step -- see
    # Geco2FinetuneDataset's own max_dynamic_exemplars docstring. 0 when
    # that feature is off (the only value possible before it existed).
    num_dynamic_exemplars: int = 0
    # Same length as ref_images/ref_boxes -- groups entries that come from
    # the SAME underlying reference photo (Track B: num_ref_scale_variants
    # > 1 produces multiple independently-degraded copies of one fixed ref
    # per step, e.g. [0,0,0,1,1,1,2,2,2] for 3 refs x 3 variants). Each
    # dynamic exemplar (if any) gets its OWN unique id -- no scale-grouping
    # there, see Geco2FinetuneDataset.__getitem__. Empty list (the only
    # value possible before num_ref_scale_variants existed) means "one
    # group per entry, in order" -- callers that don't care about Track B's
    # scale-fusion grouping can ignore this field entirely.
    ref_group_ids: list[int] = field(default_factory=list)


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
        query_downscale_range: tuple[float, float] = (1.0, 1.0),
        max_dynamic_exemplars: int = 0,
        dynamic_exemplar_box_jitter: float = 0.0,
        num_ref_scale_variants: int = 1,
        seed: int | None = None,
        hard_frame_frac: float = 0.0,
        hard_frame_top: int = 50,
    ):
        if not video_ids:
            raise ValueError("video_ids must be non-empty")
        if num_ref_scale_variants < 1:
            raise ValueError(f"num_ref_scale_variants must be >= 1, got {num_ref_scale_variants}")
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
        # Opt-in QUERY-side detail-loss augmentation -- default (1.0,1.0) is
        # a no-op. Symmetric counterpart to ref_downscale_range: until now,
        # only the reference image was ever degraded during training (the
        # query frame was always read raw), so the model only ever learned
        # "degraded exemplar vs. sharp query", never the reverse. Empirically,
        # detecting very small/distant objects at inference still requires
        # manually blurring the QUERY frame first -- i.e. the model needs
        # exactly this direction of robustness too, and never saw it during
        # training. Uses the SAME log-uniform sampler/effect as
        # ref_downscale_range (sample_ref_downscale_factor + the shared
        # _apply_ref_downscale shrink, applied to the whole query frame
        # before it reaches the model) -- not yet known to help, hence a
        # separate opt-in range rather than folding it into an existing one.
        self.query_downscale_lo, self.query_downscale_hi = query_downscale_range
        # Opt-in (0 = old behavior, exactly the fixed 3 reference photos
        # every step): at inference, stage123_geco2.dynamic_prototype
        # appends extra exemplar tokens CROPPED FROM THE QUERY VIDEO ITSELF
        # while it runs -- a token count/composition the base checkpoint
        # never saw during ITS OWN training (only ever exactly 3 studio ref
        # photos), which empirically degrades recall even for a single
        # extra, genuinely-correct token: GECO2/models/transformer.py's
        # PrototypeAttentionBlock is a real cross-attention (image features
        # as query, ALL exemplar tokens as key/value) run over EVERY
        # spatial location, so any change to the exemplar SET reshapes the
        # attention distribution for the WHOLE frame, not just scores for
        # one box -- no inference-time threshold/topk/NMS tuning can undo
        # that. Fixing it needs the model to have seen variable-count,
        # partly-video-sourced exemplar sets during training. When > 0,
        # each step samples n_extra ~ Uniform{0, ..., max_dynamic_exemplars}
        # extra exemplars, each a crop from a DIFFERENT present frame of
        # the SAME video (never the current query frame) at its own GT box
        # -- set close to your inference-time dynamic_prototype.max_tokens
        # for train/inference symmetry.
        #
        # Deviates from this module's own stated principle (see
        # docs/GECO2_FINETUNE_PLAN.md: "GT is used here only as the
        # training loss target... never fed into building the reference-
        # image exemplar, precisely because real inference will never have
        # it either"): at INFERENCE, dynamic_prototype's extra tokens come
        # from the model's OWN confirmed prediction, not GT -- a genuinely
        # noise-free GT crop is the OPTIMISTIC case, so this is a
        # deliberate approximation, not the ideal (self-supervised,
        # bootstrapped-from-the-model's-own-predictions) version of this
        # augmentation. dynamic_exemplar_box_jitter below exists to narrow
        # that train/inference gap a little without the cost of running
        # inference mid-training-step.
        self.max_dynamic_exemplars = max_dynamic_exemplars
        # Randomly perturbs each dynamic exemplar's GT box (jitter_box) so
        # it isn't a pixel-perfect crop -- approximates a real confirmed
        # detection box's own imprecision. 0.0 (default) = exact GT box.
        self.dynamic_exemplar_box_jitter = dynamic_exemplar_box_jitter
        self.dynamic_exemplar_count = 0  # realized total, for per-epoch logging (mirrors present_count/absent_count)
        # Track B (docs/GECO2_scale_domain_gap_plan.md): opt-in (1 = old
        # behavior, exactly one degraded copy per fixed ref image per step,
        # unchanged). When > 1, each of the 3 fixed reference images gets
        # this many INDEPENDENTLY-sampled (ref_downscale_factor,
        # brightness, contrast) variants in the SAME step, grouped via
        # FinetuneSample.ref_group_ids -- lets a downstream ScaleFusionGate
        # (aero_eyes/models/geco2_scale_fusion.py) learn to fuse/select
        # across scale-variants of the SAME object, closing the train/
        # inference mismatch ref_downscale_levels' inference-only flat
        # concatenation left open (see that config field's own docstring in
        # aero_eyes/config.py).
        self.num_ref_scale_variants = num_ref_scale_variants
        # Opt-in hard-frame mining (0 = uniform, unchanged): the training loop
        # reports a per-frame hardness via record_hardness(); with probability
        # hard_frame_frac the frame is drawn from the hard_frame_top hardest
        # frames recorded so far for that video and present/absent kind.
        self.hard_frame_frac = hard_frame_frac
        self.hard_frame_top = hard_frame_top
        self._hardness: dict[tuple[str, bool], dict[int, float]] = {}
        self.hard_count = 0
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

    def record_hardness(self, video_id: str, frame_idx: int, is_present: bool, value: float) -> None:
        """Latest hardness of a frame (higher = harder); only used when hard_frame_frac > 0."""
        self._hardness.setdefault((video_id, is_present), {})[int(frame_idx)] = float(value)

    def _pick_hard(self, video_id: str, is_present: bool) -> int | None:
        seen = self._hardness.get((video_id, is_present))
        if not seen or self.rng.random() >= self.hard_frame_frac:
            return None
        top = sorted(seen, key=seen.get, reverse=True)[: max(1, self.hard_frame_top)]
        self.hard_count += 1
        return int(self.rng.choice(top))

    def _sample_frame(self, video_id: str) -> tuple[int, bool]:
        present, absent, _total = self._pools[video_id]
        want_present = self.rng.random() < self.p_present
        if self.hard_frame_frac > 0 and (present if want_present else absent):
            hard = self._pick_hard(video_id, want_present)
            if hard is not None:
                return hard, want_present
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
        ref_group_ids: list[int] = []
        for ref_idx, (img, box) in enumerate(zip(native_imgs, native_boxes)):
            # num_ref_scale_variants > 1 (Track B, opt-in): independently
            # re-sample this SAME ref image's degradation this many times,
            # instead of once -- see __init__'s own docstring for why.
            for _ in range(self.num_ref_scale_variants):
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
                ref_group_ids.append(ref_idx)

        video_path = self._video_paths[video_id]
        frame_bgr = read_frame(video_path, frame_idx)
        # Query-side counterpart to the ref-side downscale above -- see
        # query_downscale_range's docstring in __init__. Uses
        # _apply_query_downscale (shrink-then-upscale-BACK), NOT
        # _apply_ref_downscale -- convert_gt_box_to_canvas below computes
        # its scale factor from frame_bgr.shape and multiplies gt_box
        # (fixed, ORIGINAL video coordinates) by it, so frame_bgr's pixel
        # DIMENSIONS must stay unchanged here or the GT target would be
        # silently corrupted (see _apply_query_downscale's docstring).
        # No-op at the default (1.0, 1.0) range.
        query_factor = sample_ref_downscale_factor(self.rng, self.query_downscale_lo, self.query_downscale_hi)
        frame_bgr = _apply_query_downscale(frame_bgr, query_factor)
        gt_box = self._gt[video_id].get(frame_idx)

        # Extra dynamic-style exemplar(s) (opt-in, see max_dynamic_exemplars
        # docstring in __init__): crop(s) from OTHER present frames of this
        # SAME video, at their own GT box -- never the current query frame
        # itself (that would let the model trivially match itself instead
        # of learning genuine appearance invariance).
        num_dynamic_exemplars = 0
        if self.max_dynamic_exemplars > 0:
            n_extra = int(self.rng.integers(0, self.max_dynamic_exemplars + 1))
            present_pool = [f for f in self._pools[video_id][0] if f != frame_idx]
            n_extra = min(n_extra, len(present_pool))
            if n_extra > 0:
                # Each dynamic exemplar is its own group (no scale-fusion
                # grouping) -- start numbering after every fixed-ref group
                # id already used above (0..len(native_imgs)-1).
                next_group_id = len(native_imgs)
                extra_idxs = self.rng.choice(present_pool, size=n_extra, replace=False)
                for extra_idx in extra_idxs:
                    extra_idx = int(extra_idx)
                    extra_frame = read_frame(video_path, extra_idx)
                    extra_box = self._gt[video_id][extra_idx]
                    extra_box_t = jitter_box(
                        self.rng, (extra_box.x1, extra_box.y1, extra_box.x2, extra_box.y2),
                        self.dynamic_exemplar_box_jitter,
                    )
                    ref_images.append(extra_frame)
                    ref_boxes.append(extra_box_t)
                    ref_group_ids.append(next_group_id)
                    next_group_id += 1
                    num_dynamic_exemplars += 1
            self.dynamic_exemplar_count += num_dynamic_exemplars

        return FinetuneSample(
            video_id=video_id, frame_idx=frame_idx, is_present=is_present,
            ref_images=ref_images, ref_boxes=ref_boxes,
            frame_bgr=frame_bgr, gt_box=gt_box,
            num_dynamic_exemplars=num_dynamic_exemplars,
            ref_group_ids=ref_group_ids,
        )


def finetune_collate(batch: list[FinetuneSample]) -> list[FinetuneSample]:
    """No padding needed (0-or-1 GT box per sample) -- returns the list
    as-is; the training loop iterates it at the Python-list level (mirrors
    GECO2/train.py's per-idx loop), since each sample also carries its own
    variable-shaped, independently-augmented reference images."""
    return list(batch)

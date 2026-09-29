"""Grounding DINO (Liu et al., arXiv:2303.05499) -- open-vocabulary,
TEXT-prompted object detector, loaded through HuggingFace transformers
(GroundingDinoForObjectDetection / AutoProcessor), never the original
IDEA-Research/GroundingDINO repo -- that repo needs a custom CUDA op
(MultiScaleDeformableAttention) compiled from source; transformers ships a
pure-PyTorch reimplementation, no compile step, consistent with this
project's general preference for the standard transformers path over a
vendored repo when one exists (see SigLIP2 vs FG-CLIP's own docstring for
the same tradeoff).

"tiny"/"base" are the original IDEA-Research checkpoints. "mm_tiny"/
"mm_base"/"mm_large" are MM-Grounding-DINO (OpenMMLab's retrain of the same
architecture on broader grounding data) -- loaded through the exact SAME
AutoModelForZeroShotObjectDetection/AutoProcessor path (no code path
difference here beyond the _HF_MAP entry below), merged into transformers
upstream as MMGroundingDinoForObjectDetection. See Stage123GDinoConfig's
own docstring (aero_eyes/config.py) for the full rationale/tradeoffs of
each variant, and why "large"/"edge" are deliberately NOT offered for the
ORIGINAL (non-mm) checkpoints specifically.

NOT YET VALIDATED on this project's own footage.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from aero_eyes.types import Box
from aero_eyes.utils.geometry import nms

log = logging.getLogger(__name__)

_HF_MAP = {
    "tiny": "IDEA-Research/grounding-dino-tiny",   # Swin-T
    "base": "IDEA-Research/grounding-dino-base",   # Swin-B
    # MM-Grounding-DINO (arXiv:2401.02361) -- same architecture, retrained
    # by OpenMMLab. Picked the checkpoint CLOSEST to each original GDino
    # checkpoint's OWN training data, not the highest-benchmark one --
    # EMPIRICALLY OBSERVED (this project's own footage): the broader-data
    # "_grit_v3det"/"_all" checkpoints (this wrapper's first pick) badly
    # under-detected the true target versus original GDino, even after
    # fixing the text=[[...]] input format and dropping input_ids from
    # post-processing (see _postprocess's own docstring) -- neither fix
    # helped, pointing at a genuine calibration/behavior shift from GRIT
    # (large-scale, long/descriptive phrase grounding, not short category
    # names) and V3Det (13000+ fine-grained categories, pushes toward
    # needing more SPECIFIC category matches) diverging from this
    # project's own short, category-name-style prompts. Swapped to the
    # narrowest available checkpoint per tier instead:
    #   original GroundingDINO-Tiny trained on O365+GoldG+Cap4M -- mm_tiny
    #     now maps to o365v1_goldg (same core O365+GoldG, missing only
    #     Cap4M -- no openmmlab checkpoint matches it exactly, this is the
    #     closest available).
    #   original GroundingDINO-Base's own exact recipe wasn't confirmed
    #     here -- mm_base maps to o365v1_goldg_v3det (still carries V3Det,
    #     no bare o365v1_goldg-only Base checkpoint is published) rather
    #     than "_all" (O365+EVERYTHING), and mm_large to
    #     o365v2_oiv6_goldg rather than "_all", for the same
    #     narrower-is-closer-to-original reasoning -- NEITHER has been
    #     confirmed on this project's own footage the way mm_tiny's
    #     failure mode was.
    "mm_tiny": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg",
    "mm_base": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
    "mm_large": "openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg",
}


def _normalize_prompt(text: str) -> str:
    """Grounding DINO's own convention: lowercase, phrases separated by
    ". ", trailing period. Only lowercases + appends a trailing "." if
    missing -- does NOT split/rejoin multi-phrase text, so a caller with
    more than one phrase must already separate them with ". " themselves."""
    text = text.strip().lower()
    if text and not text.endswith("."):
        text += "."
    return text


def _split_phrases(prompt: str) -> list[str]:
    """Splits an already-_normalize_prompt'd, ". "-joined multi-phrase
    string back into individual phrase strings -- needed for MM-Grounding-
    DINO's own `text=[[phrase, ...]]` list-of-lists input convention (see
    GroundingDinoDetector.raw_boxes_and_scores's own docstring for why).
    Drops the trailing "." and any empty pieces from a double separator."""
    return [p for p in (s.strip() for s in prompt.rstrip(".").split(". ")) if p]


class GroundingDinoDetector:
    """Wraps one HF Grounding DINO checkpoint. detect_frame() takes a BGR
    frame + a text prompt and returns Box objects in absolute pixel xyxy,
    already NMS'd and top-K'd -- the same contract
    aero_eyes.models.geco2_detector.GeCo2Detector.detect_frame has, so
    Stage 4's re-detection call sites can treat the two interchangeably."""

    def __init__(self, cfg):
        from aero_eyes.models.features import _resolve_device

        g = cfg.stage123_gdino
        if g.variant not in _HF_MAP:
            raise ValueError(f"Unknown stage123_gdino.variant '{g.variant}'. Must be one of {list(_HF_MAP)}.")
        self.variant = g.variant
        self.box_threshold = g.box_threshold
        self.text_threshold = g.text_threshold
        self.nms_iou = g.nms_iou
        self.topk_per_keyframe = g.topk_per_keyframe
        self.min_box_area_enabled = g.min_box_area_enabled
        self.min_box_area = g.min_box_area
        self.max_box_area_frac_enabled = g.max_box_area_frac_enabled
        self.max_box_area_frac = g.max_box_area_frac
        self.device = _resolve_device(cfg.device())
        self.model, self.processor = self._load(_HF_MAP[g.variant])
        log.info("Grounding DINO %s (%s) on %s", g.variant, _HF_MAP[g.variant], self.device)

    def _load(self, hf_name: str):
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "The installed transformers version doesn't ship Grounding DINO support -- "
                "run `pip install -U transformers` and retry. See Stage123GDinoConfig's own "
                "docstring (aero_eyes/config.py)."
            )
        if self.variant.startswith("mm_"):
            # MMGroundingDinoForObjectDetection was merged into transformers
            # later (~August 2025) than the original GroundingDinoForObject
            # Detection -- an older install may have the base import above
            # succeed while still not recognizing this checkpoint's
            # model_type, which otherwise surfaces as an opaque error deep
            # inside from_pretrained() below. Check explicitly for a clear
            # message pointing at the actual cause.
            try:
                from transformers import MMGroundingDinoForObjectDetection  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    f"stage123_gdino.variant='{self.variant}' needs MM-Grounding-DINO support "
                    "(MMGroundingDinoForObjectDetection), merged into transformers later than "
                    "plain Grounding DINO support -- run `pip install -U transformers` and "
                    "retry. See Stage123GDinoConfig's own docstring (aero_eyes/config.py)."
                )
        processor = AutoProcessor.from_pretrained(hf_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_name)
        model.eval().to(self.device)
        return model, processor

    def _postprocess(
        self, outputs: Any, inputs: dict, frame_shape: tuple[int, int],
        box_threshold: float, text_threshold: float,
    ) -> list[dict]:
        """Isolates the one part of this wrapper most exposed to
        transformers version skew (post_process_grounded_object_detection's
        signature/return keys have changed across releases -- e.g. `labels`
        vs newer `text_labels`) behind a single call site, so a version
        bump only needs fixing here. Takes box_threshold/text_threshold as
        explicit arguments (not self.box_threshold/self.text_threshold)
        so raw_boxes_and_scores below can request a near-zero cutoff for
        threshold-calibration tooling without touching this instance's own
        configured thresholds.

        VERIFIED (transformers source, huggingface/transformers
        src/transformers/models/grounding_dino/processing_grounding_dino.py):
        the box-score threshold kwarg was renamed `box_threshold` ->
        `threshold` in transformers>=4.51.0, and older releases (<4.51) only
        accept `box_threshold`. Tries the current name first (matches any
        transformers version this project is likely to run, including 5.x),
        falling back to the old name on a bare TypeError -- NOT wrapped in
        the outer except below, so a real signature drift beyond just this
        one rename still surfaces as a clear error instead of silently
        retrying forever.

        TRIED (unverified, no live environment to confirm against --
        EMPIRICALLY OBSERVED on this project's own footage: mm_tiny
        under-detected the true object with background scoring higher,
        opposite the direction broader training data should push):
        omits `input_ids` entirely for self.variant.startswith("mm_"),
        matching HF's own official MM Grounding DINO usage example (which
        calls post_process_grounded_object_detection WITHOUT input_ids at
        all when text was passed as list-of-lists, unlike the original
        checkpoints' own docs example, which always passes it) -- see
        raw_boxes_and_scores's own docstring for the matching input-side
        change (list-of-lists text=). Reasoning for why omitting it is
        plausibly correct rather than just copying the doc example
        blindly: input_ids here is only used to decode detected boxes'
        token spans back into human-readable label strings (this wrapper
        never reads that `labels`/`text_labels` output field, only
        `boxes`/`scores`) -- NOT re-fed into score computation itself,
        which already happened during the model's own forward pass. If
        mm_* STILL under-detects after this, this was not (or not the
        only) actual cause -- recalibrate box_threshold/text_threshold
        per checkpoint instead (scripts/calibrate_gdino_threshold.py or
        the lighter scripts/debug_gdino_raw_scores.py)."""
        h, w = frame_shape
        args = (outputs,) if self.variant.startswith("mm_") else (outputs, inputs["input_ids"])
        try:
            results = self.processor.post_process_grounded_object_detection(
                *args, threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[(h, w)],
            )
        except TypeError as e_new:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    *args, box_threshold=box_threshold,
                    text_threshold=text_threshold, target_sizes=[(h, w)],
                )
            except TypeError as e_old:
                raise RuntimeError(
                    "GroundingDinoProcessor.post_process_grounded_object_detection's signature "
                    "doesn't match what this wrapper expects with EITHER `threshold=` (current) "
                    f"or `box_threshold=` (pre-4.51) (errors: {e_new!r} / {e_old!r}). Check "
                    "this method's current signature for your installed transformers version."
                ) from e_old
        return results[0]

    def raw_boxes_and_scores(
        self, frame_bgr: np.ndarray, text_prompt: str,
        box_threshold: float = 0.0, text_threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass + post-process, BEFORE min_box_area/
        max_box_area_frac/NMS/top-K filtering -- the shared primitive behind
        detect_frame (called with this instance's OWN configured
        thresholds) and threshold-calibration tooling (called with a
        near-zero threshold, to see every candidate Grounding DINO
        considered at all -- see stage123_gdino.py's
        global_adaptive_threshold and scripts/calibrate_gdino_threshold.py).
        Returns (boxes_xyxy [N,4], scores [N]), both plain numpy, empty
        arrays if nothing cleared box_threshold/text_threshold.

        For self.variant.startswith("mm_") (MM-Grounding-DINO), `text=` is
        passed as list-of-lists (`[[phrase, ...]]`) instead of the original
        checkpoints' single ". "-joined string -- HF's own official MM
        Grounding DINO usage example uses this format (text_labels=[["a
        cat", "a remote control"]]), not the string convention shown for
        the original checkpoints on that same docs page. NOT independently
        confirmed to change scoring for a single-phrase prompt (this
        project's typical case) -- matches documented usage defensively
        rather than from an observed behavior difference in THIS wrapper.
        If real footage still shows mm_* under-detecting true positives
        after this, recalibrate box_threshold/text_threshold for that
        checkpoint specifically (scripts/calibrate_gdino_threshold.py,
        which works unchanged for any stage123_gdino.variant) before
        suspecting the input format further."""
        h, w = frame_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        prompt = _normalize_prompt(text_prompt)
        text_input = [_split_phrases(prompt)] if self.variant.startswith("mm_") else prompt
        inputs = self.processor(images=pil_img, text=text_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self._postprocess(outputs, inputs, (h, w), box_threshold, text_threshold)
        return result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()

    def detect_frame(self, frame_bgr: np.ndarray, text_prompt: str) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        boxes_xyxy, scores = self.raw_boxes_and_scores(
            frame_bgr, text_prompt, self.box_threshold, self.text_threshold,
        )
        return self.filter_boxes(boxes_xyxy, scores, (h, w))

    def filter_boxes(
        self, boxes_xyxy: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, int],
    ) -> list[Box]:
        """min_box_area/max_box_area_frac/NMS/top-K over already-decoded raw
        boxes -- the tail half of detect_frame, factored out so
        threshold-calibration tooling (stage123_gdino.py's
        global_adaptive_threshold pass 2, scripts/calibrate_gdino_threshold.py)
        can apply the SAME filtering to a raw_boxes_and_scores() call made
        with a threshold other than self.box_threshold, without duplicating
        this logic."""
        h, w = frame_shape
        boxes = [
            Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), score=float(s))
            for b, s in zip(boxes_xyxy, scores)
        ]
        if self.min_box_area_enabled:
            boxes = [b for b in boxes if b.area() >= self.min_box_area]
        if self.max_box_area_frac_enabled:
            # A near-full-frame (or large-fraction-of-background) box is
            # essentially never the real target here (a small drone-viewed
            # object) -- see this project's own SegmentationConfig.
            # max_area_frac for the same idea applied to reference-photo
            # masks. Grounding DINO has no such ceiling built in: unlike
            # YOLO/FastSAM (anchor/architecture-constrained box sizes) or
            # GeCo2 (box regression anchored to the reference exemplar's own
            # size), its open-set query regression can emit a box of ANY
            # size when nothing in the frame truly matches the text prompt,
            # and plain IoU-NMS does NOT reject it just because a smaller,
            # correct box also exists (IoU between a small box nested inside
            # a much larger one is LOW, so NMS doesn't suppress either).
            frame_area = float(h * w)
            boxes = [b for b in boxes if b.area() <= self.max_box_area_frac * frame_area]
        if not boxes:
            return []
        keep = nms(boxes, self.nms_iou)
        boxes = [boxes[i] for i in keep][: self.topk_per_keyframe]
        return boxes

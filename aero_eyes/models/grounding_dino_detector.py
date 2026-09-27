"""Grounding DINO (Liu et al., arXiv:2303.05499) -- open-vocabulary,
TEXT-prompted object detector, loaded through HuggingFace transformers
(GroundingDinoForObjectDetection / AutoProcessor), never the original
IDEA-Research/GroundingDINO repo -- that repo needs a custom CUDA op
(MultiScaleDeformableAttention) compiled from source; transformers ships a
pure-PyTorch reimplementation, no compile step, consistent with this
project's general preference for the standard transformers path over a
vendored repo when one exists (see SigLIP2 vs FG-CLIP's own docstring for
the same tradeoff).

Only "tiny" (Swin-T) and "base" (Swin-B) are wired -- see
Stage123GDinoConfig's own docstring (aero_eyes/config.py) for why "large"/
"edge" are deliberately NOT offered as variant values.

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
        processor = AutoProcessor.from_pretrained(hf_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_name)
        model.eval().to(self.device)
        return model, processor

    def _postprocess(self, outputs: Any, inputs: dict, frame_shape: tuple[int, int]) -> list[dict]:
        """Isolates the one part of this wrapper most exposed to
        transformers version skew (post_process_grounded_object_detection's
        signature/return keys have changed across releases -- e.g. `labels`
        vs newer `text_labels`) behind a single call site, so a version
        bump only needs fixing here.

        VERIFIED (transformers source, huggingface/transformers
        src/transformers/models/grounding_dino/processing_grounding_dino.py):
        the box-score threshold kwarg was renamed `box_threshold` ->
        `threshold` in transformers>=4.51.0, and older releases (<4.51) only
        accept `box_threshold`. Tries the current name first (matches any
        transformers version this project is likely to run, including 5.x),
        falling back to the old name on a bare TypeError -- NOT wrapped in
        the outer except below, so a real signature drift beyond just this
        one rename still surfaces as a clear error instead of silently
        retrying forever."""
        h, w = frame_shape
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[(h, w)],
            )
        except TypeError as e_new:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold, target_sizes=[(h, w)],
                )
            except TypeError as e_old:
                raise RuntimeError(
                    "GroundingDinoProcessor.post_process_grounded_object_detection's signature "
                    "doesn't match what this wrapper expects with EITHER `threshold=` (current) "
                    f"or `box_threshold=` (pre-4.51) (errors: {e_new!r} / {e_old!r}). Check "
                    "this method's current signature for your installed transformers version."
                ) from e_old
        return results[0]

    def detect_frame(self, frame_bgr: np.ndarray, text_prompt: str) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        prompt = _normalize_prompt(text_prompt)
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self._postprocess(outputs, inputs, (h, w))

        boxes_xyxy = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        boxes = [
            Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), score=float(s))
            for b, s in zip(boxes_xyxy, scores)
        ]
        if self.min_box_area_enabled:
            boxes = [b for b in boxes if b.area() >= self.min_box_area]
        if not boxes:
            return []
        keep = nms(boxes, self.nms_iou)
        boxes = [boxes[i] for i in keep][: self.topk_per_keyframe]
        return boxes

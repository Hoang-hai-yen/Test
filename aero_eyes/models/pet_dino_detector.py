"""PET-DINO (Fu et al., CVPR 2026, arXiv:2604.00503) -- "Unifying Visual
Cues into Grounding DINO with Prompt-Enriched Training". UNLIKE
grounding_dino_detector.GroundingDinoDetector (loaded through
`transformers`, including its own mm_tiny/mm_base/mm_large MM-Grounding-
DINO variants), PET-DINO has NOT been merged upstream into transformers --
it lives in its own MMDetection-based repo
(https://github.com/fuweifuvtoo/PET_DINO), never vendored/pip-installable
as a single package. This wraps `mmdet.apis.DetInferencer`, MMDetection's
own "load once, call many times" inferencer object (NOT a CLI-per-
invocation tool -- confirmed via that repo's own scripts/image_demo.py
source, which builds exactly one DetInferencer and calls it per image), so
a per-frame video loop is feasible the same way it is for
GroundingDinoDetector.

Setup required before this can run at all -- see Stage123PetDinoConfig's
own docstring (aero_eyes/config.py) and requirements.txt's own
"pipeline.detector == 'pet_dino'" block: clone the PET_DINO repo, install
its own (heavy, version-sensitive) mmdet/mmengine/mmcv + lvis-api stack,
download checkpoints from https://huggingface.co/fuweifu/PET-DINO.

CRITICAL CAVEAT -- READ BEFORE TRUSTING THIS WRAPPER: written from
PET_DINO's own README + scripts/image_demo.py source (fetched and read
this session, not guessed), confirming:
  - DetInferencer is the right entry point, constructed as
    DetInferencer(model=<config path>, weights=<checkpoint path>, device=...).
  - Its __call__ accepts `texts` (text prompt string) and, for visual
    prompts, `prompt_bboxes`/`prompt_bboxes_labels`/`prompt_image`/
    `prompt_visual_embedding_path` (see raw_boxes_and_scores_visual below),
    plus `pred_score_thr`.
NOT confirmed for this specific fork -- assumed from general, stable
mmdet.apis.DetInferencer convention across the library, NOT smoke-tested
here (this environment has neither mmdet nor even bare `transformers`
installed):
  - __call__'s other kwargs used to suppress per-frame disk writes
    (out_dir/no_save_vis/no_save_pred/return_vis) -- a video loop calling
    this thousands of times must NOT write a viz/pred file to disk every
    call; verify these are the right kwargs (or find the right ones)
    before running on real footage.
  - The RETURN schema: assumed here to be
    {"predictions": [{"bboxes": [[x1,y1,x2,y2], ...], "scores": [...],
    "labels": [...]}]}, one dict per input image -- matches
    mmdet.apis.DetInferencer's documented general contract, but PET-DINO's
    own fork/version was not exercised to confirm the exact key names.
Fix _parse_predictions below if a real run shows a different schema.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import nms

log = logging.getLogger(__name__)


class PetDinoDetector:
    """Wraps one PET-DINO checkpoint via mmdet.apis.DetInferencer.
    detect_frame() takes a BGR frame + a text prompt and returns Box
    objects in absolute pixel xyxy, already NMS'd and top-K'd -- the SAME
    contract aero_eyes.models.grounding_dino_detector.GroundingDinoDetector
    .detect_frame / geco2_detector.GeCo2Detector.detect_frame have. See
    this module's own docstring for which parts of this wrapper are
    confirmed vs. assumed-standard-MMDetection-convention."""

    def __init__(self, cfg):
        from aero_eyes.models.features import _resolve_device

        p = cfg.stage123_pet_dino
        self.box_threshold = p.box_threshold
        self.nms_iou = p.nms_iou
        self.topk_per_keyframe = p.topk_per_keyframe
        self.min_box_area_enabled = p.min_box_area_enabled
        self.min_box_area = p.min_box_area
        self.max_box_area_frac_enabled = p.max_box_area_frac_enabled
        self.max_box_area_frac = p.max_box_area_frac
        self.device = _resolve_device(cfg.device())
        self.inferencer = self._load(p)
        log.info("PET-DINO (%s) on %s", p.config_file, self.device)

    def _load(self, p):
        try:
            from mmdet.apis import DetInferencer
        except ImportError:
            raise RuntimeError(
                "mmdet is not installed -- pipeline.detector == 'pet_dino' needs PET-DINO's "
                "own MMDetection-based dependency stack, separate from this project's usual "
                "requirements.txt. See requirements.txt's own \"pipeline.detector == 'pet_dino'\" "
                "block and Stage123PetDinoConfig's docstring (aero_eyes/config.py) for the full "
                "setup sequence."
            )
        config_path = os.path.join(p.repo_path, p.config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"PET-DINO config file not found: {config_path} -- check "
                f"stage123_pet_dino.repo_path (currently '{p.repo_path}') points at your actual "
                f"PET_DINO clone, and config_file (currently '{p.config_file}') matches a real "
                f"file under it."
            )
        if not os.path.exists(p.weights_path):
            raise FileNotFoundError(
                f"PET-DINO checkpoint not found: {p.weights_path} -- download it from "
                f"https://huggingface.co/fuweifu/PET-DINO and point stage123_pet_dino."
                f"weights_path at it."
            )
        return DetInferencer(model=config_path, weights=p.weights_path, device=self.device)

    def _parse_predictions(self, result: dict) -> tuple[np.ndarray, np.ndarray]:
        """Isolates the one part of this wrapper most exposed to being
        wrong about PET-DINO's own DetInferencer output schema (see this
        module's own CRITICAL CAVEAT) behind a single call site, so a
        real-run correction only needs fixing here."""
        preds = result.get("predictions") or []
        if not preds:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        pred = preds[0]
        boxes_xyxy = np.asarray(pred.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(pred.get("scores", []), dtype=np.float32)
        return boxes_xyxy, scores

    def raw_boxes_and_scores(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """One DetInferencer call with a TEXT prompt, BEFORE min_box_area/
        max_box_area_frac/NMS/top-K filtering -- mirrors
        GroundingDinoDetector.raw_boxes_and_scores's own contract. Returns
        (boxes_xyxy [N,4], scores [N])."""
        result = self.inferencer(
            inputs=frame_bgr[..., ::-1],  # BGR -> RGB, mmdet's own convention
            texts=text_prompt, pred_score_thr=0.0,
            return_vis=False, no_save_vis=True, no_save_pred=True, out_dir="",
        )
        return self._parse_predictions(result)

    def raw_boxes_and_scores_visual(
        self, frame_bgr: np.ndarray, prompt_bboxes: list[list[float]], prompt_bboxes_labels: list[int],
        prompt_image: str | None = None, prompt_visual_embedding_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """VISUAL-prompt counterpart to raw_boxes_and_scores -- exposed for
        future use (see Stage123PetDinoConfig's own docstring: nothing in
        this pipeline automatically converts data.refs_subdir's reference
        photos into a visual prompt yet, no caller uses this method today).
        `prompt_image`/`prompt_visual_embedding_path` are mutually
        exclusive alternate ways PET-DINO's own CLI accepts a visual
        prompt (a reference image to crop the prompt boxes from, or a
        pre-extracted embedding .pt file) -- see this module's own
        docstring for which parts of this call are confirmed vs. assumed."""
        result = self.inferencer(
            inputs=frame_bgr[..., ::-1], prompt_type="Visual",
            prompt_bboxes=prompt_bboxes, prompt_bboxes_labels=prompt_bboxes_labels,
            prompt_image=prompt_image, prompt_visual_embedding_path=prompt_visual_embedding_path,
            pred_score_thr=0.0, return_vis=False, no_save_vis=True, no_save_pred=True, out_dir="",
        )
        return self._parse_predictions(result)

    def detect_frame(self, frame_bgr: np.ndarray, text_prompt: str) -> list[Box]:
        boxes_xyxy, scores = self.raw_boxes_and_scores(frame_bgr, text_prompt)
        h, w = frame_bgr.shape[:2]
        keep = scores >= self.box_threshold
        return self.filter_boxes(boxes_xyxy[keep], scores[keep], (h, w))

    def filter_boxes(
        self, boxes_xyxy: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, int],
    ) -> list[Box]:
        """min_box_area/max_box_area_frac/NMS/top-K over already-decoded raw
        boxes -- identical logic to GroundingDinoDetector.filter_boxes,
        duplicated rather than shared (small, self-contained, and the two
        detectors' configs are otherwise fully independent)."""
        h, w = frame_shape
        boxes = [
            Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), score=float(s))
            for b, s in zip(boxes_xyxy, scores)
        ]
        if self.min_box_area_enabled:
            boxes = [b for b in boxes if b.area() >= self.min_box_area]
        if self.max_box_area_frac_enabled:
            frame_area = float(h * w)
            boxes = [b for b in boxes if b.area() <= self.max_box_area_frac * frame_area]
        if not boxes:
            return []
        keep = nms(boxes, self.nms_iou)
        boxes = [boxes[i] for i in keep][: self.topk_per_keyframe]
        return boxes

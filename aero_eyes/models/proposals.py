"""Class-agnostic region proposals: YOLOv11n, FastSAM-s, OR GeCo2.

YOLOv8 is explicitly NOT permitted anywhere in this project.
Selected by config stage2.proposal_model.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from aero_eyes.types import Box

log = logging.getLogger(__name__)


class ProposalModel(ABC):
    @abstractmethod
    def propose(self, image_bgr: np.ndarray) -> list[Box]:
        """Return a list of candidate bounding boxes in xyxy pixel coords."""
        pass


class Yolov11nProposals(ProposalModel):
    """YOLOv11n class-agnostic proposals via Ultralytics."""

    def __init__(self, cfg):
        from ultralytics import YOLO  # type: ignore

        self.conf = cfg.conf
        self.iou = cfg.iou
        self.max_det = cfg.max_det
        self.classes = cfg.classes  # None = all classes
        self._model = YOLO(cfg.weights)
        log.info("YOLOv11n loaded: %s", cfg.weights)

    def propose(self, image_bgr: np.ndarray) -> list[Box]:
        results = self._model.predict(
            image_bgr,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )
        boxes: list[Box] = []
        for r in results:
            if r.boxes is None:
                continue
            for xyxy, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = xyxy
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2), score=float(score)))
        return boxes


class FastSamSProposals(ProposalModel):
    """FastSAM-s instance segmentation used as region proposals."""

    def __init__(self, cfg):
        from ultralytics import FastSAM  # type: ignore

        self.conf = cfg.conf
        self.iou = cfg.iou
        self.imgsz = cfg.imgsz
        self._model = FastSAM(cfg.weights)
        log.info("FastSAM-s loaded: %s", cfg.weights)

    def propose(self, image_bgr: np.ndarray) -> list[Box]:
        results = self._model(
            image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            retina_masks=True,
        )
        boxes: list[Box] = []
        for r in results:
            if r.boxes is None:
                continue
            for xyxy, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = xyxy
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2), score=float(score)))
        return boxes


class Geco2ProposalModel(ProposalModel):
    """GeCo2 few-shot exemplar detector adapter for Stage 2 proposals."""

    def __init__(self, cfg, sample_id: Optional[str] = None):
        from aero_eyes.models.geco2_detector import GeCo2Detector

        self.cfg = cfg
        self.detector = GeCo2Detector(cfg)
        self.prototype: Optional[dict[str, torch.Tensor]] = None
        self.sample_id = sample_id

        if sample_id is not None:
            self.prepare_for_sample(sample_id)

    def prepare_for_sample(self, sample_id: str) -> None:
        """Encode reference images into exemplar prototype tokens for the given sample."""
        self.sample_id = sample_id
        work_dir = Path(self.cfg.project.work_dir) / sample_id
        g_cfg = getattr(self.cfg, "stage123_geco2", None)
        cache_name = getattr(g_cfg, "prototype_cache_name", "geco2_prototype.pt") if g_cfg else "geco2_prototype.pt"
        proto_path = work_dir / cache_name

        from aero_eyes.models.geco2_detector import GeCo2Detector

        if self.cfg.project.use_cache and proto_path.exists():
            self.prototype = GeCo2Detector.load_prototype(proto_path)
            log.info("[Geco2Proposal] %s: loaded cached prototype tokens from %s", sample_id, proto_path)
            return

        data_root = Path(self.cfg.data.data_root)
        refs_dir = data_root / sample_id / self.cfg.data.refs_subdir
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ref_paths = sorted([p for p in refs_dir.iterdir() if p.suffix.lower() in exts]) if refs_dir.is_dir() else []
        ref_paths = ref_paths[: self.cfg.data.num_references]

        if len(ref_paths) < self.cfg.data.num_references:
            log.warning(
                "[Geco2Proposal] Expected %d reference images in %s, found %d.",
                self.cfg.data.num_references,
                refs_dir,
                len(ref_paths),
            )
            return

        ref_imgs = [cv2.imread(str(p)) for p in ref_paths]

        ref_boxes = None
        seg_cfg = getattr(g_cfg, "segmentation", None) if g_cfg else None
        if seg_cfg and seg_cfg.enabled:
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
            raw_boxes = []
            for m in masks:
                ys, xs = np.where(m)
                if len(xs) == 0:
                    raw_boxes.append(None)
                else:
                    raw_boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
            ref_boxes = raw_boxes

        self.prototype = self.detector.encode_exemplars(ref_imgs, ref_boxes=ref_boxes)
        work_dir.mkdir(parents=True, exist_ok=True)
        GeCo2Detector.save_prototype(self.prototype, proto_path)
        log.info("[Geco2Proposal] %s: encoded %d reference exemplars -> %s", sample_id, len(ref_imgs), proto_path)

    def propose(self, image_bgr: np.ndarray) -> list[Box]:
        if self.prototype is None:
            log.warning("[Geco2Proposal] Prototype tokens not initialized, returning empty proposals.")
            return []
        return self.detector.detect_frame(image_bgr, self.prototype)


def build_proposal_model(cfg, sample_id: Optional[str] = None) -> ProposalModel:
    """Factory: build the configured proposal model."""
    name = getattr(cfg.stage2, "proposal_model", "yolov11n").lower()
    if name == "yolov11n":
        return Yolov11nProposals(cfg.stage2.yolov11n)
    if name == "fastsam_s":
        return FastSamSProposals(cfg.stage2.fastsam_s)
    if name == "geco2":
        return Geco2ProposalModel(cfg, sample_id=sample_id)
    raise ValueError(
        f"Unknown proposal_model '{name}'. Must be 'yolov11n', 'fastsam_s', or 'geco2'. "
        "YOLOv8 is explicitly NOT allowed."
    )
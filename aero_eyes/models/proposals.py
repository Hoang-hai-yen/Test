"""Class-agnostic region proposals: YOLOv11n OR FastSAM-s.

YOLOv8 is explicitly NOT permitted anywhere in this project.
Selected by config stage2.proposal_model.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)


class ProposalModel(ABC):
    @abstractmethod
    def propose(self, image_bgr: np.ndarray) -> list[Box]:
        """Return a list of candidate bounding boxes in xyxy pixel coords."""


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


def build_proposal_model(cfg) -> ProposalModel:
    """Factory: build the configured proposal model."""
    name = cfg.stage2.proposal_model
    if name == "yolov11n":
        return Yolov11nProposals(cfg.stage2.yolov11n)
    if name == "fastsam_s":
        return FastSamSProposals(cfg.stage2.fastsam_s)
    raise ValueError(
        f"Unknown proposal_model '{name}'. Must be 'yolov11n' or 'fastsam_s'. "
        "YOLOv8 is explicitly NOT allowed."
    )

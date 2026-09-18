"""Shared consecutive-hit confirmation gate -- was private to stage4.py
(_DetectionConfirmer), now also used by GeCo2's own dynamic_prototype
online update (see aero_eyes.models.geco2_detector.GeCo2DynamicPrototypeTracker)
so a single spurious high-scoring frame can't poison either mechanism.
"""
from __future__ import annotations

from aero_eyes.types import Box


class DetectionConfirmer:
    """Requires `required_hits` consecutive, spatially-agreeing detector
    hits before trusting one -- see DetectionConfirmationConfig (config.py)
    for why. A hit that disagrees with the pending one (IoU below
    iou_threshold) restarts the streak from that new hit rather than
    discarding it outright, so a real object that has moved since the last
    detect attempt is not penalized for it.
    """

    def __init__(self, required_hits: int, iou_threshold: float):
        self.required_hits = max(1, required_hits)
        self.iou_threshold = iou_threshold
        self._pending_box: Box | None = None
        self._pending_hits = 0

    def offer(self, box: Box) -> Box | None:
        """Feed one detector hit. Returns the box once `required_hits`
        consecutive hits have agreed; otherwise None (still pending)."""
        from aero_eyes.utils.geometry import box_iou

        if self._pending_box is not None and box_iou(self._pending_box, box) >= self.iou_threshold:
            self._pending_hits += 1
        else:
            self._pending_hits = 1
        self._pending_box = box
        if self._pending_hits >= self.required_hits:
            self._pending_box = None
            self._pending_hits = 0
            return box
        return None

    def reset(self) -> None:
        """Call on a frame/attempt where the detector found nothing at all
        -- a gap breaks the "consecutive" streak."""
        self._pending_box = None
        self._pending_hits = 0

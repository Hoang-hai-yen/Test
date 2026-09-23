"""Shared data types for the serialized artifacts that flow between stages.

Each stage READS the previous stage's artifact from disk and WRITES its own,
so any stage can be run/debugged/swapped in isolation.

Artifact contract:
  Stage 1 -> prototype.npz        (prototype vector + metadata + masks)
  Stage 2 -> candidates.json      ({frame_idx: [Box, ...]} + tile info)
  Stage 3 -> detections.json      ({frame_idx: [Detection, ...]})
  Stage 4 -> tracks.json          (per-frame boxes incl. tracked frames)
  Stage 5 -> tube.json + submission file

All boxes are stored internally as absolute xyxy floats, pixel space,
0-based frame indices. Conversion to/from the user's confirmed GT &
submission formats happens ONLY at the I/O boundary (utils/io.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Box:
    """Absolute xyxy pixel box."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    # Set only by GeCo2Detector.filter_boxes_by_threshold when
    # stage123_geco2.peak_contrast_filter.enabled -- how sharply this box's
    # GeCo2 centerness peak stands out from its local neighborhood (peak -
    # neighborhood_mean) / neighborhood_std, as opposed to `score`'s raw peak
    # VALUE. None everywhere else (old behavior, no shape change to callers
    # that never set it). See Geco2PeakContrastFilterConfig's own docstring.
    peak_contrast: float | None = None

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        d = {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "score": self.score}
        if self.peak_contrast is not None:
            d["peak_contrast"] = self.peak_contrast
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(
            x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"], score=d.get("score", 1.0),
            peak_contrast=d.get("peak_contrast"),
        )

    def clip(self, w: float, h: float) -> "Box":
        return Box(
            x1=max(0.0, min(self.x1, w)),
            y1=max(0.0, min(self.y1, h)),
            x2=max(0.0, min(self.x2, w)),
            y2=max(0.0, min(self.y2, h)),
            score=self.score,
            peak_contrast=self.peak_contrast,
        )


@dataclass
class Detection:
    frame_idx: int
    box: Box
    similarity: float
    source: str  # "detect" | "track"

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "box": self.box.to_dict(),
            "similarity": self.similarity,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(
            frame_idx=d["frame_idx"],
            box=Box.from_dict(d["box"]),
            similarity=d["similarity"],
            source=d["source"],
        )


@dataclass
class SpatioTemporalTube:
    """A contiguous or gapped sequence of per-frame bounding boxes."""
    video_id: str
    # frame_idx -> Box (only present frames are stored)
    frames: dict[int, Box] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def present_frames(self) -> list[int]:
        return sorted(self.frames.keys())

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frames": {str(k): v.to_dict() for k, v in self.frames.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpatioTemporalTube":
        return cls(
            video_id=d["video_id"],
            frames={int(k): Box.from_dict(v) for k, v in d.get("frames", {}).items()},
        )

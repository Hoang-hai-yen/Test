"""Per-track windowed drift-plausibility check for Stage 4 tracking
(stage4.kalman_motion_check).

v1 of this (single-step constant-velocity Kalman filter, predicting only
from the immediately preceding frame) was swept empirically
(scripts/sweep_kalman_max_dist_ratio.py) against real footage and found to
be net HARMFUL at every ratio strict enough to ever trigger: ST-IoU dropped
monotonically as the ratio tightened, on both samples tested, with the
least-bad setting being "never trigger". Root cause: real motion in this
footage (drone camera + falling/tumbling objects) has plenty of
frame-to-frame acceleration/direction change that a 1-step constant-
velocity predictor reads as "implausible", so it fired mostly on genuine
motion, not on drift -- forcing re-detect that then often failed to
recover the box (MISSING_PRED exploded: 187 -> 1590 at the strictest ratio
tested on one sample).

v2 (this version) instead fits a robust LINEAR trend across the last
window_frames PAST positions (least-squares, so a couple of noisy/fast
frames don't dominate it the way a single previous frame did in v1),
extrapolates that trend one step forward, and only flags the CURRENT frame
if it diverges from where the established recent trend says it should be.
This damps ordinary frame-to-frame jitter/acceleration (which v1 treated
as literal ground truth for "expected velocity") while still catching a
track that abruptly departs from its own recent trajectory -- the actual
signature of latching onto a confuser mid-track. Not yet validated the
same way v1 was -- sweep max_dist_ratio (and window_frames) again on real
data before trusting this in production; the empirical claim above is
about v1, not a guarantee this redesign is better.

Catches a kind of drift verify_interval's DINOv2 cosine check cannot:
appearance-similarity is blind to WHERE the box is, so a same-shape/colour
confuser sitting far from where the real object could plausibly have moved
to by now can still pass cosine, yet is geometrically implausible given the
track's own recent trajectory. This is a cheap per-frame complement to
verify_interval, not a replacement -- see stage4.py's KalmanMotionCheckConfig
for how the two combine.

Deliberately NOT a ByteTrack-style multi-object association pipeline (built
for associating many per-frame detections across identities): this project
tracks exactly one object per video and already avoids per-frame detection
by design (tracker + occasional re-detect). Only ByteTrack's core idea --
predict expected motion, flag implausible jumps -- is borrowed here.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from aero_eyes.types import Box


class BoxDriftCheck:
    """Fits a linear trend over the last `window_frames` PAST box centers
    and flags the current frame if it diverges from that trend by more than
    `max_dist_ratio` box-diagonals. One instance per active track -- call
    init() every time the caller (re)inits its own tracker, since a fresh
    lock has no relation to the PREVIOUS track's trajectory.
    """

    def __init__(self, window_frames: int = 10):
        # Need >=2 points to fit a line at all; short of that the check is
        # a no-op (always plausible) until enough history accumulates.
        self.window_frames = max(2, window_frames)
        self._history: deque[tuple[float, float]] = deque(maxlen=self.window_frames)

    def init(self, box: Box) -> None:
        """(Re)anchor -- discard prior trajectory history and start a fresh
        window from `box`. Call this whenever the caller's own
        tracker.init() is called (new keyframe lock, post-refine
        re-anchor, or re-detect after track loss)."""
        self._history.clear()
        self._history.append(self._center(box))

    def check_and_update(self, box: Box, max_dist_ratio: float) -> bool:
        """Fit a line through the history collected so far (NOT including
        `box` itself), extrapolate it one step forward, and compare that
        prediction to `box`'s actual center -- distance normalized by
        `box`'s own diagonal, so bigger objects tolerate bigger absolute
        pixel divergence. `box` is appended to history regardless of the
        verdict (so a genuine fast-but-real move still becomes part of the
        established trend for the next check, instead of leaving the
        window stuck on stale, now-wrong positions).

        Returns True (plausible) whenever there isn't yet enough history
        (fewer than 3 points) to fit a trend worth trusting.
        """
        plausible = True
        if len(self._history) >= 3:
            ts = np.arange(len(self._history), dtype=np.float64)
            xs = np.array([p[0] for p in self._history], dtype=np.float64)
            ys = np.array([p[1] for p in self._history], dtype=np.float64)
            vx, bx = np.polyfit(ts, xs, 1)
            vy, by = np.polyfit(ts, ys, 1)
            t_next = float(len(self._history))
            pred_cx, pred_cy = vx * t_next + bx, vy * t_next + by

            cx, cy = self._center(box)
            diag = ((box.x2 - box.x1) ** 2 + (box.y2 - box.y1) ** 2) ** 0.5
            dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5
            plausible = diag <= 0 or (dist / diag) <= max_dist_ratio

        self._history.append(self._center(box))
        return plausible

    @staticmethod
    def _center(box: Box) -> tuple[float, float]:
        return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0


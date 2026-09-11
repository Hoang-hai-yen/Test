"""Per-track constant-velocity Kalman filter for a motion-plausibility
check in Stage 4 tracking (stage4.kalman_motion_check).

Catches a kind of drift verify_interval's DINOv2 cosine check cannot:
appearance-similarity is blind to WHERE the box is, so a same-shape/colour
confuser sitting far from where the real object could plausibly have moved
to by now can still pass cosine, yet is geometrically implausible given the
track's own recent velocity. This is a cheap per-frame complement to
verify_interval, not a replacement -- see stage4.py's KalmanMotionCheckConfig
for how the two combine.

Deliberately NOT a ByteTrack-style multi-object association pipeline (built
for associating many per-frame detections across identities): this project
tracks exactly one object per video and already avoids per-frame detection
by design (tracker + occasional re-detect). Only ByteTrack's core idea --
predict expected motion, flag implausible jumps -- is borrowed here.
"""
from __future__ import annotations

import cv2
import numpy as np

from aero_eyes.types import Box


class BoxMotionKalman:
    """Constant-velocity Kalman filter over a box's center (cx, cy). One
    instance per active track -- call init() every time the caller (re)inits
    its own tracker, since a fresh lock has no relation to the PREVIOUS
    track's velocity.
    """

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1.0):
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32,
        )
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32,
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self._ready = False

    def init(self, box: Box) -> None:
        """(Re)anchor the filter to `box` with zero velocity -- call this
        whenever the caller's own tracker.init() is called (new keyframe
        lock, post-refine re-anchor, or re-detect after track loss)."""
        cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
        # Explicit (4,1) column vector -- a bare 1D (4,) array is coerced
        # differently across OpenCV major versions (silently worked as a
        # column vector on 4.x here, but OpenCV 5.0's gemm inside predict()
        # asserts on the resulting shape instead of coercing it).
        self._kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._ready = True

    def check_and_update(self, box: Box, max_dist_ratio: float) -> bool:
        """Predict this frame's expected center from the track's own recent
        velocity, compare against `box`'s actual center -- distance
        normalized by `box`'s own diagonal, so bigger objects tolerate
        bigger absolute pixel jumps -- then update the filter with `box`
        regardless of the verdict (so a genuine fast-but-real move doesn't
        leave the filter stuck predicting the old, now-wrong, position; the
        caller re-anchors via init() anyway once a track is invalidated and
        re-detect finds a new lock).

        Returns True (plausible) unconditionally on the first call after
        init() -- there is no velocity yet to judge a jump against.
        """
        if not self._ready:
            self.init(box)
            return True

        predicted = self._kf.predict().reshape(-1)
        pred_cx, pred_cy = float(predicted[0]), float(predicted[1])

        cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
        diag = ((box.x2 - box.x1) ** 2 + (box.y2 - box.y1) ** 2) ** 0.5
        dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5
        plausible = diag <= 0 or (dist / diag) <= max_dist_ratio

        self._kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        return plausible

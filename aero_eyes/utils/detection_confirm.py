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


class TrackerAgreementGate:
    """Arbitrates a keyframe detection against the box the ACTIVE tracker
    holds for that very same frame (zero temporal gap, unlike
    DetectionConfirmer which compares detections a whole keyframe_interval
    apart -- too gappy for a fast-moving drone).

    judge() returns:
      "accept"     -- detection agrees with the tracker (IoU >= iou_threshold):
                      trust it immediately and re-anchor the tracker.
      "keep_track" -- disagrees, but the tracker keeps the lock for now.
      "replace"    -- disagrees and the detection wins: drop the current track,
                      re-init from the detection.

    on_mismatch decides keep_track vs replace:
      "conf_compare": the detection that last anchored the track (its
          stage-3 similarity, see anchored()) vs this detection's; the higher
          wins, ties go to the existing track. No known anchor similarity
          (e.g. track came from a re-detect) counts as -inf: detection wins.
      "hits": keep tracking through the first required_hits-1 consecutive
          mismatching keyframes; the required_hits-th one replaces the track.
          A keyframe with no detection breaks the streak (reset_streak()).
      "confirm_new_track": unconditional replace on every mismatch (no
          conf/streak comparison at all) -- but the caller (stage4.py) is
          expected to first spend `track_confirmed` (read it BEFORE calling
          judge(), which is the only thing that flips it back to True) to
          decide whether the track being replaced had itself ever survived
          one of ITS OWN agreeing keyframes since its last (re-)init. If not
          (still on probation), the caller retroactively wipes every frame
          written under it -- an unconfirmed track was never independently
          corroborated, so extending trust to its whole probation window
          just because a THIRD, unrelated detection later showed up nearby
          would be unjustified. If it HAD already been confirmed at some
          earlier keyframe, its already-written history is left untouched
          (only this point forward changes) -- that history was already
          corroborated once and this later mismatch says nothing about it.
          Either way the replacement track starts its own fresh probation
          (anchored() resets track_confirmed to False).
    """

    def __init__(self, iou_threshold: float, on_mismatch: str, required_hits: int):
        if on_mismatch not in ("conf_compare", "hits", "confirm_new_track"):
            raise ValueError(
                f"Unknown on_mismatch '{on_mismatch}'. Must be 'conf_compare', 'hits', "
                "or 'confirm_new_track'."
            )
        self.iou_threshold = iou_threshold
        self.on_mismatch = on_mismatch
        self.required_hits = max(1, required_hits)
        self.anchor_sim: float | None = None
        self._streak = 0
        # confirm_new_track only -- irrelevant (never read) for the other
        # two modes. True once THIS track (since its last anchored() call)
        # has had at least one keyframe agree with it. See judge()'s
        # "accept" branch (the only place this ever becomes True) and
        # anchored() (the only place it's reset to False).
        self.track_confirmed = False

    def anchored(self, sim: float | None) -> None:
        """The tracker was just (re-)initialized from a detection with
        stage-3 similarity `sim` (None if unknown, e.g. a re-detect).
        Starts a fresh, unconfirmed probation for confirm_new_track mode."""
        self.anchor_sim = sim
        self._streak = 0
        self.track_confirmed = False

    def reset_streak(self) -> None:
        self._streak = 0

    def judge(self, track_box: Box, det_box: Box, det_sim: float) -> str:
        from aero_eyes.utils.geometry import box_iou

        if box_iou(track_box, det_box) >= self.iou_threshold:
            self._streak = 0
            self.track_confirmed = True
            return "accept"
        self._streak += 1
        if self.on_mismatch == "confirm_new_track":
            # Caller reads self.track_confirmed (still holding its PRE-call
            # value here) to know whether to wipe the outgoing track's
            # buffered frames -- nothing more to decide on our end.
            return "replace"
        if self.on_mismatch == "conf_compare":
            anchor = float("-inf") if self.anchor_sim is None else self.anchor_sim
            keep = anchor >= det_sim
        else:
            keep = self._streak < self.required_hits
        if keep:
            return "keep_track"
        self._streak = 0
        return "replace"

"""Stage 4 — Tracking between keyframes.

Flow:  detections.json + video
       -> initialize tracker at each keyframe detection
       -> propagate boxes on intermediate frames
       -> if conf < tau: trigger re-detection
       -> every verify_interval frames (opt-in): re-embed the tracked crop
          and cross-check it against the prototype, forcing a re-detect if
          it no longer matches -- catches silent drift builtin trackers'
          placeholder confidence can't (see _track_similarity)
       -> every frame (opt-in, stage4.kalman_motion_check): cheap windowed
          drift-plausibility check, forcing a re-detect when the box departs
          from its own recent trajectory in a way verify_interval's cosine
          check wouldn't catch (see aero_eyes/utils/motion_drift_check.py)
       -> at every fresh lock (opt-in, stage4.backward_tracking): run a
          SEPARATE tracker instance BACKWARD from that lock to recover
          frames where the object was present but not yet detected (cold
          start, or a re-lock after mid-video track loss)
       -> tracks.json

Tracker options: builtin | litetrack | none
Reads:  detections.json, video
Writes: tracks.json
Viz:    annotated video (boxes per frame, detect vs track colour-coded).
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)


class _DetectionConfirmer:
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


def run_stage4(cfg, sample_id: str) -> Path:
    """Run Stage 4. Returns path to tracks.json."""
    from aero_eyes.models.trackers import NoneTracker, build_tracker
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.io import read_detections, read_detections_threshold, write_tracks
    from aero_eyes.utils.video import AnnotatedVideoWriter, frame_iterator, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = work_dir / "tracks.json"
    if cfg.project.use_cache and tracks_path.exists():
        log.info("[Stage4] %s: using cached tracks at %s", sample_id, tracks_path)
        return tracks_path

    # ---- Load detections ----
    det_path = work_dir / "detections.json"
    if not det_path.exists():
        raise FileNotFoundError(
            f"detections.json not found at {det_path}. Run Stage 3 first."
        )
    detections = read_detections(det_path)
    # Reuse the exact threshold Stage 3 matched with (adaptive z-score value
    # when enabled) so mid-video re-detection applies the same acceptance
    # bar instead of silently falling back to the fixed config default.
    match_threshold = read_detections_threshold(det_path)
    if match_threshold is None:
        match_threshold = cfg.stage3.match_threshold

    # ---- Locate video ----
    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(f"No video found for sample '{sample_id}'.")
    video_path = video_files[0]
    vinfo = video_info(video_path)
    total_frames = vinfo["total_frames"]

    # ---- Build tracker ----
    tracker = build_tracker(cfg)
    is_none_tracker = isinstance(tracker, NoneTracker)
    s4 = cfg.stage4
    use_geco2 = cfg.pipeline.detector == "geco2"

    # ---- stage4.backward_tracking: separate tracker instance + rolling
    # frame buffer, built only when the feature is actually used (a second
    # LiteTrack tracker doubles ONNX session memory) -- see
    # BackwardTrackingConfig's docstring for the recovery/stopping rules. ----
    bt_cfg = s4.backward_tracking
    backward_tracker = None
    recent_frames: "OrderedDict[int, np.ndarray]" = OrderedDict()
    if bt_cfg.enabled and not is_none_tracker:
        backward_tracker = build_tracker(cfg)

    # box_refine.apply_in_stage4 piggybacks on verify_interval's own cadence
    # (frames_since_verify below) -- needs no DINOv2 prototype, so it's set
    # up independently of the is_none_tracker/verify_interval extractor
    # block further down.
    br_cfg = cfg.box_refine
    box_refine_active = (
        br_cfg.enabled and br_cfg.apply_in_stage4
        and s4.verify_interval > 0 and not is_none_tracker
    )
    box_refine_segmenter = None
    geco2_refine_detector = None
    geco2_refine_prototype = None
    if box_refine_active and br_cfg.method in ("sam", "sam_dense"):
        from aero_eyes.models.segmentation import MobileSAMSegmenter
        box_refine_segmenter = MobileSAMSegmenter(weights_path=cfg.stage1.segmentation.weights)
    elif box_refine_active and br_cfg.method == "sam2_dense":
        from aero_eyes.models.geco2_detector import load_geco2_detector_and_prototype
        geco2_refine_detector, geco2_refine_prototype = load_geco2_detector_and_prototype(cfg, work_dir)
        if geco2_refine_detector is None:
            log.warning(
                "[Stage4] %s: box_refine.method=sam2_dense but no %s found -- "
                "refinement disabled this run (boxes left unchanged).",
                sample_id, cfg.stage123_geco2.prototype_cache_name,
            )

    # For NoneTracker, we need proposal+matching on every frame. Which
    # detector backs re-detection must match whichever one produced
    # detections.json (legacy DINOv2+YOLO/FastSAM prototype vs GeCo2
    # exemplar tokens) -- they are not interchangeable artifacts.
    proposal_model = None
    extractor = None
    prototype = None
    per_ref_features = []
    geco2_detector = None
    geco2_prototype = None
    geco2_color_sig = None
    if is_none_tracker:
        if use_geco2:
            geco2_detector, geco2_prototype, geco2_color_sig = _load_geco2(cfg, sample_id, work_dir)
        else:
            from aero_eyes.models.features import build_feature_extractor
            from aero_eyes.models.proposals import build_proposal_model
            from aero_eyes.utils.io import read_prototype

            proposal_model = build_proposal_model(cfg)
            extractor = build_feature_extractor(cfg)
            proto_path = work_dir / cfg.stage1.prototype.cache_name
            if proto_path.exists():
                prototype, _, per_ref_features = read_prototype(proto_path)
    elif s4.verify_interval > 0:
        # An active tracker (builtin/litetrack) doesn't need extractor/
        # prototype for tracking itself, but verify_interval's periodic
        # re-check (_track_similarity below) does -- load them here, from
        # prototype.npz (the DINOv2 embedding space), whenever it exists.
        # Always exists for the legacy pipeline; for pipeline.detector=geco2
        # it only exists when stage123_geco2.cosine_rescore.enabled built
        # one (via stage1.run_stage1) -- otherwise there is no DINOv2
        # embedding space to re-verify against, and this degrades to a
        # no-op (logged once) rather than erroring.
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.utils.io import read_prototype

        proto_path = work_dir / cfg.stage1.prototype.cache_name
        if proto_path.exists():
            extractor = build_feature_extractor(cfg)
            prototype, _, per_ref_features = read_prototype(proto_path)
        else:
            log.warning(
                "[Stage4] %s: stage4.verify_interval=%d but no prototype.npz found at %s -- "
                "periodic re-verification unavailable this run (needs a DINOv2 embedding "
                "space; plain pipeline.detector=geco2 without "
                "stage123_geco2.cosine_rescore.enabled doesn't build one).",
                sample_id, s4.verify_interval, proto_path,
            )

    # geco2_redetect_cosine_filter needs the same DINOv2 extractor/prototype
    # as verify_interval above, but on a different trigger (use_geco2, not
    # tracker-active-ness) -- reuse it if verify_interval already loaded it,
    # otherwise load it here. Covers tracker=none (which skips the
    # if/elif above entirely) and verify_interval=0 (which skips the elif).
    if use_geco2 and s4.geco2_redetect_cosine_filter and extractor is None:
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.utils.io import read_prototype

        proto_path = work_dir / cfg.stage1.prototype.cache_name
        if proto_path.exists():
            extractor = build_feature_extractor(cfg)
            prototype, _, per_ref_features = read_prototype(proto_path)
        else:
            log.warning(
                "[Stage4] %s: stage4.geco2_redetect_cosine_filter=true but no prototype.npz "
                "found at %s -- cosine filter unavailable this run (needs "
                "stage123_geco2.cosine_rescore.enabled to have built one).",
                sample_id, proto_path,
            )

    # backward_tracking.validate_against_boundary.cosine_arbitration needs
    # the same DINOv2 extractor/prototype as verify_interval above, loaded
    # here if nothing else already did -- see CosineArbitrationConfig.
    # Gated on validate_against_boundary TOO, not just cosine_arbitration.
    # enabled -- _arbitrate_by_cosine is only ever CALLED from inside the
    # validate_against_boundary check (boundary_check is None otherwise),
    # so cosine_arbitration.enabled alone would load a DINOv2 model that
    # never actually gets used if that flag is off.
    bt_ca_cfg = s4.backward_tracking.cosine_arbitration
    bt_ca_active = bt_ca_cfg.enabled and s4.backward_tracking.validate_against_boundary
    if bt_ca_active and extractor is None:
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.utils.io import read_prototype

        proto_path = work_dir / cfg.stage1.prototype.cache_name
        if proto_path.exists():
            extractor = build_feature_extractor(cfg)
            prototype, _, per_ref_features = read_prototype(proto_path)
        else:
            log.warning(
                "[Stage4] %s: backward_tracking.cosine_arbitration.enabled=true but no "
                "prototype.npz found at %s -- arbitration unavailable this run (falls back to "
                "always discarding the backward segment on disagreement).",
                sample_id, proto_path,
            )

    # keep_tracking_on_missed_keyframe.validate_against_next_keyframe.
    # cosine_arbitration -- same shared model, same gating pattern (needs
    # ITS OWN trigger flag too, not just cosine_arbitration.enabled -- see
    # bt_ca_active's comment above for why).
    kt_ca_cfg = s4.keep_tracking_on_missed_keyframe.cosine_arbitration
    kt_ca_active = kt_ca_cfg.enabled and s4.keep_tracking_on_missed_keyframe.validate_against_next_keyframe
    if kt_ca_active and extractor is None:
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.utils.io import read_prototype

        proto_path = work_dir / cfg.stage1.prototype.cache_name
        if proto_path.exists():
            extractor = build_feature_extractor(cfg)
            prototype, _, per_ref_features = read_prototype(proto_path)
        else:
            log.warning(
                "[Stage4] %s: keep_tracking_on_missed_keyframe.cosine_arbitration.enabled=true "
                "but no prototype.npz found at %s -- arbitration unavailable this run.",
                sample_id, proto_path,
            )

    def _load_arbitration_prototype(ca_cfg, active: bool) -> tuple:
        """Shared by both cosine_arbitration instances above:
        use_adaptive_prototype picks prototype_adapted.npz (stage3.
        dynamic_prototype's final state -- original refs + whatever it
        appended) instead of the original-only prototype/per_ref_features.
        Returns its OWN (prototype, per_ref_features) pair so
        verify_interval/geco2_redetect_cosine_filter (which share the
        plain `prototype`/`per_ref_features` above) are never affected."""
        result_prototype, result_refs = prototype, per_ref_features
        if active and ca_cfg.use_adaptive_prototype:
            from aero_eyes.utils.io import read_prototype

            adapted_path = work_dir / "prototype_adapted.npz"
            if adapted_path.exists():
                result_prototype, _, result_refs = read_prototype(adapted_path)
            else:
                log.warning(
                    "[Stage4] %s: cosine_arbitration.use_adaptive_prototype=true but no "
                    "prototype_adapted.npz found at %s (needs stage3.dynamic_prototype.enabled "
                    "to have produced one) -- falling back to the original prototype.npz.",
                    sample_id, adapted_path,
                )
        return result_prototype, result_refs

    bt_ca_prototype, bt_ca_per_ref_features = _load_arbitration_prototype(bt_ca_cfg, bt_ca_active)
    kt_ca_prototype, kt_ca_per_ref_features = _load_arbitration_prototype(kt_ca_cfg, kt_ca_active)

    # ---- Kalman motion-plausibility check (active tracker only -- see
    # KalmanMotionCheckConfig for why this doesn't apply to tracker=none,
    # which has no continuous track for motion to be judged against) ----
    motion_kf = None
    if not is_none_tracker and s4.kalman_motion_check.enabled:
        from aero_eyes.utils.motion_drift_check import BoxDriftCheck
        motion_kf = BoxDriftCheck(window_frames=s4.kalman_motion_check.window_frames)

    # ---- keep_tracking_on_missed_keyframe.validate_against_next_keyframe:
    # dedicated BoxDriftCheck instance (independent of kalman_motion_check
    # above, which may be disabled) fed the CURRENT track's positions every
    # tracked frame regardless of kept/not-kept status, so it always holds
    # the trend since the last trusted anchor -- see
    # KeepTrackingOnMissedKeyframeConfig's docstring for why this checks
    # motion plausibility instead of leaning on cosine/appearance. ----
    kt_cfg = s4.keep_tracking_on_missed_keyframe
    missed_kf_drift = None
    if not is_none_tracker and kt_cfg.enabled and kt_cfg.validate_against_next_keyframe:
        from aero_eyes.utils.motion_drift_check import BoxDriftCheck
        missed_kf_drift = BoxDriftCheck(window_frames=kt_cfg.window_frames)

    # ---- Video writer for visualizations ----
    writer = None
    if cfg.runtime.save_visualizations:
        viz_dir = work_dir / "viz" / "stage4"
        viz_dir.mkdir(parents=True, exist_ok=True)
        writer = AnnotatedVideoWriter(
            path=viz_dir / "tracking.mp4",
            fps=vinfo["fps"] or 25.0,
            width=vinfo["width"],
            height=vinfo["height"],
        )

    # ---- Main loop ----
    kf_set = set(detections.keys())
    tracks: dict[int, Box | None] = {}
    tracker_active = False
    # Human-readable reason the LAST tracker_active=False transition
    # happened -- surfaced by the keep_tracking_on_missed_keyframe give-up
    # log below so "no active track to fall back on" doesn't hide WHY the
    # track died (low confidence vs. track_age vs. drift check vs. cosine
    # re-verification all look identical from that log alone otherwise).
    track_lost_reason: str | None = None
    track_age = 0
    frames_since_verify = 0
    # Temporary diagnostic counters for box_refine.apply_in_stage4 -- this
    # path had no visibility at all (unlike scripts/check_box_refine_effect.py,
    # which counts changed/rejected boxes for the apply_in_stage3 path).
    br_attempts = 0
    br_changed = 0
    confirm_cfg = s4.confirm_detections
    confirmer = (
        _DetectionConfirmer(confirm_cfg.required_hits, confirm_cfg.iou_threshold)
        if confirm_cfg.enabled else None
    )

    # frame_idx of the first frame in an ONGOING keep_tracking_on_missed_keyframe
    # segment still awaiting validation (None = no pending segment). Set the
    # moment a missed keyframe first gets kept-through; resolved (validated
    # or cleared) the next time an INDEPENDENT box becomes available -- see
    # _resolve_pending_kept_segment below.
    kept_segment_start: int | None = None

    def _clear_pending_kept_segment() -> None:
        """No independent box available to validate a pending segment
        against (track went fully inactive without a fresh re-init) --
        leave its already-written frames as-is, just stop treating it as
        open so a LATER, unrelated segment isn't validated against it."""
        nonlocal kept_segment_start
        if kept_segment_start is not None:
            log.info(
                "[Stage4] %s: keep_tracking_on_missed_keyframe segment starting at frame %d "
                "ended without a fresh re-detect to validate against -- kept as-is, unvalidated",
                sample_id, kept_segment_start,
            )
        kept_segment_start = None

    def _arbitrate_kept_segment_by_cosine(seg_start: int, fresh_box: Box) -> bool:
        """keep_tracking_on_missed_keyframe.validate_against_next_keyframe.
        cosine_arbitration (EXPERIMENTAL, opt-in): called only when the
        motion check already flagged a disagreement between the kept-
        through segment and `fresh_box` (the independent detection at the
        CURRENT frame_idx). Unlike backward_tracking's version, `fresh_box`
        has NOT been written to tracks[] yet (the caller does that only
        after this returns) and there is no `recent_frames` buffer to pull
        an older frame's pixels from here (that buffer only exists when
        backward_tracking.enabled) -- so both crops are taken from the
        CURRENT frame_bgr: `fresh_box` as given, and the kept segment's own
        LAST box (tracks[seg_start's predecessor], i.e. frame_idx - 1)
        re-used as an approximate hypothesis for this same frame (valid
        as long as the object hasn't moved far in one frame). Returns True
        if the kept segment's hypothesis scored higher.
        """
        local_kt_ca_cfg = kt_cfg.cosine_arbitration
        kept_box = tracks.get(frame_idx - 1)
        if not local_kt_ca_cfg.enabled or extractor is None or kt_ca_prototype is None or kept_box is None:
            return False
        pooling = (
            cfg.accuracy.cheap_boosters.multi_ref_pooling
            if local_kt_ca_cfg.pooling == "match_stage3" else local_kt_ca_cfg.pooling
        )
        kept_sim = _track_similarity(
            frame_bgr, kept_box, extractor, kt_ca_prototype, kt_ca_per_ref_features, cfg, pooling=pooling,
        )
        fresh_sim = _track_similarity(
            frame_bgr, fresh_box, extractor, kt_ca_prototype, kt_ca_per_ref_features, cfg, pooling=pooling,
        )
        if kept_sim is None or fresh_sim is None or kept_sim <= fresh_sim:
            return False
        log.info(
            "[Stage4] %s: keep_tracking_on_missed_keyframe cosine arbitration -- kept-through "
            "segment (sim=%.3f) scored higher than frame %d's independent detection (sim=%.3f) "
            "-- keeping the kept-through segment%s",
            sample_id, kept_sim, frame_idx, fresh_sim,
            ", overriding (rejecting) the independent detection this frame"
            if local_kt_ca_cfg.override_boundary_on_win else "",
        )
        return True

    def _resolve_pending_kept_segment(fresh_box: Box) -> bool:
        """Call whenever an INDEPENDENT fresh box becomes available (a real
        keyframe detection, or a successful re-detect) while a
        keep_tracking_on_missed_keyframe segment is pending. Checks
        `fresh_box` for motion-plausibility against the trend fitted from
        the kept segment's OWN tracked positions; if implausible, every
        frame in [kept_segment_start, frame_idx) is retroactively marked
        absent instead of keeping a track that likely drifted onto the
        wrong object -- UNLESS cosine_arbitration is enabled and the kept
        segment's own score wins (see _arbitrate_kept_segment_by_cosine),
        in which case the segment is kept instead.

        Returns True if `fresh_box` should be trusted by the caller
        (proceed with the normal tracker re-init this frame) -- always
        true when nothing was pending, when the motion check passed, or
        when arbitration is off/loses. Returns False only when
        cosine_arbitration.override_boundary_on_win won: the kept-through
        segment is trusted INSTEAD of `fresh_box` this frame, so the
        caller should treat this frame as if the detection never
        happened (kept_segment_start stays open, tracker_active/the
        tracker's own state are left untouched, so tracking continues
        from the kept-through segment's state into subsequent frames).
        """
        nonlocal kept_segment_start
        if kept_segment_start is None:
            return True
        segment_len = frame_idx - kept_segment_start
        if missed_kf_drift is not None:
            plausible = missed_kf_drift.check_and_update(fresh_box, kt_cfg.max_dist_ratio)
            if not plausible:
                if _arbitrate_kept_segment_by_cosine(kept_segment_start, fresh_box):
                    if kt_cfg.cosine_arbitration.override_boundary_on_win:
                        return False  # kept_segment_start stays open, fresh_box rejected this frame
                    kept_segment_start = None
                    return True  # kept segment AND fresh_box both survive, untouched
                for fi in range(kept_segment_start, frame_idx):
                    tracks[fi] = None
                log.info(
                    "[Stage4] %s: keep_tracking_on_missed_keyframe segment [%d, %d) failed "
                    "motion-plausibility check against frame %d's independent detection "
                    "-- retroactively marked absent", sample_id, kept_segment_start, frame_idx, frame_idx,
                )
                kept_segment_start = None
                return True
        log.info(
            "[Stage4] %s: keep_tracking_on_missed_keyframe kept %d frame(s) [%d, %d) through a "
            "missed keyframe, validated at frame %d",
            sample_id, segment_len, kept_segment_start, frame_idx, frame_idx,
        )
        kept_segment_start = None
        return True

    def _arbitrate_by_cosine(boundary_fi: int, boundary_box: Box, recovered_frames: list[int]) -> bool:
        """backward_tracking.validate_against_boundary.cosine_arbitration
        (EXPERIMENTAL, opt-in): called only when the motion check already
        flagged a disagreement at `boundary_fi`. Re-embeds the LAST
        backward-recovered box (closest to the disagreement) and
        `boundary_box` with DINOv2, scores both against the SAME
        prototype (bt_ca_prototype/bt_ca_per_ref_features -- original-only
        or dynamic_prototype-adapted, per cosine_arbitration.
        use_adaptive_prototype), and returns True (keep the backward
        segment -- the caller skips its normal discard) if the backward
        side scored higher -- see CosineArbitrationConfig's own docstring
        for what happens to `boundary_box` itself in that case. False
        (including when unavailable/inconclusive) means fall back to the
        normal "discard the backward segment" behavior.
        """
        local_ca_cfg = bt_cfg.cosine_arbitration
        if not local_ca_cfg.enabled or not recovered_frames or extractor is None or bt_ca_prototype is None:
            return False
        pooling = (
            cfg.accuracy.cheap_boosters.multi_ref_pooling
            if local_ca_cfg.pooling == "match_stage3" else local_ca_cfg.pooling
        )
        last_fi = recovered_frames[-1]
        backward_box = tracks[last_fi]
        backward_frame = recent_frames.get(last_fi)
        boundary_frame = recent_frames.get(boundary_fi)
        if backward_frame is None or boundary_frame is None:
            return False
        backward_sim = _track_similarity(
            backward_frame, backward_box, extractor, bt_ca_prototype, bt_ca_per_ref_features, cfg, pooling=pooling,
        )
        boundary_sim = _track_similarity(
            boundary_frame, boundary_box, extractor, bt_ca_prototype, bt_ca_per_ref_features, cfg, pooling=pooling,
        )
        if backward_sim is None or boundary_sim is None or backward_sim <= boundary_sim:
            return False
        log.info(
            "[Stage4] %s: backward_tracking cosine arbitration -- backward segment (sim=%.3f) "
            "scored higher than the boundary box at frame %d (sim=%.3f) -- keeping backward "
            "segment%s",
            sample_id, backward_sim, boundary_fi, boundary_sim,
            ", discarding the boundary box" if local_ca_cfg.override_boundary_on_win else "",
        )
        if local_ca_cfg.override_boundary_on_win:
            tracks[boundary_fi] = None
        return True

    def _run_backward_recovery(anchor_frame_idx: int, anchor_box: Box) -> None:
        """stage4.backward_tracking: called right after EVERY fresh lock
        (tracker_active going False -> True, whether this is the video's
        first lock or a re-lock after mid-video track loss). Runs
        `backward_tracker` from `anchor_box` at `anchor_frame_idx`
        backward through `recent_frames`, filling in tracks[fi] for
        currently-absent frames immediately before it -- stops at the
        first frame that already has a box (a previous segment's
        territory, never overwritten), frame 0,
        bt_cfg.max_backward_frames back, a frame missing from the buffer
        (ran further back than what's been read), or the backward
        tracker's own confidence dropping below tracker_conf_threshold.
        No-op if the feature is off or nothing is buffered yet.

        bt_cfg.validate_against_boundary: "stops at a frame that already
        has a box" only guarantees the recovered segment never OVERWRITES
        that boundary -- not that it actually agrees with it (the backward
        tracker could have drifted onto a confuser the whole way and just
        happened to run out of frames right next to a real, independent
        box). When enabled, a BoxDriftCheck trend is fit from the segment's
        OWN recovered positions as it goes; the boundary box is checked
        against that trend the moment recovery reaches it, and if
        implausible, every frame this call recovered is reverted to absent
        instead of being kept sitting next to a box it disagrees with.
        """
        if backward_tracker is None:
            return
        lo = max(0, anchor_frame_idx - bt_cfg.max_backward_frames)
        backward_tracker.init(recent_frames[anchor_frame_idx], anchor_box)

        boundary_check = None
        if bt_cfg.validate_against_boundary:
            from aero_eyes.utils.motion_drift_check import BoxDriftCheck
            boundary_check = BoxDriftCheck(window_frames=bt_cfg.window_frames)
            boundary_check.init(anchor_box)

        recovered_frames: list[int] = []
        fi = anchor_frame_idx - 1
        while fi >= lo:
            existing = tracks.get(fi)
            if existing is not None:
                if boundary_check is not None and not boundary_check.check_and_update(
                    existing, bt_cfg.max_dist_ratio,
                ):
                    if not _arbitrate_by_cosine(fi, existing, recovered_frames):
                        for rf in recovered_frames:
                            tracks[rf] = None
                        log.info(
                            "[Stage4] %s: backward_tracking segment [%d, %d) failed "
                            "motion-plausibility check against the real box at frame %d "
                            "-- likely drifted onto a confuser, discarding the whole segment",
                            sample_id, anchor_frame_idx - len(recovered_frames), anchor_frame_idx, fi,
                        )
                    return
                break
            if fi not in recent_frames:
                break
            box, conf = backward_tracker.update(recent_frames[fi])
            if box is None or conf < s4.tracker_conf_threshold:
                break
            tracks[fi] = box
            recovered_frames.append(fi)
            if boundary_check is not None:
                boundary_check.check_and_update(box, bt_cfg.max_dist_ratio)
            fi -= 1
        if recovered_frames:
            log.info(
                "[Stage4] %s: backward_tracking recovered %d frame(s) [%d, %d) "
                "before the lock at frame %d",
                sample_id, len(recovered_frames), anchor_frame_idx - len(recovered_frames),
                anchor_frame_idx, anchor_frame_idx,
            )

    try:
        for frame_idx, frame_bgr in frame_iterator(video_path):
            box_out: Box | None = None
            source = "none"

            if backward_tracker is not None:
                recent_frames[frame_idx] = frame_bgr.copy()
                while len(recent_frames) > bt_cfg.max_backward_frames + 1:
                    recent_frames.popitem(last=False)

            if is_none_tracker:
                # Re-detect every frame
                if use_geco2:
                    raw_box, source = _detect_on_frame_geco2(
                        frame_bgr, geco2_detector, geco2_prototype, geco2_color_sig, cfg.stage123_geco2.color_postfilter,
                        cosine_extractor=extractor if s4.geco2_redetect_cosine_filter else None,
                        cosine_prototype=prototype if s4.geco2_redetect_cosine_filter else None,
                        per_ref_features=per_ref_features, cfg=cfg, match_threshold=match_threshold,
                    )
                else:
                    raw_box, source = _detect_on_frame(
                        frame_bgr, frame_idx, proposal_model, extractor,
                        prototype, per_ref_features, cfg, match_threshold
                    )
                if confirmer is None:
                    box_out = raw_box
                elif raw_box is None:
                    confirmer.reset()
                    box_out, source = None, "none"
                else:
                    box_out = confirmer.offer(raw_box)
                    source = "detect" if box_out is not None else "none"
            else:
                is_keyframe = frame_idx in kf_set
                dets = detections[frame_idx] if is_keyframe else None
                if is_keyframe and dets:
                    # Initialize or re-initialize tracker from detection
                    candidate = max(dets, key=lambda d: d.similarity).box
                    confirmed = confirmer.offer(candidate) if confirmer is not None else candidate
                    if confirmed is not None:
                        # This IS the independent box a pending
                        # keep_tracking_on_missed_keyframe segment (if any)
                        # was waiting for -- validate/resolve it BEFORE
                        # trusting this detection over the tracker's state.
                        # False only means cosine_arbitration overrode it in
                        # favor of the kept-through segment -- skip the
                        # reinit below entirely; box_out stays None this
                        # frame and the existing tracker/kept_segment_start
                        # are left untouched (see that function's own
                        # docstring).
                        if _resolve_pending_kept_segment(confirmed):
                            was_inactive = not tracker_active
                            tracker.init(frame_bgr, confirmed)
                            if motion_kf is not None:
                                motion_kf.init(confirmed)
                            if missed_kf_drift is not None:
                                missed_kf_drift.init(confirmed)
                            tracker_active = True
                            track_age = 0
                            frames_since_verify = 0
                            box_out = confirmed
                            source = "detect"
                            # stage4.backward_tracking: this is a FRESH lock
                            # (no track was active a moment ago) -- recover
                            # any frames right before it where the object
                            # was present but not yet detected. A routine
                            # keyframe re-anchor of an ALREADY-active track
                            # skips this (was_inactive False) -- nothing to
                            # recover there.
                            if was_inactive:
                                _run_backward_recovery(frame_idx, confirmed)
                    else:
                        tracker_active = False
                        track_lost_reason = "keyframe detection failed confirmer's consecutive-hit check"
                        _clear_pending_kept_segment()
                # stage4.keep_tracking_on_missed_keyframe: a keyframe with no
                # surviving detection falls through to the SAME
                # tracker.update() path as a non-keyframe, as long as a track
                # is already active -- see that config field's own docstring
                # for why (a missed DETECTION at one frame shouldn't override
                # the tracker's own live state, which every check below still
                # gets to judge on its own terms).
                elif tracker_active and (not is_keyframe or kt_cfg.enabled):
                    if is_keyframe and kept_segment_start is None:
                        kept_segment_start = frame_idx
                        log.info(
                            "[Stage4] %s: frame %d: keyframe had no surviving detection -- "
                            "keep_tracking_on_missed_keyframe keeping the active track alive "
                            "through it", sample_id, frame_idx,
                        )
                    box, conf = tracker.update(frame_bgr)
                    track_age += 1
                    frames_since_verify += 1
                    if missed_kf_drift is not None and box is not None:
                        # Feed the trend continuously (whether or not this
                        # frame ends up part of a kept-through segment) so
                        # whenever validation DOES trigger, it has the
                        # richest history available -- see missed_kf_drift's
                        # own setup comment. Return value unused here; only
                        # the check at _resolve_pending_kept_segment's call
                        # site (against an INDEPENDENT box) matters.
                        missed_kf_drift.check_and_update(box, kt_cfg.max_dist_ratio)
                    track_ok = (conf >= s4.tracker_conf_threshold
                                and track_age <= s4.max_track_age
                                and box is not None)
                    if not track_ok:
                        if box is None:
                            track_lost_reason = "tracker.update returned no box"
                        elif conf < s4.tracker_conf_threshold:
                            track_lost_reason = f"tracker confidence {conf:.3f} < threshold {s4.tracker_conf_threshold:.3f}"
                        else:
                            track_lost_reason = f"track_age {track_age} exceeded max_track_age {s4.max_track_age}"
                        log.debug(
                            "[Stage4] frame %d: tracker.update rejected "
                            "(conf=%.3f threshold=%.3f, track_age=%d max=%d, "
                            "box_is_none=%s) -- try re-detect",
                            frame_idx, conf, s4.tracker_conf_threshold,
                            track_age, s4.max_track_age, box is None,
                        )
                    # Set below (stage4.absence_check) when verify_interval's
                    # cosine similarity comes back so far below
                    # match_threshold that re-detect is skipped outright
                    # instead of attempted -- see that block for why.
                    skip_redetect = False

                    # Drift-plausibility check (stage4.kalman_motion_check,
                    # opt-in): cheap enough to run every frame, unlike
                    # verify_interval's DINOv2 cosine check below -- catches
                    # a confuser that looks similar enough to pass cosine but
                    # sits somewhere the box's own recent trajectory couldn't
                    # plausibly have led to. Runs first so a frame it already
                    # flags skips the more expensive cosine check too
                    # (track_ok is already False by then).
                    if track_ok and box is not None and motion_kf is not None:
                        if not motion_kf.check_and_update(box, s4.kalman_motion_check.max_dist_ratio):
                            log.debug(
                                "[Stage4] frame %d: track failed drift-plausibility check "
                                "(diverged from recent trajectory) -- forcing re-detect", frame_idx,
                            )
                            track_ok = False
                            track_lost_reason = "kalman_motion_check: diverged from recent trajectory"

                    # OpenCV's own tracker confidence is a near-constant
                    # placeholder (BuiltinTracker.update always returns 0.9
                    # on success -- OpenCV exposes no real confidence), so it
                    # cannot by itself tell a drifted lock from a correct
                    # one. Every verify_interval frames, cross-check the
                    # tracked crop against the DINOv2 prototype (the same
                    # embedding/threshold used to decide the original match)
                    # so a silently-drifted track gets caught within a few
                    # frames instead of persisting for the full
                    # max_track_age. 0 (default) = never runs, i.e. exactly
                    # the original conf/age-only logic.
                    if (track_ok and box is not None and s4.verify_interval > 0
                            and frames_since_verify >= s4.verify_interval):
                        frames_since_verify = 0

                        if extractor is not None and prototype is not None:
                            sim = _track_similarity(
                                frame_bgr, box, extractor, prototype,
                                per_ref_features, cfg,
                            )
                            if sim is not None and sim < match_threshold:
                                track_ok = False
                                track_lost_reason = f"verify_interval: re-verification cosine sim {sim:.3f} < match_threshold {match_threshold:.3f}"
                                ac_cfg = s4.absence_check
                                if ac_cfg.enabled and sim < match_threshold * ac_cfg.absence_ratio:
                                    # Similarity isn't just borderline-low --
                                    # it's far enough below match_threshold
                                    # that a re-detect attempt would almost
                                    # certainly either fail outright or lock
                                    # onto an unrelated confuser (the object
                                    # has most likely genuinely left the
                                    # frame). Skip re-detect entirely and
                                    # report absent this frame instead of
                                    # giving it a chance to extend the track
                                    # past a real departure.
                                    skip_redetect = True
                                    track_lost_reason = f"absence_check: sim {sim:.3f} well below absence threshold {match_threshold * ac_cfg.absence_ratio:.3f}, object likely gone"
                                    log.debug(
                                        "[Stage4] frame %d: track failed re-verification "
                                        "(sim=%.3f well below absence threshold %.3f) -- "
                                        "object likely gone, skipping re-detect", frame_idx,
                                        sim, match_threshold * ac_cfg.absence_ratio,
                                    )
                                else:
                                    log.debug(
                                        "[Stage4] frame %d: track failed re-verification "
                                        "(sim=%.3f < %.3f, likely drifted) -- forcing re-detect",
                                        frame_idx, sim, match_threshold,
                                    )

                        # box_refine.apply_in_stage4 (opt-in, piggybacked on
                        # this same verify_interval cadence): sharpen the
                        # tracked box to the actual object silhouette and
                        # re-anchor the tracker to it, independent of
                        # whether the cosine re-verification above ran at
                        # all (no DINOv2 prototype needed for this).
                        if track_ok and box_refine_active:
                            br_attempts += 1
                            box_before_refine = box
                            if br_cfg.method == "sam_dense":
                                from aero_eyes.utils.box_refine import refine_boxes_dense
                                box = refine_boxes_dense(
                                    box_refine_segmenter, frame_bgr, [box],
                                    min_iou_with_original=br_cfg.min_iou_with_original,
                                    context_margin=br_cfg.context_margin,
                                    adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                                )[0]
                            elif br_cfg.method == "sam2_dense":
                                if geco2_refine_detector is not None:
                                    from aero_eyes.utils.box_refine import apply_iou_gate
                                    refined = geco2_refine_detector.sam2_refine_boxes(
                                        frame_bgr, geco2_refine_prototype, [box],
                                    )[0]
                                    box = apply_iou_gate([refined], [box], br_cfg.min_iou_with_original)[0]
                            else:
                                from aero_eyes.utils.box_refine import refine_box
                                box = refine_box(
                                    br_cfg.method, frame_bgr, box, br_cfg.context_margin,
                                    segmenter=box_refine_segmenter, min_iou_with_original=br_cfg.min_iou_with_original,
                                    adaptive_context_margin_cfg=br_cfg.adaptive_context_margin,
                                )
                            if box != box_before_refine:
                                br_changed += 1
                            tracker.init(frame_bgr, box)
                            if motion_kf is not None:
                                motion_kf.init(box)

                    if track_ok:
                        box_out = box
                        source = "track"
                    elif skip_redetect:
                        # stage4.absence_check: re-verification similarity
                        # was far enough below match_threshold that the
                        # object is judged almost certainly gone -- skip
                        # re-detect and report absent outright rather than
                        # giving it a chance to lock onto a confuser and
                        # extend the track past a real departure.
                        tracker_active = False
                        box_out = None
                        source = "none"
                        if confirmer is not None:
                            confirmer.reset()
                        _clear_pending_kept_segment()
                    else:
                        # Confidence too low, track too old, or failed
                        # re-verification — try re-detect
                        tracker_active = False
                        if use_geco2:
                            if geco2_detector is None:
                                geco2_detector, geco2_prototype, geco2_color_sig = _load_geco2(cfg, sample_id, work_dir)
                            raw_box, source = _detect_on_frame_geco2(
                                frame_bgr, geco2_detector, geco2_prototype, geco2_color_sig, cfg.stage123_geco2.color_postfilter,
                                cosine_extractor=extractor if s4.geco2_redetect_cosine_filter else None,
                                cosine_prototype=prototype if s4.geco2_redetect_cosine_filter else None,
                                per_ref_features=per_ref_features, cfg=cfg, match_threshold=match_threshold,
                            )
                        else:
                            # Lazy-init for re-detect fallback -- guarded
                            # independently (not both under one "proposal_model
                            # is None" check) since verify_interval above may
                            # have already built extractor/prototype for its
                            # own periodic re-check, in which case only
                            # proposal_model (never needed by verify_interval)
                            # remains to be built here.
                            if proposal_model is None:
                                from aero_eyes.models.proposals import build_proposal_model
                                proposal_model = build_proposal_model(cfg)
                            if extractor is None:
                                from aero_eyes.models.features import build_feature_extractor
                                from aero_eyes.utils.io import read_prototype
                                extractor = build_feature_extractor(cfg)
                                proto_path = work_dir / cfg.stage1.prototype.cache_name
                                if proto_path.exists():
                                    prototype, _, per_ref_features = read_prototype(proto_path)

                            raw_box, source = _detect_on_frame(
                                frame_bgr, frame_idx, proposal_model, extractor,
                                prototype, per_ref_features, cfg, match_threshold
                            )
                        if confirmer is None:
                            box_out = raw_box
                        elif raw_box is None:
                            confirmer.reset()
                            box_out = None
                        else:
                            box_out = confirmer.offer(raw_box)
                            if box_out is None:
                                source = "none"
                        if box_out is not None:
                            # Successful re-detect IS the independent box a
                            # pending kept-through segment was waiting for.
                            # False here means cosine_arbitration overrode
                            # it in favor of the kept-through segment --
                            # reject this re-detect (report absent this
                            # frame instead) and leave kept_segment_start
                            # OPEN for a later attempt, rather than clearing
                            # it -- unlike the keyframe call site above,
                            # there is no still-running tracker to fall
                            # back on here (the original track already
                            # failed before this re-detect was even tried).
                            if _resolve_pending_kept_segment(box_out):
                                tracker.init(frame_bgr, box_out)
                                if motion_kf is not None:
                                    motion_kf.init(box_out)
                                if missed_kf_drift is not None:
                                    missed_kf_drift.init(box_out)
                                tracker_active = True
                                track_age = 0
                                frames_since_verify = 0
                                # Always a fresh lock here (tracker_active
                                # was forced False a few lines up before
                                # this re-detect attempt) -- recover any
                                # frames right before it where the object
                                # went undetected.
                                _run_backward_recovery(frame_idx, box_out)
                            else:
                                box_out = None
                                source = "none"
                        else:
                            # Re-detect also failed -- nothing independent
                            # to validate a pending segment against; leave
                            # its frames as-is (see _clear_pending_kept_segment).
                            _clear_pending_kept_segment()
                elif is_keyframe:
                    # Keyframe had no surviving detection, and either no
                    # track was active to fall back on, or
                    # keep_tracking_on_missed_keyframe is off -- give up on
                    # this keyframe exactly like the original logic did.
                    if not tracker_active:
                        reason = f"no active track to fall back on ({track_lost_reason or 'never locked on'})"
                    elif not kt_cfg.enabled:
                        reason = "keep_tracking_on_missed_keyframe is disabled"
                    else:
                        reason = "unknown"
                    log.info(
                        "[Stage4] %s: frame %d: keyframe had no surviving detection -- "
                        "giving up on it (%s)", sample_id, frame_idx, reason,
                    )
                    tracker_active = False
                    if confirmer is not None:
                        confirmer.reset()
                    _clear_pending_kept_segment()

            tracks[frame_idx] = box_out

            if writer is not None and box_out is not None:
                vis = vizmod.draw_frame_annotation(frame_bgr, box_out, source, frame_idx)
                writer.write(vis)
            elif writer is not None:
                writer.write(frame_bgr)

            log.debug("[Stage4] frame %d: %s box=%s", frame_idx, source, box_out)
    finally:
        if writer is not None:
            writer.release()

    write_tracks(tracks, tracks_path)
    elapsed = time.time() - t0
    present = sum(1 for v in tracks.values() if v is not None)
    log.info("[Stage4] %s done in %.1fs -> %s (%d/%d frames with box)",
             sample_id, elapsed, tracks_path, present, total_frames)
    if box_refine_active:
        log.info(
            "[Stage4] %s: box_refine.apply_in_stage4 fired %d time(s), actually "
            "changed the box %d time(s) (rest left unchanged by "
            "min_iou_with_original=%.2f gate or no plausible mask found)",
            sample_id, br_attempts, br_changed, br_cfg.min_iou_with_original,
        )
    return tracks_path


def _track_similarity(
    frame_bgr,
    box: Box,
    extractor,
    prototype,
    per_ref_features: list,
    cfg,
    pooling: str = "mean",
) -> float | None:
    """Re-embed the crop the tracker is CURRENTLY reporting and return its
    cosine similarity to the target, using the same DINOv2 embedding space
    Stage 3 used to decide the original match -- the real correctness check
    BuiltinTracker's own fixed placeholder confidence (see
    aero_eyes/models/trackers.py) cannot provide. Used by
    stage4.verify_interval's periodic re-check (always "mean", its own
    established behavior, unchanged) and by backward_tracking.
    validate_against_boundary.cosine_arbitration (which passes
    accuracy.cheap_boosters.multi_ref_pooling, to match Stage 3's own
    pooling convention -- see that config field's own docstring).

    Returns the raw similarity (not a match/no-match bool) so the caller
    can apply BOTH match_threshold (still the same object?) and, if
    stage4.absence_check is enabled, a stricter absence_threshold below it
    (is the object almost certainly gone, vs. merely a borderline/drifted
    match worth a re-detect attempt?) -- see stage4.absence_check's own
    config docstring for why that distinction matters. None if the crop
    couldn't be embedded at all (e.g. degenerate box) -- not evidence of
    absence, just a failed measurement.

    pooling: "mean" (default, this function's original/only behavior) or
    "max" across per_ref_features when multi-ref is active -- see
    aero_eyes.stages.stage3._pool_sims, the same convention Stage 3 itself
    uses for its OWN matching.
    """
    feats = extractor.extract_crops(
        frame_bgr, [box],
        pad_ratio=cfg.stage2.candidate.feature_crop_pad,
        batch_size=1,
    )
    if feats.shape[0] == 0:
        return None

    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )
    if use_multi_ref:
        sims = [float(feats[0] @ ref_feat) for ref_feat in per_ref_features]
        return max(sims) if pooling == "max" else float(np.mean(sims))
    return float(feats[0] @ prototype)


def _load_geco2(cfg, sample_id: str, work_dir: Path):
    """Lazily load the GeCo2 detector + its cached exemplar tokens for
    Stage 4 re-detection. Returns (detector, prototype, color_sig) --
    detector/prototype are (None, None) if the exemplar cache from Stage
    1+2+3 (stage123_geco2.py) is missing; color_sig is None if
    color_postfilter is disabled.

    Loading color_sig here (not just in stage123_geco2.py's own keyframe
    loop) matters: without it, Stage 4's OWN re-detection below
    (_detect_on_frame_geco2, triggered whenever the tracker loses
    confidence or ages out -- precisely the highest-risk moment for
    latching onto a same-shape-different-color confuser) would silently
    bypass color filtering entirely, even with color_postfilter.enabled=true.
    """
    from aero_eyes.models.geco2_detector import load_geco2_detector_and_prototype

    detector, prototype = load_geco2_detector_and_prototype(cfg, work_dir)
    if detector is None:
        log.warning(
            "[Stage4] %s not found -- GeCo2 re-detection disabled for this run.",
            work_dir / cfg.stage123_geco2.prototype_cache_name,
        )
        return None, None, None
    color_sig = None
    if cfg.stage123_geco2.color_postfilter.enabled:
        from aero_eyes.stages.stage123_geco2 import build_color_signature
        color_sig = build_color_signature(cfg, sample_id, work_dir)
    return detector, prototype, color_sig


def _detect_on_frame_geco2(
    frame_bgr, detector, prototype, color_sig=None, cpf_cfg=None,
    cosine_extractor=None, cosine_prototype=None, per_ref_features=None,
    cfg=None, match_threshold=None,
):
    """GeCo2-backed equivalent of _detect_on_frame: single best re-detection
    box on one frame, or (None, "none") if nothing passed threshold/NMS or
    the detector/prototype weren't available. Applies the same color
    post-filter as stage123_geco2.py's keyframe loop when color_sig is
    given -- see _load_geco2's docstring for why this path needs it too.

    cosine_extractor/cosine_prototype (stage4.geco2_redetect_cosine_filter):
    GeCo2's own score is relative per-frame, not a similarity to a known
    target, so it sometimes locks onto a confuser instead of reporting
    "not found". When given, every surviving candidate is additionally
    embedded with DINOv2 and dropped unless its cosine similarity to
    cosine_prototype clears match_threshold (the same adaptive/fixed
    threshold Stage 3 used) -- the best GeCo2 score among the SURVIVORS
    wins, or (None, "none") if none survive.
    """
    if detector is None or prototype is None:
        return None, "none"
    boxes = detector.detect_frame(frame_bgr, prototype)
    if color_sig is not None:
        from aero_eyes.stages.stage123_geco2 import apply_color_postfilter
        boxes = apply_color_postfilter(frame_bgr, boxes, color_sig, cpf_cfg)
    if not boxes:
        return None, "none"

    if cosine_extractor is not None and cosine_prototype is not None:
        feats = cosine_extractor.extract_crops(
            frame_bgr, boxes,
            pad_ratio=cfg.stage2.candidate.feature_crop_pad,
            batch_size=cfg.runtime.batch_size,
        )
        use_multi_ref = (
            cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
            and cfg.accuracy.cheap_boosters.multi_reference_embedding
            and per_ref_features
        )
        if use_multi_ref:
            sims = np.mean([feats @ ref_feat for ref_feat in per_ref_features], axis=0)
        else:
            sims = feats @ cosine_prototype
        boxes = [b for b, sim in zip(boxes, sims) if sim >= match_threshold]
        if not boxes:
            return None, "none"

    best = max(boxes, key=lambda b: b.score)
    return best, "detect"


def _detect_on_frame(
    frame_bgr,
    frame_idx: int,
    proposal_model,
    extractor,
    prototype,
    per_ref_features: list,
    cfg,
    match_threshold: float,
):
    """Run proposal + matching on a single frame; return (best_box, source)."""
    if proposal_model is None or extractor is None or prototype is None:
        return None, "none"

    from aero_eyes.utils.geometry import nms, remap_box_from_tile, sahi_tiles

    s2 = cfg.stage2
    h, w = frame_bgr.shape[:2]

    # Proposals
    if s2.sahi.use_sahi:
        tiles = sahi_tiles(w, h, s2.sahi.tile, s2.sahi.overlap)
        all_boxes = []
        for tile in tiles:
            tile_img = frame_bgr[tile.y1:tile.y2, tile.x1:tile.x2]
            if tile_img.size == 0:
                continue
            for b in proposal_model.propose(tile_img):
                all_boxes.append(remap_box_from_tile(b, tile))
        keep = nms(all_boxes, 0.5)
        boxes = [all_boxes[i] for i in keep]
    else:
        boxes = proposal_model.propose(frame_bgr)

    boxes = [b for b in boxes if b.area() >= s2.candidate.min_box_area]
    boxes = boxes[:s2.candidate.max_candidates_per_keyframe]
    if not boxes:
        return None, "none"

    feats = extractor.extract_crops(
        frame_bgr, boxes,
        pad_ratio=s2.candidate.feature_crop_pad,
        batch_size=cfg.runtime.batch_size,
    )

    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )
    if use_multi_ref:
        sims_per_ref = [feats @ ref_feat for ref_feat in per_ref_features]
        sims = np.mean(sims_per_ref, axis=0)
    else:
        sims = feats @ prototype

    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= match_threshold:
        return boxes[best_idx], "detect"
    return None, "none"


def main():
    p = argparse.ArgumentParser(description="Stage 4 — tracking")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--debug", action="store_true", help="Verbose per-frame tracker logging")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage4(cfg, args.sample)


if __name__ == "__main__":
    main()

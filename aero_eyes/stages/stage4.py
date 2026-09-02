"""Stage 4 — Tracking between keyframes.

Flow:  detections.json + video
       -> initialize tracker at each keyframe detection
       -> propagate boxes on intermediate frames
       -> if conf < tau: trigger re-detection
       -> every verify_interval frames (opt-in): re-embed the tracked crop
          and cross-check it against the prototype, forcing a re-detect if
          it no longer matches -- catches silent drift builtin trackers'
          placeholder confidence can't (see _track_still_matches)
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
        # re-check (_track_still_matches below) does -- load them here, from
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
    track_age = 0
    frames_since_verify = 0
    confirm_cfg = s4.confirm_detections
    confirmer = (
        _DetectionConfirmer(confirm_cfg.required_hits, confirm_cfg.iou_threshold)
        if confirm_cfg.enabled else None
    )

    try:
        for frame_idx, frame_bgr in frame_iterator(video_path):
            box_out: Box | None = None
            source = "none"

            if is_none_tracker:
                # Re-detect every frame
                if use_geco2:
                    raw_box, source = _detect_on_frame_geco2(
                        frame_bgr, geco2_detector, geco2_prototype, geco2_color_sig, cfg.stage123_geco2.color_postfilter,
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
                if frame_idx in kf_set:
                    # Initialize or re-initialize tracker from detection
                    dets = detections[frame_idx]
                    if dets:
                        candidate = max(dets, key=lambda d: d.similarity).box
                        confirmed = confirmer.offer(candidate) if confirmer is not None else candidate
                        if confirmed is not None:
                            tracker.init(frame_bgr, confirmed)
                            tracker_active = True
                            track_age = 0
                            frames_since_verify = 0
                            box_out = confirmed
                            source = "detect"
                        else:
                            tracker_active = False
                    else:
                        tracker_active = False
                        if confirmer is not None:
                            confirmer.reset()
                elif tracker_active:
                    box, conf = tracker.update(frame_bgr)
                    track_age += 1
                    frames_since_verify += 1
                    track_ok = (conf >= s4.tracker_conf_threshold
                                and track_age <= s4.max_track_age
                                and box is not None)

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
                            and extractor is not None and prototype is not None
                            and frames_since_verify >= s4.verify_interval):
                        frames_since_verify = 0
                        if not _track_still_matches(
                            frame_bgr, box, extractor, prototype,
                            per_ref_features, cfg, match_threshold,
                        ):
                            log.debug(
                                "[Stage4] frame %d: track failed re-verification "
                                "(likely drifted) -- forcing re-detect", frame_idx,
                            )
                            track_ok = False

                    if track_ok:
                        box_out = box
                        source = "track"
                    else:
                        # Confidence too low, track too old, or failed
                        # re-verification — try re-detect
                        tracker_active = False
                        if use_geco2:
                            if geco2_detector is None:
                                geco2_detector, geco2_prototype, geco2_color_sig = _load_geco2(cfg, sample_id, work_dir)
                            raw_box, source = _detect_on_frame_geco2(
                                frame_bgr, geco2_detector, geco2_prototype, geco2_color_sig, cfg.stage123_geco2.color_postfilter,
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
                            tracker.init(frame_bgr, box_out)
                            tracker_active = True
                            track_age = 0
                            frames_since_verify = 0

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
    return tracks_path


def _track_still_matches(
    frame_bgr,
    box: Box,
    extractor,
    prototype,
    per_ref_features: list,
    cfg,
    match_threshold: float,
) -> bool:
    """Re-embed the crop the tracker is CURRENTLY reporting and check it
    still resembles the target, using the same DINOv2 embedding space and
    threshold Stage 3 used to decide the original match -- the real
    correctness check BuiltinTracker's own fixed placeholder confidence
    (see aero_eyes/models/trackers.py) cannot provide. Used by
    stage4.verify_interval's periodic re-check.
    """
    feats = extractor.extract_crops(
        frame_bgr, [box],
        pad_ratio=cfg.stage2.candidate.feature_crop_pad,
        batch_size=1,
    )
    if feats.shape[0] == 0:
        return False

    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )
    if use_multi_ref:
        sim = float(np.mean([feats[0] @ ref_feat for ref_feat in per_ref_features]))
    else:
        sim = float(feats[0] @ prototype)

    return sim >= match_threshold


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
    from aero_eyes.models.geco2_detector import GeCo2Detector

    proto_path = work_dir / cfg.stage123_geco2.prototype_cache_name
    if not proto_path.exists():
        log.warning(
            "[Stage4] %s not found -- GeCo2 re-detection disabled for this run.",
            proto_path,
        )
        return None, None, None
    detector = GeCo2Detector(cfg)
    prototype = GeCo2Detector.load_prototype(proto_path)
    color_sig = None
    if cfg.stage123_geco2.color_postfilter.enabled:
        from aero_eyes.stages.stage123_geco2 import build_color_signature
        color_sig = build_color_signature(cfg, sample_id, work_dir)
    return detector, prototype, color_sig


def _detect_on_frame_geco2(frame_bgr, detector, prototype, color_sig=None, cpf_cfg=None):
    """GeCo2-backed equivalent of _detect_on_frame: single best re-detection
    box on one frame, or (None, "none") if nothing passed threshold/NMS or
    the detector/prototype weren't available. Applies the same color
    post-filter as stage123_geco2.py's keyframe loop when color_sig is
    given -- see _load_geco2's docstring for why this path needs it too.
    """
    if detector is None or prototype is None:
        return None, "none"
    boxes = detector.detect_frame(frame_bgr, prototype)
    if color_sig is not None:
        from aero_eyes.stages.stage123_geco2 import apply_color_postfilter
        boxes = apply_color_postfilter(frame_bgr, boxes, color_sig, cpf_cfg)
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
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 4 — tracking")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage4(cfg, args.sample)


if __name__ == "__main__":
    main()

"""Stage 4 — Tracking between keyframes.

Flow:  detections.json + video
       -> initialize tracker at each keyframe detection
       -> propagate boxes on intermediate frames
       -> if conf < tau: trigger re-detection
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
    if is_none_tracker:
        if use_geco2:
            geco2_detector, geco2_prototype = _load_geco2(cfg, work_dir)
        else:
            from aero_eyes.models.features import build_feature_extractor
            from aero_eyes.models.proposals import build_proposal_model
            from aero_eyes.utils.io import read_prototype

            proposal_model = build_proposal_model(cfg)
            extractor = build_feature_extractor(cfg)
            proto_path = work_dir / cfg.stage1.prototype.cache_name
            if proto_path.exists():
                prototype, _, per_ref_features = read_prototype(proto_path)

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
                    raw_box, source = _detect_on_frame_geco2(frame_bgr, geco2_detector, geco2_prototype)
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
                    if (conf >= s4.tracker_conf_threshold
                            and track_age <= s4.max_track_age
                            and box is not None):
                        box_out = box
                        source = "track"
                    else:
                        # Confidence too low or track too old — try re-detect
                        tracker_active = False
                        if use_geco2:
                            if geco2_detector is None:
                                geco2_detector, geco2_prototype = _load_geco2(cfg, work_dir)
                            raw_box, source = _detect_on_frame_geco2(frame_bgr, geco2_detector, geco2_prototype)
                        else:
                            if proposal_model is None:
                                # Lazy-init for re-detect fallback
                                from aero_eyes.models.features import build_feature_extractor
                                from aero_eyes.models.proposals import build_proposal_model
                                from aero_eyes.utils.io import read_prototype
                                proposal_model = build_proposal_model(cfg)
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


def _load_geco2(cfg, work_dir: Path):
    """Lazily load the GeCo2 detector + its cached exemplar tokens for
    Stage 4 re-detection. Returns (detector, prototype) or (None, None) if
    the exemplar cache from Stage 1+2+3 (stage123_geco2.py) is missing.
    """
    from aero_eyes.models.geco2_detector import GeCo2Detector

    proto_path = work_dir / cfg.stage123_geco2.prototype_cache_name
    if not proto_path.exists():
        log.warning(
            "[Stage4] %s not found -- GeCo2 re-detection disabled for this run.",
            proto_path,
        )
        return None, None
    detector = GeCo2Detector(cfg)
    prototype = GeCo2Detector.load_prototype(proto_path)
    return detector, prototype


def _detect_on_frame_geco2(frame_bgr, detector, prototype):
    """GeCo2-backed equivalent of _detect_on_frame: single best re-detection
    box on one frame, or (None, "none") if nothing passed threshold/NMS or
    the detector/prototype weren't available.
    """
    if detector is None or prototype is None:
        return None, "none"
    boxes = detector.detect_frame(frame_bgr, prototype)
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

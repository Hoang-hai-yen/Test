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


def _load_best_prototype(work_dir: Path, cfg):
    """CẢI TIẾN (đồng bộ Stage3 <-> Stage4): Stage 3 có thể tinh chỉnh prototype
    qua Dynamic Prototype Update và ghi ra `prototype_dynamic.npz`. Trước đây
    Stage 4 luôn đọc thẳng `prototype.npz` gốc của Stage 1 khi cần re-detect
    (mất track / NoneTracker), khiến việc tinh chỉnh ở Stage 3 không hề có tác
    dụng lúc tracking — hai stage "lệch pha" nhau. Hàm này ưu tiên bản đã tinh
    chỉnh nếu tồn tại, fallback về bản gốc nếu chưa có (ví dụ Stage 3 chưa
    chạy hoặc dynamic update không kích hoạt).
    """
    from aero_eyes.utils.io import read_prototype

    refined_path = work_dir / "prototype_dynamic.npz"
    if refined_path.exists():
        return read_prototype(refined_path)

    base_path = work_dir / cfg.stage1.prototype.cache_name
    if base_path.exists():
        return read_prototype(base_path)

    return None, None, []


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

    # For NoneTracker, we need proposal+matching on every frame. For an
    # active tracker, we still need extractor+prototype (but not
    # proposal_model, which stays lazy) up front so periodic re-verification
    # (see verify_interval below) is available from the first tracked frame,
    # not only after a track has already been deemed lost.
    proposal_model = None
    extractor = None
    prototype = None
    per_ref_features = []
    if is_none_tracker:
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.models.proposals import build_proposal_model

        proposal_model = build_proposal_model(cfg)
        extractor = build_feature_extractor(cfg)
        prototype, _, per_ref_features = _load_best_prototype(work_dir, cfg)
    elif cfg.stage4.verify_interval > 0:
        from aero_eyes.models.features import build_feature_extractor

        extractor = build_feature_extractor(cfg)
        prototype, _, per_ref_features = _load_best_prototype(work_dir, cfg)

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

    try:
        for frame_idx, frame_bgr in frame_iterator(video_path):
            box_out: Box | None = None
            source = "none"

            if is_none_tracker:
                # Re-detect every frame
                box_out, source = _detect_on_frame(
                    frame_bgr, frame_idx, proposal_model, extractor,
                    prototype, per_ref_features, cfg, match_threshold
                )
            else:
                if frame_idx in kf_set:
                    # Initialize or re-initialize tracker from detection
                    dets = detections[frame_idx]
                    if dets:
                        best = max(dets, key=lambda d: d.similarity)
                        tracker.init(frame_bgr, best.box)
                        tracker_active = True
                        track_age = 0
                        frames_since_verify = 0
                        box_out = best.box
                        source = "detect"
                    else:
                        tracker_active = False
                elif tracker_active:
                    box, conf = tracker.update(frame_bgr)
                    track_age += 1
                    frames_since_verify += 1
                    track_ok = (conf >= s4.tracker_conf_threshold
                                and track_age <= s4.max_track_age
                                and box is not None)

                    # OpenCV's own "confidence" is a near-constant placeholder
                    # (see BuiltinTracker.update) -- it cannot tell a drifted
                    # lock from a correct one. Every verify_interval frames,
                    # cross-check the tracked crop against the prototype with
                    # the same DINOv2 embedding used for detection, so a
                    # silently-drifted track gets caught within a few frames
                    # instead of persisting for the full max_track_age.
                    if (track_ok and box is not None and extractor is not None
                            and prototype is not None and s4.verify_interval > 0
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
                        if proposal_model is None:
                            # Lazy-init for re-detect fallback
                            from aero_eyes.models.proposals import build_proposal_model
                            proposal_model = build_proposal_model(cfg)
                            if extractor is None:
                                from aero_eyes.models.features import build_feature_extractor
                                extractor = build_feature_extractor(cfg)
                                prototype, _, per_ref_features = _load_best_prototype(work_dir, cfg)

                        box_out, source = _detect_on_frame(
                            frame_bgr, frame_idx, proposal_model, extractor,
                            prototype, per_ref_features, cfg, match_threshold
                        )
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
    """Re-embed the currently-tracked crop and check it still resembles the
    target. This is the real correctness check that `BuiltinTracker`'s own
    fixed placeholder confidence cannot provide (see trackers.py)."""
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

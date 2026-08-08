"""Stage 2 — Candidate generation (per keyframe).

Flow:  drone video
       -> keyframe sampling every N frames
       -> OPTIONAL SAHI tiling
       -> class-agnostic proposals via YOLOv11n OR FastSAM-s
       -> DINOv2 features per candidate crop
       -> candidates.json

Writes: <work_dir>/<sample_id>/candidates.json
Viz:    keyframes with tile grid + candidate boxes overlaid.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box, Detection

log = logging.getLogger(__name__)


def _proposals_for_frame(
    frame_bgr: np.ndarray,
    proposal_model,
    use_sahi: bool,
    sahi_tile: list[int],
    sahi_overlap: float,
    min_area: float,
    max_candidates: int,
) -> list[Box]:
    """Run proposals on a single frame, with or without SAHI tiling."""
    from aero_eyes.utils.geometry import nms, remap_box_from_tile, sahi_tiles

    h, w = frame_bgr.shape[:2]

    if not use_sahi:
        boxes = proposal_model.propose(frame_bgr)
    else:
        tiles = sahi_tiles(w, h, sahi_tile, sahi_overlap)
        all_boxes: list[Box] = []
        for tile in tiles:
            tile_img = frame_bgr[tile.y1:tile.y2, tile.x1:tile.x2]
            if tile_img.size == 0:
                continue
            tile_boxes = proposal_model.propose(tile_img)
            for b in tile_boxes:
                all_boxes.append(remap_box_from_tile(b, tile))
        # NMS across tiles
        keep = nms(all_boxes, iou_threshold=0.5)
        boxes = [all_boxes[i] for i in keep]

    # Filter by minimum area
    boxes = [b for b in boxes if b.area() >= min_area]

    # Sort by score descending, cap count
    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes[:max_candidates]


def run_stage2(cfg, sample_id: str) -> Path:
    """Run Stage 2 for the given sample. Returns path to candidates.json."""
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.proposals import build_proposal_model
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.geometry import nms, sahi_tiles
    from aero_eyes.utils.io import write_candidates
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = work_dir / "candidates.json"
    if cfg.project.use_cache and candidates_path.exists():
        log.info("[Stage2] %s: using cached candidates at %s", sample_id, candidates_path)
        return candidates_path

    # ---- Locate video ----
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(
            f"No video matching '{cfg.data.video_glob}' found in {video_dir}."
        )
    video_path = video_files[0]
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[Stage2] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    # ---- Build models ----
    proposal_model = build_proposal_model(cfg)
    extractor = build_feature_extractor(cfg)

    # ---- Keyframe indices ----
    interval = cfg.stage2.keyframe_interval
    kf_indices = set(keyframe_indices(total_frames, interval))

    # ---- Multi-scale scan (cheap_boosters) ----
    scales = [1.0]
    if cfg.accuracy.mode in ("cheap_boosters", "max_accuracy"):
        if cfg.accuracy.cheap_boosters.multi_scale_scan:
            scales = cfg.accuracy.cheap_boosters.scales

    cand_cfg = cfg.stage2.candidate
    sahi_cfg = cfg.stage2.sahi

    candidates: dict[int, list[Detection]] = {}
    viz_dir = work_dir / "viz" / "stage2"

    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue

        all_boxes: list[Box] = []
        h, w = frame_bgr.shape[:2]

        for scale in scales:
            if scale != 1.0:
                sw, sh = int(w * scale), int(h * scale)
                scaled = cv2.resize(frame_bgr, (sw, sh))
            else:
                scaled = frame_bgr

            boxes = _proposals_for_frame(
                scaled,
                proposal_model=proposal_model,
                use_sahi=sahi_cfg.use_sahi,
                sahi_tile=sahi_cfg.tile,
                sahi_overlap=sahi_cfg.overlap,
                min_area=cand_cfg.min_box_area,
                max_candidates=cand_cfg.max_candidates_per_keyframe,
            )

            # Rescale boxes back to original frame coords
            if scale != 1.0:
                for b in boxes:
                    all_boxes.append(Box(
                        b.x1 / scale, b.y1 / scale,
                        b.x2 / scale, b.y2 / scale,
                        score=b.score,
                    ))
            else:
                all_boxes.extend(boxes)

        # Final NMS and cap
        from aero_eyes.utils.geometry import nms as do_nms
        keep = do_nms(all_boxes, iou_threshold=0.5)
        final_boxes = [all_boxes[i] for i in keep]
        final_boxes = final_boxes[:cand_cfg.max_candidates_per_keyframe]

        # Extract DINOv2 features for each candidate crop
        if final_boxes:
            feats = extractor.extract_crops(
                frame_bgr, final_boxes,
                pad_ratio=cand_cfg.feature_crop_pad,
                batch_size=cfg.runtime.batch_size,
            )
        else:
            feats = np.zeros((0, extractor._feature_dim()), dtype=np.float32)

        detections: list[Detection] = []
        for i, box in enumerate(final_boxes):
            # Store feature as extra metadata in Detection (we encode as source)
            # We'll use a special source key to carry the feature vector
            d = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="candidate")
            # Attach feature as attribute (not part of dataclass, but we handle in write)
            d._feature = feats[i]  # type: ignore[attr-defined]
            detections.append(d)

        candidates[frame_idx] = detections
        log.debug("[Stage2] frame %d: %d candidates", frame_idx, len(detections))

        if cfg.runtime.save_visualizations:
            tiles_drawn = (
                sahi_tiles(w, h, sahi_cfg.tile, sahi_cfg.overlap)
                if sahi_cfg.use_sahi else None
            )
            vizmod.save_stage2_keyframe(frame_bgr, final_boxes, tiles_drawn, frame_idx, viz_dir)

    # Serialize: store features alongside detections in a combined format
    _write_candidates_with_features(candidates, candidates_path)

    elapsed = time.time() - t0
    log.info("[Stage2] %s done in %.1fs -> %s (%d keyframes)", sample_id, elapsed,
             candidates_path, len(candidates))
    return candidates_path


def _write_candidates_with_features(
    candidates: dict[int, list[Detection]],
    path: Path,
) -> None:
    """Write candidates JSON + companion NPZ for feature vectors."""
    import json
    import numpy as np

    from aero_eyes.utils.io import SCHEMA_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)

    frames_json: dict[str, list[dict]] = {}
    all_feats: list[np.ndarray] = []
    feat_index: list[tuple[int, int]] = []  # (frame_idx, local_idx) -> all_feats index

    for fi, dets in candidates.items():
        frame_dets: list[dict] = []
        for det in dets:
            d = det.to_dict()
            feat = getattr(det, "_feature", None)
            if feat is not None:
                global_idx = len(all_feats)
                all_feats.append(feat)
                feat_index.append((fi, global_idx))
                d["feat_idx"] = global_idx
            frame_dets.append(d)
        frames_json[str(fi)] = frame_dets

    payload = {"schema_version": SCHEMA_VERSION, "frames": frames_json}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    # Save feature matrix alongside
    feat_path = path.with_suffix(".feats.npz")
    if all_feats:
        np.savez_compressed(str(feat_path), features=np.stack(all_feats, axis=0))
    else:
        np.savez_compressed(str(feat_path), features=np.zeros((0, 768), dtype=np.float32))


def read_candidates_with_features(path: Path):
    """Load candidates.json + features NPZ. Returns (detections_dict, feat_matrix)."""
    import json
    import numpy as np
    from aero_eyes.types import Detection, Box

    with open(path) as f:
        payload = json.load(f)

    feat_path = path.with_suffix(".feats.npz")
    if feat_path.exists():
        feat_matrix = np.load(str(feat_path))["features"]
    else:
        feat_matrix = None

    candidates: dict[int, list[Detection]] = {}
    for fi_str, dets in payload["frames"].items():
        fi = int(fi_str)
        det_list: list[Detection] = []
        for d in dets:
            det = Detection.from_dict(d)
            if feat_matrix is not None and "feat_idx" in d:
                det._feature = feat_matrix[d["feat_idx"]]  # type: ignore[attr-defined]
            det_list.append(det)
        candidates[fi] = det_list

    return candidates, feat_matrix


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 2 — candidate generation")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage2(cfg, args.sample)


if __name__ == "__main__":
    main()

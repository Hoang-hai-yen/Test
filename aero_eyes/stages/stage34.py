"""Stage B — Match & Track (xử lý chính).

Gộp từ 2 stage cũ:
  - Stage 3 (cross-domain matching): cosine similarity (Pass 1) -> dynamic
    prototype update -> re-score (Pass 2) -> threshold filter -> NMS ->
    detections.json (+ prototype_dynamic.npz nếu dynamic update kích hoạt)
  - Stage 4 (tracking): init tracker tại mỗi keyframe detection -> propagate
    -> re-detect khi confidence thấp / track quá cũ / verify thất bại ->
    tracks.json

Lý do gộp: Stage 4 vốn đã đọc trực tiếp `prototype_dynamic.npz` và
`match_threshold` do Stage 3 ghi ra (đồng bộ hai stage qua file trung gian).
Gộp lại thành 1 stage giúp rõ ràng rằng đây là một luồng xử lý liền mạch
"match rồi track", tránh nguy cơ chạy Stage B thiếu phần matching.

Reads:  candidates.json (+ .feats.npz), prototype.npz, video
Writes: detections.json, (prototype_dynamic.npz nếu có), tracks.json
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from aero_eyes.types import Box, Detection

log = logging.getLogger(__name__)


# =====================================================================
# Phần 1 (cũ: Stage 3) — Cross-domain matching -> detections.json
# =====================================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _score_against_ref(feats: np.ndarray, ref: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return feats @ ref
    if metric == "l2":
        return -np.linalg.norm(feats - ref[None, :], axis=1)
    if metric == "l1":
        return -np.sum(np.abs(feats - ref[None, :]), axis=1)
    raise ValueError(f"Unknown metric '{metric}'")


def _run_matching(cfg, sample_id: str) -> Path:
    """(cũ: run_stage3) So khớp candidates với prototype -> detections.json."""
    from stage_a_prepare import read_candidates_with_features
    from aero_eyes.utils.io import read_prototype, write_detections, write_prototype

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[StageB/match] %s: using cached detections", sample_id)
        return det_path

    proto_path = work_dir / cfg.stage1.prototype.cache_name
    if not proto_path.exists():
        raise FileNotFoundError("prototype.npz not found. Chạy Stage A trước.")
    prototype, meta, per_ref_features = read_prototype(proto_path)

    cand_path = work_dir / "candidates.json"
    if not cand_path.exists():
        raise FileNotFoundError("candidates.json not found. Chạy Stage A trước.")
    candidates, feat_matrix = read_candidates_with_features(cand_path)

    if feat_matrix is None or feat_matrix.shape[0] == 0:
        write_detections({}, det_path)
        return det_path

    s3 = cfg.stage3
    threshold = s3.match_threshold
    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )

    detections: dict[int, list[Detection]] = {}

    all_entries: list[tuple[int, Detection, np.ndarray]] = []
    for frame_idx, cand_dets in candidates.items():
        for det in cand_dets:
            feat = getattr(det, "_feature", None)
            if feat is not None:
                all_entries.append((frame_idx, det, feat))

    if not all_entries:
        write_detections({}, det_path)
        return det_path

    all_frame_idxs = [e[0] for e in all_entries]
    all_dets = [e[1] for e in all_entries]
    all_feats = np.stack([e[2] for e in all_entries], axis=0)

    # --- LƯỢT 1: So khớp cơ bản ---
    if use_multi_ref:
        sims_per_ref = [_score_against_ref(all_feats, ref_feat, s3.similarity) for ref_feat in per_ref_features]
        all_sims = np.mean(sims_per_ref, axis=0)
    else:
        all_sims = _score_against_ref(all_feats, prototype, s3.similarity)

    # --- CẢI TIẾN 1 (mở rộng): DYNAMIC PROTOTYPE UPDATE với NGƯỠNG THÍCH ỨNG ---
    # Ngưỡng cứng 0.40 trước đây chỉ "may mắn" kích hoạt được với nhóm dễ
    # (BlackBox) vì phân phối similarity của nó cao sẵn. Với nhóm khó
    # (CardboardBox, LifeJacket) similarity hiếm khi vượt 0.40 -> cơ chế
    # không bao giờ kích hoạt -> prototype không được tinh chỉnh theo domain
    # thực tế của video. Sửa: chọn "high-confidence" theo phân vị (percentile)
    # của CHÍNH phân phối điểm số của sample này, có sàn tuyệt đối để tránh
    # kéo theo nhiễu khi toàn bộ điểm số đều thấp, và chạy nhiều vòng để
    # prototype hội tụ dần về đúng target trong video.
    dyn_cfg = getattr(s3, "dynamic_prototype", None)
    dyn_enabled = getattr(dyn_cfg, "enabled", True) if dyn_cfg is not None else True
    dyn_rounds = getattr(dyn_cfg, "rounds", 2) if dyn_cfg is not None else 2
    dyn_alpha = getattr(dyn_cfg, "alpha", 0.3) if dyn_cfg is not None else 0.3
    dyn_percentile = getattr(dyn_cfg, "high_conf_percentile", 90) if dyn_cfg is not None else 90
    dyn_abs_floor = getattr(dyn_cfg, "high_conf_abs_floor", 0.15) if dyn_cfg is not None else 0.15
    dyn_min_support = getattr(dyn_cfg, "min_support", 2) if dyn_cfg is not None else 2

    dynamic_updated = False
    if dyn_enabled:
        for round_idx in range(dyn_rounds):
            adaptive_high_thresh = max(dyn_abs_floor, float(np.percentile(all_sims, dyn_percentile)))
            high_conf_mask = all_sims >= adaptive_high_thresh

            if int(high_conf_mask.sum()) < dyn_min_support:
                break

            high_conf_feats = all_feats[high_conf_mask]
            dynamic_feat = high_conf_feats.mean(axis=0)
            dynamic_feat /= (np.linalg.norm(dynamic_feat) + 1e-8)

            log.info(
                "[StageB/match] %s: Dynamic Prototype Update vòng %d/%d — ngưỡng thích ứng=%.3f "
                "(percentile=%d), %d candidates, alpha=%.2f",
                sample_id, round_idx + 1, dyn_rounds, adaptive_high_thresh,
                dyn_percentile, int(high_conf_mask.sum()), dyn_alpha,
            )
            dynamic_updated = True

            if use_multi_ref:
                per_ref_features.append(dynamic_feat)
                sims_per_ref = [_score_against_ref(all_feats, ref_feat, s3.similarity) for ref_feat in per_ref_features]
                all_sims = np.mean(sims_per_ref, axis=0)
            else:
                prototype = (1 - dyn_alpha) * prototype + dyn_alpha * dynamic_feat
                prototype /= (np.linalg.norm(prototype) + 1e-8)
                all_sims = _score_against_ref(all_feats, prototype, s3.similarity)

    # Ghi lại prototype đã tinh chỉnh (dynamic-updated) ra file riêng
    # (prototype_dynamic.npz), KHÔNG ghi đè prototype.npz gốc (giữ tái lập
    # được / tránh trôi dạt qua nhiều lần rerun). Phần tracking bên dưới sẽ
    # ưu tiên đọc file này nếu có.
    if dynamic_updated:
        refined_proto_path = work_dir / "prototype_dynamic.npz"
        refined_meta = dict(meta) if isinstance(meta, dict) else {}
        refined_meta["dynamic_updated"] = True
        refined_meta["dynamic_rounds"] = dyn_rounds
        write_prototype(
            prototype=prototype,
            meta=refined_meta,
            per_ref_features=per_ref_features if use_multi_ref else None,
            path=refined_proto_path,
        )
        log.info("[StageB/match] %s: đã lưu prototype tinh chỉnh -> %s", sample_id, refined_proto_path)

    log.info(
        "[StageB/match] %s: scores min=%.3f p50=%.3f mean=%.3f max=%.3f",
        sample_id, float(all_sims.min()), float(np.percentile(all_sims, 50)),
        float(all_sims.mean()), float(all_sims.max())
    )

    # ---- Compute effective threshold ----
    if s3.adaptive_threshold:
        sim_mean = float(all_sims.mean())
        sim_std = float(all_sims.std())
        raw_threshold = sim_mean + s3.adaptive_z_score * sim_std
        if s3.similarity == "cosine":
            effective_threshold = max(s3.adaptive_min_floor, raw_threshold)
        else:
            effective_threshold = raw_threshold
        # Cap tại điểm số cao nhất thực tế quan sát được trong video. Công
        # thức mean + z*std chỉ là ước lượng thống kê, không phải trần cứng
        # -- với video có độ phân tán điểm số cao, nó có thể vượt max thực
        # tế một chút (từng gặp: max=0.727 vs threshold=0.736), làm loại bỏ
        # oan ứng viên tốt nhất và trả về 0 detection dù có ứng viên mạnh.
        sim_max = float(all_sims.max())
        if effective_threshold > sim_max:
            log.info(
                "[StageB/match] %s: adaptive threshold %.3f exceeds max score %.3f -- "
                "capping at max so the best candidate isn't dropped.",
                sample_id, effective_threshold, sim_max,
            )
            effective_threshold = sim_max
        log.info("[StageB/match] %s: adaptive threshold = %.3f", sample_id, effective_threshold)
    else:
        effective_threshold = threshold

    # ---- Filter by threshold ----
    keep_mask = all_sims >= effective_threshold
    selected = [
        (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
        for i in range(len(all_sims)) if keep_mask[i]
    ]

    global_topk = s3.global_topk
    if global_topk is not None and len(selected) > global_topk:
        selected.sort(key=lambda x: x[2], reverse=True)
        selected = selected[:global_topk]

    # NMS
    from collections import defaultdict
    from aero_eyes.utils.geometry import nms

    frame_groups: dict[int, list[tuple[Detection, float]]] = defaultdict(list)
    for fi, det, sim in selected:
        frame_groups[fi].append((det, sim))

    for frame_idx, det_sim_pairs in frame_groups.items():
        det_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        dets_f = [d for d, _ in det_sim_pairs]
        sims_f = [s for _, s in det_sim_pairs]

        keep_idx = nms(
            [d.box.__class__(d.box.x1, d.box.y1, d.box.x2, d.box.y2, score=s)
             for d, s in zip(dets_f, sims_f)],
            iou_threshold=s3.nms_iou,
        )
        post_nms = [(dets_f[i], sims_f[i]) for i in keep_idx][: s3.topk_per_keyframe]

        result_dets = [
            Detection(frame_idx=frame_idx, box=det.box, similarity=sim, source="detect")
            for det, sim in post_nms
        ]
        detections[frame_idx] = result_dets

    write_detections(detections, det_path, threshold=effective_threshold)
    elapsed = time.time() - t0
    log.info("[StageB/match] %s done in %.1fs -> %s (%d detection frames)",
             sample_id, elapsed, det_path, len(detections))
    return det_path


# =====================================================================
# Phần 2 (cũ: Stage 4) — Tracking -> tracks.json
# =====================================================================

def _load_best_prototype(work_dir: Path, cfg):
    """Ưu tiên đọc prototype đã tinh chỉnh (`prototype_dynamic.npz`, ghi ra
    ở phần matching phía trên) nếu tồn tại, fallback về bản gốc
    `prototype.npz` nếu chưa có (ví dụ dynamic update không kích hoạt)."""
    from aero_eyes.utils.io import read_prototype

    refined_path = work_dir / "prototype_dynamic.npz"
    if refined_path.exists():
        return read_prototype(refined_path)

    base_path = work_dir / cfg.stage1.prototype.cache_name
    if base_path.exists():
        return read_prototype(base_path)

    return None, None, []


def _track_still_matches(
    frame_bgr,
    box: Box,
    extractor,
    prototype,
    per_ref_features: list,
    cfg,
    match_threshold: float,
) -> bool:
    """Re-embed crop đang track và kiểm tra còn giống target không -- đây là
    check đúng nghĩa mà confidence cố định của BuiltinTracker không cung
    cấp được."""
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
    """Run proposal + matching trên 1 frame; trả về (best_box, source)."""
    if proposal_model is None or extractor is None or prototype is None:
        return None, "none"

    from aero_eyes.utils.geometry import nms, remap_box_from_tile, sahi_tiles

    s2 = cfg.stage2
    h, w = frame_bgr.shape[:2]

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


def _run_tracking(cfg, sample_id: str) -> Path:
    """(cũ: run_stage4) Track object giữa các keyframe."""
    from aero_eyes.models.trackers import NoneTracker, build_tracker
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.io import read_detections, read_detections_threshold, write_tracks
    from aero_eyes.utils.video import AnnotatedVideoWriter, frame_iterator, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = work_dir / "tracks.json"
    if cfg.project.use_cache and tracks_path.exists():
        log.info("[StageB/track] %s: using cached tracks at %s", sample_id, tracks_path)
        return tracks_path

    # ---- Load detections ----
    det_path = work_dir / "detections.json"
    if not det_path.exists():
        raise FileNotFoundError(
            f"detections.json not found at {det_path}. Chạy phần matching trước."
        )
    detections = read_detections(det_path)
    # Dùng lại đúng threshold mà phần matching đã dùng (giá trị adaptive
    # z-score khi bật) để re-detect giữa video áp cùng ngưỡng chấp nhận,
    # thay vì âm thầm rơi về default cố định trong config.
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
        viz_dir = work_dir / "viz" / "stageB_track"
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
                box_out, source = _detect_on_frame(
                    frame_bgr, frame_idx, proposal_model, extractor,
                    prototype, per_ref_features, cfg, match_threshold
                )
            else:
                if frame_idx in kf_set:
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
                    # -- nó không phân biệt được lock trôi dạt với lock đúng.
                    # Cứ mỗi verify_interval frame, đối chiếu crop đang track
                    # với prototype bằng cùng DINOv2 embedding dùng để
                    # detect, để bắt được track trôi dạt trong vài frame
                    # thay vì để nó tồn tại hết max_track_age.
                    if (track_ok and box is not None and extractor is not None
                            and prototype is not None and s4.verify_interval > 0
                            and frames_since_verify >= s4.verify_interval):
                        frames_since_verify = 0
                        if not _track_still_matches(
                            frame_bgr, box, extractor, prototype,
                            per_ref_features, cfg, match_threshold,
                        ):
                            log.debug(
                                "[StageB/track] frame %d: track failed re-verification "
                                "(likely drifted) -- forcing re-detect", frame_idx,
                            )
                            track_ok = False

                    if track_ok:
                        box_out = box
                        source = "track"
                    else:
                        tracker_active = False
                        if proposal_model is None:
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

            log.debug("[StageB/track] frame %d: %s box=%s", frame_idx, source, box_out)
    finally:
        if writer is not None:
            writer.release()

    write_tracks(tracks, tracks_path)
    elapsed = time.time() - t0
    present = sum(1 for v in tracks.values() if v is not None)
    log.info("[StageB/track] %s done in %.1fs -> %s (%d/%d frames with box)",
             sample_id, elapsed, tracks_path, present, total_frames)
    return tracks_path


# =====================================================================
# Entry point gộp
# =====================================================================

def run_stage_b(cfg, sample_id: str) -> Path:
    """Stage B — Match & Track. Chạy tuần tự: matching -> tracking.
    Trả về đường dẫn tracks.json."""
    _run_matching(cfg, sample_id)
    tracks_path = _run_tracking(cfg, sample_id)
    return tracks_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage B — match & track")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage_b(cfg, args.sample)


if __name__ == "__main__":
    main()
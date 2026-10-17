"""Stage B — Match & Track (xử lý chính).

Gộp từ 2 stage cũ:
  - Stage 3 (cross-domain matching): cosine similarity (Pass 1) -> dynamic
    prototype update -> re-score (Pass 2) -> threshold filter -> NMS ->
    detections.json (+ prototype_dynamic.npz nếu dynamic update kích hoạt)
  - Stage 4 (tracking): init tracker tại mỗi keyframe detection -> propagate
    -> re-detect khi confidence thấp / track quá cũ / verify thất bại ->
    tracks.json

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


def _get_pool_fn(cfg):
    """Lấy hàm gộp điểm đa tham chiếu động (max hoặc mean) từ cấu hình."""
    cheap_boosters = getattr(getattr(cfg, "accuracy", None), "cheap_boosters", None)
    pool_mode = getattr(cheap_boosters, "multi_ref_pooling", "mean") if cheap_boosters is not None else "mean"
    return np.max if pool_mode == "max" else np.mean


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


def _otsu_threshold(scores: np.ndarray, bins: int = 256) -> float:
    """Otsu split of a 1-D score distribution (maximises between-class variance)."""
    hist, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[int(np.argmax(between)) + 1])


def _adaptive_cap_threshold(scores: np.ndarray, mode: str, quantile: float = 0.25) -> float:
    """Upper cap for the adaptive threshold.

    "otsu": the Otsu split itself -- too aggressive when the target is absent
        from many frames (public BlackBox_1 top-1 F1 92.5% -> 49.9%).
    "upper_quantile": the given quantile of the scores ABOVE the Otsu split,
        i.e. only bite when mean + z*std would reject most of the high
        (target) cluster, as on Wallet_1.
    """
    otsu = _otsu_threshold(scores)
    if mode == "otsu":
        return otsu
    upper = scores[scores >= otsu]
    return float(np.quantile(upper, quantile)) if upper.size else otsu


def _online_match_and_threshold(
    s3, sample_id: str, candidates: dict, all_feats: np.ndarray, all_frame_idxs: list[int],
    prototype: np.ndarray, per_ref_features: list[np.ndarray], use_multi_ref: bool, pool_fn,
):
    """Causal (streaming) counterpart of the whole-video matching below: frames
    are processed in order and every decision for frame t uses only what was
    observed at frames < t, as a live drone feed requires.

    - Threshold: mean + z*std over a running window of the last
      adaptive_threshold_online_window similarity scores (design from
      test-geco2's OnlineAdaptiveThreshold). Cold start (fewer than
      adaptive_threshold_min_samples scores seen) rejects everything by
      default (adaptive_threshold_online_cold_start="reject"); "floor" accepts
      >= adaptive_min_floor like test-geco2.
    - adaptive_cap (otsu / upper_quantile / otsu_if_sparse) is applied to the
      same window; otsu_if_sparse uses the running candidates-per-frame.
    - Dynamic prototype: instead of 2 whole-video rounds, a running mean of
      high-confidence candidate features (>= max(high_conf_abs_floor,
      high_conf_percentile of the window)) joins the reference pool once it
      has min_support members, weighted like the `rounds` offline refs.

    Returns (all_sims, keep_mask, frame_thresholds, per_ref_features_final,
    prototype_final, dynamic_used).
    """
    from collections import defaultdict, deque

    dyn_cfg = getattr(s3, "dynamic_prototype", None)
    dyn_enabled = getattr(dyn_cfg, "enabled", True) if dyn_cfg is not None else True
    dyn_rounds = getattr(dyn_cfg, "rounds", 2) if dyn_cfg is not None else 2
    dyn_alpha = getattr(dyn_cfg, "alpha", 0.3) if dyn_cfg is not None else 0.3
    dyn_percentile = getattr(dyn_cfg, "high_conf_percentile", 90) if dyn_cfg is not None else 90
    dyn_abs_floor = getattr(dyn_cfg, "high_conf_abs_floor", 0.15) if dyn_cfg is not None else 0.15
    dyn_min_support = getattr(dyn_cfg, "min_support", 2) if dyn_cfg is not None else 2

    window: deque = deque(maxlen=s3.adaptive_threshold_online_window)
    min_samples = s3.adaptive_threshold_min_samples
    cold_start = getattr(s3, "adaptive_threshold_online_cold_start", "reject")
    floor = s3.adaptive_min_floor if s3.similarity == "cosine" else float("-inf")
    cap_mode_cfg = getattr(s3, "adaptive_cap", "none")
    max_cpf = getattr(s3, "adaptive_cap_max_cands_per_frame", 1.0)
    cap_q = getattr(s3, "adaptive_cap_quantile", 0.25)

    frame_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, fi in enumerate(all_frame_idxs):
        frame_to_idx[fi].append(i)

    n = len(all_frame_idxs)
    all_sims = np.zeros(n, dtype=np.float32)
    keep_mask = np.zeros(n, dtype=bool)
    frame_thresholds: dict[int, float] = {}
    dyn_sum = np.zeros(all_feats.shape[1], dtype=np.float64)
    dyn_n = 0
    seen_entries = seen_frames = 0
    cold = capped = 0

    def dynamic_feat():
        if not dyn_enabled or dyn_n < dyn_min_support:
            return None
        v = dyn_sum / dyn_n
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)

    for fi in sorted(candidates):
        idxs = frame_to_idx.get(fi, [])
        seen_frames += 1
        if not idxs:
            continue

        # -- score this frame against the prototype state as of frame t-1
        feats = all_feats[idxs]
        dyn = dynamic_feat()
        if use_multi_ref:
            refs = list(per_ref_features) + ([dyn] * dyn_rounds if dyn is not None else [])
            sims = pool_fn([_score_against_ref(feats, r, s3.similarity) for r in refs], axis=0)
        else:
            proto = prototype
            if dyn is not None:
                proto = (1 - dyn_alpha) * prototype + dyn_alpha * dyn
                proto = proto / (np.linalg.norm(proto) + 1e-8)
            sims = _score_against_ref(feats, proto, s3.similarity)

        # -- threshold from strictly-earlier scores only
        if len(window) < min_samples:
            thr = floor if cold_start == "floor" else float("inf")
            cold += 1
        else:
            w = np.asarray(window, dtype=np.float64)
            thr = max(floor, float(w.mean() + s3.adaptive_z_score * w.std()))
            cap_mode = cap_mode_cfg
            if cap_mode == "otsu_if_sparse":
                cap_mode = "otsu" if seen_entries / max(1, seen_frames - 1) < max_cpf else "none"
            if cap_mode != "none":
                cap_thr = _adaptive_cap_threshold(w, cap_mode, cap_q)
                if cap_thr < thr:
                    thr = max(floor, cap_thr)
                    capped += 1
            # No "cap at max score" here (the offline path has one so the
            # video's best candidate is never dropped): online, the window can
            # hold only background before the target appears, and capping at
            # its max would let the strongest background box through.

        all_sims[idxs] = sims
        keep_mask[idxs] = sims >= thr
        if np.isfinite(thr):
            frame_thresholds[fi] = float(thr)

        # -- update state AFTER this frame's decision
        if dyn_enabled and len(window) >= min_samples:
            hc_thr = max(dyn_abs_floor, float(np.percentile(np.asarray(window), dyn_percentile)))
            hc = sims >= hc_thr
            if hc.any():
                dyn_sum += feats[hc].sum(axis=0)
                dyn_n += int(hc.sum())
        window.extend(np.asarray(sims, dtype=np.float64).tolist())
        seen_entries += len(idxs)

    dyn = dynamic_feat()
    per_ref_final = list(per_ref_features) + ([dyn] * dyn_rounds if (use_multi_ref and dyn is not None) else [])
    proto_final = prototype
    if not use_multi_ref and dyn is not None:
        proto_final = (1 - dyn_alpha) * prototype + dyn_alpha * dyn
        proto_final = proto_final / (np.linalg.norm(proto_final) + 1e-8)

    thr_vals = np.array(list(frame_thresholds.values())) if frame_thresholds else np.array([floor])
    log.info(
        "[StageB/match] %s: ONLINE threshold (window=%d, min_samples=%d, cap=%s) -> "
        "thr p10/p50/p90=%.3f/%.3f/%.3f, cold-start frames=%d, capped frames=%d, "
        "dynamic feats=%d, %d / %d candidates pass",
        sample_id, s3.adaptive_threshold_online_window, min_samples, cap_mode_cfg,
        *np.percentile(thr_vals, [10, 50, 90]), cold, capped, dyn_n, int(keep_mask.sum()), n,
    )
    return all_sims, keep_mask, frame_thresholds, per_ref_final, proto_final, dyn is not None


def _run_matching(cfg, sample_id: str) -> Path:
    """(cũ: run_stage3) So khớp candidates với prototype -> detections.json."""
    from aero_eyes.stages.stage12 import read_candidates_with_features
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
    pool_fn = _get_pool_fn(cfg)

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
        all_sims = pool_fn(sims_per_ref, axis=0)
    else:
        all_sims = _score_against_ref(all_feats, prototype, s3.similarity)

    online = bool(s3.adaptive_threshold and getattr(s3, "adaptive_threshold_online", False))
    frame_thresholds = None
    if online:
        all_sims, keep_mask, frame_thresholds, per_ref_final, proto_final, dyn_used = _online_match_and_threshold(
            s3, sample_id, candidates, all_feats, all_frame_idxs, prototype, per_ref_features,
            use_multi_ref, pool_fn,
        )
        # Single number kept for logging / tools that expect one threshold;
        # Stage 4 reads the per-frame thresholds instead.
        effective_threshold = float(np.median(list(frame_thresholds.values()))) if frame_thresholds else threshold
        if dyn_used:
            refined_meta = dict(meta) if isinstance(meta, dict) else {}
            refined_meta["dynamic_updated"] = True
            refined_meta["dynamic_online"] = True
            write_prototype(
                prototype=proto_final,
                meta=refined_meta,
                per_ref_features=per_ref_final if use_multi_ref else None,
                path=work_dir / "prototype_dynamic.npz",
            )
    else:
        # --- CẢI TIẾN 1: DYNAMIC PROTOTYPE UPDATE với NGƯỠNG THÍCH ỨNG ---
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
                    all_sims = pool_fn(sims_per_ref, axis=0)
                else:
                    prototype = (1 - dyn_alpha) * prototype + dyn_alpha * dynamic_feat
                    prototype /= (np.linalg.norm(prototype) + 1e-8)
                    all_sims = _score_against_ref(all_feats, prototype, s3.similarity)

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

            cap_mode = getattr(s3, "adaptive_cap", "none")
            if cap_mode == "otsu_if_sparse":
                # Few proposals per keyframe => most candidates are the target, the
                # regime where mean + z*std overshoots (Wallet_1: 0.6 cands/frame).
                cands_per_frame = len(all_entries) / max(1, len(candidates))
                max_cpf = getattr(s3, "adaptive_cap_max_cands_per_frame", 2.0)
                cap_mode = "otsu" if cands_per_frame < max_cpf else "none"
                log.info(
                    "[StageB/match] %s: %.2f candidates/frame (sparse < %.1f) -> cap=%s",
                    sample_id, cands_per_frame, max_cpf, cap_mode,
                )
            if cap_mode != "none":
                cap_thr = _adaptive_cap_threshold(
                    all_sims, cap_mode, getattr(s3, "adaptive_cap_quantile", 0.25)
                )
                if cap_thr < effective_threshold:
                    log.info(
                        "[StageB/match] %s: adaptive threshold %.3f capped by %s -> %.3f",
                        sample_id, effective_threshold, cap_mode, cap_thr,
                    )
                    effective_threshold = max(s3.adaptive_min_floor, cap_thr)

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

    write_detections(detections, det_path, threshold=effective_threshold,
                     frame_thresholds=frame_thresholds)
    elapsed = time.time() - t0
    log.info("[StageB/match] %s done in %.1fs -> %s (%d detection frames)",
             sample_id, elapsed, det_path, len(detections))
    return det_path


# =====================================================================
# Phần 2 (cũ: Stage 4) — Tracking -> tracks.json
# =====================================================================

def _load_best_prototype(work_dir: Path, cfg):
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
    pool_fn = _get_pool_fn(cfg)

    if use_multi_ref:
        sim = float(pool_fn([feats[0] @ ref_feat for ref_feat in per_ref_features]))
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
    pool_fn = _get_pool_fn(cfg)

    if use_multi_ref:
        sims_per_ref = [feats @ ref_feat for ref_feat in per_ref_features]
        sims = pool_fn(sims_per_ref, axis=0)
    else:
        sims = feats @ prototype

    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= match_threshold:
        return boxes[best_idx], "detect"
    return None, "none"


def _run_tracking(cfg, sample_id: str) -> Path:
    from aero_eyes.models.trackers import NoneTracker, build_tracker
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.io import (
        read_detections, read_detections_frame_thresholds, read_detections_threshold, write_tracks,
    )
    from aero_eyes.utils.video import AnnotatedVideoWriter, frame_iterator, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = work_dir / "tracks.json"
    if cfg.project.use_cache and tracks_path.exists():
        log.info("[StageB/track] %s: using cached tracks at %s", sample_id, tracks_path)
        return tracks_path

    det_path = work_dir / "detections.json"
    if not det_path.exists():
        raise FileNotFoundError(
            f"detections.json not found at {det_path}. Chạy phần matching trước."
        )
    detections = read_detections(det_path)
    match_threshold = read_detections_threshold(det_path)
    if match_threshold is None:
        match_threshold = cfg.stage3.match_threshold
    # Online Stage 3 stores one threshold per keyframe; re-detection/verify at
    # frame t then uses the latest threshold decided at a keyframe <= t
    # (causal), instead of one whole-video value.
    frame_thr = read_detections_frame_thresholds(det_path)
    thr_frames = sorted(frame_thr) if frame_thr else []

    def thr_at(frame_idx: int) -> float:
        if not thr_frames:
            return match_threshold
        import bisect
        k = bisect.bisect_right(thr_frames, frame_idx) - 1
        return frame_thr[thr_frames[k]] if k >= 0 else frame_thr[thr_frames[0]]

    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(f"No video found for sample '{sample_id}'.")
    video_path = video_files[0]
    vinfo = video_info(video_path)
    total_frames = vinfo["total_frames"]

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
                    prototype, per_ref_features, cfg, thr_at(frame_idx)
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

                    if (track_ok and box is not None and extractor is not None
                            and prototype is not None and s4.verify_interval > 0
                            and frames_since_verify >= s4.verify_interval):
                        frames_since_verify = 0
                        if not _track_still_matches(
                            frame_bgr, box, extractor, prototype,
                            per_ref_features, cfg, thr_at(frame_idx),
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
                            prototype, per_ref_features, cfg, thr_at(frame_idx)
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

def run_stage34(cfg, sample_id: str) -> Path:
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
    run_stage34(cfg, args.sample)


if __name__ == "__main__":
    main()

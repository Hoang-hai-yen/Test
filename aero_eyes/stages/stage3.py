"""Stage 3 — Cross-domain matching.

Flow:  candidates.json + prototype.npz
       -> cosine similarity (Pass 1)
       -> DYNAMIC PROTOTYPE UPDATE (CẢI TIẾN 1)
       -> re-score similarity (Pass 2)
       -> threshold filter
       -> NMS across tiles
       -> detections.json
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from aero_eyes.types import Detection

log = logging.getLogger(__name__)


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


def run_stage3(cfg, sample_id: str) -> Path:
    from aero_eyes.stages.stage2 import read_candidates_with_features
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.geometry import nms
    from aero_eyes.utils.io import read_prototype, write_detections, write_prototype
    from aero_eyes.utils.video import read_frame

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage3] %s: using cached detections", sample_id)
        return det_path

    proto_path = work_dir / cfg.stage1.prototype.cache_name
    if not proto_path.exists():
        raise FileNotFoundError("prototype.npz not found.")
    prototype, meta, per_ref_features = read_prototype(proto_path)

    cand_path = work_dir / "candidates.json"
    if not cand_path.exists():
        raise FileNotFoundError("candidates.json not found.")
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
    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    video_path = video_files[0] if video_files else None
    viz_dir = work_dir / "viz" / "stage3"

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
                break  # không đủ ứng viên tin cậy -> dừng sớm, tránh làm lệch prototype

            high_conf_feats = all_feats[high_conf_mask]
            dynamic_feat = high_conf_feats.mean(axis=0)
            dynamic_feat /= (np.linalg.norm(dynamic_feat) + 1e-8)

            log.info(
                "[Stage3] %s: Dynamic Prototype Update vòng %d/%d — ngưỡng thích ứng=%.3f "
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

    # CẢI TIẾN: Ghi lại prototype đã tinh chỉnh (dynamic-updated) ra một file
    # riêng (prototype_dynamic.npz), KHÔNG ghi đè prototype.npz gốc của Stage 1
    # (giữ tái lập được / tránh trôi dạt qua nhiều lần rerun). Stage 4 sẽ ưu
    # tiên đọc file này nếu có, để tracking / re-detect dùng đúng target
    # signature đã được tinh chỉnh theo video thay vì bản gốc từ Stage 1.
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
        log.info("[Stage3] %s: đã lưu prototype tinh chỉnh -> %s", sample_id, refined_proto_path)

    # Log metrics
    log.info(
        "[Stage3] %s: scores min=%.3f p50=%.3f mean=%.3f max=%.3f",
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
        # Cap at the video's own best observed score. mean + z*std is a
        # statistical estimate, not a hard ceiling -- on a video where the
        # score distribution has high spread, it can end up a hair above
        # the actual max (seen in practice: max=0.727 vs threshold=0.736),
        # rejecting a genuinely good top match and producing zero detections
        # for the whole video even though a strong candidate existed. The
        # top candidate should never be discarded purely because the
        # threshold formula overshot the data range.
        sim_max = float(all_sims.max())
        if effective_threshold > sim_max:
            log.info(
                "[Stage3] %s: adaptive threshold %.3f exceeds max score %.3f -- "
                "capping at max so the best candidate isn't dropped.",
                sample_id, effective_threshold, sim_max,
            )
            effective_threshold = sim_max
        log.info("[Stage3] %s: adaptive threshold = %.3f", sample_id, effective_threshold)
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
    log.info("[Stage3] %s done in %.1fs -> %s (%d detection frames)",
             sample_id, elapsed, det_path, len(detections))
    return det_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage3(cfg, args.sample)


if __name__ == "__main__":
    main()

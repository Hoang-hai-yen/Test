"""Stage 3 — Cross-domain matching.

Flow:  candidates.json + prototype.npz
       -> cosine similarity
       -> threshold filter
       -> NMS across tiles
       -> top-K per keyframe
       -> detections.json

Reads:  prototype.npz, candidates.json (+.feats.npz)
Writes: detections.json
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
    """Cosine similarity between two 1-D vectors (both assumed L2-normalized)."""
    return float(np.dot(a, b))


def run_stage3(cfg, sample_id: str) -> Path:
    """Run Stage 3 for the given sample. Returns path to detections.json."""
    from aero_eyes.stages.stage2 import read_candidates_with_features
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.geometry import nms
    from aero_eyes.utils.io import read_prototype, write_detections
    from aero_eyes.utils.video import read_frame, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage3] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    # ---- Load prototype ----
    proto_path = work_dir / cfg.stage1.prototype.cache_name
    if not proto_path.exists():
        raise FileNotFoundError(
            f"prototype.npz not found at {proto_path}. Run Stage 1 first."
        )
    prototype, meta, per_ref_features = read_prototype(proto_path)

    # ---- Load candidates ----
    cand_path = work_dir / "candidates.json"
    if not cand_path.exists():
        raise FileNotFoundError(
            f"candidates.json not found at {cand_path}. Run Stage 2 first."
        )
    candidates, feat_matrix = read_candidates_with_features(cand_path)

    if feat_matrix is None or feat_matrix.shape[0] == 0:
        log.warning("[Stage3] No candidate features found — writing empty detections.")
        write_detections({}, det_path)
        return det_path

    # ---- Stage 3 config ----
    s3 = cfg.stage3
    threshold = s3.match_threshold
    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )

    # ---- Match: global top-K or per-keyframe threshold ----
    detections: dict[int, list[Detection]] = {}
    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    video_path = video_files[0] if video_files else None
    viz_dir = work_dir / "viz" / "stage3"

    # Build flat list of (frame_idx, det, feat) for all candidates
    all_entries: list[tuple[int, Detection, np.ndarray]] = []
    for frame_idx, cand_dets in candidates.items():
        for det in cand_dets:
            feat = getattr(det, "_feature", None)
            if feat is not None:
                all_entries.append((frame_idx, det, feat))

    if not all_entries:
        write_detections({}, det_path)
        log.warning("[Stage3] %s: no candidate features found", sample_id)
        return det_path

    all_frame_idxs = [e[0] for e in all_entries]
    all_dets = [e[1] for e in all_entries]
    all_feats = np.stack([e[2] for e in all_entries], axis=0)  # [N, D]

    # Compute similarity for every candidate at once
    if use_multi_ref:
        sims_per_ref = [all_feats @ ref_feat for ref_feat in per_ref_features]
        all_sims = np.mean(sims_per_ref, axis=0)
    else:
        all_sims = all_feats @ prototype  # [N]

    # CD-ViTO domain prompter (max_accuracy)
    if (cfg.accuracy.mode == "max_accuracy"
            and cfg.accuracy.max_accuracy.domain_prompter.enabled):
        all_sims = _apply_domain_prompter(all_feats, prototype, all_sims, cfg)

    # Always log the raw similarity distribution — the ground-to-aerial domain
    # gap means a fixed match_threshold tuned on one dataset can silently pass
    # zero candidates on another; this makes that visible instead of a mute
    # "0 detection frames" result.
    log.info(
        "[Stage3] %s: candidate similarity stats — min=%.3f p50=%.3f mean=%.3f "
        "std=%.3f p95=%.3f max=%.3f (n=%d)",
        sample_id, float(all_sims.min()), float(np.percentile(all_sims, 50)),
        float(all_sims.mean()), float(all_sims.std()),
        float(np.percentile(all_sims, 95)), float(all_sims.max()), len(all_sims),
    )

    # ---- Compute effective threshold ----
    if s3.adaptive_threshold:
        sim_mean = float(all_sims.mean())
        sim_std = float(all_sims.std())
        effective_threshold = max(
            s3.adaptive_min_floor,
            sim_mean + s3.adaptive_z_score * sim_std,
        )
        log.info(
            "[Stage3] %s: adaptive threshold = max(floor=%.3f, %.3f + %.1f*%.3f) = %.3f",
            sample_id, s3.adaptive_min_floor,
            sim_mean, s3.adaptive_z_score, sim_std, effective_threshold,
        )
    else:
        effective_threshold = threshold

    # ---- Filter by threshold ----
    keep_mask = all_sims >= effective_threshold
    selected = [
        (all_frame_idxs[i], all_dets[i], float(all_sims[i]))
        for i in range(len(all_sims)) if keep_mask[i]
    ]
    log.info("[Stage3] %s: threshold=%.3f → %d / %d candidates pass",
             sample_id, effective_threshold, len(selected), len(all_sims))

    # ---- Apply global_topk cap (after threshold, not instead of it) ----
    global_topk = s3.global_topk
    if global_topk is not None and len(selected) > global_topk:
        selected.sort(key=lambda x: x[2], reverse=True)
        selected = selected[:global_topk]
        log.info("[Stage3] %s: capped to global_topk=%d", sample_id, global_topk)

    # Group by frame, apply NMS + topk_per_keyframe
    from collections import defaultdict
    frame_groups: dict[int, list[tuple[Detection, float]]] = defaultdict(list)
    for fi, det, sim in selected:
        frame_groups[fi].append((det, sim))

    for frame_idx, det_sim_pairs in frame_groups.items():
        det_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        dets_f = [d for d, _ in det_sim_pairs]
        sims_f = [s for _, s in det_sim_pairs]

        # NMS
        keep_idx = nms(
            [d.box.__class__(d.box.x1, d.box.y1, d.box.x2, d.box.y2, score=s)
             for d, s in zip(dets_f, sims_f)],
            iou_threshold=s3.nms_iou,
        )
        post_nms = [(dets_f[i], sims_f[i]) for i in keep_idx]

        # Top-K per keyframe
        post_nms = post_nms[: s3.topk_per_keyframe]

        result_dets = [
            Detection(frame_idx=frame_idx, box=det.box, similarity=sim, source="detect")
            for det, sim in post_nms
        ]
        detections[frame_idx] = result_dets

        if cfg.runtime.save_visualizations and video_path:
            try:
                frame_bgr = read_frame(video_path, frame_idx)
                vizmod.save_stage3_detections(
                    frame_bgr, [d.box for d in result_dets],
                    [d.similarity for d in result_dets],
                    frame_idx, viz_dir,
                )
            except Exception:
                pass

    write_detections(detections, det_path)
    elapsed = time.time() - t0
    log.info("[Stage3] %s done in %.1fs -> %s (%d detection frames)",
             sample_id, elapsed, det_path, len(detections))
    return det_path


def _apply_domain_prompter(
    feats: np.ndarray,
    prototype: np.ndarray,
    sims: np.ndarray,
    cfg,
) -> np.ndarray:
    """CD-ViTO-style domain feature alignment (simplified).

    Synthesizes 'imaginary domain' feature shifts by interpolating between
    the candidate feature distribution and the prototype direction,
    then re-scores using the shifted features.
    """
    dp = cfg.accuracy.max_accuracy.domain_prompter
    strength = dp.strength

    # Compute the mean domain gap: shift candidate features toward prototype style
    # by blending them with the prototype direction
    proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    shifted = feats + strength * proto_norm[None]
    # Re-normalize
    norms = np.linalg.norm(shifted, axis=-1, keepdims=True).clip(min=1e-8)
    shifted = shifted / norms
    new_sims = shifted @ prototype
    # Blend original and new scores
    return 0.5 * sims + 0.5 * new_sims


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 3 — cross-domain matching")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage3(cfg, args.sample)


if __name__ == "__main__":
    main()

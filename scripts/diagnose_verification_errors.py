"""Diagnostic (Phase 0 of docs/GECO2_precision_improvements_plan.md): WHY is
Stage 3 precision capped on real footage -- a small number of RECURRING
confuser objects (systematic, fixable with hard-negative-style suppression)
or DIFFUSE/scattered false positives (evidence the underlying appearance
signal itself, not the decision mechanism, is the bottleneck)? This decides
which of the 5 follow-up techniques in that plan is worth investing in first
(item 3, a better score, vs. item 6, structural multi-frame identity
tracking) -- run this BEFORE implementing either.

Read-only: loads existing candidates.json/detections.json/prototype.npz,
makes no pipeline changes. Reuses this project's own established
TP/FN/FP/TN semantics (scripts/check_cosine_effect.py::compute_prf1) so
every number here is directly comparable to every P/R/F1 already reported
by that script and scripts/check_stage_prf1_progression.py.

For every FALSE POSITIVE (a detection surviving on a frame the target is
NOT present in at all -- see compute_prf1's own docstring) and every TRUE
POSITIVE, recovers the original candidate feature vector (candidates.json's
own companion .feats.npz) by matching the detection's box back to
candidates.json's same-frame boxes via IoU (handles box_refine having
nudged coordinates slightly since Stage 3 first picked them).

Reports:
  1. FP self-clustering: how many distinct confuser "identities" exist
     among all FPs, and how concentrated they are.
  2. FP spatial recurrence: do FPs cluster around a fixed frame region
     (a static background object) or scatter uniformly.
  3. Global "oracle" separability upper bound: cluster ALL TPs + FPs +
     exemplars together in one shot (full-video hindsight, cheating on
     purpose) -- if even this can't separate TP from FP, no per-keyframe
     or windowed online mechanism ever will either.
  4. TP vs. FP similarity score distributions, side by side.
  5. Representative FP samples from the largest clusters, for manual
     video inspection (frame_idx + box so you can look up the actual crop).

Usage:
    python -m scripts.diagnose_verification_errors --config configs/config.yaml --sample IDCard_0
    python -m scripts.diagnose_verification_errors --config configs/config.yaml --sample IDCard_0 --iou-threshold 0.3
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from aero_eyes.types import Box, Detection
from aero_eyes.utils.geometry import box_iou
from aero_eyes.stages.stage2 import read_candidates_with_features
from aero_eyes.utils.io import load_gt, read_detections, read_prototype
from scripts.check_cosine_effect import _print_result, compute_prf1

log = logging.getLogger(__name__)


def _find_matching_feature(box: Box, cand_dets: list[Detection]) -> np.ndarray | None:
    """Best-IoU candidate in the same frame that has a stored feature --
    box_refine may have nudged the detection's box slightly since Stage 3
    first picked it from this candidate, so exact coordinate equality
    isn't assumed. Returns None if no candidate overlaps at all."""
    best_iou, best_feat = 0.0, None
    for cd in cand_dets:
        feat = getattr(cd, "_feature", None)
        if feat is None:
            continue
        iou = box_iou(box, cd.box)
        if iou > best_iou:
            best_iou, best_feat = iou, feat
    return best_feat


def _collect_tp_fp_entries(
    detections: dict[int, list[Detection]],
    candidates: dict[int, list[Detection]],
    gt: dict[int, Box],
    processed_frames: set[int],
    iou_threshold: float,
) -> tuple[list[tuple[int, Box, float, np.ndarray]], list[tuple[int, Box, float, np.ndarray]], int]:
    """Returns (tp_entries, fp_entries, n_unmatched) -- each entry is
    (frame_idx, box, similarity, feature)."""
    tp_entries: list[tuple[int, Box, float, np.ndarray]] = []
    fp_entries: list[tuple[int, Box, float, np.ndarray]] = []
    n_unmatched = 0

    for fi in sorted(processed_frames):
        dets = detections.get(fi, [])
        if not dets:
            continue
        if fi in gt:
            best_det = max(dets, key=lambda d: box_iou(gt[fi], d.box))
            if box_iou(gt[fi], best_det.box) >= iou_threshold:
                feat = _find_matching_feature(best_det.box, candidates.get(fi, []))
                if feat is None:
                    n_unmatched += 1
                else:
                    tp_entries.append((fi, best_det.box, best_det.similarity, feat))
            # else: FN (wrong localization or nothing close enough) -- no
            # box to analyze here, a fundamentally different failure mode
            # from "detected something on a GT-absent frame."
        else:
            for d in dets:
                feat = _find_matching_feature(d.box, candidates.get(fi, []))
                if feat is None:
                    n_unmatched += 1
                else:
                    fp_entries.append((fi, d.box, d.similarity, feat))

    return tp_entries, fp_entries, n_unmatched


def _report_fp_clustering(fp_entries: list, distance_threshold: float) -> np.ndarray:
    """Deliberately uses AgglomerativeClustering with a FIXED distance
    threshold here, not HDBSCAN (unlike the production cluster_verify_
    candidates primitive) -- confirmed empirically while building this
    script that HDBSCAN's density-based extraction is unreliable at the
    small sample sizes a single video's FP count typically produces (it
    can label an obviously tight, well-separated 8-point cluster entirely
    as noise at N=13 total points -- the same small-N fragility documented
    throughout tests/test_cluster_verify.py). A fixed distance threshold
    has no such small-N failure mode and is more appropriate for a one-off
    diagnostic than a density-based method tuned for the production
    per-keyframe decision."""
    from sklearn.cluster import AgglomerativeClustering

    fp_feats = np.stack([e[3] for e in fp_entries])
    fp_feats = fp_feats / np.linalg.norm(fp_feats, axis=1, keepdims=True)
    sim = fp_feats @ fp_feats.T
    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    n = len(fp_entries)
    if n < 2:
        labels = np.zeros(n, dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="average", metric="precomputed",
        ).fit_predict(dist)

    unique_labels, counts = np.unique(labels, return_counts=True)
    n_singletons = int((counts == 1).sum())
    print(f"\n--- 1. FP self-clustering (cosine distance threshold={distance_threshold}) ---")
    print(f"{len(unique_labels)} distinct confuser identit(y/ies) found among {n} FP(s); "
          f"{n_singletons} appear only once ({100 * n_singletons / n:.1f}%)")

    sizes = sorted(zip(unique_labels.tolist(), counts.tolist()), key=lambda x: -x[1])
    for lbl, size in sizes[:10]:
        print(f"  identity {lbl}: {size} FP(s) ({100 * size / n:.1f}% of all FPs)")

    top3_frac = sum(s for _, s in sizes[:3]) / n
    verdict = "SYSTEMATIC (a few recurring confuser(s) dominate)" if top3_frac > 0.5 else "DIFFUSE (no dominant confuser -- scattered)"
    print(f"Top-3 identities cover {100 * top3_frac:.1f}% of all FPs -> {verdict}")
    return labels


def _report_spatial_recurrence(fp_entries: list) -> None:
    print("\n--- 2. FP spatial recurrence (8x8 grid over the observed extent of FP box centers) ---")
    centers = np.array([[(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2] for _, b, _, _ in fp_entries])
    x_min, x_max = centers[:, 0].min(), centers[:, 0].max()
    y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
    grid = np.zeros((8, 8), dtype=int)
    for x, y in centers:
        gx = min(7, int(8 * (x - x_min) / (x_max - x_min + 1e-6)))
        gy = min(7, int(8 * (y - y_min) / (y_max - y_min + 1e-6)))
        grid[gy, gx] += 1
    n = len(fp_entries)
    max_cell = int(grid.max())
    print(f"Max single-cell occupancy: {max_cell}/{n} ({100 * max_cell / n:.1f}%) -- "
          f"{'a static background region likely recurs' if max_cell / n > 0.2 else 'no obvious spatial hot-spot'}")
    for row in grid:
        print("  " + " ".join(f"{v:3d}" for v in row))


def _report_oracle_separability(
    tp_entries: list, fp_entries: list, per_ref_features: list, prototype: np.ndarray, min_cluster_size: int,
) -> None:
    from aero_eyes.config import ClusterVerificationConfig
    from aero_eyes.utils.cluster_verify import cluster_verify_candidates

    print("\n--- 3. Global oracle separability upper bound (ALL TP+FP+exemplars clustered at once, full hindsight) ---")
    n_tp, n_fp = len(tp_entries), len(fp_entries)
    tp_feats = np.stack([e[3] for e in tp_entries]) if tp_entries else np.zeros((0, fp_entries[0][3].shape[0]))
    fp_feats = np.stack([e[3] for e in fp_entries])
    all_feats = np.concatenate([tp_feats, fp_feats], axis=0)
    all_feats = all_feats / np.linalg.norm(all_feats, axis=1, keepdims=True)

    ref_feats = np.stack(per_ref_features) if per_ref_features else prototype[None, :]
    ref_feats = ref_feats / np.linalg.norm(ref_feats, axis=1, keepdims=True)

    oracle_cfg = ClusterVerificationConfig(
        enabled=True, cluster_method="hdbscan", min_cluster_size=min_cluster_size, max_candidates_for_cluster=None,
    )
    keep_mask, method_label = cluster_verify_candidates(
        all_feats, ref_feats, oracle_cfg,
        fallback_keep_mask_fn=lambda c, r: np.zeros(c.shape[0], dtype=bool),
    )
    tp_kept = int(keep_mask[:n_tp].sum()) if n_tp else 0
    fp_kept = int(keep_mask[n_tp:].sum())
    print(f"method={method_label}")
    if n_tp:
        print(f"  {tp_kept}/{n_tp} TP would be kept ({100 * tp_kept / n_tp:.1f}%)")
    print(f"  {fp_kept}/{n_fp} FP would be kept ({100 * fp_kept / n_fp:.1f}%)")
    if n_tp:
        margin = tp_kept / n_tp - fp_kept / n_fp
        verdict = ("clustering CAN separate TP/FP given full-video context -- a structural, multi-frame "
                   "mechanism (plan item 6) is worth prototyping") if margin > 0.3 else (
            "even full-hindsight clustering struggles -- the embedding/score itself is the bottleneck, "
            "not the decision mechanism (favor plan item 3, a better score, over item 6)")
        print(f"  oracle TP/FP retention margin: {100 * margin:.1f} points -> {verdict}")


def _report_score_distributions(tp_entries: list, fp_entries: list) -> None:
    print("\n--- 4. TP vs FP similarity score distributions ---")
    fp_sims = np.array([e[2] for e in fp_entries])
    print(f"  FP: n={fp_sims.size} mean={fp_sims.mean():.3f} std={fp_sims.std():.3f} "
          f"min={fp_sims.min():.3f} max={fp_sims.max():.3f}")
    if tp_entries:
        tp_sims = np.array([e[2] for e in tp_entries])
        print(f"  TP: n={tp_sims.size} mean={tp_sims.mean():.3f} std={tp_sims.std():.3f} "
              f"min={tp_sims.min():.3f} max={tp_sims.max():.3f}")
        overlap = fp_sims.max() >= tp_sims.min()
        print(f"  Ranges overlap: {overlap} (if True, no fixed cutoff on this raw score alone could separate every TP/FP)")
    else:
        print("  TP: none collected (0 true positives with a matched feature)")


def _report_representative_samples(fp_entries: list, labels: np.ndarray, top_n: int = 3, per_cluster: int = 5) -> None:
    print(f"\n--- 5. Representative FP samples from the top {top_n} identities (for manual video inspection) ---")
    unique_labels = sorted(set(labels.tolist()))
    sizes = sorted(((lbl, int((labels == lbl).sum())) for lbl in unique_labels), key=lambda x: -x[1])
    for lbl, size in sizes[:top_n]:
        if size < 2:
            break  # remaining identities are singletons -- nothing "representative" to show
        members = [fp_entries[i] for i in range(len(fp_entries)) if labels[i] == lbl]
        print(f"  identity {lbl} ({size} member(s)), showing up to {per_cluster}:")
        for fi, box, sim, _ in members[:per_cluster]:
            print(f"    frame_idx={fi:<6} box=({box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}) similarity={sim:.3f}")


def diagnose_sample(
    cfg, sample_id: str, iou_threshold: float, fp_cluster_distance_threshold: float, oracle_min_cluster_size: int,
) -> None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    cand_path = work_dir / "candidates.json"
    det_path = work_dir / "detections.json"
    proto_path = work_dir / cfg.stage1.prototype.cache_name

    if not cand_path.exists() or not det_path.exists() or not proto_path.exists():
        print(f"{sample_id}: missing candidates.json/detections.json/prototype.npz in {work_dir} -- run the pipeline first.")
        return

    try:
        gt = load_gt(cfg.data.gt.global_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {cfg.data.gt.global_file}, skipping.")
        return

    candidates, _ = read_candidates_with_features(cand_path)
    detections = read_detections(det_path)
    prototype, _, per_ref_features = read_prototype(proto_path)
    processed_frames = set(candidates.keys())

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) ===")
    r = compute_prf1(detections, gt, iou_threshold, processed_frames=processed_frames)
    _print_result("Stage 3 detections.json", r)

    tp_entries, fp_entries, n_unmatched = _collect_tp_fp_entries(
        detections, candidates, gt, processed_frames, iou_threshold,
    )
    print(f"\nCollected {len(tp_entries)} TP feature(s), {len(fp_entries)} FP feature(s) "
          f"({n_unmatched} detection(s) couldn't be matched back to a candidate feature -- "
          "usually box_refine moving the box further than any surviving candidate's own box).")

    if not fp_entries:
        print("No FP entries collected -- nothing further to diagnose here (either FP=0, or feature-matching "
              "failed for all of them; see the unmatched count above).")
        return

    labels = _report_fp_clustering(fp_entries, fp_cluster_distance_threshold)
    _report_spatial_recurrence(fp_entries)
    _report_oracle_separability(tp_entries, fp_entries, per_ref_features, prototype, oracle_min_cluster_size)
    _report_score_distributions(tp_entries, fp_entries)
    _report_representative_samples(fp_entries, labels)


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Diagnose whether Stage 3 false positives are dominated by a few recurring confusers "
                    "(systematic) or are diffuse/scattered (weak appearance signal) -- see "
                    "docs/GECO2_precision_improvements_plan.md's Phase 0."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as a correct localization (default: 0.5, match "
                    "check_cosine_effect.py's own default)")
    p.add_argument("--fp-cluster-distance-threshold", type=float, default=0.3,
                    help="Section 1 (FP self-clustering): cosine-distance threshold for grouping FPs into the "
                    "same confuser identity via AgglomerativeClustering (default: 0.3, i.e. cosine "
                    "similarity > 0.7 -- lower = stricter/more identities, higher = looser/fewer)")
    p.add_argument("--oracle-min-cluster-size", type=int, default=3,
                    help="Section 3 (global oracle separability): HDBSCAN min_cluster_size, reusing the SAME "
                    "production cluster_verify_candidates primitive stage3.py/geco2_detector.py use "
                    "(default: 3)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        diagnose_sample(cfg, sid, args.iou_threshold, args.fp_cluster_distance_threshold, args.oracle_min_cluster_size)


if __name__ == "__main__":
    main()

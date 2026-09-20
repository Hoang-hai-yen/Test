"""Full-pipeline comparison: DAVE (arXiv:2404.16622) module (ii)-style
exemplar-cluster verification vs. this project's existing Z-score-based
candidate verification, for BOTH mechanisms it replaces -- see
docs/GECO2_cluster_verification_plan.md for the full rationale.

Both mechanisms below only take effect when stage123_geco2.cosine_rescore
is enabled (GeCo2 becomes a wide-recall candidate generator; the accept/
reject decision is made elsewhere) -- same entry point,
run_stage12_geco2_candidates -> run_stage3, for both:

  Part A (stage3.verification_method): "threshold" (today's
    adaptive_threshold z_score=2.0 baseline, the validated default per
    configs/config.yaml's own sweep notes) vs "cluster" with
    cluster_method=hdbscan vs cluster_method=spectral.

  Part B (stage123_geco2.dynamic_prototype): today's topk_fusion (Z-score
    fusion, enabled) vs cluster_verification with cluster_method=hdbscan vs
    cluster_method=spectral. stage3.verification_method is held at
    "threshold" while sweeping this axis, to isolate Part B's effect on the
    dynamic-prototype-enriched exemplar set from Part A's own choice of
    verification method.

For Part B, also captures each run's dynamic_prototype log_summary() line
(via a temporary logging handler -- these counters are not persisted to
disk anywhere else) and reports, per sample: how many offer_topk() calls
verified nothing (reported absent) vs. fell back to the plain cosine gate
-- the concrete "how much did fixing cold-start help" signal topk_fusion's
own cold-start problem doesn't have an equivalent for.

Metric: ST-IoU (this project's own evaluation metric -- see
aero_eyes/evaluate.py; precision/recall are not separately computed here).

Usage:
    python -m scripts.compare_cluster_vs_zscore_verification --config configs/config.yaml \
        --set data.data_root=... --set data.gt.global_file=... \
        --set project.work_dir=/kaggle/working/runs/exp001
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_CLUSTER_SUMMARY_RE = re.compile(
    r"dynamic_prototype cluster_verification summary \(cluster_method=(?P<method>\w+)\) -- "
    r"(?P<offers>\d+) offer_topk\(\) call\(s\), (?P<unverified>\d+) keyframe\(s\) verified NOTHING "
    r"\(reported absent\), (?P<fallback>\d+) fell back"
)


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _clear_cache(work_dir: Path, sample_ids: list[str], prototype_cache_name: str) -> None:
    for sid in sample_ids:
        sdir = work_dir / sid
        for name in ("candidates.json", "candidates.feats.npz", "detections.json", "detections_prerefine.json",
                     "tracks.json", "submission.json", "prototype.npz", "prototype_adapted.npz",
                     prototype_cache_name):
            f = sdir / name
            if f.exists():
                f.unlink()


def _run_full_pipeline(cfg, sample_ids: list[str], capture: _LogCapture | None = None) -> dict[str, float]:
    """Returns {video_id: st_iou} for the CURRENT cfg. Runs the
    cosine_rescore entry point (run_stage12_geco2_candidates -> run_stage3),
    the only path either verification_method or dynamic_prototype.
    cluster_verification actually affects (see GeCo2DynamicPrototypeTracker.
    offer_topk's own docstring -- it's never wired into the plain
    run_stage123_geco2 path)."""
    from aero_eyes.stages.stage123_geco2 import run_stage12_geco2_candidates
    from aero_eyes.stages.stage3 import run_stage3
    from aero_eyes.stages.stage4 import run_stage4
    from aero_eyes.stages.stage5 import run_stage5
    from aero_eyes.evaluate import evaluate_dataset

    geco2_logger = logging.getLogger("aero_eyes.models.geco2_detector")
    if capture is not None:
        geco2_logger.addHandler(capture)
    try:
        work_dir = Path(cfg.project.work_dir)
        for sid in sample_ids:
            run_stage12_geco2_candidates(cfg, sid)
            run_stage3(cfg, sid)
            run_stage4(cfg, sid)
            run_stage5(cfg, sid)
    finally:
        if capture is not None:
            geco2_logger.removeHandler(capture)

    all_preds = []
    for sid in sample_ids:
        sub_path = work_dir / sid / cfg.data.submission.path_name
        if sub_path.exists():
            all_preds.extend(json.loads(sub_path.read_text()))
    combined_path = work_dir / "all_submissions_cluster_verify_compare.json"
    combined_path.write_text(json.dumps(all_preds))

    report = evaluate_dataset(combined_path, cfg.data.gt.global_file, cfg=cfg)
    return report["per_video"]


def _parse_cluster_summaries(messages: list[str]) -> dict:
    """Best-effort parse of the LAST cluster_verification summary line seen
    (one per sample, per GeCo2DynamicPrototypeTracker.log_summary() call) --
    returns {} if cluster_verification wasn't enabled this run (no such
    line was ever logged)."""
    parsed = None
    for msg in messages:
        m = _CLUSTER_SUMMARY_RE.search(msg)
        if m:
            parsed = {
                "cluster_method": m.group("method"),
                "offer_topk_calls": int(m.group("offers")),
                "keyframes_verified_nothing": int(m.group("unverified")),
                "keyframes_fell_back_to_plain_gate": int(m.group("fallback")),
            }
    return parsed or {}


def run_variant(cfg, sample_ids: list[str], label: str, track_cluster_summary: bool = False) -> dict:
    work_dir = Path(cfg.project.work_dir)
    _clear_cache(work_dir, sample_ids, cfg.stage123_geco2.prototype_cache_name)
    capture = _LogCapture() if track_cluster_summary else None
    per_video = _run_full_pipeline(cfg, sample_ids, capture=capture)
    mean = sum(per_video.values()) / len(per_video)
    log.info("%s -> mean ST-IoU=%.4f per-video=%s", label, mean, per_video)
    print(f"{label:<40}  mean ST-IoU={mean:.4f}  {per_video}")
    result = {"per_video": per_video, "mean": mean}
    if track_cluster_summary:
        result["cluster_verification_stats"] = _parse_cluster_summaries(capture.messages)
    return result


def run_part_a(cfg, sample_ids: list[str]) -> dict:
    print("\n--- Part A: stage3.verification_method (cosine_rescore candidates -> Stage 3) ---")
    cfg.stage123_geco2.cosine_rescore.enabled = True
    cfg.stage123_geco2.dynamic_prototype.enabled = False  # isolate Part A from Part B

    cfg.stage3.verification_method = "threshold"
    cfg.stage3.adaptive_threshold = True
    cfg.stage3.adaptive_threshold_method = "z_score"
    cfg.stage3.adaptive_z_score = 2.0  # validated default per configs/config.yaml's own sweep
    baseline = run_variant(cfg, sample_ids, "Part A: threshold (z_score=2.0, today's baseline)")

    cfg.stage3.verification_method = "cluster"
    cfg.stage3.cluster_verification.cluster_method = "hdbscan"
    hdbscan_result = run_variant(cfg, sample_ids, "Part A: cluster (hdbscan)")

    cfg.stage3.cluster_verification.cluster_method = "spectral"
    spectral_result = run_variant(cfg, sample_ids, "Part A: cluster (spectral)")

    cfg.stage3.verification_method = "threshold"  # reset for Part B
    return {"baseline_threshold": baseline, "cluster_hdbscan": hdbscan_result, "cluster_spectral": spectral_result}


def run_part_b(cfg, sample_ids: list[str]) -> dict:
    print("\n--- Part B: stage123_geco2.dynamic_prototype (topk_fusion vs cluster_verification) ---")
    cfg.stage123_geco2.cosine_rescore.enabled = True
    cfg.stage3.verification_method = "threshold"  # hold Part A fixed while sweeping Part B
    cfg.stage123_geco2.dynamic_prototype.enabled = True
    cfg.stage123_geco2.dynamic_prototype.cross_check_source = "feature_extractor"

    cfg.stage123_geco2.dynamic_prototype.topk_fusion.enabled = True
    cfg.stage123_geco2.dynamic_prototype.cluster_verification.enabled = False
    baseline = run_variant(cfg, sample_ids, "Part B: topk_fusion (Z-score fusion, today's baseline)")

    cfg.stage123_geco2.dynamic_prototype.topk_fusion.enabled = False
    cfg.stage123_geco2.dynamic_prototype.cluster_verification.enabled = True
    cfg.stage123_geco2.dynamic_prototype.cluster_verification.cluster_method = "hdbscan"
    hdbscan_result = run_variant(cfg, sample_ids, "Part B: cluster_verification (hdbscan)", track_cluster_summary=True)

    cfg.stage123_geco2.dynamic_prototype.cluster_verification.cluster_method = "spectral"
    spectral_result = run_variant(cfg, sample_ids, "Part B: cluster_verification (spectral)", track_cluster_summary=True)

    cfg.stage123_geco2.dynamic_prototype.enabled = False  # reset
    return {"baseline_topk_fusion": baseline, "cluster_hdbscan": hdbscan_result, "cluster_spectral": spectral_result}


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="Compare DAVE-style cluster verification vs. this project's Z-score-based baselines "
                    "(stage3.verification_method and stage123_geco2.dynamic_prototype.topk_fusion)"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--part", choices=["a", "b", "both"], default="both")
    p.add_argument("--out", default=None,
                    help="path to write JSON report (default: <work_dir>/cluster_vs_zscore_verification_report.json)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if cfg.project.use_cache:
        log.warning("project.use_cache=true in config -- forcing false for this comparison script.")
        cfg.project.use_cache = False

    data_root = Path(cfg.data.data_root)
    sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]
    print(f"Samples: {sample_ids}")

    report: dict = {"samples": sample_ids}
    if args.part in ("a", "both"):
        report["part_a"] = run_part_a(cfg, sample_ids)
    if args.part in ("b", "both"):
        report["part_b"] = run_part_b(cfg, sample_ids)

    print("\n" + "=" * 70)
    for part_key, part_label in (("part_a", "Part A (stage3.verification_method)"),
                                  ("part_b", "Part B (dynamic_prototype)")):
        if part_key not in report:
            continue
        print(f"\n{part_label}:")
        for variant_key, variant in report[part_key].items():
            print(f"  {variant_key:<22} mean ST-IoU={variant['mean']:.4f}")
            stats = variant.get("cluster_verification_stats")
            if stats:
                print(f"    {stats}")
    print("=" * 70)

    out_path = Path(args.out) if args.out else Path(cfg.project.work_dir) / "cluster_vs_zscore_verification_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()

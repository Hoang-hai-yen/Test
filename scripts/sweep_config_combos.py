"""One-factor-at-a-time sweep across the combos catalogued in
reports/Combo config thu nghiem threshold va encoder.md -- encoder,
prototype construction, adaptive_threshold_method, adaptive_threshold_online_method,
secondary filters, multi_ref_pooling -- to find the best-performing setup on
your own footage.

Only runs Stage 1 (when a combo touches the encoder/prototype/reference
preprocessing) + Stage 3 (cosine matching -> detections.json). Reuses each
sample's EXISTING candidates.json (Stage 1+2 output) instead of
regenerating it -- run Stage 1+2 normally first if candidates.json doesn't
exist yet. Score is P/R/F1 on detections.json vs ground truth, reusing
compute_prf1 (from check_cosine_effect.py) and check_stage_prf1_progression's
own check_sample() for the printed per-combo progression view, exactly the
tool this project already uses for this kind of comparison.

Every combo layers on top of a fixed BASELINE (the one already validated:
adaptive_threshold + adaptive_threshold_online + window_stat + z_score=2.0)
-- so a combo like "otsu" only changes adaptive_threshold_method, everything
else stays at the validated baseline. Combos that change the encoder or
reference-image preprocessing (feature_extractor.*, prototype.fusion,
domain_calibration.filter_target_like_frames, aerial_sim.*) rerun Stage 1,
which overwrites each sample's prototype.npz; combos that also change the
embedding (encoder swaps) additionally force
stage3.recompute_candidate_features=true, which overwrites candidates.json's
own cached per-candidate features in place. Both are backed up per sample
before the sweep starts and restored (byte-for-byte) once the sweep for that
sample finishes -- including on a crash -- so your existing
prototype.npz/candidates.json/detections.json are never left modified after
this script exits. All actual results only ever live in the JSON/Markdown
report this script writes, never in the pipeline's own working files.

A combo that raises (missing dependency, gated HF weights not downloaded,
OOM, ...) is logged and skipped -- one bad combo (e.g. a gated encoder you
haven't requested HF access for) does not abort the whole sweep.

Usage:
    # preview the combo list without running anything
    python -m scripts.sweep_config_combos --config configs/config.yaml --dry-run

    # full sweep, all samples under data.data_root, default z/downscale values
    python -m scripts.sweep_config_combos --config configs/config.yaml \\
        --set project.work_dir=/kaggle/working/runs/exp001

    # single sample, skip the heavy/gated encoders (fgclip/radio), custom z sweep
    python -m scripts.sweep_config_combos --config configs/config.yaml \\
        --sample IDCard_0 --z-values 2.0,1.5,1.0

    # include fgclip/radio too (needs their own deps/HF access already set up)
    python -m scripts.sweep_config_combos --config configs/config.yaml --include-heavy-encoders
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline: the one combo already validated on real footage (F1=0.706 on
# IDCard_0). Every other combo's overrides are layered ON TOP of this, so a
# combo dict only needs to state what it changes relative to it.
# ---------------------------------------------------------------------------
BASELINE_OVERRIDES: dict[str, Any] = {
    "stage3.adaptive_threshold": True,
    "stage3.adaptive_threshold_online": True,
    "stage3.adaptive_threshold_online_method": "window_stat",
    "stage3.adaptive_threshold_method": "z_score",
    "stage3.adaptive_z_score": 2.0,
}

# Files a Stage-1-touching combo can mutate in place -- backed up/restored
# per sample around the whole sweep.
MUTABLE_FILES = ("prototype.npz", "candidates.json", "detections.json")


@dataclass(frozen=True)
class Combo:
    id: str
    group: str
    overrides: dict[str, Any] = field(default_factory=dict)
    needs_stage1: bool = False
    recompute: bool = False   # force stage3.recompute_candidate_features=true
    sweep_z: bool = False     # expand into one variant per --z-values
    sweep_downscale: bool = False   # expand into one variant per --downscale-values
    heavy: bool = False       # gated/extra-deps encoder -- skipped unless --include-heavy-encoders
    note: str = ""


def build_combos(include_heavy: bool) -> list[Combo]:
    combos = [
        # ---- baseline + adaptive_threshold_method family (mục 3) ----
        Combo("baseline", "threshold", {}, sweep_z=True,
              note="baseline: z_score on the window_stat online mechanism"),
        Combo("T2_zscore_robust", "threshold", {"stage3.adaptive_threshold_robust": True}, sweep_z=True,
              note="median/MAD instead of mean/std -- less outlier-sensitive"),
        Combo("T3_otsu", "threshold", {"stage3.adaptive_threshold_method": "otsu"},
              note="no bimodal check -- risky on a near-unimodal window"),
        Combo("T4_gmm", "threshold", {"stage3.adaptive_threshold_method": "gmm"},
              note="bimodal check via BIC + min_separation_std, falls back to percentile"),

        # ---- adaptive_threshold_online_method family (mục 4) ----
        Combo("O2_aci", "online_method", {"stage3.adaptive_threshold_online_method": "aci"}),
        Combo("O3_saffron", "online_method", {"stage3.adaptive_threshold_online_method": "saffron"}),
        Combo("O4_corruption_compensated", "online_method",
              {"stage3.adaptive_threshold_online_method": "corruption_compensated"}),

        # ---- secondary filters (mục 5) ----
        Combo("F1_negative_prototype", "filter", {"stage3.negative_prototype_filter.enabled": True}),
        Combo("F2_identity_chain_sw0", "filter", {
            "stage3.identity_chain_filter.enabled": True,
            "stage3.identity_chain_filter.spatial_weight": 0.0,
        }),
        Combo("F2b_identity_chain_sw03", "filter", {
            "stage3.identity_chain_filter.enabled": True,
            "stage3.identity_chain_filter.spatial_weight": 0.3,
        }),
        Combo("F3_rmd", "filter", {"stage3.similarity": "rmd"}),
        Combo("F4_margin_cluster", "filter", {
            "stage3.margin_verification.enabled": True,
            "stage3.margin_verification.tau_margin": 0.05,
            "stage3.cluster_secondary_filter.enabled": True,
            "stage3.cluster_secondary_filter.window_admission_min_consecutive_hits": 2,
        }),
        Combo("F5_stacked", "filter", {
            "stage3.negative_prototype_filter.enabled": True,
            "stage3.cluster_secondary_filter.enabled": True,
            "stage3.cluster_secondary_filter.window_admission_min_consecutive_hits": 2,
            "stage3.identity_chain_filter.enabled": True,
            "stage3.identity_chain_filter.spatial_weight": 0.3,
            "stage3.margin_verification.enabled": True,
            "stage3.margin_verification.tau_margin": 0.05,
        }),

        # ---- multi_ref_pooling (mục 6) -- "max" == baseline, not repeated ----
        Combo("M1_mean", "multi_ref_pooling", {"accuracy.cheap_boosters.multi_ref_pooling": "mean"}),
        Combo("M3_min", "multi_ref_pooling", {"accuracy.cheap_boosters.multi_ref_pooling": "min"}),
        Combo("M4_agreement_weighted", "multi_ref_pooling",
              {"accuracy.cheap_boosters.multi_ref_pooling": "agreement_weighted"}),

        # ---- prototype construction (mục 2) -- needs Stage 1, same encoder ----
        Combo("P1_agreement_weighted_fusion", "prototype",
              {"stage1.prototype.fusion": "agreement_weighted"}, needs_stage1=True),
        Combo("P2_filter_target_like_frames", "prototype",
              {"stage1.domain_calibration.filter_target_like_frames": True}, needs_stage1=True),
        Combo("P3_P1_plus_P2", "prototype", {
            "stage1.prototype.fusion": "agreement_weighted",
            "stage1.domain_calibration.filter_target_like_frames": True,
        }, needs_stage1=True),
        Combo("P4_aerial_sim", "prototype", {"stage1.aerial_sim.enabled": True},
              needs_stage1=True, sweep_downscale=True),
        Combo("P5_background_keep_real", "prototype",
              {"stage1.segmentation.background_mode": "keep_real"}, needs_stage1=True),
        Combo("P6_crop_to_object_margin0.0", "prototype",
              {"stage1.crop_to_object": True, "stage1.crop_context_margin": 0.0}, needs_stage1=True),
        Combo("P7_crop_to_object_margin0.2", "prototype",
              {"stage1.crop_to_object": True, "stage1.crop_context_margin": 0.2}, needs_stage1=True),

        # ---- encoder swaps (mục 1) -- needs Stage 1 + recompute ----
        Combo("E1a_dinov3_lvd1689m", "encoder", {
            "stage1.feature_extractor.model": "dinov3",
            "stage1.feature_extractor.dinov3_pretrain_dataset": "lvd1689m",
        }, needs_stage1=True, recompute=True,
              note="architecture change ALONE (v2->v3), default pretrain -- isolates arch from domain"),
        Combo("E1b_dinov3_sat493m", "encoder", {
            "stage1.feature_extractor.model": "dinov3",
            "stage1.feature_extractor.dinov3_pretrain_dataset": "sat493m",
        }, needs_stage1=True, recompute=True,
              note="architecture change + satellite-domain pretrain together"),
        Combo("E2_dinov2_registers", "encoder",
              {"stage1.feature_extractor.dinov2_use_registers": True}, needs_stage1=True, recompute=True),
        Combo("E3_multiscale_attn", "encoder",
              {"stage1.feature_extractor.dinov2_pooling": "multiscale_attn"},
              needs_stage1=True, recompute=True),
        Combo("E4_clip", "encoder", {
            "stage1.feature_extractor.model": "clip",
            "stage1.feature_extractor.clip_variant": "vit-l/14",
        }, needs_stage1=True, recompute=True),
        Combo("E4b_siglip", "encoder",
              {"stage1.feature_extractor.model": "siglip"}, needs_stage1=True, recompute=True),
        Combo("E5a_ensemble_dinov2_clip", "encoder", {
            "stage1.feature_extractor.model": "ensemble",
            "stage1.feature_extractor.ensemble_dino_model": "dinov2",
        }, needs_stage1=True, recompute=True,
              note="original ensemble baseline (DINOv2+CLIP), never A/B tested before"),
        Combo("E5b_ensemble_dinov3_lvd1689m_clip", "encoder", {
            "stage1.feature_extractor.model": "ensemble",
            "stage1.feature_extractor.ensemble_dino_model": "dinov3",
            "stage1.feature_extractor.dinov3_pretrain_dataset": "lvd1689m",
        }, needs_stage1=True, recompute=True),
        Combo("E5c_ensemble_dinov3_sat493m_clip", "encoder", {
            "stage1.feature_extractor.model": "ensemble",
            "stage1.feature_extractor.ensemble_dino_model": "dinov3",
            "stage1.feature_extractor.dinov3_pretrain_dataset": "sat493m",
        }, needs_stage1=True, recompute=True),
        Combo("E6_fgclip", "encoder", {"stage1.feature_extractor.model": "fgclip"},
              needs_stage1=True, recompute=True, heavy=True,
              note="project already saw plain CLIP/SigLIP underperform DINOv2/v3"),
        Combo("E7_radio", "encoder", {"stage1.feature_extractor.model": "radio"},
              needs_stage1=True, recompute=True, heavy=True,
              note="empirical bet only -- no retrieval evidence in the paper"),
    ]
    if not include_heavy:
        combos = [c for c in combos if not c.heavy]
    return combos


def expand_sweeps(combos: list[Combo], z_values: list[float], downscale_values: list[float]) -> list[Combo]:
    expanded: list[Combo] = []
    for c in combos:
        if c.sweep_z:
            for z in z_values:
                ov = dict(c.overrides)
                ov["stage3.adaptive_z_score"] = z
                expanded.append(replace(c, id=f"{c.id}_z{z}", overrides=ov, sweep_z=False))
        elif c.sweep_downscale:
            for d in downscale_values:
                ov = dict(c.overrides)
                ov["stage1.aerial_sim.downscale_factor"] = d
                expanded.append(replace(c, id=f"{c.id}_ds{d}", overrides=ov, sweep_downscale=False))
        else:
            expanded.append(c)
    return expanded


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _override_strings(overrides: dict[str, Any]) -> list[str]:
    return [f"{k}={_fmt(v)}" for k, v in overrides.items()]


def _backup_files(sample_dir: Path) -> dict[str, Path]:
    backups: dict[str, Path] = {}
    for name in MUTABLE_FILES:
        src = sample_dir / name
        if src.exists():
            dst = sample_dir / f"{name}.pre_sweep_bak"
            shutil.copy2(src, dst)
            backups[name] = dst
    return backups


def _restore_files(sample_dir: Path, backups: dict[str, Path]) -> None:
    for name in MUTABLE_FILES:
        dst = sample_dir / name
        bak = backups.get(name)
        if bak is not None and bak.exists():
            shutil.copy2(bak, dst)
            bak.unlink()
        elif dst.exists():
            # Created by the sweep itself (wasn't there before) -- remove so
            # the sample directory ends up exactly as it started.
            dst.unlink()


def run_combo(config_path: str, base_overrides: list[str], combo: Combo,
              sample_id: str, iou_threshold: float) -> dict[str, Any]:
    from aero_eyes.config import load_config
    from aero_eyes.stages.stage3 import run_stage3
    from aero_eyes.utils.io import load_gt, read_candidates, read_detections
    from scripts.check_cosine_effect import compute_prf1
    from scripts.check_stage_prf1_progression import check_sample

    override_strs = (
        list(base_overrides)
        + _override_strings(BASELINE_OVERRIDES)
        + _override_strings(combo.overrides)
        + (["stage3.recompute_candidate_features=true"] if combo.recompute else [])
    )
    cfg = load_config(config_path, override_strs)

    if combo.needs_stage1:
        from aero_eyes.stages.stage1 import run_stage1
        run_stage1(cfg, sample_id)

    run_stage3(cfg, sample_id)

    print(f"\n--- {sample_id} / {combo.id} ---")
    check_sample(cfg, sample_id, iou_threshold)

    work_dir = Path(cfg.project.work_dir) / sample_id
    candidates = read_candidates(work_dir / "candidates.json")
    detections = read_detections(work_dir / "detections.json")
    gt = load_gt(cfg.data.gt.global_file, sample_id)
    r = compute_prf1(detections, gt, iou_threshold, processed_frames=set(candidates.keys()))
    return {
        "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
    }


def run_sweep(config_path: str, base_overrides: list[str], sample_ids: list[str],
              combos: list[Combo], iou_threshold: float) -> list[dict[str, Any]]:
    from aero_eyes.config import load_config

    results: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        cfg0 = load_config(config_path, base_overrides)
        sample_dir = Path(cfg0.project.work_dir) / sample_id
        if not (sample_dir / "candidates.json").exists():
            log.warning("%s: candidates.json not found at %s -- skipping (run Stage 1+2 first).",
                        sample_id, sample_dir)
            continue

        backups = _backup_files(sample_dir)
        try:
            for combo in combos:
                try:
                    metrics = run_combo(config_path, base_overrides, combo, sample_id, iou_threshold)
                    results.append({"sample": sample_id, "combo": combo.id, "group": combo.group,
                                     "overrides": combo.overrides, "error": None, **metrics})
                except Exception as exc:  # noqa: BLE001 -- one bad combo must not kill the sweep
                    log.error("%s / %s failed: %s", sample_id, combo.id, exc)
                    traceback.print_exc()
                    results.append({"sample": sample_id, "combo": combo.id, "group": combo.group,
                                     "overrides": combo.overrides, "error": str(exc)})
        finally:
            _restore_files(sample_dir, backups)
    return results


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean P/R/F1 per combo id across all samples that produced a result."""
    by_combo: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if r.get("error") is None:
            by_combo.setdefault(r["combo"], []).append(r)

    summary = []
    for combo_id, rows in by_combo.items():
        n = len(rows)
        summary.append({
            "combo": combo_id,
            "group": rows[0]["group"],
            "n_samples": n,
            "mean_precision": sum(x["precision"] for x in rows) / n,
            "mean_recall": sum(x["recall"] for x in rows) / n,
            "mean_f1": sum(x["f1"] for x in rows) / n,
        })
    summary.sort(key=lambda x: x["mean_f1"], reverse=True)
    return summary


def write_markdown_leaderboard(path: Path, summary: list[dict[str, Any]], sample_ids: list[str]) -> None:
    lines = [
        "# Combo sweep leaderboard",
        "",
        f"Samples: {', '.join(sample_ids)}",
        "",
        "| Rank | Combo | Group | N samples | Mean P | Mean R | Mean F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(summary, 1):
        lines.append(
            f"| {i} | `{row['combo']}` | {row['group']} | {row['n_samples']} | "
            f"{row['mean_precision']:.3f} | {row['mean_recall']:.3f} | {row['mean_f1']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Sweep config combos from reports/Combo config thu nghiem threshold va "
                     "encoder.md, Stage 1+3 only, reusing existing candidates.json."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", action="append", default=[], help="repeat for multiple; omit for all samples in data_root")
    p.add_argument("--set", action="append", default=[], help="base overrides applied before every combo (e.g. data_root/work_dir)")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--z-values", default="2.5,2.0,1.5,1.0,0.5")
    p.add_argument("--downscale-values", default="1.0,0.5,0.1")
    p.add_argument("--include-heavy-encoders", action="store_true",
                    help="also try fgclip/radio (extra deps / gated weights required)")
    p.add_argument("--dry-run", action="store_true", help="print the combo list and exit")
    p.add_argument("--out", default=None, help="JSON report path (default: <work_dir>/combo_sweep_report.json)")
    p.add_argument("--out-md", default=None, help="Markdown leaderboard path (default: <work_dir>/combo_sweep_leaderboard.md)")
    args = p.parse_args()

    from aero_eyes.config import load_config

    z_values = [float(z) for z in args.z_values.split(",")]
    downscale_values = [float(d) for d in args.downscale_values.split(",")]
    combos = expand_sweeps(build_combos(args.include_heavy_encoders), z_values, downscale_values)

    if args.dry_run:
        print(f"{len(combos)} combo(s) would run:")
        for c in combos:
            tag = " [needs Stage 1]" if c.needs_stage1 else ""
            tag += " [recompute features]" if c.recompute else ""
            print(f"  {c.group:16s} {c.id:35s}{tag}  overrides={c.overrides}")
        return

    cfg0 = load_config(args.config, args.set)
    sample_ids = args.sample or [d.name for d in sorted(Path(cfg0.data.data_root).iterdir()) if d.is_dir()]
    print(f"Samples: {sample_ids}")
    print(f"{len(combos)} combo(s) per sample.\n")

    results = run_sweep(args.config, args.set, sample_ids, combos, args.iou_threshold)
    summary = summarize(results)

    print("\n" + "=" * 78)
    print(f"{'rank':<5}{'combo':<38}{'group':<16}{'P':>7}{'R':>7}{'F1':>7}")
    for i, row in enumerate(summary, 1):
        print(f"{i:<5}{row['combo']:<38}{row['group']:<16}"
              f"{row['mean_precision']:>7.3f}{row['mean_recall']:>7.3f}{row['mean_f1']:>7.3f}")
    print("=" * 78)

    n_errors = sum(1 for r in results if r.get("error"))
    if n_errors:
        print(f"\n{n_errors} combo run(s) failed and were skipped -- see log above / \"error\" field in the JSON report.")

    work_dir = Path(cfg0.project.work_dir)
    out_path = Path(args.out) if args.out else work_dir / "combo_sweep_report.json"
    out_md_path = Path(args.out_md) if args.out_md else work_dir / "combo_sweep_leaderboard.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "sample_ids": sample_ids,
        "iou_threshold": args.iou_threshold,
        "z_values": z_values,
        "downscale_values": downscale_values,
        "results": results,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    write_markdown_leaderboard(out_md_path, summary, sample_ids)
    print(f"\nWrote JSON report: {out_path}")
    print(f"Wrote leaderboard: {out_md_path}")


if __name__ == "__main__":
    main()

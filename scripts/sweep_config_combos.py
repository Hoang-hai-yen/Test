"""Full matrix sweep across the combos catalogued in reports/Combo config
thu nghiem threshold va encoder.md -- crosses every Stage-1-touching combo
(encoder swap, prototype fusion, reference-image preprocessing) against
every Stage-3-only combo (threshold method, online update mechanism,
verification_method=cluster, secondary filters, multi_ref_pooling) to find
the best-performing setup on your own footage.

WHY A MATRIX, NOT ONE-FACTOR-AT-A-TIME: Stage 1 and Stage 3 run as separate
processes here, but Stage 1's own config choices (which encoder, how the
prototype is fused, how reference images are preprocessed) determine the
EMBEDDING SPACE every Stage 3 decision (threshold, filter, cluster) then
operates on. A Stage 3 technique validated only under the default encoder
is not guaranteed to hold under a different one -- e.g. a filter tuned
against DINOv2's own similarity distribution might behave differently
against RADIO's. Testing the two axes independently (plain OFAT) never
actually exercises that interaction. Full cross-product IS still one-
factor-at-a-time WITHIN each axis (no combo changes more than one Stage-1
setting or more than one Stage-3 setting at once, except the deliberately-
stacked F5/P3) -- the matrix is what CROSSES those two already-atomic axes.

Cost control: Stage 1 is expensive (re-runs the encoder over 3 reference
photos, or re-extracts every candidate's feature when the embedding itself
changes) but its own OUTPUT (prototype.npz, and candidates.json's cached
features when the encoder changed) is reused for EVERY Stage-3-only combo
paired with it -- so each Stage-1-touching combo costs exactly ONE Stage 1
run (+ ONE feature-recompute pass when its encoder differs from what's
cached) no matter how many Stage-3-only combos are tested against it.

Pass --ofat-only to fall back to the cheaper old behavior instead: every
Stage-1 combo tested ONLY against the plain baseline Stage-3 config, and
every Stage-3-only combo tested ONLY against config.yaml's own current
stage1 settings (the "stage1_default" combo -- freshly rerun, NOT whatever
prototype.npz happens to already be on disk, see build_stage1_combos's own
docstring) -- the "row 0 + column 0" slice of the same matrix, ~53 runs
instead of ~713.

Only runs Stage 1 (when a combo touches the encoder/prototype/reference
preprocessing) + Stage 3 (cosine matching -> detections.json). Reuses each
sample's EXISTING candidates.json (Stage 1+2 output) instead of
regenerating it -- run Stage 1+2 normally first if candidates.json doesn't
exist yet. Score is P/R/F1 on detections.json vs ground truth, reusing
compute_prf1 (from check_cosine_effect.py), the same tool this project
already uses for this kind of comparison.

Every combo layers on top of a fixed BASELINE (the one already validated:
adaptive_threshold + adaptive_threshold_online + window_stat + z_score=2.0)
-- so a combo like "otsu" only changes adaptive_threshold_method, everything
else stays at the validated baseline. Combos that change the encoder or
reference-image preprocessing (feature_extractor.*, prototype.fusion,
domain_calibration.filter_target_like_frames, aerial_sim.*, segmentation.*,
crop_to_object) rerun Stage 1, which overwrites each sample's
prototype.npz; combos that also change the embedding (encoder swaps)
additionally force stage3.recompute_candidate_features=true ONCE per
Stage-1 combo (see cost control above), which overwrites candidates.json's
own cached per-candidate features in place. Both are backed up per sample
before the sweep starts and restored (byte-for-byte) once the sweep for that
sample finishes -- including on a crash -- so your existing
prototype.npz/candidates.json/detections.json are never left modified after
this script exits. All actual results only ever live in the JSON/Markdown
report this script writes, never in the pipeline's own working files.

A combo that raises (missing dependency, gated HF weights not downloaded,
OOM, ...) is logged and skipped -- one bad combo (e.g. a gated encoder you
haven't requested HF access for) does not abort the whole sweep. If a
Stage-1 combo's OWN Stage 1 run fails, every Stage-3-only combo paired with
it is recorded as failed too (nothing to pair them against).

Usage:
    # preview the combo list + pair count without running anything
    python -m scripts.sweep_config_combos --config configs/config.yaml --dry-run

    # full matrix sweep, all samples under data.data_root, default z/downscale values
    python -m scripts.sweep_config_combos --config configs/config.yaml \\
        --set project.work_dir=/kaggle/working/runs/exp001

    # cheaper row-0+column-0 sweep instead of the full matrix
    python -m scripts.sweep_config_combos --config configs/config.yaml --ofat-only

    # single sample, skip the heavy/gated encoders (fgclip/radio/evaclip), custom z sweep
    python -m scripts.sweep_config_combos --config configs/config.yaml \\
        --sample IDCard_0 --z-values 2.0,1.5,1.0

    # include fgclip/radio/evaclip too (needs their own deps/HF access already set up)
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
    # project.use_cache defaults to true -- run_stage1/run_stage3 both
    # short-circuit and return the EXISTING prototype.npz/detections.json
    # unchanged when one is already on disk (see their own cache-check at
    # the top of each function), which is exactly what a normal pipeline
    # run in this SAME work_dir would have left behind. Without forcing
    # this off, every combo here would silently read back that one
    # pre-existing file instead of ever actually re-running Stage 1/3 with
    # its own overrides -- every combo would score identically, which is
    # what this comment exists to prevent (confirmed to happen in
    # practice: 3 different z_score combos all returned the exact same
    # cached detections.json before this override was added).
    "project.use_cache": False,
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
    recompute: bool = False   # force stage3.recompute_candidate_features=true (once per Stage-1 combo)
    sweep_z: bool = False     # expand into one variant per --z-values
    sweep_downscale: bool = False   # expand into one variant per --downscale-values
    heavy: bool = False       # gated/extra-deps encoder -- skipped unless --include-heavy-encoders
    note: str = ""


def build_stage1_combos(include_heavy: bool) -> list[Combo]:
    """Combos that touch Stage 1 (encoder + prototype construction +
    reference-image preprocessing) -- each costs one Stage 1 rerun, reused
    across every Stage-3-only combo paired with it (see module docstring).
    Always includes "stage1_default" first: NO stage1.* overrides at all,
    so it reruns Stage 1 with exactly whatever configs/config.yaml's own
    stage1: section currently says -- the encoder every Stage-3-only combo
    is tested against. Deliberately NOT "reuse whatever prototype.npz is
    already on disk" (needs_stage1=False) -- a work_dir from an earlier
    manual run, or an earlier session with a different config.yaml, can
    easily hold a prototype.npz built from a DIFFERENT stage1 config than
    what's currently written, with no way to tell from the file alone;
    forcing a rerun (project.use_cache is already forced off in
    BASELINE_OVERRIDES, so this is never skipped) guarantees "stage1_default"
    actually means "this file's own current stage1 settings", not
    "whatever happened to be cached from some earlier, possibly different,
    run"."""
    combos = [
        Combo("stage1_default", "encoder", {}, needs_stage1=True,
              note="no stage1.* overrides -- reruns Stage 1 fresh with config.yaml's own current "
                   "stage1 settings, so it's guaranteed consistent with THIS run, not whatever a "
                   "stale on-disk prototype.npz from some earlier config happens to hold"),

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
        Combo("E8_siglip2", "encoder", {"stage1.feature_extractor.model": "siglip2"},
              needs_stage1=True, recompute=True,
              note="Global-Local + Masked Prediction losses over SigLIP -- 2nd-strongest "
                   "fine-grained/near-duplicate evidence after FG-CLIP, standard transformers "
                   "classes (no trust_remote_code, lower integration risk than fgclip)"),
        Combo("E9_evaclip", "encoder", {"stage1.feature_extractor.model": "evaclip"},
              needs_stage1=True, recompute=True, heavy=True,
              note="needs a NEW dependency (open_clip_torch) not used by any other extractor; "
                   "weaker fine-grained evidence than fgclip/siglip2, only zero-shot classification numbers"),
        Combo("E10_dinotxt", "encoder", {"stage1.feature_extractor.model": "dinotxt"},
              needs_stage1=True, recompute=True,
              note="LiT-aligned text encoder over a frozen DINOv2 ViT-L/14 -- adds language/"
                   "semantic grounding without leaving the DINO family; no new dependency, but "
                   "preprocessing/output details not independently verified without a live download"),

        # ---- prototype construction (mục 2) -- needs Stage 1, same encoder ----
        # fusion="mean" is config.yaml's own default, already covered by
        # stage1_default (no override needed) -- "max"/"concat_then_pca"/
        # "agreement_weighted" are the 3 non-default alternatives.
        Combo("P1a_fusion_max", "prototype",
              {"stage1.prototype.fusion": "max"}, needs_stage1=True),
        Combo("P1b_fusion_concat_then_pca", "prototype",
              {"stage1.prototype.fusion": "concat_then_pca"}, needs_stage1=True,
              note="literature favors this for COMPLEMENTARY views -- weaker fit + "
                   "statistically unstable for 3 near-duplicate close-ups per this project's own notes"),
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
    ]
    if not include_heavy:
        combos = [c for c in combos if not c.heavy]
    return combos


def build_stage3_combos() -> list[Combo]:
    """Combos that only touch Stage 3 -- never rerun Stage 1, safe/cheap to
    pair against every Stage-1 combo above."""
    return [
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

        # ---- verification_method=cluster (mục 7) -- a THIRD, mutually
        # exclusive decision mechanism vs. the threshold/online_method/
        # filter groups above (never combines with them -- verification_
        # method IS the switch here, cluster_verification.enabled is NOT).
        # OFAT star design around the hdbscan+cosine default: vary
        # cluster_method holding pairwise_metric=cosine (V1 vs V4), vary
        # pairwise_metric holding cluster_method=hdbscan (V1 vs V2 vs V3).
        Combo("V1_cluster_hdbscan_cosine", "cluster_verification",
              {"stage3.verification_method": "cluster"},
              note="hdbscan+cosine defaults -- already A/B tested and UNDERPERFORMED threshold "
                   "(TP/FP margin +10.5pp vs +61.4/+74.5pp), kept as the reference point"),
        Combo("V2_cluster_hdbscan_l1", "cluster_verification", {
            "stage3.verification_method": "cluster",
            "stage3.cluster_verification.pairwise_metric": "l1",
        }),
        Combo("V3_cluster_hdbscan_mahalanobis", "cluster_verification", {
            "stage3.verification_method": "cluster",
            "stage3.cluster_verification.pairwise_metric": "mahalanobis",
        }),
        Combo("V4_cluster_spectral_cosine", "cluster_verification", {
            "stage3.verification_method": "cluster",
            "stage3.cluster_verification.cluster_method": "spectral",
        }),

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
        # stage3.similarity: "cosine" is the baseline default (no combo
        # needed); "rmd" (F3) is the documented/rationale-backed alternative,
        # "l1"/"l2" (negated distances, unbounded/typically negative --
        # adaptive_min_floor auto-skips for any non-cosine metric) are the
        # 2 simpler built-in alternatives, included for completeness.
        Combo("F3_rmd", "filter", {"stage3.similarity": "rmd"}),
        Combo("F3a_l1", "filter", {"stage3.similarity": "l1"}),
        Combo("F3b_l2", "filter", {"stage3.similarity": "l2"}),
        Combo("F4_margin_cluster", "filter", {
            "stage3.margin_verification.enabled": True,
            "stage3.margin_verification.tau_margin": 0.05,
            "stage3.cluster_secondary_filter.enabled": True,
            "stage3.cluster_secondary_filter.window_admission_min_consecutive_hits": 2,
        }),
        Combo("F4c_cluster_secondary_no_accumulate", "filter", {
            "stage3.cluster_secondary_filter.enabled": True,
            "stage3.cluster_secondary_filter.accumulate_new_anchors": False,
        }, note="isolates whether accumulating trusted anchors helps or hurts -- suspected "
                "culprit for cluster_secondary_filter not meaningfully improving precision in "
                "real-footage testing: a single early FP slipping past admission could poison "
                "the window and keep attracting texturally-similar confusers for the rest of the video"),
        Combo("F5_stacked", "filter", {
            "stage3.negative_prototype_filter.enabled": True,
            "stage3.cluster_secondary_filter.enabled": True,
            "stage3.cluster_secondary_filter.window_admission_min_consecutive_hits": 2,
            "stage3.identity_chain_filter.enabled": True,
            "stage3.identity_chain_filter.spatial_weight": 0.3,
            "stage3.margin_verification.enabled": True,
            "stage3.margin_verification.tau_margin": 0.05,
        }),

        # ---- multi_ref_pooling (mục 6) -- "max" == stage1_default's own
        # config.yaml default, not repeated here ----
        Combo("M1_mean", "multi_ref_pooling", {"accuracy.cheap_boosters.multi_ref_pooling": "mean"}),
        Combo("M3_min", "multi_ref_pooling", {"accuracy.cheap_boosters.multi_ref_pooling": "min"}),
        Combo("M4_agreement_weighted", "multi_ref_pooling",
              {"accuracy.cheap_boosters.multi_ref_pooling": "agreement_weighted"}),
    ]


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


def build_pairs(stage1_combos: list[Combo], stage3_combos: list[Combo],
                 matrix: bool) -> list[tuple[Combo, Combo]]:
    """(stage1_combo, stage3_combo) pairs to actually run.

    matrix=True: full cross product -- every Stage-1 combo x every Stage-3
    combo (see module docstring for why this matters).
    matrix=False (--ofat-only): the "row 0 + column 0" slice of that same
    matrix -- every Stage-1 combo paired ONLY with the plain baseline
    Stage-3 config, and stage1_default paired with EVERY Stage-3 combo.
    """
    if matrix:
        return [(s1, s3) for s1 in stage1_combos for s3 in stage3_combos]

    stage1_default = next(c for c in stage1_combos if c.id == "stage1_default")
    baseline_stage3 = Combo("baseline", "threshold", {})
    pairs = [(s1, baseline_stage3) for s1 in stage1_combos if s1.id != "stage1_default"]
    pairs += [(stage1_default, s3) for s3 in stage3_combos]
    return pairs


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


def run_stage1_once(config_path: str, base_overrides: list[str], s1: Combo, sample_id: str) -> None:
    from aero_eyes.config import load_config
    from aero_eyes.stages.stage1 import run_stage1

    override_strs = list(base_overrides) + _override_strings(BASELINE_OVERRIDES) + _override_strings(s1.overrides)
    cfg = load_config(config_path, override_strs)
    run_stage1(cfg, sample_id)


def run_pair(config_path: str, base_overrides: list[str], s1: Combo, s3: Combo,
             sample_id: str, iou_threshold: float, force_recompute: bool) -> dict[str, Any]:
    from aero_eyes.config import load_config
    from aero_eyes.stages.stage3 import run_stage3
    from aero_eyes.utils.io import load_gt, read_candidates, read_detections
    from scripts.check_cosine_effect import compute_prf1

    override_strs = (
        list(base_overrides)
        + _override_strings(BASELINE_OVERRIDES)
        + _override_strings(s1.overrides)
        + _override_strings(s3.overrides)
        + (["stage3.recompute_candidate_features=true"] if force_recompute else [])
    )
    cfg = load_config(config_path, override_strs)
    run_stage3(cfg, sample_id)

    work_dir = Path(cfg.project.work_dir) / sample_id
    candidates = read_candidates(work_dir / "candidates.json")
    detections = read_detections(work_dir / "detections.json")
    gt = load_gt(cfg.data.gt.global_file, sample_id)
    r = compute_prf1(detections, gt, iou_threshold, processed_frames=set(candidates.keys()))
    print(f"  {sample_id} / {s1.id} + {s3.id}: "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}"
          f"{' [recompute]' if force_recompute else ''}")
    return {
        "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
    }


def run_sweep(config_path: str, base_overrides: list[str], sample_ids: list[str],
              stage1_combos: list[Combo], stage3_combos: list[Combo],
              matrix: bool, iou_threshold: float) -> list[dict[str, Any]]:
    from aero_eyes.config import load_config

    # Group pairs by their Stage-1 combo so each one's Stage 1 rerun (+ its
    # ONE forced feature-recompute, if any) happens exactly once, reused for
    # every Stage-3 combo paired with it.
    pairs = build_pairs(stage1_combos, stage3_combos, matrix)
    by_stage1: dict[str, tuple[Combo, list[Combo]]] = {}
    for s1, s3 in pairs:
        entry = by_stage1.setdefault(s1.id, (s1, []))
        entry[1].append(s3)

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
            for s1, s3_list in by_stage1.values():
                if s1.needs_stage1:
                    try:
                        run_stage1_once(config_path, base_overrides, s1, sample_id)
                    except Exception as exc:  # noqa: BLE001
                        log.error("%s / stage1(%s) failed: %s -- skipping %d paired Stage-3 combo(s).",
                                  sample_id, s1.id, exc, len(s3_list))
                        traceback.print_exc()
                        for s3 in s3_list:
                            results.append({
                                "sample": sample_id, "combo": f"{s1.id}__{s3.id}",
                                "group": f"{s1.group}+{s3.group}", "error": f"stage1 failed: {exc}",
                            })
                        continue

                recompute_done = not s1.recompute
                for s3 in s3_list:
                    combo_id = f"{s1.id}__{s3.id}"
                    try:
                        force_recompute = s1.recompute and not recompute_done
                        metrics = run_pair(config_path, base_overrides, s1, s3, sample_id,
                                            iou_threshold, force_recompute)
                        if force_recompute:
                            recompute_done = True
                        results.append({"sample": sample_id, "combo": combo_id,
                                         "group": f"{s1.group}+{s3.group}", "error": None, **metrics})
                    except Exception as exc:  # noqa: BLE001 -- one bad pair must not kill the sweep
                        log.error("%s / %s failed: %s", sample_id, combo_id, exc)
                        traceback.print_exc()
                        results.append({"sample": sample_id, "combo": combo_id,
                                         "group": f"{s1.group}+{s3.group}", "error": str(exc)})
        finally:
            _restore_files(sample_dir, backups)
    return results


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean P/R/F1 per (stage1, stage3) combo pair across all samples that
    produced a result."""
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
        "| Rank | Stage1 + Stage3 combo | Group | N samples | Mean P | Mean R | Mean F1 |",
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
        description="Full matrix sweep of config combos from reports/Combo config thu nghiem "
                     "threshold va encoder.md, Stage 1+3 only, reusing existing candidates.json."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", action="append", default=[], help="repeat for multiple; omit for all samples in data_root")
    p.add_argument("--set", action="append", default=[], help="base overrides applied before every combo (e.g. data_root/work_dir)")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--z-values", default="2.5,2.0,1.5,1.0,0.5")
    p.add_argument("--downscale-values", default="1.0,0.5,0.1")
    p.add_argument("--include-heavy-encoders", action="store_true",
                    help="also try fgclip/radio/evaclip (extra deps / gated weights required)")
    p.add_argument("--ofat-only", action="store_true",
                    help="cheaper row-0+column-0 slice instead of the full Stage1 x Stage3 matrix "
                         "(~52 runs instead of ~680) -- see module docstring")
    p.add_argument("--dry-run", action="store_true", help="print the combo/pair counts and exit")
    p.add_argument("--out", default=None, help="JSON report path (default: <work_dir>/combo_sweep_report.json)")
    p.add_argument("--out-md", default=None, help="Markdown leaderboard path (default: <work_dir>/combo_sweep_leaderboard.md)")
    args = p.parse_args()

    from aero_eyes.config import load_config

    z_values = [float(z) for z in args.z_values.split(",")]
    downscale_values = [float(d) for d in args.downscale_values.split(",")]
    stage1_combos = expand_sweeps(build_stage1_combos(args.include_heavy_encoders), z_values, downscale_values)
    stage3_combos = expand_sweeps(build_stage3_combos(), z_values, downscale_values)
    matrix = not args.ofat_only
    pairs = build_pairs(stage1_combos, stage3_combos, matrix)

    if args.dry_run:
        mode = "FULL MATRIX" if matrix else "OFAT-only (row 0 + column 0)"
        print(f"Mode: {mode}")
        print(f"{len(stage1_combos)} Stage-1 combo(s) x {len(stage3_combos)} Stage-3 combo(s) "
              f"-> {len(pairs)} pair(s) per sample.\n")
        print("Stage-1 combos:")
        for c in stage1_combos:
            tag = " [recompute]" if c.recompute else ""
            print(f"  {c.group:12s} {c.id:35s}{tag}  overrides={c.overrides}")
        print("\nStage-3 combos:")
        for c in stage3_combos:
            print(f"  {c.group:20s} {c.id:35s}  overrides={c.overrides}")
        return

    cfg0 = load_config(args.config, args.set)
    sample_ids = args.sample or [d.name for d in sorted(Path(cfg0.data.data_root).iterdir()) if d.is_dir()]
    mode = "full matrix" if matrix else "OFAT-only"
    print(f"Samples: {sample_ids}")
    print(f"Mode: {mode} -- {len(pairs)} pair(s) per sample.\n")

    results = run_sweep(args.config, args.set, sample_ids, stage1_combos, stage3_combos, matrix, args.iou_threshold)
    summary = summarize(results)

    print("\n" + "=" * 90)
    print(f"{'rank':<5}{'combo':<55}{'group':<24}{'P':>7}{'R':>7}{'F1':>7}")
    for i, row in enumerate(summary, 1):
        print(f"{i:<5}{row['combo']:<55}{row['group']:<24}"
              f"{row['mean_precision']:>7.3f}{row['mean_recall']:>7.3f}{row['mean_f1']:>7.3f}")
    print("=" * 90)

    n_errors = sum(1 for r in results if r.get("error"))
    if n_errors:
        print(f"\n{n_errors} pair(s) failed and were skipped -- see log above / \"error\" field in the JSON report.")

    work_dir = Path(cfg0.project.work_dir)
    out_path = Path(args.out) if args.out else work_dir / "combo_sweep_report.json"
    out_md_path = Path(args.out_md) if args.out_md else work_dir / "combo_sweep_leaderboard.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "mode": mode,
        "sample_ids": sample_ids,
        "iou_threshold": args.iou_threshold,
        "z_values": z_values,
        "downscale_values": downscale_values,
        "n_stage1_combos": len(stage1_combos),
        "n_stage3_combos": len(stage3_combos),
        "n_pairs": len(pairs),
        "results": results,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    write_markdown_leaderboard(out_md_path, summary, sample_ids)
    print(f"\nWrote JSON report: {out_path}")
    print(f"Wrote leaderboard: {out_md_path}")


if __name__ == "__main__":
    main()

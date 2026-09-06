"""Diagnostic: does accuracy.cheap_boosters.multi_ref_pooling="max" let a
candidate win purely by matching ONE reference image well, even though it
mismatches the other reference(s) badly? That's the exact mechanism
suspected behind dynamic_prototype absorbing a confuser (e.g. a static
background object -- a yellow rectangular road marking, in this project's
LifeJacket_1 sample -- that happens to closely resemble ONE particular
reference photo's color/lighting/angle even though it looks nothing like
the other reference(s)).

For every candidate in candidates.json, computes its cosine (or l1/l2)
score against EACH reference image INDIVIDUALLY (via
aero_eyes.stages.stage3._score_against_ref, imported -- not reimplemented),
not just the single pooled value Stage 3 actually uses. Also replays
stage3.dynamic_prototype's own rounds (via run_dynamic_prototype_rounds,
imported -- a no-op if disabled) BEFORE computing per-ref scores/spread,
so both reflect the FINAL reference set / prototype Stage 3's own
effective threshold was actually computed from -- dynamic_prototype can
append new references (or blend a new prototype) each round, and scoring
against only the ORIGINAL references would badly understate real scores
once that has happened, making any threshold comparison meaningless (this
was a real bug in an earlier version of this script: pooled scores came
out far below the real effective_threshold for EVERY group, including
genuine matches, because they were being compared on different scales).

Then splits candidates into 3 groups by ground truth:
  genuine_match   -- GT present on this frame AND IoU >= --iou-threshold
  wrong_location  -- GT present on this frame but IoU < --iou-threshold
                      (some other box, or a badly-placed one)
  unverifiable    -- no GT entry at all on this frame (object confirmed
                      ABSENT there, not merely unlabeled -- see project
                      convention) -- this is where a static background
                      confuser's candidates land, since it's never the
                      real target.

For each group, reports the POOLED score distribution (using whatever
accuracy.cheap_boosters.multi_ref_pooling is CURRENTLY set to) and the
PER-REFERENCE SPREAD (max_ref_score - min_ref_score) distribution -- a
LARGE spread means "matches one reference much better than the others"
(exactly what max-pooling can be fooled by); a small spread means "agrees
across every reference view" (much harder to fake by accident). If
unverifiable/wrong_location candidates show a systematically larger
spread than genuine_match ones, that's direct evidence an agreement-based
gate would help.

Also SIMULATES an agreement gate at a few floor values: a candidate's
pooled score gets replaced by its own MIN per-ref score (a hard veto)
whenever that min falls below the floor, and reports, at each floor, how
many confuser-suspect candidates would drop below the effective match
threshold vs how many GENUINE matches would ALSO get demoted (the gate's
cost) -- so you can pick a concrete multi_ref_pooling / floor setting
backed by actual numbers on this dataset instead of guessing.

The "effective match threshold" used for this simulation is read directly
from detections.json (the ACTUAL bar Stage 3 last used -- adaptive_threshold's
real per-video value when enabled, not stage3.match_threshold's static
config default, which can be wildly irrelevant when adaptive_threshold is
on: raw cosine scores from this project's ground-to-aerial domain gap can
sit as low as ~0.05-0.15, while match_threshold's own default is 0.55).
Falls back to stage3.match_threshold with a loud warning only if
detections.json is missing or didn't record a threshold.

Read-only: does not touch detections.json/tracks.json/prototype.npz.
Needs >=2 cached reference-image features (per_ref_features in
prototype.npz) -- i.e. accuracy.cheap_boosters.multi_reference_embedding
must have been on when Stage 1/1b built the prototype, regardless of
whether it's on in the CURRENT config being checked.

Usage:
    python -m scripts.check_multi_ref_agreement --config configs/config.yaml --sample LifeJacket_1
    python -m scripts.check_multi_ref_agreement --config configs/config.yaml --sample LifeJacket_1 \\
        --set accuracy.cheap_boosters.multi_ref_pooling=mean   # compare against max side by side
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from aero_eyes.stages.stage3 import _pool_sims, _score_against_ref, run_dynamic_prototype_rounds
from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_detections_threshold, read_prototype

log = logging.getLogger(__name__)

_DEFAULT_FLOORS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def check_sample(
    cfg, sample_id: str, iou_threshold: float,
    agreement_floors: tuple = _DEFAULT_FLOORS,
) -> None:
    from aero_eyes.stages.stage2 import read_candidates_with_features

    work_dir = Path(cfg.project.work_dir) / sample_id
    proto_path = work_dir / cfg.stage1.prototype.cache_name
    cand_path = work_dir / "candidates.json"

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}, "
          f"multi_ref_pooling={cfg.accuracy.cheap_boosters.multi_ref_pooling}) ===")

    if not proto_path.exists():
        print(f"  prototype.npz not found at {proto_path} -- run Stage 1 first.")
        return
    if not cand_path.exists():
        print(f"  candidates.json not found at {cand_path} -- run Stage 2 (legacy) or the "
              "GeCo2 candidate stage first.")
        return

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"  not found in {gt_file}, skipping.")
        return

    prototype, _, per_ref_features = read_prototype(proto_path)
    if len(per_ref_features) < 2:
        print(f"  only {len(per_ref_features)} cached reference feature(s) found in "
              f"{proto_path} -- per-reference agreement needs >=2 (re-run Stage 1 with "
              "accuracy.cheap_boosters.multi_reference_embedding=true and >=2 reference "
              "images if this is unexpected).")
        return

    candidates, _ = read_candidates_with_features(cand_path)
    all_entries = [
        (fi, det, getattr(det, "_feature", None))
        for fi, dets in candidates.items() for det in dets
    ]
    all_entries = [(fi, det, feat) for fi, det, feat in all_entries if feat is not None]
    if not all_entries:
        print("  no candidate features found -- nothing to check.")
        return

    all_frame_idxs = [e[0] for e in all_entries]
    all_dets = [e[1] for e in all_entries]
    all_feats = np.stack([e[2] for e in all_entries], axis=0)

    s3 = cfg.stage3
    multi_ref_pooling = cfg.accuracy.cheap_boosters.multi_ref_pooling
    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )

    if use_multi_ref:
        sims_per_ref = [_score_against_ref(all_feats, ref, s3.similarity) for ref in per_ref_features]
        initial_pooled = _pool_sims(sims_per_ref, multi_ref_pooling)
    else:
        initial_pooled = _score_against_ref(all_feats, prototype, s3.similarity)

    # Replay dynamic_prototype's rounds (no-op if disabled) so `pooled` below
    # reflects the FINAL reference set / prototype Stage 3's own effective
    # threshold was actually computed from -- scoring only against the
    # ORIGINAL references (as an earlier version of this script did) badly
    # understates real scores once rounds have appended new references or
    # blended a new prototype, making any threshold comparison meaningless.
    _, pooled, final_per_ref_features = run_dynamic_prototype_rounds(
        sample_id, all_feats, initial_pooled, prototype, list(per_ref_features),
        use_multi_ref, multi_ref_pooling, s3.similarity, s3.dynamic_prototype,
    )

    if use_multi_ref:
        sims_per_ref_final = np.stack(
            [_score_against_ref(all_feats, ref, s3.similarity) for ref in final_per_ref_features], axis=0,
        )  # [R, N] -- R may be larger than len(per_ref_features) if rounds appended any
        per_ref_min = sims_per_ref_final.min(axis=0)
        spread = sims_per_ref_final.max(axis=0) - per_ref_min
        if len(final_per_ref_features) > len(per_ref_features):
            print(
                f"  note: stage3.dynamic_prototype appended "
                f"{len(final_per_ref_features) - len(per_ref_features)} dynamically-derived "
                f"reference(s) during its rounds -- scores/spread below use the FULL final set "
                f"of {len(final_per_ref_features)} references (what Stage 3's own final "
                "threshold decision actually used), not just the original ones. See "
                "check_dynamic_prototype_purity.py --export-crops to inspect which candidates "
                "got appended."
            )
    else:
        sims_per_ref_final = None
        per_ref_min = pooled
        spread = np.zeros_like(pooled)
        print(
            "  note: accuracy.cheap_boosters.multi_reference_embedding / accuracy.mode "
            "currently means Stage 3 does NOT use per-reference scoring -- dynamic_prototype "
            "(if enabled) blends into ONE prototype vector instead. Per-reference spread below "
            "is not applicable (always 0); only the pooled-score groups and the threshold gate "
            "simulation are meaningful in this mode."
        )

    groups: dict[str, list[int]] = {"genuine_match": [], "wrong_location": [], "unverifiable": []}
    for i in range(len(all_dets)):
        fi = all_frame_idxs[i]
        if fi in gt:
            iou = box_iou(gt[fi], all_dets[i].box)
            groups["genuine_match" if iou >= iou_threshold else "wrong_location"].append(i)
        else:
            groups["unverifiable"].append(i)

    print("  per-group pooled score & per-reference spread (spread = best-ref score - worst-ref score):")
    for label in ("genuine_match", "wrong_location", "unverifiable"):
        idxs = groups[label]
        if not idxs:
            print(f"    {label}: n=0")
            continue
        idxs_arr = np.array(idxs)
        print(
            f"    {label}: n={len(idxs)}  "
            f"pooled(mean={pooled[idxs_arr].mean():.3f}, p50={np.percentile(pooled[idxs_arr], 50):.3f})  "
            f"spread(mean={spread[idxs_arr].mean():.3f}, p50={np.percentile(spread[idxs_arr], 50):.3f})"
        )

    for label in ("unverifiable", "wrong_location"):
        if sims_per_ref_final is None:
            break  # not use_multi_ref -- no per-ref breakdown to show, already noted above
        idxs = groups[label]
        if not idxs:
            continue
        top = sorted(idxs, key=lambda i: spread[i], reverse=True)[:5]
        print(f"  highest-spread {label} candidates (confuser suspects -- matches one ref far "
              "better than the others):")
        for i in top:
            per_ref_str = ", ".join(f"{s:.3f}" for s in sims_per_ref_final[:, i])
            print(
                f"    frame {all_frame_idxs[i]}: per-ref=[{per_ref_str}] "
                f"pooled({multi_ref_pooling})={pooled[i]:.3f} spread={spread[i]:.3f}"
            )

    if groups["genuine_match"]:
        g_spread_p50 = np.percentile(spread[np.array(groups["genuine_match"])], 50)
        for label in ("unverifiable", "wrong_location"):
            if not groups[label]:
                continue
            l_spread_p50 = np.percentile(spread[np.array(groups[label])], 50)
            if l_spread_p50 > g_spread_p50 * 1.5:
                print(
                    f"  -> {label} candidates have a MUCH larger typical per-ref spread "
                    f"({l_spread_p50:.3f}) than genuine matches ({g_spread_p50:.3f}) -- consistent "
                    "with single-reference overfitting via multi_ref_pooling=max; an agreement "
                    "gate (see simulation below) is likely to help here."
                )
            elif l_spread_p50 > g_spread_p50 * 1.1:
                print(
                    f"  -> {label} candidates have a SOMEWHAT larger typical per-ref spread "
                    f"({l_spread_p50:.3f}) than genuine matches ({g_spread_p50:.3f}) -- a real but "
                    "modest effect; single-reference overfitting is probably only PART of the "
                    "story here, not the whole explanation for these candidates' scores."
                )
            else:
                print(
                    f"  -> {label} candidates' typical per-ref spread ({l_spread_p50:.3f}) is NOT "
                    f"meaningfully larger than genuine matches' ({g_spread_p50:.3f}) -- an "
                    "agreement gate is unlikely to selectively filter these; the confuser is "
                    "matching ALL references somewhat evenly, not overfitting to just one."
                )

    det_path = work_dir / "detections.json"
    effective_threshold = read_detections_threshold(det_path) if det_path.exists() else None
    if effective_threshold is None:
        effective_threshold = s3.match_threshold
        print(
            f"  warning: could not read the real effective threshold from {det_path} "
            "(missing, or Stage 3 didn't record one) -- falling back to the static "
            f"stage3.match_threshold={s3.match_threshold:.3f}, which is almost certainly "
            "WRONG if stage3.adaptive_threshold=true (the gate simulation below would then "
            "compare against a threshold far from what Stage 3 actually used -- run Stage 3 "
            "at least once for this sample first)."
        )
    print(
        f"  simulated agreement gate (pooled score replaced by the candidate's own MIN "
        f"per-ref score whenever that min < floor -- vetoes a single-reference-only match; "
        f"pass/fail measured against effective_threshold={effective_threshold:.3f}):"
    )
    confuser_idx = np.array(groups["unverifiable"] + groups["wrong_location"], dtype=int)
    genuine_idx = np.array(groups["genuine_match"], dtype=int)
    for floor in agreement_floors:
        gated = np.where(per_ref_min < floor, per_ref_min, pooled)
        n_conf_before = int((pooled[confuser_idx] >= effective_threshold).sum()) if confuser_idx.size else 0
        n_conf_after = int((gated[confuser_idx] >= effective_threshold).sum()) if confuser_idx.size else 0
        n_gen_before = int((pooled[genuine_idx] >= effective_threshold).sum()) if genuine_idx.size else 0
        n_gen_after = int((gated[genuine_idx] >= effective_threshold).sum()) if genuine_idx.size else 0
        print(
            f"    floor={floor:.2f}: confuser-suspects passing threshold {n_conf_before} -> {n_conf_after}  "
            f"|  genuine matches passing threshold {n_gen_before} -> {n_gen_after} "
            f"(cost: {n_gen_before - n_gen_after} genuine match(es) newly demoted below threshold)"
        )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Check whether multi_ref_pooling=max lets a candidate win by matching "
        "only ONE reference image well, and simulate an agreement-based gate against it."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as a correct localization (default: 0.5)")
    p.add_argument("--agreement-floors", default=",".join(str(f) for f in _DEFAULT_FLOORS),
                    help="Comma-separated floor values to simulate (default: "
                    f"{','.join(str(f) for f in _DEFAULT_FLOORS)})")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    floors = tuple(float(x) for x in args.agreement_floors.split(","))

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        check_sample(cfg, sid, args.iou_threshold, agreement_floors=floors)


if __name__ == "__main__":
    main()

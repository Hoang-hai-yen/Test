"""Fit and evaluate PCA whitening on DINOv3 CLS embeddings of this project's own
labeled crops, to see whether whitening makes cosine(reference prototype,
video crop) separate the target from clutter better than raw cosine.

What it does
------------
1. Embeds, for every video in annotations.json, the reference photos (prepared
   like stage1, exactly as scripts/train_lora_dinov3.py does), the GT crops
   ("pos") and the clutter crops from candidates.json ("neg"), with the
   CURRENT stage1.feature_extractor settings (so --set ...dinov3_lora_weights_path
   works too). Embeddings are cached (--cache), so re-running with different
   whitening settings does not touch the GPU again.
2. Fits a whitening transform  z = normalize(((x - mean) @ V_k) * (lambda_k + eps*mean(lambda))^(-power/2))
   on a pooled set of embeddings (--fit-on), for every combination of the grid
   options --n-components/--drop-top/--power/--eps.
3. Evaluates each combination OUT OF SAMPLE and compares with raw cosine:
   per held-out video, cosine(prototype from that video's refs, crop) of GT
   crops vs clutter crops -> AUROC, TPR@FPR 1% / 0.1% (same metrics as the
   LoRA script). --eval loo = leave-one-object-out (whitening fitted without
   the object it is scored on), --eval split = one fixed split like
   train_lora_dinov3's --val-objects/--val-suffix.
4. Saves the best combination, refitted on ALL videos, to <out-dir>/whitening.npz
   (+ results.json). Enable it in Stage 3 with --set stage3.whitening.enabled=true
   --set stage3.whitening.weights_path=<out-dir>/whitening.npz.
5. Optionally (--probe-refs/--probe-crops) prints raw vs whitened cosine for
   hand-picked crops, e.g. an ID-card crop next to a blank-paper crop.

Usage:
    python -m scripts.fit_pca_whitening --config configs/config.yaml \
        --set stage1.feature_extractor.model=dinov3 \
        --set stage1.feature_extractor.preprocess_mode=resize_then_crop \
        --set stage1.feature_extractor.candidate_preprocess_mode=pad_to_square \
        --eval loo --n-components 64 128 256 --power 0.5 1.0 \
        --probe-refs data/IDCard_1/object_images --probe-crops crops/card.jpg crops/paper.jpg
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aero_eyes.models.whitening import Whitener, WhiteningParams, fit_whitener  # noqa: F401 (re-exported)

log = logging.getLogger("fit_pca_whitening")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIT_SOURCES = {"pos": ("pos",), "neg": ("neg",), "pos+neg": ("pos", "neg"), "all": ("refs", "pos", "neg")}


# ---------------------------------------------------------------------------
# Evaluation on cached embeddings (pure numpy, unit-tested)
# ---------------------------------------------------------------------------
# emb: {video_id: {"obj": str, "refs": [R,D], "pos": [P,D], "neg": [N,D]}}

def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def eval_video(d: dict, transform) -> dict:
    """Same protocol as train_lora_dinov3.eval_video, on cached embeddings."""
    from scripts.train_lora_dinov3 import separation_metrics

    proto = _unit(transform(d["refs"]).mean(axis=0))
    pos = transform(d["pos"]) @ proto if len(d["pos"]) else np.zeros(0)
    neg = transform(d["neg"]) @ proto if len(d["neg"]) else np.zeros(0)
    return separation_metrics(pos, neg)


def _identity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32) / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def summarize(per_video: dict[str, dict]) -> dict:
    def mean(key):
        vals = [m[key] for m in per_video.values() if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {"auroc": mean("auroc"), "tpr_fpr1pct": mean("tpr_fpr1pct"), "tpr_fpr0p1pct": mean("tpr_fpr0p1pct")}


def pooled(emb: dict, video_ids: list[str], fit_on: str) -> np.ndarray:
    parts = [emb[v][s] for v in video_ids for s in FIT_SOURCES[fit_on] if len(emb[v][s])]
    if not parts:
        raise ValueError(f"No embeddings to fit on for --fit-on {fit_on} over {video_ids}.")
    return np.concatenate(parts)


def make_folds(emb: dict, mode: str, val_objects=(), val_suffix=None) -> list[tuple[list[str], list[str]]]:
    from scripts.train_lora_dinov3 import split_videos

    ids = sorted(emb)
    if mode == "loo":
        objs = sorted({emb[v]["obj"] for v in ids})
        return [([v for v in ids if emb[v]["obj"] != o], [v for v in ids if emb[v]["obj"] == o]) for o in objs]
    return [split_videos(ids, val_objects, val_suffix)]


def evaluate_grid(emb: dict, grid: list[WhiteningParams], fit_on: str, folds) -> dict:
    """Out-of-sample metrics for raw cosine ('baseline') and for every
    WhiteningParams in grid, over all folds."""
    val_ids = [v for _, val in folds for v in val]
    baseline = {v: eval_video(emb[v], _identity) for v in val_ids}
    results = {"baseline": {"per_video": baseline, **summarize(baseline)}, "grid": []}
    for params in grid:
        per_video = {}
        for train_ids, val in folds:
            w = fit_whitener(pooled(emb, train_ids, fit_on), params)
            for v in val:
                per_video[v] = eval_video(emb[v], w.transform)
        results["grid"].append({"params": asdict(params), "per_video": per_video, **summarize(per_video)})
    return results


# ---------------------------------------------------------------------------
# Embedding collection (GPU / disk I/O)
# ---------------------------------------------------------------------------

def collect_embeddings(cfg, ext, ids: list[str], args) -> dict:
    from scripts.train_lora_dinov3 import build_video_data

    rng = np.random.default_rng(args.seed)
    dim = ext._dim()
    emb = {}
    for vid in ids:
        vd = build_video_data(cfg, vid, args, rng)

        def enc(imgs, mode):
            return ext.extract(imgs, batch_size=args.micro_batch, preprocess_mode=mode) if imgs \
                else np.zeros((0, dim), np.float32)

        emb[vid] = {
            "obj": vd.obj,
            "refs": enc(vd.refs, ext.preprocess_mode),
            "pos": enc(vd.pos, ext.candidate_preprocess_mode),
            "neg": enc(vd.neg, ext.candidate_preprocess_mode),
        }
        del vd
    return emb


def save_embeddings(emb: dict, path: Path) -> None:
    flat = {}
    for vid, d in emb.items():
        flat[f"{vid}|obj"] = np.array(d["obj"])
        for s in ("refs", "pos", "neg"):
            flat[f"{vid}|{s}"] = d[s]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_embeddings(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    emb: dict = {}
    for key in z.files:
        vid, field = key.split("|")
        emb.setdefault(vid, {})["obj" if field == "obj" else field] = str(z[key]) if field == "obj" else z[key]
    return emb


# ---------------------------------------------------------------------------
# Reporting / probe
# ---------------------------------------------------------------------------

def print_table(results: dict) -> list[dict]:
    rows = [{"name": "raw cosine (baseline)", **{k: results["baseline"][k] for k in ("auroc", "tpr_fpr1pct", "tpr_fpr0p1pct")}}]
    for g in results["grid"]:
        p = g["params"]
        rows.append({"name": f"K={p['n_components']} drop={p['drop_top']} pow={p['power']} eps={p['eps']}",
                     "auroc": g["auroc"], "tpr_fpr1pct": g["tpr_fpr1pct"], "tpr_fpr0p1pct": g["tpr_fpr0p1pct"]})
    width = max(len(r["name"]) for r in rows)
    print(f"\n{'setting'.ljust(width)}   AUROC   TPR@1%  TPR@0.1%   (mean over held-out videos)")
    for r in rows:
        print(f"{r['name'].ljust(width)}  {r['auroc']:.4f}  {r['tpr_fpr1pct']:.4f}  {r['tpr_fpr0p1pct']:.4f}")
    return rows


def _collect_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p])
    return out


def probe(ext, whitener: Whitener, ref_paths: list[str], crop_paths: list[str], micro_batch: int) -> None:
    import cv2

    refs, crops = _collect_images(ref_paths), _collect_images(crop_paths)
    r = ext.extract([cv2.imread(str(p)) for p in refs], batch_size=micro_batch, preprocess_mode=ext.preprocess_mode)
    c = ext.extract([cv2.imread(str(p)) for p in crops], batch_size=micro_batch, preprocess_mode=ext.candidate_preprocess_mode)
    raw = (c @ r.T).mean(axis=1)
    wh = (whitener.transform(c) @ whitener.transform(r).T).mean(axis=1)
    print(f"\nprobe (mean cosine over {len(refs)} ref(s))")
    print(f"{'crop'.ljust(28)}      raw  whitened")
    for p, a, b in zip(crops, raw, wh):
        print(f"{p.name[:28].ljust(28)}  {a:7.4f}  {b:8.4f}")


# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--annotations", default=None, help="default: cfg.data.gt.global_file")
    ap.add_argument("--out-dir", default="runs/pca_whitening")
    ap.add_argument("--cache", default=None, help="embeddings cache (.npz); default <out-dir>/embeddings.npz. "
                    "Delete it after changing the extractor/LoRA/crop settings -- it is reused as-is.")
    # evaluation
    ap.add_argument("--eval", choices=["loo", "split", "none"], default="loo",
                    help="loo = leave-one-object-out; split = --val-objects/--val-suffix; none = no evaluation, "
                         "fit the first grid value only")
    ap.add_argument("--val-objects", default="")
    ap.add_argument("--val-suffix", default=None)
    ap.add_argument("--select-by", choices=["auroc", "tpr_fpr1pct", "tpr_fpr0p1pct"], default="tpr_fpr1pct")
    # whitening grid
    ap.add_argument("--fit-on", choices=list(FIT_SOURCES), default="pos+neg")
    ap.add_argument("--n-components", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--drop-top", type=int, nargs="+", default=[0])
    ap.add_argument("--power", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1])
    ap.add_argument("--no-save", action="store_true")
    # data (same meaning/defaults as scripts/train_lora_dinov3.py)
    ap.add_argument("--max-pos", type=int, default=300, help="GT crops kept per video")
    ap.add_argument("--max-neg", type=int, default=600, help="negative crops kept per video")
    ap.add_argument("--jitter-copies", type=int, default=0)
    ap.add_argument("--neg-iou-max", type=float, default=0.1)
    ap.add_argument("--pos-iou-min", type=float, default=0.6)
    ap.add_argument("--raw-refs", action="store_true", help="skip stage1-style ref masking/crop/aerial_sim")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    # probe
    ap.add_argument("--probe-refs", nargs="+", default=None)
    ap.add_argument("--probe-crops", nargs="+", default=None)
    args = ap.parse_args()

    grid = [WhiteningParams(k, d, p, e) for k, d, p, e in
            itertools.product(args.n_components, args.drop_top, args.power, args.eps)]
    out_dir = Path(args.out_dir)
    cache = Path(args.cache) if args.cache else out_dir / "embeddings.npz"

    from aero_eyes.config import load_config
    from aero_eyes.models.features import DINOv3FeatureExtractor, build_feature_extractor
    from aero_eyes.utils.io import list_video_ids

    cfg = load_config(args.config, args.set)
    if cfg.stage1.feature_extractor.model != "dinov3":
        raise SystemExit("Set --set stage1.feature_extractor.model=dinov3.")
    ext = build_feature_extractor(cfg)
    if not isinstance(ext, DINOv3FeatureExtractor):
        raise SystemExit(f"Got {type(ext).__name__}; disable projection_head / candidate_background_masking.")
    if args.annotations is None:
        args.annotations = cfg.data.gt.global_file
    if cfg.stage1.feature_extractor.dinov3_lora_weights_path:
        log.warning("A LoRA is loaded: it was trained on these same videos, so out-of-sample numbers here are "
                    "optimistic for the LoRA itself (leave-one-object-out only holds out the WHITENING).")

    if cache.exists():
        emb = load_embeddings(cache)
        log.info("Loaded cached embeddings for %d videos from %s (delete it to re-embed).", len(emb), cache)
    else:
        t0 = time.time()
        emb = collect_embeddings(cfg, ext, list_video_ids(args.annotations), args)
        save_embeddings(emb, cache)
        log.info("Embedded %d videos in %.0fs -> %s", len(emb), time.time() - t0, cache)
    for vid, d in emb.items():
        log.info("%s: %d refs, %d pos, %d neg", vid, len(d["refs"]), len(d["pos"]), len(d["neg"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    best = grid[0]
    if args.eval != "none":
        folds = make_folds(emb, args.eval, [o for o in args.val_objects.split(",") if o], args.val_suffix)
        results = evaluate_grid(emb, grid, args.fit_on, folds)
        print_table(results)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        best_row = max(results["grid"], key=lambda g: (-np.inf if np.isnan(g[args.select_by]) else g[args.select_by]))
        best = WhiteningParams(**best_row["params"])
        base_v = results["baseline"][args.select_by]
        log.info("best by %s: %s (%.4f vs raw %.4f)%s", args.select_by, best, best_row[args.select_by], base_v,
                 "" if best_row[args.select_by] > base_v else "  -- NOT better than raw cosine")

    final = fit_whitener(pooled(emb, sorted(emb), args.fit_on), best)
    if not args.no_save:
        final.save(out_dir / "whitening.npz", {"fit_on": args.fit_on, "videos": sorted(emb)})
        log.info("saved %s (k=%d); use it via stage3.whitening.weights_path", out_dir / "whitening.npz", final.out_dim)
    if args.probe_refs and args.probe_crops:
        probe(ext, final, args.probe_refs, args.probe_crops, args.micro_batch)


if __name__ == "__main__":
    main()

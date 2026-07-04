"""Stage 1 — Reference processing (offline, runs once per target).

Flow:  3 reference images
       -> MobileSAM foreground mask
       -> DINOv2 ViT-B/14 patch features
       -> fuse across views -> multi-view prototype
       -> write prototype.npz

Writes: <work_dir>/<sample_id>/prototype.npz
Viz:    <work_dir>/<sample_id>/viz/stage1/ (when save_visualizations=true)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def run_stage1(cfg, sample_id: str) -> Path:
    """Run Stage 1 for the given sample. Returns path to prototype.npz."""
    from aero_eyes.config import load_config
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.segmentation import MobileSAMSegmenter
    from aero_eyes.utils.geometry import generate_synth_views
    from aero_eyes.utils.io import write_prototype
    from aero_eyes.utils import viz as vizmod

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    proto_path = work_dir / cfg.stage1.prototype.cache_name
    if cfg.project.use_cache and proto_path.exists():
        log.info("[Stage1] %s: using cached prototype at %s", sample_id, proto_path)
        return proto_path

    # ---- 1. Load reference images ----
    data_root = Path(cfg.data.data_root)
    refs_dir = data_root / sample_id / cfg.data.refs_subdir
    # Case-insensitive match across common extensions (.jpg/.JPG/.jpeg/.png/.bmp/.webp).
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in exts
    )
    if len(ref_paths) < cfg.data.num_references:
        raise FileNotFoundError(
            f"Expected {cfg.data.num_references} reference images in {refs_dir}, "
            f"found {len(ref_paths)}. Files present: "
            f"{[p.name for p in refs_dir.iterdir()] if refs_dir.is_dir() else '(directory does not exist)'}"
        )
    ref_paths = ref_paths[: cfg.data.num_references]
    ref_imgs = [cv2.imread(str(p)) for p in ref_paths]

    # ---- 2. MobileSAM masking ----
    seg_cfg = cfg.stage1.segmentation
    segmenter = MobileSAMSegmenter(
        weights_path=seg_cfg.weights,
        fallback_if_missing=seg_cfg.fallback_if_missing,
    ) if seg_cfg.enabled else None

    masked_imgs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for img in ref_imgs:
        if segmenter is not None:
            mask = segmenter.segment(img)
        else:
            mask = np.ones(img.shape[:2], dtype=bool)
        masks.append(mask)
        masked = img.copy()
        masked[~mask] = 0
        masked_imgs.append(masked)

    if cfg.runtime.save_visualizations:
        viz_dir = work_dir / "viz" / "stage1"
        vizmod.save_stage1_refs(ref_imgs, masks, viz_dir)

    # ---- 3. Collect images to extract features from ----
    feat_cfg = cfg.stage1.feature_extractor
    extractor = build_feature_extractor(cfg)

    images_per_ref: list[list[np.ndarray]] = []
    for i, (masked, mask) in enumerate(zip(masked_imgs, masks)):
        imgs_this_ref = [masked]
        # Synthetic viewpoint augmentation
        acc = cfg.accuracy
        if acc.mode == "max_accuracy" and acc.max_accuracy.synthetic_viewpoint_aug.enabled:
            sva = acc.max_accuracy.synthetic_viewpoint_aug
            synth = generate_synth_views(
                masked, mask,
                method=sva.method,
                num_views=sva.num_synth_views,
                pitch_range_deg=sva.pitch_range_deg,
                seed=cfg.project.seed + i,
            )
            imgs_this_ref.extend(synth)
        images_per_ref.append(imgs_this_ref)

    # ---- 4. Extract features ----
    per_ref_features: list[np.ndarray] = []
    for imgs in images_per_ref:
        feats = extractor.extract(imgs, batch_size=cfg.runtime.batch_size)
        # Average over augmented views for this ref
        avg_feat = feats.mean(axis=0)
        per_ref_features.append(avg_feat)

    per_ref_array = np.stack(per_ref_features, axis=0)  # [num_refs, D]

    # ---- 5. Fuse prototype ----
    fusion = cfg.stage1.prototype.fusion
    if fusion == "mean":
        prototype = per_ref_array.mean(axis=0)
    elif fusion == "max":
        prototype = per_ref_array.max(axis=0)
    elif fusion == "concat_then_pca":
        flat = per_ref_array.reshape(1, -1).squeeze()
        # Simple PCA reduction to the per-ref feature dimension
        from sklearn.decomposition import PCA  # type: ignore
        n_components = per_ref_array.shape[1]
        pca = PCA(n_components=min(n_components, per_ref_array.shape[0]))
        pca.fit(per_ref_array)
        prototype = pca.components_[0]
    else:
        raise ValueError(f"Unknown fusion method '{fusion}'")

    if cfg.stage1.prototype.l2_normalize:
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm
        # Also L2-normalize per-ref features
        for i in range(len(per_ref_features)):
            n = np.linalg.norm(per_ref_features[i])
            if n > 0:
                per_ref_features[i] = per_ref_features[i] / n

    # ---- 6. Write prototype ----
    meta = {
        "sample_id": sample_id,
        "num_refs": len(ref_paths),
        "fusion": fusion,
        "feature_model": feat_cfg.model,
        "accuracy_mode": cfg.accuracy.mode,
    }
    # Save per-ref features only when multi-ref embedding is active
    save_per_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
    )
    write_prototype(
        prototype=prototype,
        meta=meta,
        per_ref_features=per_ref_features if save_per_ref else None,
        path=proto_path,
    )

    elapsed = time.time() - t0
    log.info("[Stage1] %s done in %.1fs -> %s", sample_id, elapsed, proto_path)
    return proto_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 1 — reference processing")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True, help="sample id under data_root")
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage1(cfg, args.sample)


if __name__ == "__main__":
    main()

"""Stage 1 — Reference processing (offline, runs once per target).

Flow:  3 reference images
       -> MobileSAM foreground mask (center-crop fallback if implausible,
          opt-in via stage1.segmentation.center_crop_fallback)
       -> multi-scale pyramid (1.0x/0.75x/0.5x) -> DINOv2 ViT-B/14 features
       -> mask-area-weighted fusion across views -> multi-view prototype
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


def _apply_aerial_sim(img: np.ndarray, downscale_factor: float, blur_ksize: int) -> np.ndarray:
    """Degrade a reference image toward what a distant aerial camera would
    capture: shrink-then-upscale to destroy fine detail, optionally followed
    by a Gaussian blur. Both are no-ops at their default (1.0 / 0) values.
    """
    out = img
    if downscale_factor < 1.0:
        h, w = out.shape[:2]
        small_w = max(1, int(round(w * downscale_factor)))
        small_h = max(1, int(round(h * downscale_factor)))
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if blur_ksize > 0:
        k = blur_ksize | 1  # Gaussian blur requires an odd kernel size
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


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
            if seg_cfg.center_crop_fallback:
                mask_ratio = float(mask.sum()) / float(mask.size)
                if mask_ratio < seg_cfg.min_valid_mask_ratio or mask_ratio > seg_cfg.max_valid_mask_ratio:
                    from aero_eyes.utils.geometry import center_box_mask
                    log.warning(
                        "[Stage1] %s: MobileSAM mask area implausible (%.1f%% of frame), "
                        "using center-crop fallback (ratio=%.2f) instead of passthrough.",
                        sample_id, mask_ratio * 100.0, seg_cfg.center_fallback_ratio,
                    )
                    mask = center_box_mask(img.shape, seg_cfg.center_fallback_ratio)
        else:
            mask = np.ones(img.shape[:2], dtype=bool)
        masks.append(mask)
        masked = img.copy()
        # Fill background with the image's own mean color rather than pure
        # black: a large solid-black region is out-of-distribution for
        # ImageNet-pretrained backbones (DINOv2/CLIP/SigLIP) and can distort
        # the embedding more than the background clutter it was meant to
        # remove.
        bg_color = img.reshape(-1, 3).mean(axis=0)
        masked[~mask] = bg_color
        masked_imgs.append(masked)

    if cfg.runtime.save_visualizations:
        viz_dir = work_dir / "viz" / "stage1"
        vizmod.save_stage1_refs(ref_imgs, masks, viz_dir)

    # ---- 2b. Aerial-view simulation (shrink/blur to match drone domain) ----
    sim_cfg = cfg.stage1.aerial_sim
    if sim_cfg.enabled:
        masked_imgs = [
            _apply_aerial_sim(m, sim_cfg.downscale_factor, sim_cfg.blur_ksize)
            for m in masked_imgs
        ]

    # ---- 3. Collect images to extract features from ----
    feat_cfg = cfg.stage1.feature_extractor
    extractor = build_feature_extractor(cfg)

    images_per_ref: list[list[np.ndarray]] = []
    for i, (masked, mask) in enumerate(zip(masked_imgs, masks)):
        # Multi-scale pyramid: extract features at 1.0x/0.75x/0.5x of the
        # masked reference image, then average over them (below) along with
        # any synthetic views -- makes the fused per-ref feature more robust
        # to how large the object appears (drone altitude/zoom varies the
        # query video's apparent object scale in a way a single fixed-scale
        # reference photo can't hedge against on its own).
        imgs_this_ref = []
        for scale in (1.0, 0.75, 0.5):
            if scale == 1.0:
                imgs_this_ref.append(masked)
            else:
                h, w = masked.shape[:2]
                scaled = cv2.resize(masked, (max(1, int(w * scale)), max(1, int(h * scale))))
                imgs_this_ref.append(scaled)
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
        # Weight each ref's contribution by its own mask's foreground area
        # ratio instead of a plain unweighted mean -- a ref whose segmenter
        # mask covers more of the frame is (empirically) a more confident/
        # reliable segmentation, less likely to have background bleeding
        # into what was measured as "the object", so it should count for
        # more when fusing the 3 refs into one prototype.
        weights = np.array([m.sum() / m.size + 1e-4 for m in masks])
        weights = weights / weights.sum()
        log.info("[Stage1] %s: mask-area-weighted fusion, weights=%s", sample_id, weights)
        prototype = np.average(per_ref_array, axis=0, weights=weights)
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

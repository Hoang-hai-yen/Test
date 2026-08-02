%%writefile /content/aero_eyes/aero_eyes/stages/stage1.py
"""Stage 1 — Reference processing (offline, runs once per target).

Flow:  3 reference images
       -> MobileSAM foreground mask
       -> Multi-scale crops (CẢI TIẾN 3)
       -> DINOv2 ViT-B/14 patch features
       -> Weighted Fusion based on Mask Area (CẢI TIẾN 5)
       -> write prototype.npz

Writes: <work_dir>/<sample_id>/prototype.npz
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _center_box_mask(shape: tuple, ratio: float) -> np.ndarray:
    """CẢI TIẾN 6 (root-cause fix cho CardboardBox_0-style failure):
    Fallback an toàn khi MobileSAM cho ra mask diện tích phi lý (quá nhỏ hoặc
    quá lớn so với khung ảnh). Thay vì passthrough toàn khung (kéo theo nhiễu
    nền/môi trường vào thẳng Prototype ở Stage 1 -> hỏng luôn matching ở
    Stage 3), ta giả định ảnh reference (object_images) luôn có vật thể nằm
    ở trung tâm khung hình (đặc thù của ảnh chụp cận cảnh tham chiếu), nên
    dùng 1 vùng crop trung tâm với padding hợp lý làm mask thay thế.
    """
    h, w = shape[:2]
    mh, mw = int(round(h * ratio)), int(round(w * ratio))
    y0, x0 = max(0, (h - mh) // 2), max(0, (w - mw) // 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y0 + mh, x0:x0 + mw] = True
    return mask


def _apply_aerial_sim(img: np.ndarray, downscale_factor: float, blur_ksize: int) -> np.ndarray:
    out = img
    if downscale_factor < 1.0:
        h, w = out.shape[:2]
        small_w = max(1, int(round(w * downscale_factor)))
        small_h = max(1, int(round(h * downscale_factor)))
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if blur_ksize > 0:
        k = blur_ksize | 1  
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def run_stage1(cfg, sample_id: str) -> Path:
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
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in exts
    )
    if len(ref_paths) < cfg.data.num_references:
        raise FileNotFoundError(f"Expected {cfg.data.num_references} ref images.")
    ref_paths = ref_paths[: cfg.data.num_references]
    ref_imgs = [cv2.imread(str(p)) for p in ref_paths]

    # ---- 2. MobileSAM masking ----
    seg_cfg = cfg.stage1.segmentation
    segmenter = MobileSAMSegmenter(
        weights_path=seg_cfg.weights,
        fallback_if_missing=seg_cfg.fallback_if_missing,
    ) if seg_cfg.enabled else None

    # Ngưỡng hợp lệ cho diện tích mask (tỷ lệ so với khung ảnh reference).
    # Có default an toàn nên KHÔNG cần sửa file config để phát huy tác dụng.
    mask_min_ratio = getattr(seg_cfg, "min_valid_mask_ratio", 0.03)
    mask_max_ratio = getattr(seg_cfg, "max_valid_mask_ratio", 0.92)
    center_fallback_ratio = getattr(seg_cfg, "center_fallback_ratio", 0.75)

    masked_imgs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for img in ref_imgs:
        if segmenter is not None:
            mask = segmenter.segment(img)
            mask_ratio = float(mask.sum()) / float(mask.size)
            if mask_ratio < mask_min_ratio or mask_ratio > mask_max_ratio:
                log.warning(
                    "[Stage1] %s: MobileSAM mask area implausible (%.1f%% of frame), "
                    "dùng center-crop fallback (ratio=%.2f) thay vì passthrough toàn khung.",
                    sample_id, mask_ratio * 100.0, center_fallback_ratio,
                )
                mask = _center_box_mask(img.shape, center_fallback_ratio)
        else:
            mask = np.ones(img.shape[:2], dtype=bool)
        masks.append(mask)
        masked = img.copy()
        bg_color = img.reshape(-1, 3).mean(axis=0)
        masked[~mask] = bg_color
        masked_imgs.append(masked)

    if cfg.runtime.save_visualizations:
        viz_dir = work_dir / "viz" / "stage1"
        vizmod.save_stage1_refs(ref_imgs, masks, viz_dir)

    sim_cfg = cfg.stage1.aerial_sim
    if sim_cfg.enabled:
        masked_imgs = [
            _apply_aerial_sim(m, sim_cfg.downscale_factor, sim_cfg.blur_ksize)
            for m in masked_imgs
        ]

    # ---- 3. Collect images to extract features from (CẢI TIẾN 3: Multi-scale) ----
    feat_cfg = cfg.stage1.feature_extractor
    extractor = build_feature_extractor(cfg)

    images_per_ref: list[list[np.ndarray]] = []
    for i, (masked, mask) in enumerate(zip(masked_imgs, masks)):
        imgs_this_ref = []
        
        # Tạo Multi-scale Pyramid (1.0x, 0.75x, 0.5x)
        for scale in [1.0, 0.75, 0.5]:
            if scale == 1.0:
                imgs_this_ref.append(masked)
            else:
                h, w = masked.shape[:2]
                scaled_img = cv2.resize(masked, (max(1, int(w*scale)), max(1, int(h*scale))))
                imgs_this_ref.append(scaled_img)

        # Synthetic viewpoint augmentation
        acc = cfg.accuracy
        if acc.mode == "max_accuracy" and acc.max_accuracy.synthetic_viewpoint_aug.enabled:
            sva = acc.max_accuracy.synthetic_viewpoint_aug
            synth = generate_synth_views(
                masked, mask, method=sva.method, num_views=sva.num_synth_views,
                pitch_range_deg=sva.pitch_range_deg, seed=cfg.project.seed + i,
            )
            imgs_this_ref.extend(synth)
            
        images_per_ref.append(imgs_this_ref)

    # ---- 4. Extract features ----
    per_ref_features: list[np.ndarray] = []
    for imgs in images_per_ref:
        feats = extractor.extract(imgs, batch_size=cfg.runtime.batch_size)
        avg_feat = feats.mean(axis=0)
        per_ref_features.append(avg_feat)

    per_ref_array = np.stack(per_ref_features, axis=0) 

    # ---- 5. Fuse prototype (CẢI TIẾN 5: Weighted Fusion) ----
    fusion = cfg.stage1.prototype.fusion
    
    # Tính toán trọng số dựa trên tỷ lệ diện tích Foreground (Mask area)
    weights = np.array([m.sum() / m.size + 1e-4 for m in masks])
    weights = weights / weights.sum()

    if fusion == "mean":
        log.info(f"[Stage1] Áp dụng Weighted Fusion với trọng số: {weights}")
        prototype = np.average(per_ref_array, axis=0, weights=weights)
    elif fusion == "max":
        prototype = per_ref_array.max(axis=0)
    elif fusion == "concat_then_pca":
        flat = per_ref_array.reshape(1, -1).squeeze()
        from sklearn.decomposition import PCA 
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
    save_per_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
    )
    write_prototype(
        prototype=prototype, meta=meta,
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
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage1(cfg, args.sample)


if __name__ == "__main__":
    main()

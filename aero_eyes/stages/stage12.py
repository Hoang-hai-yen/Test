"""Stage A — Prepare (tiền xử lý).

Gộp từ 2 stage cũ:
  - Stage 1 (reference processing): 3 ref images -> MobileSAM mask ->
    multi-scale crops -> DINOv2 features -> weighted fusion -> prototype.npz
  - Stage 2 (candidate generation): video -> keyframe sampling -> proposals
    (+ optional SAHI tiling) -> optional dense-scan fallback (dùng
    prototype.npz vừa tạo ở phần trên) -> DINOv2 features -> candidates.json

Lý do gộp: Stage 2 (dense_scan) vốn đã phụ thuộc trực tiếp vào prototype.npip
do Stage 1 ghi ra, nên tách thành 2 lệnh CLI riêng chỉ tạo thêm rủi ro
"chạy thiếu bước 1". Gộp lại thành 1 stage duy nhất, chạy tuần tự nội bộ,
vẫn giữ nguyên các file trung gian (prototype.npz, candidates.json +
candidates.feats.npz) để không phá cache / visualization / khả năng debug
từng bước như cũ.

Writes:
  <work_dir>/<sample_id>/prototype.npz
  <work_dir>/<sample_id>/candidates.json (+ .feats.npz)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box, Detection

log = logging.getLogger(__name__)


# =====================================================================
# Phần 1 (cũ: Stage 1) — Reference processing -> prototype.npz
# =====================================================================

def _center_box_mask(shape: tuple, ratio: float) -> np.ndarray:
    """CẢI TIẾN 6 (root-cause fix cho CardboardBox_0-style failure):
    Fallback an toàn khi MobileSAM cho ra mask diện tích phi lý (quá nhỏ hoặc
    quá lớn so với khung ảnh). Thay vì passthrough toàn khung (kéo theo nhiễu
    nền/môi trường vào thẳng Prototype -> hỏng luôn matching ở phần
    matching), ta giả định ảnh reference (object_images) luôn có vật thể nằm
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


def _run_reference_processing(cfg, sample_id: str) -> Path:
    """(cũ: run_stage1) Sinh prototype.npz từ ảnh reference."""
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
        log.info("[StageA/ref] %s: using cached prototype at %s", sample_id, proto_path)
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
                    "[StageA/ref] %s: MobileSAM mask area implausible (%.1f%% of frame), "
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
        viz_dir = work_dir / "viz" / "stageA_ref"
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
                scaled_img = cv2.resize(masked, (max(1, int(w * scale)), max(1, int(h * scale))))
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
        log.info(f"[StageA/ref] Áp dụng Weighted Fusion với trọng số: {weights}")
        prototype = np.average(per_ref_array, axis=0, weights=weights)
    elif fusion == "max":
        prototype = per_ref_array.max(axis=0)
    elif fusion == "concat_then_pca":
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
    log.info("[StageA/ref] %s done in %.1fs -> %s", sample_id, elapsed, proto_path)
    return proto_path


# =====================================================================
# Phần 2 (cũ: Stage 2) — Candidate generation -> candidates.json
# =====================================================================

def _proposals_for_frame(
    frame_bgr: np.ndarray,
    proposal_model,
    use_sahi: bool,
    sahi_tile: list[int],
    sahi_overlap: float,
    min_area: float,
    max_candidates: int,
) -> list[Box]:
    """Run proposals on a single frame, with or without SAHI tiling."""
    from aero_eyes.utils.geometry import nms, remap_box_from_tile, sahi_tiles

    h, w = frame_bgr.shape[:2]

    if not use_sahi:
        boxes = proposal_model.propose(frame_bgr)
    else:
        tiles = sahi_tiles(w, h, sahi_tile, sahi_overlap)
        all_boxes: list[Box] = []
        for tile in tiles:
            tile_img = frame_bgr[tile.y1:tile.y2, tile.x1:tile.x2]
            if tile_img.size == 0:
                continue
            tile_boxes = proposal_model.propose(tile_img)
            for b in tile_boxes:
                all_boxes.append(remap_box_from_tile(b, tile))
        keep = nms(all_boxes, iou_threshold=0.5)
        boxes = [all_boxes[i] for i in keep]

    boxes = [b for b in boxes if b.area() >= min_area]
    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes[:max_candidates]


def _dense_scan_boxes(
    frame_bgr: np.ndarray,
    extractor,
    prototype: np.ndarray,
    use_sahi: bool,
    sahi_tile: list[int],
    sahi_overlap: float,
    sim_threshold: float,
    min_blob_patches: int,
    min_area: float,
    max_candidates: int,
    use_fpn_pyramid: bool = False,
) -> list[Box]:
    """DINOv2 patch-similarity scan against the prototype -- fallback
    candidate source for when proposal_model starves a keyframe."""
    from aero_eyes.utils.geometry import dense_patches_to_boxes, nms, sahi_tiles

    h, w = frame_bgr.shape[:2]
    all_boxes: list[Box] = []
    extract = extractor.extract_pyramid_grid if use_fpn_pyramid else extractor.extract_dense_grid

    if use_sahi:
        for tile in sahi_tiles(w, h, sahi_tile, sahi_overlap):
            tile_img = frame_bgr[tile.y1:tile.y2, tile.x1:tile.x2]
            if tile_img.size == 0:
                continue
            grid, scale_x, scale_y = extract(tile_img)
            all_boxes.extend(dense_patches_to_boxes(
                grid, prototype, sim_threshold, min_blob_patches,
                scale_x, scale_y, offset_x=tile.x1, offset_y=tile.y1,
            ))
        keep = nms(all_boxes, iou_threshold=0.5)
        all_boxes = [all_boxes[i] for i in keep]
    else:
        grid, scale_x, scale_y = extract(frame_bgr)
        all_boxes = dense_patches_to_boxes(
            grid, prototype, sim_threshold, min_blob_patches, scale_x, scale_y,
        )

    all_boxes = [b for b in all_boxes if b.area() >= min_area]
    all_boxes.sort(key=lambda b: b.score, reverse=True)
    return all_boxes[:max_candidates]


def _run_candidate_generation(cfg, sample_id: str) -> Path:
    """(cũ: run_stage2) Sinh candidates.json từ video, có thể dùng
    prototype.npz (đã tạo ở _run_reference_processing) cho dense-scan."""
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.proposals import build_proposal_model
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.geometry import sahi_tiles
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = work_dir / "candidates.json"
    if cfg.project.use_cache and candidates_path.exists():
        log.info("[StageA/cand] %s: using cached candidates at %s", sample_id, candidates_path)
        return candidates_path

    # ---- Locate video ----
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(
            f"No video matching '{cfg.data.video_glob}' found in {video_dir}."
        )
    video_path = video_files[0]
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[StageA/cand] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    # ---- Build models ----
    proposal_model = build_proposal_model(cfg)
    extractor = build_feature_extractor(cfg)

    # ---- Keyframe indices ----
    interval = cfg.stage2.keyframe_interval
    kf_indices = set(keyframe_indices(total_frames, interval))

    # ---- Multi-scale scan (cheap_boosters) ----
    scales = [1.0]
    if cfg.accuracy.mode in ("cheap_boosters", "max_accuracy"):
        if cfg.accuracy.cheap_boosters.multi_scale_scan:
            scales = cfg.accuracy.cheap_boosters.scales

    cand_cfg = cfg.stage2.candidate
    sahi_cfg = cfg.stage2.sahi
    dscfg = getattr(cfg.stage2, 'dense_scan', None)

    # ---- Optional dense-scan fallback candidate source ----
    dense_prototype = None
    if dscfg is not None and getattr(dscfg, 'enabled', False):
        if cfg.stage1.feature_extractor.model != "dinov2":
            log.warning(
                "[StageA/cand] %s: stage2.dense_scan.enabled=true but "
                "stage1.feature_extractor.model='%s' (only 'dinov2' supports "
                "the dense patch-token scan) -- dense scan disabled for this run.",
                sample_id, cfg.stage1.feature_extractor.model,
            )
        else:
            proto_path = work_dir / cfg.stage1.prototype.cache_name
            if not proto_path.exists():
                raise FileNotFoundError(
                    f"stage2.dense_scan.enabled=true but prototype.npz not found "
                    f"at {proto_path}. Reference processing (phần 1 của StageA) "
                    f"chưa chạy hoặc chưa ghi ra file."
                )
            from aero_eyes.utils.io import read_prototype
            dense_prototype, _, _ = read_prototype(proto_path)

    candidates: dict[int, list[Detection]] = {}
    viz_dir = work_dir / "viz" / "stageA_cand"

    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue

        all_boxes: list[Box] = []
        h, w = frame_bgr.shape[:2]

        for scale in scales:
            if scale != 1.0:
                sw, sh = int(w * scale), int(h * scale)
                scaled = cv2.resize(frame_bgr, (sw, sh))
            else:
                scaled = frame_bgr

            boxes = _proposals_for_frame(
                scaled,
                proposal_model=proposal_model,
                use_sahi=sahi_cfg.use_sahi,
                sahi_tile=sahi_cfg.tile,
                sahi_overlap=sahi_cfg.overlap,
                min_area=cand_cfg.min_box_area,
                max_candidates=cand_cfg.max_candidates_per_keyframe,
            )

            if scale != 1.0:
                for b in boxes:
                    all_boxes.append(Box(
                        b.x1 / scale, b.y1 / scale,
                        b.x2 / scale, b.y2 / scale,
                        score=b.score,
                    ))
            else:
                all_boxes.extend(boxes)

        from aero_eyes.utils.geometry import nms as do_nms
        keep = do_nms(all_boxes, iou_threshold=0.5)
        final_boxes = [all_boxes[i] for i in keep]
        final_boxes = final_boxes[:cand_cfg.max_candidates_per_keyframe]
        box_sources = ["candidate"] * len(final_boxes)

        if dense_prototype is not None and len(final_boxes) < dscfg.trigger_min_proposals:
            dense_boxes = _dense_scan_boxes(
                frame_bgr, extractor, dense_prototype,
                use_sahi=sahi_cfg.use_sahi, sahi_tile=sahi_cfg.tile,
                sahi_overlap=sahi_cfg.overlap,
                sim_threshold=dscfg.sim_threshold, min_blob_patches=dscfg.min_blob_patches,
                min_area=cand_cfg.min_box_area,
                max_candidates=dscfg.max_dense_candidates_per_keyframe,
                use_fpn_pyramid=dscfg.use_fpn_pyramid,
            )
            if dense_boxes:
                log.info(
                    "[StageA/cand] %s frame %d: proposal path found only %d candidate(s) "
                    "(< trigger_min_proposals=%d) -- dense scan added %d more.",
                    sample_id, frame_idx, len(final_boxes),
                    dscfg.trigger_min_proposals, len(dense_boxes),
                )
                dense_id_set = {id(b) for b in dense_boxes}
                merged = final_boxes + dense_boxes
                keep = do_nms(merged, iou_threshold=0.5)
                final_boxes = [merged[i] for i in keep][:cand_cfg.max_candidates_per_keyframe]
                box_sources = [
                    "candidate_dense" if id(b) in dense_id_set else "candidate"
                    for b in final_boxes
                ]

        if final_boxes:
            feats = extractor.extract_crops(
                frame_bgr, final_boxes,
                pad_ratio=cand_cfg.feature_crop_pad,
                batch_size=cfg.runtime.batch_size,
            )
        else:
            feats = np.zeros((0, extractor._feature_dim()), dtype=np.float32)

        detections: list[Detection] = []
        for i, box in enumerate(final_boxes):
            d = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source=box_sources[i])
            d._feature = feats[i]  # type: ignore[attr-defined]
            detections.append(d)

        candidates[frame_idx] = detections
        log.debug("[StageA/cand] frame %d: %d candidates", frame_idx, len(detections))

        if cfg.runtime.save_visualizations:
            tiles_drawn = (
                sahi_tiles(w, h, sahi_cfg.tile, sahi_cfg.overlap)
                if sahi_cfg.use_sahi else None
            )
            vizmod.save_stage2_keyframe(frame_bgr, final_boxes, tiles_drawn, frame_idx, viz_dir)

    _write_candidates_with_features(candidates, candidates_path)

    elapsed = time.time() - t0
    log.info("[StageA/cand] %s done in %.1fs -> %s (%d keyframes)", sample_id, elapsed,
             candidates_path, len(candidates))
    return candidates_path


def _write_candidates_with_features(
    candidates: dict[int, list[Detection]],
    path: Path,
) -> None:
    """Write candidates JSON + companion NPZ for feature vectors."""
    import json

    from aero_eyes.utils.io import SCHEMA_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)

    frames_json: dict[str, list[dict]] = {}
    all_feats: list[np.ndarray] = []
    feat_index: list[tuple[int, int]] = []

    for fi, dets in candidates.items():
        frame_dets: list[dict] = []
        for det in dets:
            d = det.to_dict()
            feat = getattr(det, "_feature", None)
            if feat is not None:
                global_idx = len(all_feats)
                all_feats.append(feat)
                feat_index.append((fi, global_idx))
                d["feat_idx"] = global_idx
            frame_dets.append(d)
        frames_json[str(fi)] = frame_dets

    payload = {"schema_version": SCHEMA_VERSION, "frames": frames_json}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    feat_path = path.with_suffix(".feats.npz")
    if all_feats:
        np.savez_compressed(str(feat_path), features=np.stack(all_feats, axis=0))
    else:
        np.savez_compressed(str(feat_path), features=np.zeros((0, 768), dtype=np.float32))


def read_candidates_with_features(path: Path):
    """Load candidates.json + features NPZ. Returns (detections_dict, feat_matrix).
    Dùng lại ở Stage B (matching)."""
    import json
    from aero_eyes.types import Detection

    with open(path) as f:
        payload = json.load(f)

    feat_path = path.with_suffix(".feats.npz")
    if feat_path.exists():
        feat_matrix = np.load(str(feat_path))["features"]
    else:
        feat_matrix = None

    candidates: dict[int, list[Detection]] = {}
    for fi_str, dets in payload["frames"].items():
        fi = int(fi_str)
        det_list: list[Detection] = []
        for d in dets:
            det = Detection.from_dict(d)
            if feat_matrix is not None and "feat_idx" in d:
                det._feature = feat_matrix[d["feat_idx"]]  # type: ignore[attr-defined]
            det_list.append(det)
        candidates[fi] = det_list

    return candidates, feat_matrix


# =====================================================================
# Entry point gộp
# =====================================================================

def run_stage12(cfg, sample_id: str) -> tuple[Path, Path]:
    """Stage A — Prepare. Chạy tuần tự: reference processing -> candidate
    generation. Trả về (prototype_path, candidates_path)."""
    proto_path = _run_reference_processing(cfg, sample_id)
    candidates_path = _run_candidate_generation(cfg, sample_id)
    return proto_path, candidates_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage A — prepare (reference + candidates)")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage12(cfg, args.sample)


if __name__ == "__main__":
    main()
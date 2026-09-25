"""Light LoRA fine-tune of DINOv3 on this project's own labeled crops, so that
cosine(reference-photo prototype, video crop) separates the target from
clutter better than the frozen backbone does.

What it trains
--------------
The DINOv3 backbone stays frozen; only low-rank adapters on a few attention
projections (default q_proj/v_proj of the last 4 blocks, rank 8) are trained.
Data is POOLED across every video listed in annotations.json (objects are
inferred from the video id: "Laptop_0", "Laptop_1" -> object "Laptop"):

  * reference photos per video (prepared like stage1: segmentation /
    crop_to_object / aerial_sim as configured), embedded through the model
    being trained -> one prototype per object (mean of L2-normalized refs);
  * positives  = GT crops of that object's videos (plus detector proposals
    with IoU >= --pos-iou-min, when candidates.json exists);
  * negatives  = clutter crops (candidates.json boxes with IoU < --neg-iou-max
    against GT; falls back to random background boxes without it).

Loss: every crop is classified against [object prototypes..., background] by
cosine/temperature -- so a target crop is pulled toward ITS object's ref
prototype and away from the other objects', and a clutter crop is pushed away
from ALL prototypes. That is the same "cosine to the ref prototype" decision
the pipeline makes at Stage 3, trained directly.

Pushing clutter away (all optional, off by default)
--------------------------------------------------
  --hard-neg-pool P    hard-negative mining: train on the clutter crops with the
                       highest cosine to the prototype (P crops drawn per object
                       per step, the top --neg-per-object kept).
  --neg-margin m       adds a hinge on clutter: relu(cos(clutter, own prototype) - m),
                       weight --neg-margin-weight, so clutter keeps being pushed
                       below m even after cross-entropy is satisfied.
  --select-by tpr_fpr1pct  picks lora_best.pt by the share of targets kept at
                       1% clutter leak instead of AUROC (the high-cosine tail is
                       what leaks false positives; AUROC is dominated by easy clutter).
TPR@FPR 1% / 0.1% are always computed and logged/saved in metrics.json.

Other loss / score options (all default to the behaviour above)
---------------------------------------------------------------
  --loss triplet       replaces the cross-entropy with a triplet loss on the same
                       object-vs-crop scores: relu(--triplet-margin - s(target crop)
                       + s(clutter crop or other object's crop)), averaged over the
                       violating pairs; a satisfied pair gives no gradient, so it
                       moves the backbone less than cross-entropy (--triplet-mining hard
                       keeps only the hardest negative).
  --score patch|both   train (and validate) on the Chamfer/OT PATCH-token score that
                       stage3.patch_matching computes, instead of the CLS cosine.
                       both = --cls-weight * CLS + (1 - that) * patch. Then run the
                       pipeline with the SAME setup: stage3.patch_matching.enabled=true,
                       reuse_cls_preprocess=true, method/layers/symmetric matching
                       --patch-method/--patch-layers/--patch-one-way (and cls_weight
                       for both); a mismatch logs a warning at load time.
                       Costs more per step than CLS (crops x refs patch matrices);
                       lower --patch-chunk / --micro-batch if the GPU runs out of memory.
                       --hard-neg-pool still mines on CLS cosine.

Evaluating honestly (only 7 objects, 2 videos each!)
----------------------------------------------------
No flag = train on everything and report nothing about generalization.
  --val-suffix _1      train on *_0 videos, validate on *_1: measures
                       SAME object, NEW video. Objects are shared between
                       train and val, so this says nothing about new objects.
  --val-objects A,B    hold out whole objects (all their videos): measures
                       NEW objects. Rotate through the 7 objects yourself
                       (leave-one-object-out) and average -- 7 points is a
                       weak estimate, only trust a consistent gain.
Validation AUROC/AP compares, per held-out video, cosine(prototype from that
video's refs) of GT crops vs clutter crops -- before training (epoch 0, i.e.
the frozen backbone) and after each epoch.

Usage:
    python -m scripts.train_lora_dinov3 --config configs/config.yaml \
        --set stage1.feature_extractor.model=dinov3 --val-suffix _1
Then run the pipeline with
    --set stage1.feature_extractor.dinov3_lora_weights_path=<out>/lora_best.pt \
    --set project.use_cache=false
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("train_lora_dinov3")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def object_id(video_id: str) -> str:
    """'Laptop_1' -> 'Laptop' (trailing _<digits> is the video index)."""
    return re.sub(r"_\d+$", "", video_id)


def split_videos(
    video_ids: list[str], val_objects: tuple[str, ...] | list[str] = (), val_suffix: str | None = None,
) -> tuple[list[str], list[str]]:
    if val_objects and val_suffix:
        raise ValueError("--val-objects and --val-suffix are mutually exclusive.")
    if val_objects:
        held = set(val_objects)
        unknown = held - {object_id(v) for v in video_ids}
        if unknown:
            raise ValueError(f"--val-objects {sorted(unknown)} not found in annotations.")
        val = [v for v in video_ids if object_id(v) in held]
    elif val_suffix:
        val = [v for v in video_ids if v.endswith(val_suffix)]
        if not val:
            raise ValueError(f"--val-suffix {val_suffix!r} matches no video id.")
    else:
        val = []
    train = [v for v in video_ids if v not in set(val)]
    if not train:
        raise ValueError("Nothing left to train on after applying the validation split.")
    return train, val


def split_candidates(cands: dict, gt: dict, neg_iou_max: float, pos_iou_min: float):
    """candidates.json boxes -> (negatives, extra positives) as (frame, Box).
    Boxes with IoU in [neg_iou_max, pos_iou_min) against GT are ambiguous
    (mislocalized target) and dropped."""
    from aero_eyes.utils.geometry import box_iou

    neg, pos = [], []
    for fi, dets in cands.items():
        g = gt.get(fi)
        for d in dets:
            if g is None:
                neg.append((fi, d.box))
                continue
            iou = box_iou(d.box, g)
            if iou < neg_iou_max:
                neg.append((fi, d.box))
            elif iou >= pos_iou_min:
                pos.append((fi, d.box))
    return neg, pos


def even_subsample(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def random_subsample(items: list, n: int, rng: np.random.Generator) -> list:
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted(rng.choice(len(items), size=n, replace=False).tolist())]


def jitter_box(box, rng: np.random.Generator, min_iou: float, tries: int = 20):
    """A loosened/shifted copy of a GT box (each side scaled 0.8-1.3x, center
    moved up to 15% of the box size), kept only if it still overlaps the GT
    with IoU >= min_iou -- i.e. an imperfect-but-still-correct detection.
    None if no draw qualifies."""
    from aero_eyes.types import Box
    from aero_eyes.utils.geometry import box_iou

    w, h = box.x2 - box.x1, box.y2 - box.y1
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    for _ in range(tries):
        nw, nh = w * rng.uniform(0.8, 1.3), h * rng.uniform(0.8, 1.3)
        ncx, ncy = cx + rng.uniform(-0.15, 0.15) * w, cy + rng.uniform(-0.15, 0.15) * h
        cand = Box(ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2)
        if box_iou(cand, box) >= min_iou:
            return cand
    return None


def score_bg_loss(
    scores: torch.Tensor, labels: torch.Tensor, bg_logit: torch.Tensor, tau: float,
    owner: torch.Tensor | None = None, neg_margin: float | None = None, margin_weight: float = 1.0,
) -> torch.Tensor:
    """(K+1)-way cross-entropy on `scores` [C, K] (cosine-like similarity of
    every crop to every object) / tau plus one learnable background logit.
    labels in [0,K]; K = clutter. Target crops and clutter crops are averaged
    separately so a clutter-heavy batch cannot drown out the target term.

    Cross-entropy alone stops caring once a clutter crop merely loses to the
    (learnable!) background logit, so it never asks for clutter to be FAR
    from the prototype. neg_margin (with `owner`: the object index each crop
    belongs to) adds margin_weight * mean(relu(score(clutter, its own
    object) - neg_margin)): a hinge in plain cosine units that keeps pushing
    every clutter crop below neg_margin. None = off."""
    K = scores.shape[1]
    logits = torch.cat([scores / tau, bg_logit.reshape(1, 1).expand(len(scores), 1)], dim=1)
    ce = F.cross_entropy(logits, labels, reduction="none")
    is_target = labels < K
    parts = [ce[m].mean() for m in (is_target, ~is_target) if bool(m.any())]
    loss = torch.stack(parts).mean()
    if neg_margin is not None and owner is not None and bool((~is_target).any()):
        clutter = ~is_target
        cos_own = scores[clutter].gather(1, owner[clutter].unsqueeze(1)).squeeze(1)
        loss = loss + margin_weight * F.relu(cos_own - neg_margin).mean()
    return loss


def prototype_bg_loss(
    crop_emb: torch.Tensor, labels: torch.Tensor, protos: torch.Tensor,
    bg_logit: torch.Tensor, tau: float,
    owner: torch.Tensor | None = None, neg_margin: float | None = None, margin_weight: float = 1.0,
) -> torch.Tensor:
    """score_bg_loss on CLS cosines: scores = crop_emb @ protos.T."""
    return score_bg_loss(crop_emb @ protos.t(), labels, bg_logit, tau, owner, neg_margin, margin_weight)


def triplet_loss(
    scores: torch.Tensor, labels: torch.Tensor, owner: torch.Tensor, margin: float = 0.1, mining: str = "all",
) -> torch.Tensor:
    """Triplet loss on the same [C, K] scores: for every object k the anchor is
    its reference set, the positives are k's target crops and the negatives
    are k's clutter crops (owner == k) plus the OTHER objects' target crops.
    Each (positive, negative) pair costs relu(margin - s_pos + s_neg): once a
    pair satisfies the margin it contributes nothing (no gradient) -- softer
    than cross-entropy, which never stops pushing. Averaged over the
    violating pairs only (so easy pairs don't dilute the gradient).
    mining="hard" keeps only each object's hardest negative."""
    if mining not in ("all", "hard"):
        raise ValueError(f"Unknown triplet mining {mining!r}. Must be 'all' or 'hard'.")
    K = scores.shape[1]
    hinges = []
    for k in range(K):
        pos = scores[labels == k, k]
        neg = scores[((labels == K) & (owner == k)) | ((labels < K) & (labels != k)), k]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        if mining == "hard":
            hinges.append(F.relu(margin - pos + neg.max()))
        else:
            hinges.append(F.relu(margin - pos[:, None] + neg[None, :]).reshape(-1))
    if not hinges:
        return scores.sum() * 0.0
    h = torch.cat(hinges)
    return h.sum() / (h > 0).sum().clamp(min=1)


def combine_scores_loss(scores: torch.Tensor, labels, owner, bg_logit, args) -> torch.Tensor:
    """args.loss dispatch: "ce" (default, score_bg_loss + optional neg-margin
    hinge) or "triplet"."""
    if getattr(args, "loss", "ce") == "triplet":
        return triplet_loss(
            scores, labels, owner, getattr(args, "triplet_margin", 0.1), getattr(args, "triplet_mining", "all"),
        )
    return score_bg_loss(
        scores, labels, bg_logit, args.tau, owner=owner,
        neg_margin=getattr(args, "neg_margin", None), margin_weight=getattr(args, "neg_margin_weight", 1.0),
    )


def separation_metrics(pos_cos: np.ndarray, neg_cos: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    from sklearn.metrics import roc_curve

    if len(pos_cos) == 0 or len(neg_cos) == 0:
        nan = float("nan")
        return {"auroc": nan, "ap": nan, "pos_cos": nan, "neg_cos": nan,
                "tpr_fpr1pct": nan, "tpr_fpr0p1pct": nan}
    y = np.concatenate([np.ones(len(pos_cos)), np.zeros(len(neg_cos))])
    s = np.concatenate([pos_cos, neg_cos])
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return {
        "auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
        "pos_cos": float(np.mean(pos_cos)), "neg_cos": float(np.mean(neg_cos)),
        # Fraction of targets kept while letting at most 1% / 0.1% of clutter through --
        # the high-cosine clutter TAIL, which is what leaks FPs into the pipeline
        # (AUROC is dominated by the easy bulk of clutter).
        "tpr_fpr1pct": float(tpr[fpr <= 0.01].max()), "tpr_fpr0p1pct": float(tpr[fpr <= 0.001].max()),
    }


def augment_crop(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    gain = rng.uniform(0.8, 1.2)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def degrade_ref(img: np.ndarray, factor: float) -> np.ndarray:
    if factor >= 1.0:
        return img
    from aero_eyes.stages.stage1 import apply_ref_degradation

    return apply_ref_degradation(img, factor, 0, 100)


# ---------------------------------------------------------------------------
# Data building (video / disk I/O)
# ---------------------------------------------------------------------------

@dataclass
class VideoData:
    video_id: str
    obj: str
    refs: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    neg: list = field(default_factory=list)


def prepare_ref_images(cfg, ref_imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Mirror run_stage1 steps 2/2a/2b for the reference photos (background
    masking, crop_to_object, aerial_sim, as configured) so training sees the
    refs exactly as the pipeline embeds them."""
    from aero_eyes.models.segmentation import build_segmenter
    from aero_eyes.stages.stage1 import _apply_aerial_sim
    from aero_eyes.utils.geometry import apply_background_mode, center_box_mask, crop_to_object, mask_bbox

    seg = cfg.stage1.segmentation
    segmenter = build_segmenter(seg, cfg) if seg.enabled else None
    out = []
    for img in ref_imgs:
        if segmenter is not None:
            mask = segmenter.segment(img)
            if seg.center_crop_fallback:
                ratio = float(mask.sum()) / float(mask.size)
                if ratio < seg.min_valid_mask_ratio or ratio > seg.max_valid_mask_ratio:
                    mask = center_box_mask(img.shape, seg.center_fallback_ratio)
        else:
            mask = np.ones(img.shape[:2], dtype=bool)
        masked = apply_background_mode(img, mask, seg.background_mode, seg.blur_sigma)
        if seg.enabled and cfg.stage1.crop_to_object:
            box = mask_bbox(mask)
            if box is not None:
                masked, _ = crop_to_object(masked, box, cfg.stage1.crop_context_margin)
        sim = cfg.stage1.aerial_sim
        if sim.enabled:
            masked = _apply_aerial_sim(masked, sim.downscale_factor, sim.blur_ksize)
        out.append(masked)
    return out


def _random_negative_box(gt_box, shape, rng: np.random.Generator, iou_max: float):
    from aero_eyes.types import Box
    from aero_eyes.utils.geometry import box_iou

    h, w = shape[:2]
    bw, bh = gt_box.x2 - gt_box.x1, gt_box.y2 - gt_box.y1
    for _ in range(30):
        s = rng.uniform(0.6, 1.6)
        nw, nh = max(8.0, bw * s), max(8.0, bh * s)
        if nw >= w or nh >= h:
            continue
        x1, y1 = rng.uniform(0, w - nw), rng.uniform(0, h - nh)
        box = Box(x1, y1, x1 + nw, y1 + nh)
        if box_iou(box, gt_box) < iou_max:
            return box
    return None


def build_video_data(cfg, vid: str, args, rng: np.random.Generator) -> VideoData:
    from aero_eyes.stages.stage2 import read_candidates_with_features
    from aero_eyes.utils.geometry import crop_with_pad
    from aero_eyes.utils.io import load_gt
    from aero_eyes.utils.video import frame_iterator

    vdir = Path(cfg.data.data_root) / vid
    videos = list(vdir.glob(cfg.data.video_glob))
    if not videos:
        raise FileNotFoundError(f"No video matching '{cfg.data.video_glob}' in {vdir}.")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    refs_dir = vdir / cfg.data.refs_subdir
    ref_paths = sorted(p for p in (refs_dir.iterdir() if refs_dir.is_dir() else []) if p.suffix.lower() in exts)
    ref_paths = ref_paths[: cfg.data.num_references]
    if not ref_paths:
        raise FileNotFoundError(f"No reference images in {refs_dir}.")
    ref_imgs = [cv2.imread(str(p)) for p in ref_paths]
    refs = ref_imgs if args.raw_refs else prepare_ref_images(cfg, ref_imgs)

    gt = load_gt(args.annotations, vid)
    gt_frames = sorted(gt)
    pos_items = [(f, gt[f]) for f in even_subsample(gt_frames, args.max_pos)]
    for f, b in list(pos_items):
        for _ in range(getattr(args, "jitter_copies", 0)):
            jb = jitter_box(b, rng, args.pos_iou_min)
            if jb is not None:
                pos_items.append((f, jb))
    neg_items: list = []
    cand_path = Path(cfg.project.work_dir) / vid / "candidates.json"
    if cand_path.exists():
        cands, _ = read_candidates_with_features(cand_path)
        negs, cpos = split_candidates(cands, gt, args.neg_iou_max, args.pos_iou_min)
        neg_items = random_subsample(negs, args.max_neg, rng)
        pos_items += random_subsample(cpos, args.max_pos // 2, rng)
    else:
        log.warning("%s: no candidates.json at %s -- using random background boxes as negatives.", vid, cand_path)
        neg_items = [(f, None) for f in random_subsample(gt_frames, min(args.max_neg, len(gt_frames)), rng)]

    needed = {f for f, _ in pos_items} | {f for f, _ in neg_items}
    pad = cfg.stage2.candidate.feature_crop_pad
    pos_by_frame, neg_by_frame = defaultdict(list), defaultdict(list)
    for f, b in pos_items:
        pos_by_frame[f].append(b)
    for f, b in neg_items:
        neg_by_frame[f].append(b)

    vd = VideoData(video_id=vid, obj=object_id(vid), refs=refs)
    last = max(needed) if needed else -1
    for fi, frame in frame_iterator(videos[0]):
        if fi > last:
            break
        if fi not in needed:
            continue
        for b in pos_by_frame.get(fi, []):
            c = crop_with_pad(frame, b, pad)
            if min(c.shape[:2]) >= 4:
                vd.pos.append(c)
        for b in neg_by_frame.get(fi, []):
            if b is None:
                b = _random_negative_box(gt[fi], frame.shape, rng, args.neg_iou_max)
                if b is None:
                    continue
            c = crop_with_pad(frame, b, pad)
            if min(c.shape[:2]) >= 4:
                vd.neg.append(c)
    log.info("%s: %d refs, %d positive crops, %d negative crops", vid, len(vd.refs), len(vd.pos), len(vd.neg))
    return vd


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def embed(ext, images: list, mode: str, micro_batch: int, amp: bool) -> torch.Tensor:
    """L2-normalized embeddings through the CURRENT (LoRA-wrapped) model,
    differentiable, using the extractor's own preprocessing."""
    dev = torch.device(ext.device)
    outs = []
    for i in range(0, len(images), micro_batch):
        pv = ext.pixel_values(images[i:i + micro_batch], mode).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            z = ext.forward_cls(pv)
        outs.append(F.normalize(z.float(), dim=-1))
    return torch.cat(outs)


def embed_both(ext, images: list, mode: str, micro_batch: int, amp: bool, layers, need_patch: bool):
    """(L2-normalised CLS [B,D], L2-normalised patch tokens [B,N,D'] or None)
    from ONE forward pass per micro-batch, differentiable, through the
    CURRENT (LoRA-wrapped) model with the extractor's own preprocessing."""
    dev = torch.device(ext.device)
    cls_out, patch_out = [], []
    for i in range(0, len(images), micro_batch):
        pv = ext.pixel_values(images[i:i + micro_batch], mode).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            if need_patch:
                c, p = ext.forward_tokens(pv, layers)
            else:
                c, p = ext.forward_cls(pv), None
        cls_out.append(F.normalize(c.float(), dim=-1))
        if need_patch:
            patch_out.append(F.normalize(p.float(), dim=-1))
    return torch.cat(cls_out), (torch.cat(patch_out) if need_patch else None)


def _patch_pair_scores(crop_patch: torch.Tensor, ref_patch: torch.Tensor, args) -> torch.Tensor:
    from aero_eyes.models.patch_match import patch_pair_scores

    return patch_pair_scores(
        crop_patch, ref_patch, getattr(args, "patch_method", "chamfer"), not getattr(args, "patch_one_way", False),
        getattr(args, "patch_ot_epsilon", 0.05), getattr(args, "patch_ot_iters", 20), getattr(args, "patch_chunk", 8),
    )


def patch_scores_by_object(crop_patch: torch.Tensor, ref_patch: torch.Tensor, ref_owner: torch.Tensor, K: int, args):
    """[C, K] patch score of every crop against every object: the mean over
    that object's reference images of the Chamfer/OT patch score (the same
    'mean' multi-ref pooling stage3.patch_matching uses by default)."""
    pair = _patch_pair_scores(crop_patch, ref_patch, args)
    return torch.stack([pair[:, ref_owner == k].mean(dim=1) for k in range(K)], dim=1)


def _pick(lst: list, n: int, rng: np.random.Generator) -> list:
    return [lst[i] for i in rng.integers(0, len(lst), size=n)]


def mine_hard_negatives(ext, pool: list, proto: torch.Tensor, n: int, mode: str, micro_batch: int, amp: bool) -> list:
    """The n crops of `pool` with the highest cosine to `proto` under the
    CURRENT model (scored without gradients) -- the clutter that would leak
    through. Caveat: mining amplifies label noise (a target crop wrongly
    listed as clutter, e.g. in an unannotated frame, is exactly what gets
    picked), so keep the pool moderate."""
    with torch.no_grad():
        scores = embed(ext, pool, mode, micro_batch, amp) @ proto.detach()
    return [pool[i] for i in torch.topk(scores, min(n, len(pool))).indices.tolist()]


def train_step(ext, objs: list[str], obj_data: dict, args, rng, bg_logit) -> torch.Tensor:
    K = len(objs)
    score_mode = getattr(args, "score", "cls")        # cls | patch | both
    need_patch = score_mode != "cls"
    layers = getattr(args, "patch_layers", [-1])
    ref_imgs, ref_owner = [], []
    for k, o in enumerate(objs):
        for img in _pick(obj_data[o].refs, args.refs_per_object, rng):
            ref_imgs.append(degrade_ref(img, float(rng.choice(args.ref_factors))))
            ref_owner.append(k)
    ref_emb, ref_patch = embed_both(ext, ref_imgs, ext.preprocess_mode, args.micro_batch, args.amp, layers, need_patch)
    ref_owner_t = torch.tensor(ref_owner, device=ref_emb.device)
    protos = torch.stack([F.normalize(ref_emb[ref_owner_t == k].mean(0), dim=0) for k in range(K)])

    pool_n = getattr(args, "hard_neg_pool", 0)
    crops, labels, owner = [], [], []
    for k, o in enumerate(objs):
        d = obj_data[o]
        for img in _pick(d.pos, args.pos_per_object, rng):
            crops.append(augment_crop(img, rng))
            labels.append(k)
            owner.append(k)
        if d.neg:
            if pool_n > args.neg_per_object:
                # mined on CLS cosine even for patch/both training (a proxy for the patch score)
                negs = mine_hard_negatives(
                    ext, _pick(d.neg, pool_n, rng), protos[k], args.neg_per_object,
                    ext.candidate_preprocess_mode, args.micro_batch, args.amp,
                )
            else:
                negs = _pick(d.neg, args.neg_per_object, rng)
            for img in negs:
                crops.append(augment_crop(img, rng))
                labels.append(K)
                owner.append(k)
    crop_emb, crop_patch = embed_both(
        ext, crops, ext.candidate_preprocess_mode, args.micro_batch, args.amp, layers, need_patch,
    )
    y = torch.tensor(labels, device=crop_emb.device)
    own = torch.tensor(owner, device=crop_emb.device)
    if score_mode == "cls":
        return combine_scores_loss(crop_emb @ protos.t(), y, own, bg_logit, args)
    patch_scores = patch_scores_by_object(crop_patch, ref_patch, ref_owner_t, K, args)
    patch_loss = combine_scores_loss(patch_scores, y, own, bg_logit, args)
    if score_mode == "patch":
        return patch_loss
    w = getattr(args, "cls_weight", 0.3)
    return w * combine_scores_loss(crop_emb @ protos.t(), y, own, bg_logit, args) + (1.0 - w) * patch_loss


@torch.no_grad()
def eval_video(ext, vd: VideoData, args) -> dict:
    """Separation of GT crops vs clutter for one video, with the SAME score the
    model is trained on: CLS cosine (score=cls), mean-over-refs patch score
    (patch), or their cls_weight blend (both)."""
    score_mode = getattr(args, "score", "cls")
    layers = getattr(args, "patch_layers", [-1])
    refs = [degrade_ref(r, args.eval_ref_factor) for r in vd.refs]
    need_patch = score_mode != "cls"
    ref_emb, ref_patch = embed_both(ext, refs, ext.preprocess_mode, args.micro_batch, False, layers, need_patch)
    proto = F.normalize(ref_emb.mean(0), dim=0)
    w = getattr(args, "cls_weight", 0.3)

    def score(images: list):
        if not images:
            return np.zeros(0)
        out = []
        for i in range(0, len(images), args.micro_batch):
            c, p = embed_both(
                ext, images[i:i + args.micro_batch], ext.candidate_preprocess_mode, args.micro_batch, False,
                layers, need_patch,
            )
            if score_mode == "cls":
                s = c @ proto
            else:
                sp = _patch_pair_scores(p, ref_patch, args).mean(dim=1)
                s = sp if score_mode == "patch" else w * (c @ proto) + (1.0 - w) * sp
            out.append(s.cpu())
        return torch.cat(out).numpy()

    return separation_metrics(score(vd.pos), score(vd.neg))


def evaluate(ext, val_data: list[VideoData], args) -> dict:
    per_video = {v.video_id: eval_video(ext, v, args) for v in val_data}
    def _mean(key):
        vals = [m[key] for m in per_video.values() if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "per_video": per_video, "mean_auroc": _mean("auroc"), "mean_tpr_fpr1pct": _mean("tpr_fpr1pct"),
        "mean_tpr_fpr0p1pct": _mean("tpr_fpr0p1pct"),
    }


def run_training(ext, train_data: list[VideoData], val_data: list[VideoData], args) -> dict:
    from aero_eyes.models.lora import apply_lora, lora_parameters, save_lora

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    obj_data: dict[str, VideoData] = {}
    for v in train_data:
        d = obj_data.setdefault(v.obj, VideoData(video_id=v.obj, obj=v.obj))
        d.refs += v.refs
        d.pos += v.pos
        d.neg += v.neg
    objs = [o for o, d in obj_data.items() if d.refs and d.pos]
    if len(objs) < 2:
        raise ValueError(f"Need at least 2 objects with refs and positive crops to train, got {objs}.")
    log.info("Training objects (%d): %s", len(objs), objs)

    if hasattr(ext.model, "_lora_meta"):
        log.info("Model already has LoRA layers (from dinov3_lora_weights_path) -- continuing from them.")
    else:
        wrapped = apply_lora(ext.model, args.targets, args.rank, args.alpha, args.last_n_blocks)
        log.info("LoRA on %d Linear layers (rank=%d).", len(wrapped), args.rank)
    params = lora_parameters(ext.model)
    bg_logit = torch.nn.Parameter(torch.tensor(args.bg_init_cos / args.tau, device=torch.device(ext.device)))
    opt = torch.optim.AdamW(params + [bg_logit], lr=args.lr, weight_decay=args.weight_decay)
    log.info("Trainable: %d LoRA params tensors, %d values.", len(params), sum(p.numel() for p in params))

    # Stored in the checkpoint; DINOv3FeatureExtractor warns at load when the
    # running preprocessing differs from what the adapters were trained under.
    train_config = {
        k: getattr(ext, k, None)
        for k in ("preprocess_mode", "candidate_preprocess_mode", "image_size", "variant", "pretrain_dataset")
    }
    train_config.update(getattr(args, "extra_meta", None) or {})
    log.info("Training under: %s", train_config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    select_key = {"auroc": "mean_auroc", "tpr_fpr1pct": "mean_tpr_fpr1pct"}[getattr(args, "select_by", "auroc")]
    fmt = lambda ev: f"val mean AUROC {ev['mean_auroc']:.4f} | TPR@FPR1% {ev['mean_tpr_fpr1pct']:.4f}"
    if val_data:
        base = evaluate(ext, val_data, args)
        log.info("epoch 0 (frozen backbone) %s", fmt(base))
        history.append({"epoch": 0, "train_loss": None, **base})
        best = base[select_key]

    for epoch in range(1, args.epochs + 1):
        losses = []
        for _ in range(args.steps_per_epoch):
            loss = train_step(ext, objs, obj_data, args, rng, bg_logit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        msg = f"epoch {epoch}: train loss {rec['train_loss']:.4f}"
        if val_data:
            ev = evaluate(ext, val_data, args)
            rec.update(ev)
            msg += " | " + fmt(ev)
            if ev[select_key] > best:
                best = ev[select_key]
                save_lora(ext.model, out_dir / "lora_best.pt", train_config)
                msg += "  (best, saved)"
        log.info(msg)
        history.append(rec)
        save_lora(ext.model, out_dir / "lora_last.pt", train_config)
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "best_val_score": best}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="LoRA fine-tune DINOv3 on pooled project crops")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--annotations", default=None, help="default: cfg.data.gt.global_file")
    p.add_argument("--out-dir", default="runs/lora_dinov3")
    p.add_argument("--val-objects", default="", help="comma-separated objects to hold out entirely")
    p.add_argument("--val-suffix", default=None, help="hold out videos ending with this, e.g. _1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--last-n-blocks", type=int, default=4, help="0 = all blocks")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--bg-init-cos", type=float, default=0.3)
    p.add_argument("--refs-per-object", type=int, default=3)
    p.add_argument("--pos-per-object", type=int, default=8)
    p.add_argument("--neg-per-object", type=int, default=8)
    p.add_argument("--max-pos", type=int, default=300, help="GT crops kept per video")
    p.add_argument("--max-neg", type=int, default=600, help="negative crops kept per video")
    p.add_argument("--jitter-copies", type=int, default=0,
                   help="extra loosened/shifted copies of each sampled GT box added as positives (imperfect crops)")
    p.add_argument("--hard-neg-pool", type=int, default=0,
                   help="per object per step, draw this many clutter crops and train on the --neg-per-object "
                        "with the highest cosine to the prototype (hard-negative mining); 0 = off")
    p.add_argument("--neg-margin", type=float, default=None,
                   help="hinge on clutter: penalize cosine(clutter, own prototype) above this (e.g. 0.15); off if unset")
    p.add_argument("--neg-margin-weight", type=float, default=1.0)
    p.add_argument("--loss", choices=["ce", "triplet"], default="ce",
                   help="ce = (K+1)-way cross-entropy on score/tau with a learnable background logit (default); "
                        "triplet = relu(margin - s(pos) + s(neg)) on the same scores, no gradient once satisfied")
    p.add_argument("--triplet-margin", type=float, default=0.1, help="margin in score (cosine) units, --loss triplet")
    p.add_argument("--triplet-mining", choices=["all", "hard"], default="all",
                   help="all = every (positive, negative) pair, averaged over violating pairs; hard = hardest negative only")
    p.add_argument("--score", choices=["cls", "patch", "both"], default="cls",
                   help="what similarity the loss/validation use: cls = CLS cosine (default); patch = Chamfer/OT "
                        "patch-token score, i.e. what stage3.patch_matching computes; both = --cls-weight blend")
    p.add_argument("--cls-weight", type=float, default=0.3,
                   help="--score both: weight of the CLS cosine in the loss and in validation (patch gets 1 - this). "
                        "Use the same value for stage3.patch_matching.cls_weight")
    p.add_argument("--patch-method", choices=["chamfer", "ot"], default="chamfer",
                   help="patch score for --score patch|both (ot = Sinkhorn, much slower per step)")
    p.add_argument("--patch-layers", default="-1",
                   help='comma-separated hidden_states indices the patch tokens come from, e.g. "6,9,-1" (ViT-B has 12 blocks)')
    p.add_argument("--patch-one-way", action="store_true", help="ref->crop Chamfer only (default: symmetric)")
    p.add_argument("--patch-ot-epsilon", type=float, default=0.05)
    p.add_argument("--patch-ot-iters", type=int, default=20, help="Sinkhorn iterations (training default lower than inference's 50)")
    p.add_argument("--patch-chunk", type=int, default=8, help="crops per patch-score block (lower = less GPU memory)")
    p.add_argument("--select-by", choices=["auroc", "tpr_fpr1pct"], default="auroc",
                   help="validation metric that picks lora_best.pt; tpr_fpr1pct = share of targets kept at 1 percent clutter leak")
    p.add_argument("--neg-iou-max", type=float, default=0.1)
    p.add_argument("--pos-iou-min", type=float, default=0.6)
    p.add_argument("--ref-factors", default="1.0", help="extra ref downscale factors sampled in training, e.g. 1.0,0.3")
    p.add_argument("--eval-ref-factor", type=float, default=1.0)
    p.add_argument("--raw-refs", action="store_true", help="skip stage1-style ref masking/crop/aerial_sim")
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list-linear-names", action="store_true")
    args = p.parse_args()
    args.targets = [t for t in args.targets.split(",") if t]
    args.last_n_blocks = args.last_n_blocks or None
    args.ref_factors = [float(x) for x in args.ref_factors.split(",") if x]
    args.patch_layers = [int(x) for x in args.patch_layers.split(",") if x.strip()]
    if 0 < args.hard_neg_pool <= args.neg_per_object:
        log.warning("--hard-neg-pool %d <= --neg-per-object %d: nothing to mine, ignoring.",
                    args.hard_neg_pool, args.neg_per_object)

    from aero_eyes.config import load_config
    from aero_eyes.models.features import DINOv3FeatureExtractor, build_feature_extractor
    from aero_eyes.utils.io import list_video_ids

    cfg = load_config(args.config, args.set)
    if cfg.stage1.feature_extractor.model != "dinov3":
        raise SystemExit("Set --set stage1.feature_extractor.model=dinov3 (dinov3_source=huggingface).")
    ext = build_feature_extractor(cfg)
    if not isinstance(ext, DINOv3FeatureExtractor):
        raise SystemExit(
            f"build_feature_extractor returned {type(ext).__name__}; disable wrappers "
            "(projection_head / candidate_background_masking) for LoRA training."
        )
    if ext.source != "huggingface":
        raise SystemExit("LoRA training needs dinov3_source=huggingface.")
    if args.list_linear_names:
        for n, m in ext.model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(n)
        return
    if args.annotations is None:
        args.annotations = cfg.data.gt.global_file

    ids = list_video_ids(args.annotations)
    val_objects = [o for o in args.val_objects.split(",") if o]
    train_ids, val_ids = split_videos(ids, val_objects, args.val_suffix)
    if not val_ids:
        log.warning("No validation split: this trains on ALL videos and reports nothing about "
                    "generalization. Use --val-suffix or --val-objects to measure it.")
    elif args.val_suffix:
        log.warning("--val-suffix shares objects between train and val: measures same-object/new-video "
                    "generalization only, not new objects.")
    log.info("train videos: %s | val videos: %s", train_ids, val_ids)
    sim = cfg.stage1.aerial_sim
    args.extra_meta = {  # informational (the extractor only compares the keys it knows)
        "segmentation_enabled": cfg.stage1.segmentation.enabled,
        "crop_to_object": cfg.stage1.crop_to_object,
        "aerial_sim_downscale": sim.downscale_factor if sim.enabled else None,
        "feature_crop_pad": cfg.stage2.candidate.feature_crop_pad,
        "raw_refs": args.raw_refs,
        "ref_factors": args.ref_factors,
        "loss": args.loss, "triplet_margin": args.triplet_margin, "triplet_mining": args.triplet_mining,
        "score": args.score, "cls_weight": args.cls_weight, "patch_method": args.patch_method,
        "patch_layers": args.patch_layers, "patch_symmetric": not args.patch_one_way,
        "patch_ot_epsilon": args.patch_ot_epsilon, "patch_ot_iters": args.patch_ot_iters,
    }

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    train_data = [build_video_data(cfg, v, args, rng) for v in train_ids]
    val_data = [build_video_data(cfg, v, args, rng) for v in val_ids]
    log.info("data built in %.0fs", time.time() - t0)

    result = run_training(ext, train_data, val_data, args)
    log.info("done. best val %s: %.4f. Checkpoints in %s", args.select_by, result["best_val_score"], args.out_dir)


if __name__ == "__main__":
    main()

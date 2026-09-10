"""Train a small projection head on top of a frozen backbone embedding
(stage1.feature_extractor.model) to close the domain gap between close-up
reference photos and tiny aerial crops -- see
aero_eyes/models/projection_head.py and
aero_eyes/models/features.py::ProjectedFeatureExtractor for how the
resulting weights get used at inference (opt-in via
stage1.feature_extractor.projection_head.enabled=true /
.weights_path=<this script's --output>).

Data is built automatically from what the project already has -- no new
labeling needed:
  positive      = (mean of the 3 reference images' embeddings, a crop at
                   the GT box on a GT-labeled frame of that SAME sample's
                   video)
  hard negative = a candidate box from that sample's candidates.json
                   (produced by Stage 2 / Stage12-GeCo2) whose IoU with GT
                   on its own frame is < --iou-threshold, or that lands on
                   a frame with no GT at all -- i.e. exactly the
                   "confuser" candidates scripts/check_multi_ref_agreement.py
                   and check_dynamic_prototype_purity.py flag as suspects.
                   Skipped (logged, not fatal) for a sample with no
                   candidates.json yet -- the in-batch negatives below
                   still apply.
  in-batch negative = every OTHER sample's positive crop in the same
                   training batch -- free, no extra mining needed.

The backbone stays FROZEN throughout; only the head (a single Linear layer
by default) is trained. Deliberate, not a shortcut: this project's actual
matching task is few-shot/open-set (the object at eval time was never seen
during training), so a large fine-tune risks overfitting to the training
categories and destroying the backbone's zero-shot generalization -- a
tiny head trained on top is much harder to overfit with a handful of
categories.

IMPORTANT -- --train-samples and --val-samples must be DISJOINT object
categories, never a frame-level split of the same sample: the only
question that matters is "does this generalize to an object the head
never saw", and splitting frames within one category cannot answer that.

This script trains, prints an in-memory proxy validation metric (top-1
retrieval accuracy across val samples), and saves projection_head.pt. It
is NOT a substitute for scripts/check_cosine_effect.py -- after training,
re-run the actual pipeline with
stage1.feature_extractor.projection_head.enabled=true on the val samples
and compare check_cosine_effect.py's real P/R/F1 against the
no-projection-head baseline before trusting this end to end.

Usage:
    python -m scripts.train_projection_head --config configs/config.yaml \\
        --train-samples BlackBox_0,BoxRefine_2,Cooler_1 --val-samples LifeJacket_1 \\
        --output runs/projection_head.pt
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _load_ref_embedding(cfg, sample_id: str, extractor) -> np.ndarray:
    """Mean-then-L2-normalize embedding of the sample's reference images.
    Deliberately simpler than Stage 1 (no MobileSAM masking/multi-scale
    pyramid) to keep this script self-contained -- the mismatch vs. what
    Stage 1 actually feeds the backbone at inference is a known
    simplification, not a correctness bug; revisit if it measurably hurts
    the val metric below.
    """
    data_root = Path(cfg.data.data_root)
    refs_dir = data_root / sample_id / cfg.data.refs_subdir
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in exts
    )[: cfg.data.num_references]
    if not ref_paths:
        raise FileNotFoundError(f"No reference images found in {refs_dir}")
    ref_imgs = [cv2.imread(str(p)) for p in ref_paths]
    feats = extractor.extract(ref_imgs, batch_size=len(ref_imgs))
    mean = feats.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm > 0 else mean).astype(np.float32)


def _collect_sample_crops(
    cfg, sample_id: str, extractor, iou_threshold: float,
    max_positives: int, max_hard_negatives: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ref_emb [D], pos_embs [P,D], neg_embs [N,D]) for one
    sample. neg_embs may be empty ([0,D]) if candidates.json is missing or
    has no confuser-like candidates."""
    from aero_eyes.utils.geometry import box_iou, crop_with_pad
    from aero_eyes.utils.io import load_gt, read_candidates
    from aero_eyes.utils.video import read_frame

    work_dir = Path(cfg.project.work_dir) / sample_id
    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(f"No video found for sample '{sample_id}'.")
    video_path = video_files[0]

    gt = load_gt(cfg.data.gt.global_file, sample_id)
    if not gt:
        raise ValueError(f"No GT frames for sample '{sample_id}' in {cfg.data.gt.global_file}.")

    ref_emb = _load_ref_embedding(cfg, sample_id, extractor)
    rng = random.Random(seed)

    # ---- positives: crop at the GT box, evenly subsampled across the GT frame range ----
    gt_frames = sorted(gt.keys())
    if len(gt_frames) > max_positives:
        pick = sorted(rng.sample(range(len(gt_frames)), max_positives))
        gt_frames = [gt_frames[i] for i in pick]

    pos_crops = []
    for fi in gt_frames:
        try:
            frame = read_frame(video_path, fi)
        except Exception:
            continue
        pos_crops.append(crop_with_pad(frame, gt[fi], cfg.stage2.candidate.feature_crop_pad))
    if not pos_crops:
        raise ValueError(f"Could not read any GT-frame crop for sample '{sample_id}'.")
    pos_embs = extractor.extract(pos_crops, batch_size=cfg.runtime.batch_size)

    # ---- hard negatives: candidates.json boxes that miss GT (or sit on a GT-absent frame) ----
    feat_dim = pos_embs.shape[1]
    neg_embs = np.zeros((0, feat_dim), dtype=np.float32)
    cand_path = work_dir / "candidates.json"
    if cand_path.exists():
        candidates = read_candidates(cand_path)
        neg_candidates = []
        for fi, dets in candidates.items():
            for det in dets:
                if fi in gt:
                    if box_iou(gt[fi], det.box) < iou_threshold:
                        neg_candidates.append((fi, det.box))
                else:
                    neg_candidates.append((fi, det.box))
        if neg_candidates:
            if len(neg_candidates) > max_hard_negatives:
                neg_candidates = rng.sample(neg_candidates, max_hard_negatives)
            neg_crops = []
            for fi, box in neg_candidates:
                try:
                    frame = read_frame(video_path, fi)
                except Exception:
                    continue
                neg_crops.append(crop_with_pad(frame, box, cfg.stage2.candidate.feature_crop_pad))
            if neg_crops:
                neg_embs = extractor.extract(neg_crops, batch_size=cfg.runtime.batch_size)
    else:
        log.warning(
            "[%s] no candidates.json found at %s -- skipping hard-negative mining for this sample "
            "(run Stage 2 / Stage12-GeCo2 first for confuser-aware negatives; training will still "
            "use in-batch cross-sample negatives from other samples).", sample_id, cand_path,
        )

    return ref_emb, pos_embs.astype(np.float32), neg_embs.astype(np.float32)


def info_nce_loss(ref_proj: torch.Tensor, pos_proj: torch.Tensor, neg_proj: torch.Tensor, temperature: float) -> torch.Tensor:
    """ref_proj/pos_proj: [B,D], index i is the anchor-positive pair for
    sample i in this batch. neg_proj: [M,D] hard negatives pooled from
    every sample in the batch (a negative mined for one sample is still a
    valid, if weaker, negative for every other sample's reference too).
    Symmetric cross-entropy over the [B,B] positive block; the hard
    negatives only feed the ref->crop direction (they only mean "this crop
    is not sample i's target", not the reverse)."""
    B = ref_proj.shape[0]
    crop_proj = torch.cat([pos_proj, neg_proj], dim=0) if neg_proj.shape[0] > 0 else pos_proj
    logits_r2c = ref_proj @ crop_proj.t() / temperature   # [B, B+M]
    logits_c2r = pos_proj @ ref_proj.t() / temperature    # [B, B]
    labels = torch.arange(B, device=ref_proj.device)
    return 0.5 * (F.cross_entropy(logits_r2c, labels) + F.cross_entropy(logits_c2r, labels))


@torch.no_grad()
def eval_top1(
    head, val_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]], device: str,
    val_ids: list[str] | None = None,
) -> tuple[float, float]:
    """Top-1 retrieval accuracy: pool EVERY val sample's positive crops +
    hard negatives into one set, and check whether each sample's
    reference's single nearest neighbor in that pool belongs to its OWN
    sample -- a direct proxy for "does the head separate this sample's
    target from every other sample's target/confusers". NOT a substitute
    for scripts/check_cosine_effect.py end to end (see module docstring).

    Returns (sample_acc, category_acc). sample_acc requires the exact same
    sample id to win; category_acc (only meaningful when val_ids is given)
    additionally counts a match against a DIFFERENT sample of the SAME
    object category (e.g. Lifering_0's ref matching a Lifering_1 crop) as
    correct too -- when a category has >1 video in val (this dataset's
    "_0"/"_1" pairs are near-duplicate videos of the SAME physical object,
    see video_category), sample_acc alone artificially penalizes the model
    for a match that is semantically correct, understating real
    cross-category separation. Watch category_acc, not sample_acc, in that
    situation.
    """
    if len(val_data) < 2:
        log.warning(
            "Only %d val sample(s) -- top-1 retrieval needs >=2 to have any cross-category "
            "negative to confuse against; the metric below is close to meaningless.", len(val_data),
        )

    ref_embs = torch.from_numpy(np.stack([r for r, _, _ in val_data])).to(device)
    ref_proj = head(ref_embs)  # [S, D]

    pool_chunks = []
    pool_owner: list[int] = []
    for j, (_, pos_j, neg_j) in enumerate(val_data):
        pj = head(torch.from_numpy(pos_j).to(device))
        pool_chunks.append(pj)
        pool_owner += [j] * pj.shape[0]
        if neg_j.shape[0] > 0:
            nj = head(torch.from_numpy(neg_j).to(device))
            pool_chunks.append(nj)
            pool_owner += [-1] * nj.shape[0]  # -1 = never counts as a correct match
    pool = torch.cat(pool_chunks, dim=0)  # [T, D]

    if val_ids is not None:
        from aero_eyes.models.geco2_finetune_data import video_category
        val_cats = [video_category(s) for s in val_ids]
    else:
        val_cats = None

    correct_sample = 0
    correct_category = 0
    for i in range(len(val_data)):
        sims = ref_proj[i] @ pool.t()
        best = int(torch.argmax(sims).item())
        owner = pool_owner[best]
        if owner == i:
            correct_sample += 1
            correct_category += 1
        elif val_cats is not None and owner != -1 and val_cats[owner] == val_cats[i]:
            correct_category += 1
    n = len(val_data)
    return correct_sample / n, correct_category / n


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--train-samples", required=True, help="comma-separated sample ids under data_root")
    p.add_argument("--val-samples", required=True,
                    help="comma-separated sample ids, DISJOINT from --train-samples -- required, "
                    "not optional, see module docstring on why a frame-level split doesn't work here")
    p.add_argument("--output", required=True, help="where to save the trained head (.pt)")
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8, help="samples per batch (capped at len(train-samples))")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="a candidate scoring below this IoU against GT on its own frame counts as a hard negative")
    p.add_argument("--max-positives-per-sample", type=int, default=30)
    p.add_argument("--max-hard-negatives-per-sample", type=int, default=30)
    p.add_argument("--early-stop-patience", type=int, default=20,
                    help="stop once val_top1_category hasn't matched or beaten its best value for "
                         "this many CONSECUTIVE epochs. Ties count as 'still fine' (patience resets), "
                         "not just strict improvement -- with only a handful of val samples, "
                         "val_top1_category legitimately plateaus at a GOOD value for the entire run "
                         "(discrete metric, few possible values); this must not be mistaken for "
                         "stagnation. Only a sustained STRICT DROP below the best-ever value (e.g. the "
                         "head catastrophically overfitting train categories within ~10 epochs and "
                         "never recovering, observed in practice: val_top1 1.0->0.0 and flat for 190 "
                         "more epochs) triggers this -- saves the remaining GPU time in exactly that "
                         "failure case instead of running all --epochs regardless.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    args = p.parse_args()

    from aero_eyes.models.geco2_finetune_data import video_category

    train_ids = [s.strip() for s in args.train_samples.split(",") if s.strip()]
    val_ids = [s.strip() for s in args.val_samples.split(",") if s.strip()]
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            f"--train-samples and --val-samples share sample(s) {overlap} -- they must be disjoint "
            "object categories (see module docstring: this measures generalization to unseen objects)."
        )
    # Category-level check, NOT just sample-id equality: 'Person1_0' (train)
    # and 'Person1_1' (val) are different sample ids but -- per this
    # dataset's own naming convention (see video_category's docstring in
    # geco2_finetune_data.py) -- almost certainly two takes of the SAME
    # physical object, sharing very similar/identical reference photos.
    # Putting one in train and the other in val leaks that object's
    # appearance into "validation", making the generalization check
    # meaningless for exactly that category -- the same footgun this
    # project already learned to guard against for the GeCo2 finetune
    # train/val split (split_train_val), now guarded here too.
    train_cats = {video_category(s) for s in train_ids}
    val_cats = {video_category(s) for s in val_ids}
    cat_overlap = train_cats & val_cats
    if cat_overlap:
        raise ValueError(
            f"--train-samples and --val-samples share object CATEGORY/categories {cat_overlap} "
            "(e.g. 'Person1_0' in train + 'Person1_1' in val) -- these are almost certainly two "
            "videos of the SAME physical object, which leaks its appearance into validation. Move "
            "every sample of an affected category to the same side (all-train or all-val)."
        )

    from aero_eyes.config import load_config
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.projection_head import ProjectionHead

    cfg = load_config(args.config, args.set)
    if cfg.stage1.feature_extractor.projection_head.enabled:
        raise ValueError(
            "The config passed to this script has stage1.feature_extractor.projection_head.enabled=true "
            "-- this script needs the RAW backbone embedding to train against, not an already-projected "
            "one. Pass a config with it off (or add --set stage1.feature_extractor.projection_head.enabled=false)."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    extractor = build_feature_extractor(cfg)
    in_dim = extractor._feature_dim()
    device = cfg.device()

    log.info("Building training pairs for %d train sample(s), %d val sample(s)...", len(train_ids), len(val_ids))
    train_data = []
    for sid in train_ids:
        try:
            train_data.append(_collect_sample_crops(
                cfg, sid, extractor, args.iou_threshold,
                args.max_positives_per_sample, args.max_hard_negatives_per_sample, args.seed,
            ))
        except Exception as e:
            log.warning("Skipping train sample '%s': %s", sid, e)
    val_data = []
    for sid in val_ids:
        try:
            val_data.append(_collect_sample_crops(
                cfg, sid, extractor, args.iou_threshold,
                args.max_positives_per_sample, args.max_hard_negatives_per_sample, args.seed,
            ))
        except Exception as e:
            log.warning("Skipping val sample '%s': %s", sid, e)

    if len(train_data) < 2:
        raise ValueError(
            f"Only {len(train_data)} usable train sample(s) -- InfoNCE needs multiple categories per "
            "batch to have any negative to contrast against; add more --train-samples."
        )
    if not val_data:
        raise ValueError("0 usable val samples -- cannot verify generalization, refusing to train blind. Fix --val-samples.")

    log.info("Train: %d samples, %d total positive crops, %d total hard negatives.",
              len(train_data), sum(pe.shape[0] for _, pe, _ in train_data), sum(ne.shape[0] for _, _, ne in train_data))
    log.info("Val:   %d samples, %d total positive crops, %d total hard negatives.",
              len(val_data), sum(pe.shape[0] for _, pe, _ in val_data), sum(ne.shape[0] for _, _, ne in val_data))

    head = ProjectionHead(in_dim=in_dim, out_dim=args.output_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rng = random.Random(args.seed)
    batch_size = min(args.batch_size, len(train_data))
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            if len(batch_idx) < 2:
                continue  # InfoNCE needs >=2 anchors to have an in-batch negative

            refs, poss, negs = [], [], []
            for idx in batch_idx:
                ref_emb, pos_embs, neg_embs = train_data[idx]
                refs.append(ref_emb)
                poss.append(pos_embs[rng.randrange(pos_embs.shape[0])])
                if neg_embs.shape[0] > 0:
                    negs.append(neg_embs[rng.randrange(neg_embs.shape[0])])

            ref_t = torch.from_numpy(np.stack(refs)).to(device)
            pos_t = torch.from_numpy(np.stack(poss)).to(device)

            ref_proj = head(ref_t)
            pos_proj = head(pos_t)
            if negs:
                neg_t = torch.from_numpy(np.stack(negs)).to(device)
                neg_proj = head(neg_t)
            else:
                neg_proj = torch.zeros((0, args.output_dim), device=device)

            loss = info_nce_loss(ref_proj, pos_proj, neg_proj, args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        head.eval()
        val_acc_sample, val_acc_cat = eval_top1(head, val_data, device, val_ids=val_ids)
        # Select on CATEGORY accuracy, not sample accuracy: when a val
        # category has >1 video (this dataset's "_0"/"_1" near-duplicate
        # pairs), sample accuracy wrongly penalizes a match against the
        # SIBLING video of the same physical object -- see eval_top1's
        # docstring. >= (not strict >): with few val samples, this metric
        # plateaus easily and legitimately ties for many epochs in a row;
        # strict `>` would freeze best_state at the FIRST epoch that ever
        # reached the plateau -- observed in practice: an early run hit its
        # plateau at epoch 1 and never moved, silently saving essentially
        # the UNTRAINED epoch-1 head despite 200 epochs of continued loss
        # improvement. >= keeps the LATEST tying epoch instead, which is at
        # least as converged.
        if val_acc_cat >= best_val_acc:
            best_val_acc = val_acc_cat
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_every = max(1, args.epochs // 20)
        if epoch % log_every == 0 or epoch == 1:
            log.info("epoch %d/%d: train_loss=%.4f val_top1_sample=%.3f val_top1_category=%.3f (best_category=%.3f)",
                      epoch, args.epochs, epoch_loss / max(1, n_batches), val_acc_sample, val_acc_cat, best_val_acc)

        if epochs_without_improvement >= args.early_stop_patience:
            log.info(
                "Early stopping at epoch %d: val_top1_category has been STRICTLY below its best "
                "(%.3f) for %d consecutive epochs -- this is a sustained drop, not a plateau at a "
                "good value (which ties every epoch and would never trigger this). Saving the best "
                "state seen (likely an early epoch) instead of continuing to train on what looks "
                "like a collapsed/overfit trajectory.",
                epoch, best_val_acc, epochs_without_improvement,
            )
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.save(args.output)
    log.info("Saved projection head (best val_top1_category=%.3f) -> %s", best_val_acc, args.output)
    log.info(
        "This in-memory top-1 metric is a PROXY, not the real answer. Now run the actual pipeline "
        "with --set stage1.feature_extractor.projection_head.enabled=true --set "
        "stage1.feature_extractor.projection_head.weights_path=%s --set "
        "stage1.feature_extractor.projection_head.output_dim=%d --set "
        "stage1.feature_extractor.projection_head.hidden_dim=%s on your VAL samples, and compare "
        "scripts/check_cosine_effect.py's real P/R/F1 against the no-projection-head baseline.",
        args.output, args.output_dim, args.hidden_dim,
    )


if __name__ == "__main__":
    main()

"""Single-GPU finetuning entrypoint for GeCo2 on the AERO EYES dataset.

Requires the compiled MultiScaleDeformableAttention CUDA extension (see
notebooks/train_geco2_aeroeyes_vastai.ipynb) -- cannot run on a machine
without it built (no CPU fallback exists). See docs/GECO2_FINETUNE_PLAN.md
for the full design this implements.

Usage:
    python -m scripts.train_geco2_aeroeyes \
        --config configs/config.yaml \
        --set data.data_root=/path/to/data \
        --set data.gt.global_file=/path/to/annotations.json \
        --set project.work_dir=/path/to/work \
        --dry-run --dry-run-steps 2

    python -m scripts.train_geco2_aeroeyes \
        --config configs/config.yaml \
        --set data.data_root=/path/to/data \
        --set data.gt.global_file=/path/to/annotations.json \
        --set project.work_dir=/path/to/work \
        --epochs 15 --steps-per-epoch 400
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aero_eyes.models.geco2_finetune_data import (
    DEFAULT_HOLDOUT_CATEGORIES,
    Geco2FinetuneDataset,
    RefImageCache,
    convert_gt_box_to_canvas,
    finetune_collate,
    split_train_val,
    validate_training_video_ids,
)

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_targets(gt_box_canvas: tuple[float, float, float, float] | None, image_size: float) -> torch.Tensor:
    """[0,4] for absent frames, [1,4] normalized xyxy in [0,1] for present."""
    if gt_box_canvas is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor([gt_box_canvas], dtype=torch.float32).clamp(0, image_size) / image_size


def compute_step_loss(
    criterion, out, target_boxes: torch.Tensor, aux_weight: float, image_size: float,
    aux_size_threshold_px: float = 25.0,
):
    """Loss recipe adapted from GECO2/train.py, with two deliberate,
    documented deviations from the reference implementation (see
    docs/GECO2_FINETUNE_PLAN.md point 8):

    1. Explicit `numel() == 0` guard for the absent-frame case, instead of
       relying on train.py's implicit behavior (empty tensor .mean() ->
       nan -> `nan < 25` evaluates False in Python, so it "works" today
       only by accident of NaN-comparison semantics -- fragile across
       torch versions, not something to import uncritically).
    2. Uses `l1['loss_bbox']` (the AUX head's own box loss) in the aux
       term, not `l['loss_bbox']` (the main head's) -- train.py has an
       apparent copy-paste bug reusing the main head's box loss for the
       aux term; fixed rather than reproduced.

    `aux_size_threshold_px` (train.py hardcodes this at 25) gates the aux
    loss to present frames whose mean GT box dimension (on the 1024 canvas)
    is below this many pixels -- a small-object emphasis that made sense
    for FSC147's mixed-size counting targets. AERO EYES objects are
    typically 20-150px, so a lot of present frames fall ABOVE 25px and
    currently give the aux head literally zero gradient signal all run --
    exposed as a tunable here (not hardcoded) so this can be raised (e.g.
    to 100-150) to actually exercise the aux head on this dataset, an
    identified low-risk lever from the first finetune attempt's post-mortem.
    """
    targets = [{"boxes": target_boxes, "labels": torch.zeros(len(target_boxes), dtype=torch.long)}]
    l = criterion(out.main, targets, out.centerness, out.ref_points)
    l1 = criterion(out.aux, targets, out.centerness_aux, out.ref_points_aux)

    if target_boxes.numel() == 0:
        alpha = 0.0
    else:
        mean_h = (target_boxes[:, 3] - target_boxes[:, 1]).mean().item() * image_size
        mean_w = (target_boxes[:, 2] - target_boxes[:, 0]).mean().item() * image_size
        alpha = aux_weight if min(mean_h, mean_w) < aux_size_threshold_px else 0.0

    main_loss = l["loss_giou"] + l["loss_ce"] + l["loss_bbox"]
    aux_loss = alpha * (l1["loss_giou"] + l1["loss_ce"] + l1["loss_bbox"])
    return main_loss + aux_loss


def run_epoch(model, loader, criterion, optimizer, args, image_size, device, train: bool):
    from aero_eyes.models.geco2_train_wrapper import (
        encode_exemplars_grad,
        forward_train_step,
        set_train_mode,
    )

    set_train_mode(model, train)
    total_loss, n_present, n_absent, n_samples = 0.0, 0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch_losses = []
            for sample in batch:
                if sample.is_present:
                    n_present += 1
                else:
                    n_absent += 1

                proto = encode_exemplars_grad(
                    model, sample.ref_images, sample.ref_boxes, image_size, device,
                )
                out = forward_train_step(model, sample.frame_bgr, proto, image_size, device)

                # Reuses the SAME coordinate-conversion function the
                # visualization sanity-check cell uses (see
                # docs/GECO2_FINETUNE_PLAN.md point 7) -- the small
                # duplicated resize_and_pad call vs. forward_train_step's
                # own is a deliberate correctness-over-perf tradeoff: it
                # guarantees the loss target and the visual check can never
                # silently diverge.
                _, gt_box_canvas, _ = convert_gt_box_to_canvas(sample.frame_bgr, sample.gt_box, image_size)
                target_boxes = build_targets(gt_box_canvas, image_size).to(device)

                loss = compute_step_loss(
                    criterion, out, target_boxes, args.aux_weight, image_size,
                    aux_size_threshold_px=args.aux_size_threshold_px,
                )
                batch_losses.append(loss)
                n_samples += 1

            batch_loss = torch.stack(batch_losses).sum()
            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.max_grad_norm,
                    )
                optimizer.step()

            total_loss += float(batch_loss.item())

    mean_loss = total_loss / max(1, n_samples)
    return mean_loss, n_present, n_absent


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finetune GeCo2 on the AERO EYES dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--holdout-categories", nargs="+", default=list(DEFAULT_HOLDOUT_CATEGORIES))
    # Bumped from 15 -> 25 after the first finetune attempt: best val_loss
    # was still improving as late as epoch 12/15 (early stopping never
    # triggered, the run simply ran out of budgeted epochs), so the
    # previous cap was likely leaving real improvement on the table.
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--aux-size-threshold-px", type=float, default=25.0,
                    help="Aux head loss only applies when mean GT box dim (on the 1024 canvas) is "
                         "below this. train.py's original 25px default starves the aux head of any "
                         "signal on most AERO EYES objects (typically 20-150px) -- try e.g. 100-150 "
                         "if the aux head appears under-trained.")
    p.add_argument("--p-present", type=float, default=0.5)
    p.add_argument("--ref-downscale-lo", type=float, default=0.03)
    p.add_argument("--ref-downscale-hi", type=float, default=1.0)
    p.add_argument("--lr-patience", type=int, default=3,
                    help="Epochs with no val_loss improvement before ReduceLROnPlateau halves LR -- "
                         "added after the first finetune attempt showed train+val loss oscillating "
                         "together (both rising for 2+ epochs at a stretch) rather than converging "
                         "smoothly, a classic too-high-LR-for-this-stage symptom.")
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--base-checkpoint", default="./GECO2/CNTQG_multitrain_ca44.pth",
                    help="Base (pretrained-on-FSC147) checkpoint to finetune from. "
                         "Overrides stage123_geco2.weights_path from --config/--set.")
    p.add_argument("--out-checkpoint", default="./GECO2/CNTQG_aeroeyes_finetuned.pth",
                    help="Where to save the finetuned checkpoint. Never overwrites --base-checkpoint.")
    p.add_argument("--early-stop-patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                    help="Run a handful of steps only, assert finite loss, save nothing. "
                         "Run this before any full training commit -- wrapper bugs can only "
                         "surface on a real GPU box, there is no local CPU fallback.")
    p.add_argument("--dry-run-steps", type=int, default=2)
    return p


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    from aero_eyes.config import load_config
    from aero_eyes.utils.io import list_video_ids

    cfg = load_config(args.config, args.set)

    video_ids = list_video_ids(cfg.data.gt.global_file)
    # Hard guard against the training/test GT-file mix-up -- a real raise,
    # not a warning (see docs/GECO2_FINETUNE_PLAN.md point 12).
    validate_training_video_ids(video_ids)
    train_ids, val_ids = split_train_val(video_ids, tuple(args.holdout_categories))
    log.info("Train videos (%d): %s", len(train_ids), train_ids)
    log.info("Val videos (%d): %s", len(val_ids), val_ids)

    # --base-checkpoint is the dedicated CLI flag for the starting-point
    # checkpoint; it intentionally overrides whatever stage123_geco2.weights_path
    # came from --config/--set.
    cfg.stage123_geco2.weights_path = args.base_checkpoint

    device = torch.device(cfg.device())

    from aero_eyes.models.geco2_train_wrapper import build_training_model
    model = build_training_model(cfg, device=device)

    ref_cache = RefImageCache(cfg, video_ids)

    steps_per_epoch = args.dry_run_steps if args.dry_run else args.steps_per_epoch
    train_ds = Geco2FinetuneDataset(
        cfg, train_ids, ref_cache, steps_per_epoch=steps_per_epoch,
        p_present=args.p_present, ref_downscale_range=(args.ref_downscale_lo, args.ref_downscale_hi),
        seed=args.seed,
    )
    val_ds = Geco2FinetuneDataset(
        cfg, val_ids, ref_cache, steps_per_epoch=max(1, steps_per_epoch // 4),
        p_present=args.p_present, ref_downscale_range=(args.ref_downscale_lo, args.ref_downscale_hi),
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=finetune_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=finetune_collate, num_workers=0)

    from models.matcher import PointLossHungarianMatcher  # GECO2/models/matcher.py
    from utils.losses import SetCriterion  # GECO2/utils/losses.py

    # Same defaults as GECO2/utils/arg_parser.py.
    matcher = PointLossHungarianMatcher(cost_class=2.0, cost_bbox=1.0, cost_giou=2.0)
    criterion = SetCriterion(0, matcher, {"loss_giou": 2.0}, ["bboxes", "ce"], focal_alpha=0.25)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # See --lr-patience help text: addresses the oscillating train+val loss
    # observed in the first finetune attempt (both rising together for
    # multiple epochs at a stretch, rather than converging), by halving LR
    # once val_loss plateaus instead of holding a single flat LR for the
    # whole run.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_decay_factor, patience=args.lr_patience,
    )

    image_size = float(cfg.stage123_geco2.image_size)

    if args.dry_run:
        log.info("--- DRY RUN: %d step(s), no checkpoint will be saved ---", args.dry_run_steps)
        t0 = time.time()
        mean_loss, n_present, n_absent = run_epoch(
            model, train_loader, criterion, optimizer, args, image_size, device, train=True,
        )
        assert np.isfinite(mean_loss), f"dry run produced a non-finite loss: {mean_loss}"
        log.info("Dry run OK in %.1fs: mean_loss=%.4f present=%d absent=%d",
                 time.time() - t0, mean_loss, n_present, n_absent)
        return

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    out_checkpoint = Path(args.out_checkpoint)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_present, train_absent = run_epoch(
            model, train_loader, criterion, optimizer, args, image_size, device, train=True,
        )
        val_loss, val_present, val_absent = run_epoch(
            model, val_loader, criterion, optimizer, args, image_size, device, train=False,
        )
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        log.info(
            "epoch %d/%d: train_loss=%.4f (present=%d absent=%d) val_loss=%.4f "
            "(present=%d absent=%d) lr=%.2e [%.1fs]",
            epoch + 1, args.epochs, train_loss, train_present, train_absent,
            val_loss, val_present, val_absent, lr_after, time.time() - t0,
        )
        if lr_after < lr_before:
            log.info("LR decayed: %.2e -> %.2e (val_loss plateaued for %d epochs)",
                      lr_before, lr_after, args.lr_patience)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss, "args": vars(args)},
                out_checkpoint,
            )
            log.info("Saved new best checkpoint (val_loss=%.4f) -> %s", val_loss, out_checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                log.info("Early stopping: no val improvement for %d epoch(s)", epochs_without_improvement)
                break

    log.info("Training done. Best val_loss=%.4f, checkpoint at %s", best_val_loss, out_checkpoint)


if __name__ == "__main__":
    main()

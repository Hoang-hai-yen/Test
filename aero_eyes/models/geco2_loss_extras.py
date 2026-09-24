"""Opt-in loss variants for GeCo2 finetuning (scripts/train_geco2_aeroeyes.py).

Why the stock loss (GECO2/utils/losses.py SetCriterion) hurts here
------------------------------------------------------------------
* The centerness term is an MSE summed over every supervised peak and divided
  by the number of GT boxes (1). A present frame has ONE positive (target 1)
  and dozens of negatives (target 0), an absent frame only negatives -- so
  the gradient is dominated by "push scores to 0", and the more absent frames
  are sampled (lower --p-present) the more the whole score map collapses to
  <= 0. Inference drops non-positive maps (max<=0 -> no detection), which is
  exactly the "recall 0.07 at p_present=0.6" collapse.
* Extra peaks INSIDE the GT box are unmatched (Hungarian is one-to-one) and
  are forced to 0 although NMS merges them at inference.
* The TP target is 1 regardless of box quality, so a high score does not mean
  a tight box; the giou weight (2.0 in weight_dict) is never applied.
* Nothing rewards the RELATIVE order TP > FP, which is what inference uses.

Everything here is a pure tensor function (no CUDA extension needed) and
every option defaults to "off" (LossOptions() reproduces the stock path).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossOptions:
    giou_weight: float = 1.0          # stock effective value is 1.0 (weight_dict is never applied)
    group_norm_ce: bool = False       # centerness: mean over positives + mean over negatives
    mask_gt_peaks: bool = False       # unmatched peaks whose centre is inside a GT box are not pushed to 0
    tp_target_iou_floor: float | None = None  # TP centerness target = clamp(IoU(pred,gt), floor) instead of 1
    multi_peak_box: float = 0.0       # >0: box loss on every peak inside the central `frac` of the GT box
    rank_weight: float = 0.0          # relu(margin + s_FP - s_TP) on the hardest FP peaks
    rank_margin: float = 0.2
    rank_topk: int = 3
    pos_weight: float = 0.0           # relu(pos_margin - s_TP): keeps TPs high independent of p_present
    pos_margin: float = 0.5
    absent_topk: int = 0              # >0: absent frames penalise only the top-k peaks ...
    absent_ceiling: float = 0.0       # ... above this score (squared hinge)
    absent_weight: float = 1.0
    tp_min_iou: float = 0.3           # matched peaks below this IoU are not treated as TP for rank/pos

    def custom_main(self) -> bool:
        return (
            self.group_norm_ce or self.mask_gt_peaks or self.tp_target_iou_floor is not None
            or self.multi_peak_box > 0 or self.rank_weight > 0 or self.pos_weight > 0
            or self.absent_topk > 0
        )


def pair_giou(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise IoU and GIoU of [M,4] xyxy boxes (row i vs row i)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-9)
    lt_c = torch.min(a[:, :2], b[:, :2])
    rb_c = torch.max(a[:, 2:], b[:, 2:])
    hull = (rb_c - lt_c).clamp(min=0).prod(-1)
    return iou, iou - (hull - union) / hull.clamp(min=1e-9)


def _cell_xy(ref_points: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """[2,N] (row=y, col=x) integer grid cells -> [N,2] normalised (x, y) in [0,1)."""
    h, w = grid_hw
    return torch.stack([ref_points[1].float() / w, ref_points[0].float() / h], dim=-1)


def points_in_boxes(xy: torch.Tensor, boxes: torch.Tensor, frac: float = 1.0) -> torch.Tensor:
    """[N,G] bool: point i lies inside the central `frac` portion of box g."""
    if boxes.numel() == 0 or xy.numel() == 0:
        return torch.zeros((len(xy), len(boxes)), dtype=torch.bool, device=xy.device)
    c = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) / 2 * frac
    d = (xy[:, None, :] - c[None]).abs()
    return (d <= half[None]).all(-1)


def _mean_or_zero(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.mean() if x.numel() else like.sum() * 0.0


def custom_main_loss(
    scores: torch.Tensor, pred_boxes: torch.Tensor, centerness: torch.Tensor, ref_points: torch.Tensor,
    target_boxes: torch.Tensor, tp_pred: torch.Tensor, tp_gt: torch.Tensor, fn_gt: torch.Tensor,
    opts: LossOptions,
) -> tuple[torch.Tensor, dict]:
    """scores [N], pred_boxes [N,4] normalised xyxy (both differentiable),
    centerness [1,H,W], ref_points [2,N], target_boxes [G,4] normalised xyxy,
    tp_pred/tp_gt: matched peak / GT indices, fn_gt: unmatched GT indices.
    Returns (loss, stats)."""
    dev = scores.device
    n = scores.shape[0]
    g = target_boxes.shape[0]
    grid_hw = (centerness.shape[-2], centerness.shape[-1])
    tp_pred = tp_pred.to(dev).long()
    tp_gt = tp_gt.to(dev).long()
    fn_gt = fn_gt.to(dev).long()
    is_matched = torch.zeros(n, dtype=torch.bool, device=dev)
    is_matched[tp_pred] = True

    xy = _cell_xy(ref_points, grid_hw) if n else torch.zeros((0, 2), device=dev)
    inside = points_in_boxes(xy, target_boxes, 1.0)                      # [N,G]
    inside_any = inside.any(-1) if g else torch.zeros(n, dtype=torch.bool, device=dev)

    # ---- absent frame: only "how high do the worst peaks get" matters ----
    if g == 0:
        if opts.absent_topk > 0 and n:
            top = scores.topk(min(opts.absent_topk, n)).values
            loss = opts.absent_weight * torch.relu(top - opts.absent_ceiling).pow(2).mean()
        elif n:
            loss = scores.pow(2).mean() if opts.group_norm_ce else scores.pow(2).sum()
        else:
            loss = scores.sum() * 0.0
        return loss, {"n_tp": 0, "n_neg": n}

    # ---- present frame ----
    iou_tp = torch.zeros(0, device=dev)
    if len(tp_pred):
        iou_tp, _ = pair_giou(pred_boxes[tp_pred], target_boxes[tp_gt])
    tp_valid = iou_tp >= opts.tp_min_iou
    tp_s = scores[tp_pred]

    neg_mask = ~is_matched
    if opts.mask_gt_peaks:
        neg_mask = neg_mask & ~inside_any
    neg_s = scores[neg_mask]

    # centerness
    if opts.tp_target_iou_floor is not None:
        tp_target = iou_tp.detach().clamp(min=opts.tp_target_iou_floor)
    else:
        tp_target = torch.ones_like(tp_s)
    if len(fn_gt):
        ctr = (target_boxes[fn_gt, :2] + target_boxes[fn_gt, 2:]) / 2
        cx = (ctr[:, 0] * grid_hw[1]).long().clamp(0, grid_hw[1] - 1)
        cy = (ctr[:, 1] * grid_hw[0]).long().clamp(0, grid_hw[0] - 1)
        fn_s = centerness[0, cy, cx]
    else:
        fn_s = scores[:0]
    pos_err = torch.cat([(tp_s - tp_target).pow(2), (fn_s - 1.0).pow(2)])
    neg_err = neg_s.pow(2)
    if opts.group_norm_ce:
        loss_ce = _mean_or_zero(pos_err, scores) + _mean_or_zero(neg_err, scores)
    else:
        loss_ce = (pos_err.sum() + neg_err.sum()) / max(g, 1)

    # boxes: matched peaks (+ optionally every peak near the GT centre)
    sel_pred, sel_gt = [tp_pred], [tp_gt]
    if opts.multi_peak_box > 0 and n:
        near = points_in_boxes(xy, target_boxes, opts.multi_peak_box) & ~is_matched[:, None]
        rows, cols = near.nonzero(as_tuple=True)
        if len(rows):
            keep = torch.ones_like(rows, dtype=torch.bool)
            keep[1:] = rows[1:] != rows[:-1]      # nonzero() is row-major: first GT per peak
            sel_pred.append(rows[keep])
            sel_gt.append(cols[keep])
    sel_pred_t, sel_gt_t = torch.cat(sel_pred), torch.cat(sel_gt)
    if len(sel_pred_t):
        src, tgt = pred_boxes[sel_pred_t], target_boxes[sel_gt_t]
        _, giou = pair_giou(src, tgt)
        loss_box = (src - tgt).abs().sum(-1).mean() + opts.giou_weight * (1 - giou).mean()
    else:
        loss_box = scores.sum() * 0.0

    loss = loss_ce + loss_box

    # ranking / positive hinge
    good = tp_s[tp_valid]
    if opts.rank_weight > 0 and len(good) and len(neg_s):
        hard = neg_s.topk(min(opts.rank_topk, len(neg_s))).values
        loss = loss + opts.rank_weight * torch.relu(opts.rank_margin + hard[None, :] - good[:, None]).mean()
    if opts.pos_weight > 0 and len(good):
        loss = loss + opts.pos_weight * torch.relu(opts.pos_margin - good).mean()

    return loss, {"n_tp": int(tp_valid.sum()), "n_neg": int(neg_mask.sum())}


def frame_detection_stats(
    scores: torch.Tensor, pred_boxes: torch.Tensor, ref_points: torch.Tensor, grid_hw: tuple[int, int],
    target_boxes: torch.Tensor, good_iou: float = 0.5, bad_iou: float = 0.3,
) -> dict:
    """Detached per-frame diagnostics used for checkpoint selection / hard-frame mining.
    Present frame: top1_hit (highest-scoring peak has IoU>=good_iou), margin
    (best good peak - highest bad peak outside the GT, clipped to [-1,1]),
    empty. Absent frame: max_score (-inf when no peaks)."""
    scores, pred_boxes = scores.detach(), pred_boxes.detach()
    n = scores.shape[0]
    if target_boxes.shape[0] == 0:
        return {"present": False, "empty": n == 0, "max_score": float(scores.max()) if n else float("-inf")}
    if n == 0:
        return {"present": True, "empty": True, "top1_hit": False, "margin": -1.0, "hardness": 2.0}
    gt = target_boxes[:1].expand(n, 4)
    iou, _ = pair_giou(pred_boxes, gt)
    xy = _cell_xy(ref_points, grid_hw)
    inside = points_in_boxes(xy, target_boxes[:1], 1.0)[:, 0]
    good = iou >= good_iou
    bad = (iou < bad_iou) & ~inside
    best_good = float(scores[good].max()) if bool(good.any()) else None
    worst_bad = float(scores[bad].max()) if bool(bad.any()) else None
    if best_good is None:
        margin = -1.0
    elif worst_bad is None:
        margin = min(1.0, best_good)
    else:
        margin = max(-1.0, min(1.0, best_good - worst_bad))
    return {
        "present": True, "empty": False, "top1_hit": bool(good[int(scores.argmax())]),
        "margin": margin, "hardness": 1.0 - margin,
    }

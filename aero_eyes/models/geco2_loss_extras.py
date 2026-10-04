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

import numpy as np
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
    recall_weight: float = 0.0        # relu(recall_margin - best score near each GT centre), every present frame
    recall_margin: float = 0.3
    recall_center_frac: float = 0.5   # "near" = peak centre inside this central fraction of the GT box

    def custom_main(self) -> bool:
        return (
            self.group_norm_ce or self.mask_gt_peaks or self.tp_target_iou_floor is not None
            or self.multi_peak_box > 0 or self.rank_weight > 0 or self.pos_weight > 0
            or self.absent_topk > 0 or self.recall_weight > 0
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
    if opts.recall_weight > 0:
        best = recall_anchor_scores(scores, xy, centerness, target_boxes, opts.recall_center_frac)
        loss = loss + opts.recall_weight * torch.relu(opts.recall_margin - best).mean()

    return loss, {"n_tp": int(tp_valid.sum()), "n_neg": int(neg_mask.sum())}


def recall_anchor_scores(
    scores: torch.Tensor, xy: torch.Tensor, centerness: torch.Tensor, target_boxes: torch.Tensor,
    center_frac: float = 0.5,
) -> torch.Tensor:
    """[G] differentiable "is this GT detectable at all" score: the highest
    peak whose centre lies in the central `center_frac` of the GT box, or --
    when no peak is there -- the centerness map value at the GT centre cell.
    No Hungarian match and no IoU condition, so frames with a poor box (where
    the pos/rank terms are skipped) still get pushed to keep a peak on the
    object; the recall hinge relu(margin - this) is what keeps TPs above the
    operating threshold."""
    h, w = centerness.shape[-2], centerness.shape[-1]
    near = points_in_boxes(xy, target_boxes, center_frac)          # [N,G]
    out = []
    for gi in range(target_boxes.shape[0]):
        idx = near[:, gi].nonzero(as_tuple=True)[0]
        if len(idx):
            out.append(scores[idx].max())
        else:
            c = (target_boxes[gi, :2] + target_boxes[gi, 2:]) / 2
            cx = (c[0] * w).long().clamp(0, w - 1)
            cy = (c[1] * h).long().clamp(0, h - 1)
            out.append(centerness[0, cy, cx])
    return torch.stack(out) if out else scores[:0]


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
    top = float(scores.max()) if n else float("-inf")
    if target_boxes.shape[0] == 0:
        return {"present": False, "empty": n == 0, "max_score": top, "top_score": top}
    if n == 0:
        return {"present": True, "empty": True, "top1_hit": False, "margin": -1.0, "hardness": 2.0,
                "top_score": top, "tp_score": None}
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
        "top_score": top, "tp_score": best_good,
    }


def frame_level_detection_metrics(rows: list[dict], threshold: float) -> dict:
    """Cross-frame metrics from frame_detection_stats rows, treating each
    frame's top-1 peak as THE detection (one target per video, as the
    pipeline uses it) and ranking all frames -- present and absent -- by that
    score, so a TP only counts as separated if it beats the strongest peak of
    OTHER frames too (the in-frame `margin` cannot see that):

      frame_ap        average precision over that ranking (TP = present frame
                      whose top-1 has IoU>=0.5; FP = any other frame's top-1);
                      denominator = number of present frames. Threshold-free.
      best_f1 / best_f1_thr  best F1 over all thresholds, and where it is.
      recall_at_t     present frames whose top-1 is correct AND >= threshold
                      ("TPs still detected" at a fixed operating point).
      absent_fp_at_t  absent frames whose top-1 >= threshold.
      tp_score        mean score of the best good (IoU>=0.5) peak, over
                      present frames that have one.
      oracle_recall   present frames with ANY good (IoU>=0.5) peak, whatever
                      its score -- the recall ceiling if scoring were perfect.
                      High oracle_recall but low recall_at_t/top1_hit = the
                      object is localized but scored too low (the heads can
                      fix that); low oracle_recall = no peak lands on it at
                      all (adapt_features has to change)."""
    items: list[tuple[float, bool]] = []
    n_pos = n_neg = 0
    rec_hits = fp_hits = 0
    tp_scores = []
    for r in rows:
        top = r.get("top_score", r.get("max_score", float("-inf")))
        has_pred = not r.get("empty") and top is not None and np.isfinite(top)
        if r.get("present"):
            n_pos += 1
            if r.get("tp_score") is not None:
                tp_scores.append(r["tp_score"])
            if has_pred:
                items.append((top, bool(r.get("top1_hit"))))
                rec_hits += int(bool(r.get("top1_hit")) and top >= threshold)
        else:
            n_neg += 1
            if has_pred:
                items.append((top, False))
                fp_hits += int(top >= threshold)
    items.sort(key=lambda t: -t[0])
    tp = fp = 0
    ap = best_f1 = 0.0
    best_thr = float("nan")
    for score, is_tp in items:
        tp, fp = tp + is_tp, fp + (not is_tp)
        precision, recall = tp / (tp + fp), (tp / n_pos if n_pos else 0.0)
        if is_tp:
            ap += precision
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, score
    nan = float("nan")
    return {
        "frame_ap": ap / n_pos if n_pos else nan,
        "best_f1": best_f1 if n_pos else nan,
        "best_f1_thr": best_thr,
        "recall_at_t": rec_hits / n_pos if n_pos else nan,
        "absent_fp_at_t": fp_hits / n_neg if n_neg else nan,
        "tp_score": float(np.mean(tp_scores)) if tp_scores else nan,
        "oracle_recall": len(tp_scores) / n_pos if n_pos else nan,
    }


def matched_tp_scores(
    matcher, main_out: dict, target_boxes: torch.Tensor, min_iou: float = 0.3,
) -> torch.Tensor:
    """Differentiable scores of the Hungarian-matched peaks whose box has
    IoU >= min_iou with their GT (same TP definition as the rank/pos terms);
    empty tensor when there is no GT, no peak, or no good match."""
    scores = main_out["box_v"][0]
    if target_boxes.shape[0] == 0 or scores.shape[0] == 0:
        return scores[:0]
    targets = [{"boxes": target_boxes, "labels": torch.zeros(len(target_boxes), dtype=torch.long)}]
    with torch.no_grad():
        indices, _fn, _fp = matcher(main_out, targets)
    tp_pred = torch.as_tensor(indices[0][0], dtype=torch.long, device=scores.device)
    tp_gt = torch.as_tensor(indices[0][1], dtype=torch.long, device=scores.device)
    if tp_pred.numel() == 0:
        return scores[:0]
    iou, _ = pair_giou(main_out["pred_boxes"][0][tp_pred].detach(), target_boxes[tp_gt])
    return scores[tp_pred][iou >= min_iou]


def cross_frame_rank_loss(
    tp_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float, topk: int = 1,
) -> torch.Tensor:
    """Score-space triplet across frames: the anchor is the shared exemplar
    set, the positive a TP peak of a present frame, the negatives the
    `topk` strongest peaks of an ABSENT frame of the same video scored with
    the SAME exemplars -> mean relu(margin + s_neg - s_tp). Unlike the
    in-frame rank term this compares against other frames, which is what a
    global score threshold at inference needs. Zero (graph kept) when either
    side is empty."""
    if tp_scores.numel() == 0 or negative_scores.numel() == 0:
        return tp_scores.sum() * 0.0 + negative_scores.sum() * 0.0
    hard = negative_scores.topk(min(topk, negative_scores.numel())).values
    return torch.relu(margin + hard[None, :] - tp_scores[:, None]).mean()

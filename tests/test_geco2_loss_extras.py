import numpy as np
import pytest
import torch

from aero_eyes.models.geco2_loss_extras import (
    LossOptions, custom_main_loss, frame_detection_stats, pair_giou, points_in_boxes,
)

GT = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
H = W = 100


def _rp(cells):
    """[(y, x), ...] grid cells -> [2,N]"""
    return torch.tensor(cells, dtype=torch.long).t().contiguous()


def _scene(tp_score=0.9, fp_scores=(0.3, 0.2), dup_score=None, tp_box=(0.4, 0.4, 0.6, 0.6)):
    """peak 0 = TP at the GT centre; then far-away FPs; optionally a duplicate peak inside the GT."""
    cells = [(50, 50)] + [(10 + 5 * i, 10) for i in range(len(fp_scores))]
    scores = [tp_score, *fp_scores]
    boxes = [list(tp_box)] + [[0.05, 0.05, 0.15, 0.15]] * len(fp_scores)
    if dup_score is not None:
        cells.append((48, 52))
        scores.append(dup_score)
        boxes.append([0.41, 0.41, 0.59, 0.59])
    s = torch.tensor(scores, requires_grad=True)
    b = torch.tensor(boxes, requires_grad=True)
    return s, b, torch.zeros(1, H, W), _rp(cells)


def _run(opts, **kw):
    s, b, c, rp = _scene(**kw)
    loss, info = custom_main_loss(
        s, b, c, rp, GT, torch.tensor([0]), torch.tensor([0]), torch.zeros(0, dtype=torch.long), opts,
    )
    return loss, info, s


def test_pair_giou_identical_and_disjoint():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.1, 0.1]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.8, 0.8, 0.9, 0.9]])
    iou, giou = pair_giou(a, b)
    assert iou[0] == pytest.approx(1.0) and giou[0] == pytest.approx(1.0)
    assert iou[1] == 0.0 and giou[1] < 0.0


def test_points_in_boxes_central_fraction():
    xy = torch.tensor([[0.5, 0.5], [0.59, 0.5]])
    assert points_in_boxes(xy, GT, 1.0).squeeze(-1).tolist() == [True, True]
    assert points_in_boxes(xy, GT, 0.5).squeeze(-1).tolist() == [True, False]


def test_default_options_are_not_custom():
    assert LossOptions().custom_main() is False


def test_group_norm_stops_negatives_from_swamping_the_positive():
    many = tuple([0.3] * 50)
    stock, _, _ = _run(LossOptions(group_norm_ce=False, tp_min_iou=0.3), fp_scores=many)
    grouped, _, _ = _run(LossOptions(group_norm_ce=True), fp_scores=many)
    assert grouped < stock / 5


def test_mask_gt_peaks_ignores_duplicates_inside_the_gt():
    a, _, _ = _run(LossOptions(group_norm_ce=True), dup_score=0.8)
    b, _, _ = _run(LossOptions(group_norm_ce=True, mask_gt_peaks=True), dup_score=0.8)
    assert b < a


def test_ranking_loss_only_when_fp_within_margin():
    base = LossOptions(group_norm_ce=True)
    ok, _, _ = _run(base, tp_score=0.9, fp_scores=(0.3,))
    r_ok, _, _ = _run(LossOptions(group_norm_ce=True, rank_weight=1.0, rank_margin=0.2), tp_score=0.9, fp_scores=(0.3,))
    assert r_ok.item() == pytest.approx(ok.item(), abs=1e-6)
    bad, _, _ = _run(base, tp_score=0.5, fp_scores=(0.6,))
    r_bad, _, _ = _run(LossOptions(group_norm_ce=True, rank_weight=1.0, rank_margin=0.2), tp_score=0.5, fp_scores=(0.6,))
    assert r_bad.item() == pytest.approx(bad.item() + (0.2 + 0.6 - 0.5), abs=1e-5)


def test_ranking_gradient_raises_tp_and_lowers_hardest_fp():
    opts = LossOptions(group_norm_ce=True, rank_weight=1.0, rank_margin=0.2)
    loss, _, s = _run(opts, tp_score=0.5, fp_scores=(0.6, 0.1))
    loss.backward()
    assert s.grad[0] < 0          # descending the loss raises the TP score
    assert s.grad[1] > 0          # ...and lowers the hardest FP


def test_pos_hinge_keeps_tp_above_margin():
    loss0, _, _ = _run(LossOptions(group_norm_ce=True), tp_score=0.2)
    loss1, _, _ = _run(LossOptions(group_norm_ce=True, pos_weight=2.0, pos_margin=0.5), tp_score=0.2)
    assert loss1.item() == pytest.approx(loss0.item() + 2.0 * 0.3, abs=1e-5)


def test_iou_target_lowers_target_for_loose_box():
    opts = LossOptions(group_norm_ce=True, tp_target_iou_floor=0.3)
    tight, _, _ = _run(opts, tp_score=0.95, fp_scores=(), tp_box=(0.4, 0.4, 0.6, 0.6))
    loose, _, _ = _run(opts, tp_score=0.95, fp_scores=(), tp_box=(0.3, 0.3, 0.7, 0.7))   # IoU 0.25 -> floor 0.3
    assert loose.item() > tight.item()


def test_multi_peak_box_adds_box_supervision_to_duplicates():
    a, _, _ = _run(LossOptions(group_norm_ce=True, mask_gt_peaks=True), dup_score=0.8)
    b, _, _ = _run(LossOptions(group_norm_ce=True, mask_gt_peaks=True, multi_peak_box=1.0), dup_score=0.8)
    assert b.item() != pytest.approx(a.item())


def test_absent_topk_penalises_only_the_highest_peaks():
    s = torch.tensor([0.9, 0.8, 0.05, 0.05, 0.05], requires_grad=True)
    rp = _rp([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)])
    b = torch.zeros(5, 4)
    empty = torch.zeros(0, dtype=torch.long)
    loss, _ = custom_main_loss(
        s, b, torch.zeros(1, H, W), rp, torch.zeros(0, 4), empty, empty, empty,
        LossOptions(absent_topk=2, absent_ceiling=0.1),
    )
    assert loss.item() == pytest.approx((0.8 ** 2 + 0.7 ** 2) / 2, abs=1e-6)
    loss.backward()
    assert s.grad[2:].abs().sum() == 0


def test_absent_frame_without_peaks_is_zero_loss():
    empty = torch.zeros(0, dtype=torch.long)
    loss, _ = custom_main_loss(
        torch.zeros(0, requires_grad=True), torch.zeros(0, 4), torch.zeros(1, H, W), torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, 4), empty, empty, empty, LossOptions(absent_topk=3),
    )
    assert loss.item() == 0.0


def test_fn_gt_center_is_supervised_towards_one():
    c = torch.zeros(1, H, W, requires_grad=True)
    empty = torch.zeros(0, dtype=torch.long)
    loss, _ = custom_main_loss(
        torch.zeros(0), torch.zeros(0, 4), c, torch.zeros(2, 0, dtype=torch.long), GT, empty, empty,
        torch.tensor([0]), LossOptions(group_norm_ce=True),
    )
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert c.grad[0, 50, 50] < 0


def test_frame_stats_top1_and_margin_present():
    s, b, _, rp = _scene(tp_score=0.9, fp_scores=(0.3,))
    st = frame_detection_stats(s, b, rp, (H, W), GT)
    assert st["top1_hit"] is True and st["margin"] == pytest.approx(0.6)
    s2, b2, _, rp2 = _scene(tp_score=0.2, fp_scores=(0.7,))
    st2 = frame_detection_stats(s2, b2, rp2, (H, W), GT)
    assert st2["top1_hit"] is False and st2["margin"] < 0


def test_frame_stats_absent_and_empty():
    empty = frame_detection_stats(torch.zeros(0), torch.zeros(0, 4), torch.zeros(2, 0, dtype=torch.long), (H, W), torch.zeros(0, 4))
    assert empty["present"] is False and empty["empty"] is True and empty["max_score"] == float("-inf")
    st = frame_detection_stats(torch.zeros(0), torch.zeros(0, 4), torch.zeros(2, 0, dtype=torch.long), (H, W), GT)
    assert st["empty"] is True and st["top1_hit"] is False


def test_hard_frame_sampling_prefers_recorded_hard_frames():
    from aero_eyes.models.geco2_finetune_data import Geco2FinetuneDataset

    ds = Geco2FinetuneDataset.__new__(Geco2FinetuneDataset)
    ds.rng = np.random.default_rng(0)
    ds.p_present = 0.0
    ds.hard_frame_frac = 1.0
    ds.hard_frame_top = 2
    ds.hard_count = 0
    ds._hardness = {}
    ds._pools = {"v": ([1, 2], list(range(100, 200)), 200)}
    ds.record_hardness("v", 150, False, 0.9)
    ds.record_hardness("v", 151, False, 0.8)
    ds.record_hardness("v", 152, False, 0.1)
    picks = {ds._sample_frame("v")[0] for _ in range(50)}
    assert picks <= {150, 151} and ds.hard_count == 50

"""Fusion of overlapping candidate boxes (cosine_rescore.candidate_fusion).

Two modes, for two different situations:

  "union" -- the boxes are PARTS of one object (e.g. wheel / handlebar / body
    of a motorbike, each detected separately). Boxes are linked when one lies
    (mostly) inside the other -- intersection over the SMALLER box's area, which
    plain IoU cannot express since a nested box has a low IoU -- and every
    connected group becomes one enclosing box. Guarded by max_union_area_ratio
    so a chain of overlaps can't swallow separate objects.

  "wbf" -- the boxes are near-duplicates of the SAME extent. Weighted Boxes
    Fusion (Solovyev et al., 2021): boxes are clustered by IoU in descending
    score order and each cluster becomes the score-weighted mean of its
    coordinates. It averages coordinates instead of unioning, so it does NOT
    turn parts into a whole: use "union" for that. Since boxes reaching this
    step have already survived NMS at nms_iou, iou_thresh must be BELOW
    nms_iou or no pair can ever be linked.
"""
from __future__ import annotations

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou


def _intersection(a: Box, b: Box) -> float:
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return iw * ih


def box_iomin(a: Box, b: Box) -> float:
    """Intersection over the SMALLER box's area (1.0 when one contains the other)."""
    smaller = min(a.area(), b.area())
    return _intersection(a, b) / smaller if smaller > 0 else 0.0


def _peak_contrast(members: list[Box]) -> float | None:
    """max over members, or None if any member has none (keeps downstream
    peakiness fusion from seeing a half-filled set)."""
    vals = [m.peak_contrast for m in members]
    return None if any(v is None for v in vals) else max(vals)


def merge_union(members: list[Box]) -> Box:
    return Box(
        min(m.x1 for m in members), min(m.y1 for m in members),
        max(m.x2 for m in members), max(m.y2 for m in members),
        score=max(m.score for m in members), peak_contrast=_peak_contrast(members),
    )


def merge_wbf(members: list[Box]) -> Box:
    w = [max(m.score, 1e-9) for m in members]
    tot = sum(w)
    return Box(
        sum(wi * m.x1 for wi, m in zip(w, members)) / tot, sum(wi * m.y1 for wi, m in zip(w, members)) / tot,
        sum(wi * m.x2 for wi, m in zip(w, members)) / tot, sum(wi * m.y2 for wi, m in zip(w, members)) / tot,
        score=sum(m.score for m in members) / len(members), peak_contrast=_peak_contrast(members),
    )


def _union_clusters(boxes: list[Box], containment_thresh: float) -> list[list[int]]:
    """Connected components of the graph 'iomin >= containment_thresh'."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iomin(boxes[i], boxes[j]) >= containment_thresh:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _wbf_clusters(boxes: list[Box], iou_thresh: float) -> list[list[int]]:
    """Greedy WBF clustering: boxes in descending score order join the first
    cluster whose CURRENT fused box they overlap with IoU > iou_thresh."""
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i].score)
    clusters: list[list[int]] = []
    fused: list[Box] = []
    for i in order:
        for c, fb in enumerate(fused):
            if box_iou(boxes[i], fb) > iou_thresh:
                clusters[c].append(i)
                fused[c] = merge_wbf([boxes[k] for k in clusters[c]])
                break
        else:
            clusters.append([i])
            fused.append(boxes[i])
    return clusters


def fuse_overlapping_boxes(boxes: list[Box], cfg) -> list[Box]:
    """Apply cfg (a CandidateFusionConfig) to one frame's boxes.

    Clusters with fewer than cfg.min_boxes members are left untouched. For
    every fused cluster: keep_originals=True keeps the member boxes AND appends
    the fused box after them (so boxes[0] stays the top GeCo2 box); False drops
    the members and returns the result best-score-first. In "union" mode a
    cluster is only fused if its enclosing box is at most
    max_union_area_ratio x the area of its largest member."""
    if not cfg.enabled or len(boxes) < max(cfg.min_boxes, 2):
        return list(boxes)
    if cfg.mode == "union":
        clusters = _union_clusters(boxes, cfg.containment_thresh)
        merge = merge_union
    else:
        clusters = _wbf_clusters(boxes, cfg.iou_thresh)
        merge = merge_wbf

    kept: list[Box] = []
    fused_boxes: list[Box] = []
    for idxs in clusters:
        members = [boxes[i] for i in idxs]
        fused = merge(members) if len(members) >= cfg.min_boxes else None
        if fused is not None and cfg.mode == "union":
            if fused.area() > cfg.max_union_area_ratio * max(m.area() for m in members):
                fused = None                                # spans more than one object's worth
        if fused is None:
            kept.extend(members)
            continue
        fused.fused = True  # type: ignore[attr-defined]  -- only for viz (a distinct color/label); not serialized
        fused_boxes.append(fused)
        if cfg.keep_originals:
            kept.extend(members)

    if cfg.keep_originals:
        order = {id(b): k for k, b in enumerate(boxes)}
        return sorted(kept, key=lambda b: order[id(b)]) + fused_boxes
    return sorted(kept + fused_boxes, key=lambda b: -b.score)

"""Reference foreground masking (Stage 1 / stage123_geco2.segmentation) --
MobileSAM (default), FastSAM, or standalone SAM2, selected by
SegmentationConfig.model. See build_segmenter() for the dispatch factory.

If weights are unavailable and fallback_if_missing == "passthrough",
returns an all-ones mask without crashing (FastSAM/SAM2's segment() always
behaves this way -- neither has a fallback_if_missing="error" mode).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)


def _pick_best_full_frame_mask(
    masks: np.ndarray, scores: np.ndarray, use_point_prompt: bool, cx: int, cy: int,
    min_area_frac: float, max_area_frac: float, score_ratio_floor: float, max_border_touch_frac: float,
) -> np.ndarray:
    """Shared "which of a SAM-family promptable decoder's multimask_output
    candidates is the genuine whole-object mask" selection -- the exact
    algorithm MobileSAMSegmenter.segment() uses (isolate a connected
    component per candidate, prefer the LARGEST among candidates the model
    itself scored within score_ratio_floor of its own best AND that pass
    the area/border-touch plausibility gates, else fall back to the single
    highest-scoring candidate). Factored out so SAM2Segmenter.segment() can
    reuse it exactly instead of drifting out of sync with MobileSAM's own
    tuning -- see MobileSAMSegmenter.segment()'s own docstring for the full
    rationale behind each step.
    """
    from aero_eyes.utils.geometry import isolate_component_at_point

    if use_point_prompt:
        cleaned = [isolate_component_at_point(m, cx, cy) for m in masks]
    else:
        cleaned = [MobileSAMSegmenter._isolate_largest_component(m) for m in masks]
    areas = [float(m.mean()) for m in cleaned]
    border_touch = [MobileSAMSegmenter._border_touch_frac(m) for m in cleaned]
    max_score = float(np.max(scores))
    confident = [i for i, s in enumerate(scores) if s >= max_score * score_ratio_floor]
    plausible = [
        i for i in confident
        if min_area_frac <= areas[i] <= max_area_frac and border_touch[i] <= max_border_touch_frac
    ]
    best_idx = max(plausible, key=lambda i: areas[i]) if plausible else int(np.argmax(scores))
    return cleaned[best_idx]


class MobileSAMSegmenter:
    """Segment the largest/most-central object in a reference image."""

    def __init__(self, weights_path: str | None = None, fallback_if_missing: str = "passthrough",
                 min_area_frac: float = 0.05, max_area_frac: float = 0.95,
                 score_ratio_floor: float = 0.85, max_border_touch_frac: float = 0.02,
                 use_point_prompt: bool = True, reject_implausible_mask: bool = True):
        self.weights_path = weights_path
        self.fallback_if_missing = fallback_if_missing
        # See SegmentationConfig.reject_implausible_mask's own docstring --
        # False skips the area/border-touch pass/fail gate below entirely,
        # returning whatever segment()'s candidate-selection picked as-is.
        self.reject_implausible_mask = reject_implausible_mask
        # Center-point prompt assumes the geometric center pixel is
        # foreground -- breaks down for ring/donut-shaped objects (e.g. a
        # life ring) whose center is a HOLLOW interior (background), which
        # can bias SAM's mask proposals toward confused/leaky boundaries
        # (confirmed empirically: life-ring reference photos showed both
        # border-touching passthrough failures AND loose/over-inclusive
        # masks on the candidate that WAS accepted). Set False to prompt
        # with the box alone (no point) for object shapes like this.
        self.use_point_prompt = use_point_prompt
        # Guardrail: a single center-point prompt sometimes locks onto a tiny
        # spurious region (a shadow, a logo) or ~the whole frame (no real
        # segmentation). Either extreme is worse than no masking at all, so
        # reject implausible mask sizes and fall back to passthrough for
        # that image instead of feeding a near-all-black or no-op crop
        # downstream.
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        # Among area-plausible candidates, only "largest wins" among those
        # SAM itself scored within this ratio of the best -- otherwise the
        # largest candidate can be a low-confidence over-segmentation that
        # bleeds into the background.
        self.score_ratio_floor = score_ratio_floor
        # Guardrail: the box prompt is inset `margin` (5%) from the true
        # image edges, so a candidate whose mask actually touches the real
        # image border is virtually never the object itself -- reference
        # photos frame the subject with margin, but a background/ground
        # plane commonly runs off-frame. This catches the case score+area
        # alone cannot: a mask that is genuinely one connected, high-scoring
        # blob because SAM's boundary leaked from the object into contiguous
        # background (confirmed empirically -- a leaked candidate touched
        # 79% of one edge while the correct candidate touched 0% of all
        # four).
        self.max_border_touch_frac = max_border_touch_frac
        self._sam = None
        self._predictor = None
        self._available = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
            model_type = "vit_t"
            ckpt = self.weights_path
            if ckpt is None:
                # Try auto-download path
                import os
                ckpt = os.path.join(os.path.expanduser("~"), ".cache", "mobile_sam",
                                    "mobile_sam.pt")
            if not self._file_exists(ckpt):
                self._maybe_download(ckpt)
            self._sam = sam_model_registry[model_type](checkpoint=ckpt)
            self._sam.eval()
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._sam.to(device)
            except Exception:
                pass
            self._predictor = SamPredictor(self._sam)
            self._available = True
            log.info("MobileSAM loaded from %s", ckpt)
        except Exception as e:
            if self.fallback_if_missing == "passthrough":
                log.warning(
                    "MobileSAM unavailable (%s). Using passthrough (full-image mask).", e
                )
                self._available = False
            else:
                raise RuntimeError(
                    f"MobileSAM could not be loaded and fallback_if_missing != 'passthrough'. "
                    f"Error: {e}. Install mobile-sam or set stage1.segmentation.fallback_if_missing=passthrough."
                ) from e

    @staticmethod
    def _isolate_component_at_point(mask: np.ndarray, px: int, py: int) -> np.ndarray:
        """See aero_eyes.utils.geometry.isolate_component_at_point (shared
        with GeCo2Detector's sam2_refine_boxes wrapper -- same logic
        regardless of which model produced the mask)."""
        from aero_eyes.utils.geometry import isolate_component_at_point
        return isolate_component_at_point(mask, px, py)

    @staticmethod
    def _isolate_largest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected foreground component -- used
        instead of _isolate_component_at_point when there is no reliable
        known-foreground point to anchor on (use_point_prompt=False), e.g.
        for ring/donut-shaped objects where the geometric center is the
        hollow interior, not the object material."""
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
        if num_labels <= 2:
            return mask
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        return labels == largest_label

    @staticmethod
    def _border_touch_frac(mask: np.ndarray) -> float:
        """Fraction of the image's outer-edge pixels (all 4 sides) that are
        foreground. Near 0 for a well-framed subject; large when the mask
        has leaked into a background plane that runs off-frame.
        """
        edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        return float(edges.mean())

    @staticmethod
    def _file_exists(path: str) -> bool:
        import os
        return os.path.isfile(path)

    def _maybe_download(self, ckpt: str) -> None:
        import os
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
            log.info("Downloading MobileSAM weights to %s ...", ckpt)
            urllib.request.urlretrieve(url, ckpt)
        except Exception as e:
            log.warning("MobileSAM download failed: %s", e)

    def set_frame(self, frame_bgr: np.ndarray) -> bool:
        """Encode `frame_bgr` ONCE via SAM's own image encoder, caching the
        embedding for subsequent segment_box_cached() calls on THIS frame
        -- the "dense feature" style of box refinement (box_refine.method
        == "sam_dense"): one encode per FRAME, reused for every candidate
        box on it, instead of a separate crop+re-encode per box (see
        segment_box() above). Avoids the small/low-res crop reliability
        problem confirmed in practice on tiny detection boxes, by giving
        the encoder full spatial context -- the same principle GeCo2's own
        SAM2-based sam_mask module uses (reusing its already-computed
        dense backbone features instead of re-encoding a crop), just with
        MobileSAM's own encoder instead of requiring a separate SAM2
        checkpoint (GeCo2's Hiera features aren't compatible with
        MobileSAM's SAM1-style decoder -- different training distribution).

        Returns False (caller should treat refinement as unavailable for
        this frame) if MobileSAM is unavailable or encoding fails.
        """
        if not self._available:
            return False
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(frame_rgb)
            return True
        except Exception as e:
            log.warning("MobileSAM set_frame failed (%s).", e)
            return False

    def segment_box_cached(
        self, box: Box, margin: float = 0.0, use_center_point: bool = False,
    ) -> np.ndarray | None:
        """Refine `box` using the image embedding set_frame() already
        cached for the CURRENT frame -- must be called after set_frame()
        for that frame. `box` is in that frame's own pixel coordinates
        (SamPredictor applies its own internal resize transform).

        `margin`: expand `box` by this fraction of its own width/height on
        each side BEFORE using it as SAM's box prompt (0.0 = prompt with
        `box` exactly as given). SAM's box-prompted decoder treats the box
        fairly literally -- if the incoming box already UNDERSIZES the real
        object (a common detector failure mode: box covers e.g. only ~60%
        of the true object), prompting with that same tight box gives SAM
        no visual room to recognize the object continues past it, so the
        returned mask tends to stay close to the input box regardless of
        box_refine.min_iou_with_original (confirmed in practice: relaxing
        that gate to 0.0 made no difference for such boxes, because the
        candidate SAM proposed was already nearly identical to the
        original -- the bottleneck was the prompt, not the gate). A small
        margin costs nothing extra here (the whole frame is already
        encoded by set_frame(), unlike segment_box()'s per-call crop).

        `use_center_point` (box_refine.use_center_point_prompt): also pass
        the ORIGINAL (pre-margin) box's center as a positive point prompt
        alongside the expanded box. A bigger margin gives SAM room to reach
        the true boundary but also more background/confuser area it could
        latch onto instead -- the center point pins down which blob in that
        wider region is actually the target, without shrinking the margin
        itself. Off by default: this project's OWN reference-image
        segmentation (segment()'s use_point_prompt) found a center point
        unreliable for ring/donut-shaped objects (a hollow center is
        background, not foreground) -- only enable this if none of your
        tracked object classes are shaped like that.

        Unlike segment_box() (which applies area/border-touch plausibility
        gates calibrated for a small CROP), this trusts SAM's own
        predicted-IoU score directly and relies on the caller's
        box_refine.min_iou_with_original gate (see aero_eyes.utils.
        box_refine.refine_boxes_dense) as the safety check instead --
        those crop-relative gates don't translate to a full-frame mask,
        where a legitimate small object occupies a tiny fraction of the
        whole image.

        Returns a full-frame-sized mask, or None if unavailable, inference
        fails, or the best mask is empty.
        """
        if not self._available:
            return None
        try:
            bw, bh = box.x2 - box.x1, box.y2 - box.y1
            mx, my = bw * margin, bh * margin
            box_arr = np.array([box.x1 - mx, box.y1 - my, box.x2 + mx, box.y2 + my])
            if use_center_point:
                point_coords = np.array([[(box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0]])
                point_labels = np.array([1])
            else:
                point_coords = None
                point_labels = None
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_arr,
                multimask_output=True,
            )
            best_idx = int(scores.argmax())
            mask = masks[best_idx]
            if use_center_point:
                # A full-frame mask is far more likely than a small crop to
                # contain an unrelated same-colored blob elsewhere in the
                # scene -- isolate to the component the point actually
                # anchors, same reasoning as segment()'s own point-prompt
                # path (see _isolate_component_at_point's docstring).
                px, py = point_coords[0]
                mask = self._isolate_component_at_point(mask, int(px), int(py))
            if not mask.any():
                return None
            return mask
        except Exception as e:
            log.warning("MobileSAM box-refine (cached-frame) inference failed (%s).", e)
            return None

    def segment_box(
        self, image_bgr: np.ndarray, box: Box, context_margin: float = 0.2,
        use_center_point: bool = False,
    ) -> tuple[np.ndarray | None, tuple[int, int] | None]:
        """Refine an approximate `box` (from a detector or tracker) to a
        tight mask, via SAM prompted with the box itself -- used to sharpen
        imprecise detection/tracking boxes (see aero_eyes.utils.box_refine),
        NOT for whole-photo reference segmentation (see segment() above,
        which assumes a close-up centered subject and uses a different set
        of heuristics tuned for that).

        Unlike segment(), this crops a small PADDED region around `box`
        first -- SAM's own image encoder then only runs on that small crop,
        not the full video frame, so refining many boxes stays cheap. The
        box is already a real (if imprecise) localization, so it's used
        directly as the prompt; `use_center_point` (box_refine.
        use_center_point_prompt, default off) additionally passes `box`'s
        own center as a positive point -- see segment_box_cached's own
        docstring for the full rationale/caveat (avoid enabling this for
        ring/donut-shaped object classes, same pitfall documented in
        __init__).

        Returns (mask, crop_offset) where mask is HxW bool over the
        CROPPED region and crop_offset=(x1,y1) locates that crop within
        image_bgr -- or (None, None) if unavailable, inference fails, or
        no plausible mask is found (caller should fall back to the
        original box unchanged).
        """
        if not self._available:
            return None, None

        h, w = image_bgr.shape[:2]
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        if bw <= 0 or bh <= 0:
            return None, None
        mx, my = bw * context_margin, bh * context_margin
        cx1 = max(0, int(box.x1 - mx))
        cy1 = max(0, int(box.y1 - my))
        cx2 = min(w, int(box.x2 + mx))
        cy2 = min(h, int(box.y2 + my))
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None
        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, None

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(crop_rgb)

            box_local = np.array([
                box.x1 - cx1, box.y1 - cy1, box.x2 - cx1, box.y2 - cy1,
            ])
            center_local = ((box_local[0] + box_local[2]) / 2.0, (box_local[1] + box_local[3]) / 2.0)
            if use_center_point:
                point_coords = np.array([center_local])
                point_labels = np.array([1])
            else:
                point_coords = None
                point_labels = None
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_local,
                multimask_output=True,
            )
            if use_center_point:
                cleaned = [
                    self._isolate_component_at_point(m, int(center_local[0]), int(center_local[1]))
                    for m in masks
                ]
            else:
                # No known-foreground point to anchor on (box-only prompt) --
                # keep each candidate's largest connected component, same as
                # segment()'s use_point_prompt=False path.
                cleaned = [self._isolate_largest_component(m) for m in masks]
            areas = [float(m.mean()) for m in cleaned]
            border_touch = [self._border_touch_frac(m) for m in cleaned]
            plausible = [
                i for i in range(len(cleaned))
                if self.min_area_frac <= areas[i] <= self.max_area_frac
                and border_touch[i] <= self.max_border_touch_frac
            ]
            if not plausible:
                return None, None
            # Trust SAM's own predicted-IoU score here (unlike segment()'s
            # "prefer largest among confident"): the box prompt already
            # pins down roughly where and how big the object is, so there's
            # far less risk of a high-scoring candidate being a bloated
            # background-leaked blob than in the whole-photo case.
            best_idx = max(plausible, key=lambda i: scores[i])
            return cleaned[best_idx], (cx1, cy1)
        except Exception as e:
            log.warning("MobileSAM box-refine inference failed (%s).", e)
            return None, None

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a binary mask (HxW bool) for the primary foreground object.

        Falls back to all-ones if MobileSAM is unavailable.
        """
        h, w = image_bgr.shape[:2]
        if not self._available:
            return np.ones((h, w), dtype=bool)

        try:
            import torch
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(image_rgb)

            # Central point prompt (good heuristic: reference photos are
            # typically subject-centred) COMBINED WITH a near-full-frame box
            # prompt. A lone point prompt often locks onto a small sub-part
            # of the object (a logo, a highlight, a shadow) instead of the
            # whole thing, especially with multimask_output picking whichever
            # candidate scores highest -- not necessarily the whole subject.
            # The box anchors SAM to "segment the dominant thing filling
            # roughly this region", which is far more reliable for close-up
            # reference photos where the subject fills most of the frame.
            #
            # use_point_prompt=False drops the point prompt entirely (box
            # only) -- the center-pixel assumption breaks down for
            # ring/donut-shaped objects (e.g. a life ring) whose center is a
            # HOLLOW interior, not object material; asserting "foreground
            # here" at a background pixel can bias SAM's mask proposals
            # toward confused/leaky boundaries (confirmed empirically: see
            # SegmentationConfig.use_point_prompt).
            margin = 0.05
            box = np.array([w * margin, h * margin, w * (1 - margin), h * (1 - margin)])
            if self.use_point_prompt:
                cx, cy = w // 2, h // 2
                point_coords, point_labels = np.array([[cx, cy]]), np.array([1])
            else:
                point_coords, point_labels = None, None
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
            # Strip any blob not connected to the point prompt FIRST, for
            # every candidate -- SAM sometimes tacks on a same-colored patch
            # of background as a disjoint extra region within an otherwise
            # correct mask. Left in, that blob inflates the candidate's area
            # and makes it win the "prefer largest" comparison below purely
            # for being big, even though it's the highest-scoring candidate
            # that is bloated, not the tight one (confirmed by inspecting
            # real SAM output: the highest-score, largest-area candidate had
            # a disconnected background chunk; a lower-scoring, smaller
            # candidate was the correct, tight one -- no score/area
            # threshold on the RAW masks can prefer the latter, only
            # cleaning first can). Without a point prompt there is no
            # "known-foreground pixel" to anchor the isolation on, so fall
            # back to keeping each candidate's largest connected component.
            if self.use_point_prompt:
                cleaned = [self._isolate_component_at_point(m, cx, cy) for m in masks]
            else:
                cleaned = [self._isolate_largest_component(m) for m in masks]

            # multimask_output=True returns 3 candidates at different
            # granularities (roughly: whole object / a part / a sub-part).
            # SAM's own predicted-IoU score does NOT reliably track "most
            # complete" -- a smaller, cleaner-edged sub-part regularly
            # outscores the full object, which was cutting off most of the
            # subject in practice. But blindly picking the LARGEST
            # area-plausible candidate overcorrects the other way: it can
            # pick a low-confidence candidate that bled into the background
            # just because it happens to be big ("cut too much excess").
            # So: only let "prefer largest" pick among candidates SAM itself
            # scored within score_ratio_floor of the best score; among
            # those, take the largest (by CLEANED area). Falls back to the
            # single highest-scoring candidate if nothing clears both bars.
            areas = [float(m.mean()) for m in cleaned]
            border_touch = [self._border_touch_frac(m) for m in cleaned]
            max_score = float(np.max(scores))
            confident = [
                i for i, s in enumerate(scores)
                if s >= max_score * self.score_ratio_floor
            ]
            plausible = [
                i for i in confident
                if self.min_area_frac <= areas[i] <= self.max_area_frac
                and border_touch[i] <= self.max_border_touch_frac
            ]
            if plausible:
                best_idx = max(plausible, key=lambda i: areas[i])
            else:
                best_idx = int(np.argmax(scores))
            mask = cleaned[best_idx]
            if not self.reject_implausible_mask:
                return mask
            area_frac = mask.mean()
            if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                log.warning(
                    "MobileSAM mask area implausible (%.1f%% of frame), using passthrough mask.",
                    area_frac * 100,
                )
                return np.ones((h, w), dtype=bool)
            if self._border_touch_frac(mask) > self.max_border_touch_frac:
                log.warning(
                    "MobileSAM mask touches the image border (%.1f%% of edge pixels), "
                    "likely leaked into background; using passthrough mask.",
                    self._border_touch_frac(mask) * 100,
                )
                return np.ones((h, w), dtype=bool)
            return mask
        except Exception as e:
            log.warning("MobileSAM inference failed (%s), using passthrough mask.", e)
            return np.ones((h, w), dtype=bool)


class FastSAMSegmenter:
    """box_refine.method == "fastsam_dense": reuses FastSAM-s (already
    loaded as stage2.proposal_model=fastsam_s's box-proposal engine --
    aero_eyes.models.proposals.FastSamSProposals -- but that wrapper
    discards the per-instance MASKS it already computes, keeping only
    boxes) as a box-conditioned refiner.

    Unlike SAM/SAM2's promptable decoder (segment_box_cached above,
    GeCo2Detector.sam2_refine_boxes), FastSAM has NO prompt-conditioned
    mask generation at all -- Ultralytics' own box/point prompting for it
    works exactly the way this class does: run "segment everything" ONCE
    per frame, then pick whichever ALREADY-PRODUCED instance mask best
    matches a given box afterward. set_frame()/segment_box_cached() mirror
    MobileSAMSegmenter's own two-call shape (one shared "encode", reused
    per box) so aero_eyes.utils.box_refine.refine_boxes_dense drives either
    segmenter identically -- FastSAM's "everything" pass just IS the encode
    step here, there's no separate prompt-conditioned decode after it.

    Ceiling this hits that the SAM/SAM2-based methods don't: it can only
    SELECT among masks the everything-pass already produced for this frame
    -- if the true object wasn't cleanly segmented as its own instance
    there (merged with a neighbor, or missed outright -- a known FastSAM
    weak point on small objects, and this project's own GT survey found
    real objects as thin as a 2px side), no amount of matching recovers
    it, unlike a promptable decoder that generates a NEW mask conditioned
    on exactly where it's prompted. Not yet benchmarked on this project's
    own dataset -- compare with scripts/check_box_refine_effect.py before
    trusting it, same as every other box_refine.method choice.
    """

    def __init__(
        self, weights: str, conf: float = 0.2, iou: float = 0.7, imgsz: int = 640,
        min_area_frac: float = 0.05, max_area_frac: float = 0.95,
        max_border_touch_frac: float = 0.02, use_point_prompt: bool = True,
        reject_implausible_mask: bool = True,
    ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.reject_implausible_mask = reject_implausible_mask
        # Only used by segment() (SegmentationConfig.model="fastsam") --
        # segment_box_cached() above (box_refine.method="fastsam_dense")
        # trusts its own box-IoU match directly instead, same as every
        # other box_refine method. Same names/semantics as
        # MobileSAMSegmenter's own plausibility gates -- see its __init__
        # for the full rationale.
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.max_border_touch_frac = max_border_touch_frac
        self.use_point_prompt = use_point_prompt
        self._model = None
        try:
            from ultralytics import FastSAM  # type: ignore
            self._model = FastSAM(weights)
            log.info("FastSAM-s loaded for box_refine.method=fastsam_dense: %s", weights)
        except Exception as e:
            log.warning(
                "FastSAM load failed (%s) -- box_refine.method=fastsam_dense unavailable "
                "this run, boxes left unchanged.", e,
            )
        self._cached_masks: list[np.ndarray] = []  # each HxW bool, full-frame-sized
        self._cached_boxes: list[Box] = []

    def set_frame(self, frame_bgr: np.ndarray) -> bool:
        """Run FastSAM's segment-everything pass ONCE for this frame,
        caching every instance's (mask, box) for segment_box_cached() to
        match prompts against. Returns False (segment_box_cached then
        always returns None) if FastSAM is unavailable, inference fails,
        or finds no instances at all."""
        self._cached_masks = []
        self._cached_boxes = []
        if self._model is None:
            return False
        try:
            results = self._model(
                frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                verbose=False, retina_masks=True,
            )
        except Exception as e:
            log.warning("FastSAM segment-everything pass failed (%s).", e)
            return False
        for r in results:
            if r.masks is None or r.boxes is None:
                continue
            masks_np = r.masks.data.cpu().numpy() > 0.5  # [K, H, W], full-frame-sized (retina_masks=True)
            boxes_np = r.boxes.xyxy.cpu().numpy()
            for m, b in zip(masks_np, boxes_np):
                self._cached_masks.append(m)
                self._cached_boxes.append(Box(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        return len(self._cached_masks) > 0

    def segment_box_cached(
        self, box: Box, margin: float = 0.0, use_center_point: bool = False,
    ) -> np.ndarray | None:
        """Match `box` (expanded by `margin`, same semantics as
        MobileSAMSegmenter.segment_box_cached's own prompt-box expansion)
        against the instances set_frame() cached for the CURRENT frame --
        must be called after set_frame(). Returns whichever cached mask
        matches best, or None if nothing was cached or nothing plausibly
        overlaps (caller then leaves the box unchanged, same fallback
        every box_refine method uses).

        `use_center_point` (box_refine.use_center_point_prompt): among
        cached instances whose mask actually CONTAINS the ORIGINAL
        (pre-margin) box's own center point, picks the one with the best
        box-IoU against the (margin-expanded) prompt box -- more robust
        than plain IoU alone when a larger neighboring instance's box
        happens to overlap the prompt box well without actually containing
        its center. Falls back to plain box-IoU matching (ignoring the
        point) if no cached instance's mask contains it.
        """
        if not self._cached_masks:
            return None
        from aero_eyes.utils.geometry import box_iou

        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        mx, my = bw * margin, bh * margin
        expanded = Box(box.x1 - mx, box.y1 - my, box.x2 + mx, box.y2 + my)

        candidates = range(len(self._cached_masks))
        if use_center_point:
            cx, cy = int((box.x1 + box.x2) / 2), int((box.y1 + box.y2) / 2)
            containing = [
                i for i in candidates
                if 0 <= cy < self._cached_masks[i].shape[0]
                and 0 <= cx < self._cached_masks[i].shape[1]
                and self._cached_masks[i][cy, cx]
            ]
            if containing:
                candidates = containing

        best_i = max(candidates, key=lambda i: box_iou(self._cached_boxes[i], expanded))
        if box_iou(self._cached_boxes[best_i], expanded) <= 0.0:
            return None
        return self._cached_masks[best_i]

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """SegmentationConfig.model == "fastsam" (reference-image
        foreground masking, Stage 1 / stage123_geco2.segmentation) --
        different job from segment_box_cached above (box_refine, matching
        against an already-detected box): here there is no box yet, so
        this matches the same synthetic near-full-frame box (+ optional
        center-point containment via use_center_point) MobileSAMSegmenter.
        segment()/SAM2Segmenter.segment() prompt with, against whatever
        instance set_frame()'s "segment everything" pass already produced
        -- see class docstring for FastSAM's own ceiling here (can only
        SELECT among those, never generate a new mask conditioned on the
        prompt). Falls back to an all-ones passthrough mask if FastSAM is
        unavailable, finds nothing, nothing matches, or the picked mask
        fails the same area/border-touch plausibility gates MobileSAM/SAM2
        apply.
        """
        h, w = image_bgr.shape[:2]
        if self._model is None:
            return np.ones((h, w), dtype=bool)
        try:
            if not self.set_frame(image_bgr):
                return np.ones((h, w), dtype=bool)
            margin = 0.05
            synthetic_box = Box(w * margin, h * margin, w * (1 - margin), h * (1 - margin))
            mask = self.segment_box_cached(synthetic_box, margin=0.0, use_center_point=self.use_point_prompt)
            if mask is None:
                return np.ones((h, w), dtype=bool)
            if not self.reject_implausible_mask:
                return mask
            area_frac = mask.mean()
            if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                log.warning(
                    "FastSAM mask area implausible (%.1f%% of frame), using passthrough mask.",
                    area_frac * 100,
                )
                return np.ones((h, w), dtype=bool)
            border_touch = MobileSAMSegmenter._border_touch_frac(mask)
            if border_touch > self.max_border_touch_frac:
                log.warning(
                    "FastSAM mask touches the image border (%.1f%% of edge pixels), likely "
                    "leaked into background; using passthrough mask.", border_touch * 100,
                )
                return np.ones((h, w), dtype=bool)
            return mask
        except Exception as e:
            log.warning("FastSAM segment() inference failed (%s), using passthrough mask.", e)
            return np.ones((h, w), dtype=bool)


class SAM2Segmenter:
    """box_refine.method == "sam2_native": a GENUINELY standalone SAM2 --
    its OWN image encoder (Hiera-base-plus) AND its OWN mask decoder, both
    loaded from the SAME public checkpoint and run independently of GeCo2
    -- unlike box_refine.method="sam2_dense" (GeCo2Detector.sam2_refine_boxes),
    which decodes masks from GeCo2's OWN detection-backbone features
    instead of running a fresh SAM2 encoder pass.

    Why this exists despite that being numerically almost the SAME
    computation: GECO2/train.sh and train_1gpu.sh both pass
    `--backbone_lr 0` (see GECO2/train.py's optimizer param groups), so
    GeCo2's backbone is FROZEN at its original sam2_hiera_base_plus.pt
    init -- meaning sam2_dense's "encoder" and this class's encoder are the
    SAME frozen weights either way. This class exists so that fact can be
    verified EMPIRICALLY (run both, compare) instead of assumed, and so it
    keeps working correctly as a clean, properly-independent alternative if
    a future GeCo2 checkpoint ever DOES fine-tune the backbone (at which
    point sam2_dense and this class would no longer be equivalent, and
    sam2_dense's masks would reflect a genuine encoder/decoder mismatch).

    Mirrors MobileSAMSegmenter/FastSAMSegmenter's set_frame()/
    segment_box_cached() shape so aero_eyes.utils.box_refine.
    refine_boxes_dense drives any of the three identically -- add
    "sam2_native" wherever "sam_dense"/"fastsam_dense" are already dispatch
    targets.

    Needs the vendored GECO2/sam2 package's OWN dependencies (hydra-core,
    omegaconf -- see GECO2/req.txt / GECO2/install.sh) in addition to
    whatever pipeline.detector=geco2 already requires; unavailable
    (falls back to leaving boxes unchanged, same as every other segmenter
    here) if those aren't installed.
    """

    _SAM2_CONFIG_NAME = "sam2_hiera_b+"
    _SAM2_CHECKPOINT_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    )

    def __init__(
        self, geco2_repo_path: str, device: str | None = None,
        min_area_frac: float = 0.05, max_area_frac: float = 0.95,
        score_ratio_floor: float = 0.85, max_border_touch_frac: float = 0.02,
        use_point_prompt: bool = True, reject_implausible_mask: bool = True,
    ):
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.reject_implausible_mask = reject_implausible_mask
        # Only used by segment() (SegmentationConfig.model="sam2") --
        # segment_box_cached() below (box_refine.method="sam2_native")
        # trusts its own predicted-IoU score directly instead, same as
        # every other box_refine method. Same names/semantics as
        # MobileSAMSegmenter's own plausibility gates -- see its __init__
        # for the full rationale.
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.score_ratio_floor = score_ratio_floor
        self.max_border_touch_frac = max_border_touch_frac
        self.use_point_prompt = use_point_prompt
        self._predictor = None
        self._available = False
        try:
            self._predictor = self._build_predictor(geco2_repo_path, self.device)
            self._available = True
            log.info("SAM2 (standalone, %s) loaded for box_refine.method=sam2_native on %s",
                      self._SAM2_CONFIG_NAME, self.device)
        except Exception as e:
            log.warning(
                "SAM2Segmenter unavailable (%s) -- box_refine.method=sam2_native disabled "
                "this run, boxes left unchanged. Needs the vendored GECO2/sam2 package's own "
                "deps (hydra-core, omegaconf) -- see GECO2/install.sh.", e,
            )

    @classmethod
    def _load_sam2_package(cls, geco2_repo_path: str):
        """Loads GECO2/sam2/sam2 (the vendored sam2 PACKAGE directory,
        note the doubled path -- GECO2/sam2 is the vendored repo checkout,
        GECO2/sam2/sam2 is the actual Python package inside it) directly by
        file path and registers it under BOTH 'sam2' and 'sam2.sam2' in
        sys.modules, WITHOUT modifying anything inside GECO2/.

        Why both names are needed -- confirmed empirically, not
        theoretical: this vendored copy's OWN internal self-imports are
        inconsistent between files. Some (e.g. modeling/backbones/
        hieradet.py) use `from sam2.modeling... import ...` (correct if
        GECO2/sam2 itself were the sys.path entry, i.e. `sam2` == this
        package directly -- the layout the ORIGINAL un-vendored repo has).
        Others (e.g. modeling/sam2_base.py, and GECO2/models/sam_mask.py's
        own already-working `from sam2.sam2.modeling... import ...`) use
        the doubled prefix (correct if GECO2 itself is the sys.path entry,
        i.e. `sam2` resolves to the OUTER empty folder and `sam2.sam2` is
        this package -- which is how GeCo2Detector's own sam2_dense path
        already imports it via _ensure_geco2_on_path). No single sys.path
        arrangement satisfies both conventions at once; aliasing 'sam2.sam2'
        to the SAME module object as 'sam2' does, regardless of which
        convention a given vendored file happens to use.

        Forcibly overwrites any PRE-EXISTING sys.modules['sam2'] (e.g. if
        GeCo2Detector's own sam2_dense path already imported the OLD,
        differently-resolved 'sam2' namespace-package in this same
        process) rather than relying on import-order/sys.path-priority
        alone, which would only work if this runs BEFORE any GeCo2Detector
        use in the same process -- confirmed by testing that classes
        already imported under the old registration (e.g.
        GECO2.models.sam_mask.MaskProcessor) keep working fine afterward,
        since only FUTURE imports consult the swapped sys.modules entry.
        """
        import importlib.util
        import sys
        from pathlib import Path

        sam2_repo = Path(geco2_repo_path).resolve() / "sam2"
        pkg_dir = sam2_repo / "sam2"
        already_correct = (
            "sam2" in sys.modules
            and getattr(sys.modules["sam2"], "__file__", None) == str(pkg_dir / "__init__.py")
        )
        if already_correct:
            return sys.modules["sam2"]

        spec = importlib.util.spec_from_file_location(
            "sam2", str(pkg_dir / "__init__.py"), submodule_search_locations=[str(pkg_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["sam2"] = module
        sys.modules["sam2.sam2"] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _build_predictor(cls, geco2_repo_path: str, device: str):
        """Builds a real sam2.sam2_image_predictor.SAM2ImagePredictor from
        the vendored GECO2/sam2 package, WITHOUT modifying anything inside
        GECO2/ -- same "reach in from outside" approach GeCo2Detector.
        sam2_refine_boxes already uses for the mask_decoder it reuses.

        Bypasses build_sam2()'s own hydra compose()/initialize_config_dir
        flow entirely -- that function's config's `_target_` strings
        (e.g. "sam2.modeling.sam2_base.SAM2Base") are the SINGLE-prefix
        convention, which only resolves correctly if GECO2/sam2 itself
        (not GECO2) were the sys.path entry -- the opposite of what
        _load_sam2_package needs for the doubled-prefix files (see its own
        docstring). Loading the yaml directly via OmegaConf and calling
        hydra.utils.instantiate() on it directly sidesteps compose()'s own
        config-search-path machinery, which is what needed that convention
        in the first place.

        Checkpoint: loaded the SAME way GECO2/models/sam_mask.py and
        GECO2/models/counter_infer.py already load their own copies of this
        exact checkpoint (torch.hub download -> strict=False state_dict
        load) rather than build_sam2()'s own ckpt_path loading (which
        raises on ANY missing/unexpected key -- too brittle here). Confirmed
        empirically: 0 missing / 0 unexpected against this exact config.
        """
        from pathlib import Path

        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from aero_eyes.models.geco2_detector import _ensure_geco2_on_path

        _ensure_geco2_on_path(geco2_repo_path)  # GECO2's own bare `models`/`utils` imports
        cls._load_sam2_package(geco2_repo_path)

        config_path = Path(geco2_repo_path).resolve() / "sam2" / "sam2_configs" / f"{cls._SAM2_CONFIG_NAME}.yaml"
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        checkpoint = torch.hub.load_state_dict_from_url(cls._SAM2_CHECKPOINT_URL, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            log.warning(
                "SAM2Segmenter: checkpoint load left %d missing / %d unexpected param(s) -- "
                "expected to be 0 for a matching hiera_base_plus checkpoint+config; results "
                "may be unreliable.", len(missing), len(unexpected),
            )
        model = model.to(device)
        model.eval()

        from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(model)

    def set_frame(self, frame_bgr: np.ndarray) -> bool:
        """Encode `frame_bgr` ONCE via SAM2's own image encoder -- same
        shape/contract as MobileSAMSegmenter.set_frame. Returns False if
        SAM2 is unavailable or encoding fails."""
        if not self._available:
            return False
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(frame_rgb)
            return True
        except Exception as e:
            log.warning("SAM2 (standalone) set_frame failed (%s).", e)
            return False

    def segment_box_cached(
        self, box: Box, margin: float = 0.0, use_center_point: bool = False,
    ) -> np.ndarray | None:
        """Same contract/semantics as MobileSAMSegmenter.segment_box_cached
        (margin-expanded box prompt, optional center-point prompt +
        connected-component isolation, best-of-multimask-output picked by
        SAM2's own predicted IoU) -- see that method's own docstring for
        the full rationale, identical here since both wrap a SAM-family
        box-prompted decoder through the same predict() shape."""
        if not self._available:
            return None
        try:
            bw, bh = box.x2 - box.x1, box.y2 - box.y1
            mx, my = bw * margin, bh * margin
            box_arr = np.array([box.x1 - mx, box.y1 - my, box.x2 + mx, box.y2 + my])
            if use_center_point:
                point_coords = np.array([[(box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0]])
                point_labels = np.array([1])
            else:
                point_coords = None
                point_labels = None
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels, box=box_arr,
                multimask_output=True,
            )
            best_idx = int(scores.argmax())
            mask = masks[best_idx].astype(bool)
            if use_center_point:
                from aero_eyes.utils.geometry import isolate_component_at_point
                px, py = point_coords[0]
                mask = isolate_component_at_point(mask, int(px), int(py))
            if not mask.any():
                return None
            return mask
        except Exception as e:
            log.warning("SAM2 (standalone) box-refine inference failed (%s).", e)
            return None

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """SegmentationConfig.model == "sam2" (reference-image foreground
        masking, Stage 1 / stage123_geco2.segmentation) -- same near-
        full-frame box (+ optional center-point) prompt and candidate-
        selection algorithm as MobileSAMSegmenter.segment (see
        _pick_best_full_frame_mask), just decoded by SAM2's own promptable
        decoder instead of MobileSAM's. Falls back to an all-ones
        passthrough mask if SAM2 is unavailable or inference fails
        (SAM2Segmenter has no fallback_if_missing="error" mode -- see class
        docstring).
        """
        h, w = image_bgr.shape[:2]
        if not self._available:
            return np.ones((h, w), dtype=bool)
        try:
            if not self.set_frame(image_bgr):
                return np.ones((h, w), dtype=bool)
            margin = 0.05
            box = np.array([w * margin, h * margin, w * (1 - margin), h * (1 - margin)])
            if self.use_point_prompt:
                cx, cy = w // 2, h // 2
                point_coords, point_labels = np.array([[cx, cy]]), np.array([1])
            else:
                cx = cy = 0
                point_coords, point_labels = None, None
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=True,
            )
            mask = _pick_best_full_frame_mask(
                masks, scores, self.use_point_prompt, cx, cy,
                self.min_area_frac, self.max_area_frac, self.score_ratio_floor, self.max_border_touch_frac,
            )
            if not self.reject_implausible_mask:
                return mask
            area_frac = mask.mean()
            if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                log.warning(
                    "SAM2 (standalone) mask area implausible (%.1f%% of frame), using passthrough mask.",
                    area_frac * 100,
                )
                return np.ones((h, w), dtype=bool)
            border_touch = MobileSAMSegmenter._border_touch_frac(mask)
            if border_touch > self.max_border_touch_frac:
                log.warning(
                    "SAM2 (standalone) mask touches the image border (%.1f%% of edge pixels), "
                    "likely leaked into background; using passthrough mask.", border_touch * 100,
                )
                return np.ones((h, w), dtype=bool)
            return mask
        except Exception as e:
            log.warning("SAM2 (standalone) segment() inference failed (%s), using passthrough mask.", e)
            return np.ones((h, w), dtype=bool)


def build_segmenter(seg_cfg, cfg):
    """Factory: builds the configured reference-image segmentation backend
    (seg_cfg.model: mobilesam | fastsam | sam2 -- see SegmentationConfig's
    own docstring for the tradeoffs) for stage1.segmentation /
    stage123_geco2.segmentation. Takes the FULL config (not just seg_cfg)
    because fastsam/sam2 need fields that live outside SegmentationConfig
    itself: fastsam reuses stage2.fastsam_s's weights/conf/iou/imgsz (the
    SAME FastSAM-s checkpoint stage2's own proposal model uses, if
    configured -- avoids requiring a second, separately-configured FastSAM
    checkpoint just for this), sam2 needs stage123_geco2.repo_path (the
    vendored GECO2/sam2 package)."""
    if seg_cfg.model == "fastsam":
        fs_cfg = cfg.stage2.fastsam_s
        return FastSAMSegmenter(
            weights=fs_cfg.weights, conf=fs_cfg.conf, iou=fs_cfg.iou, imgsz=fs_cfg.imgsz,
            min_area_frac=seg_cfg.min_area_frac, max_area_frac=seg_cfg.max_area_frac,
            max_border_touch_frac=seg_cfg.max_border_touch_frac, use_point_prompt=seg_cfg.use_point_prompt,
            reject_implausible_mask=seg_cfg.reject_implausible_mask,
        )
    if seg_cfg.model == "sam2":
        return SAM2Segmenter(
            cfg.stage123_geco2.repo_path,
            min_area_frac=seg_cfg.min_area_frac, max_area_frac=seg_cfg.max_area_frac,
            score_ratio_floor=seg_cfg.score_ratio_floor, max_border_touch_frac=seg_cfg.max_border_touch_frac,
            use_point_prompt=seg_cfg.use_point_prompt, reject_implausible_mask=seg_cfg.reject_implausible_mask,
        )
    return MobileSAMSegmenter(
        weights_path=seg_cfg.weights, fallback_if_missing=seg_cfg.fallback_if_missing,
        min_area_frac=seg_cfg.min_area_frac, max_area_frac=seg_cfg.max_area_frac,
        score_ratio_floor=seg_cfg.score_ratio_floor, max_border_touch_frac=seg_cfg.max_border_touch_frac,
        use_point_prompt=seg_cfg.use_point_prompt, reject_implausible_mask=seg_cfg.reject_implausible_mask,
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

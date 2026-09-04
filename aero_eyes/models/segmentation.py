"""MobileSAM wrapper for reference foreground masking (Stage 1).

If weights are unavailable and fallback_if_missing == "passthrough",
returns an all-ones mask without crashing.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)


class MobileSAMSegmenter:
    """Segment the largest/most-central object in a reference image."""

    def __init__(self, weights_path: str | None = None, fallback_if_missing: str = "passthrough",
                 min_area_frac: float = 0.05, max_area_frac: float = 0.95,
                 score_ratio_floor: float = 0.85, max_border_touch_frac: float = 0.02,
                 use_point_prompt: bool = True):
        self.weights_path = weights_path
        self.fallback_if_missing = fallback_if_missing
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
        """Keep only the connected component touching (px, py) (the point
        prompt) -- discards any disconnected blob SAM tacked on elsewhere in
        the frame (e.g. a same-colored patch of background), even if that
        blob is large enough to pass the area/score gates below. Falls back
        to the largest component if the point itself isn't foreground in
        this particular mask.
        """
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
        if num_labels <= 2:  # 0=background + at most 1 foreground component
            return mask
        py = min(max(py, 0), mask.shape[0] - 1)
        px = min(max(px, 0), mask.shape[1] - 1)
        label_at_point = labels[py, px]
        if label_at_point == 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            label_at_point = int(np.argmax(counts))
        return labels == label_at_point

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

    def segment_box(
        self, image_bgr: np.ndarray, box: Box, context_margin: float = 0.2,
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
        directly as the prompt -- no oracle guess needed, and no point
        prompt (a box alone is a reliable enough anchor here, and avoids
        the same center-pixel pitfall documented in __init__ for
        ring/donut-shaped objects).

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
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_local,
                multimask_output=True,
            )
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

"""MobileSAM wrapper for reference foreground masking (Stage 1).

If weights are unavailable and fallback_if_missing == "passthrough",
returns an all-ones mask without crashing.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class MobileSAMSegmenter:
    """Segment the largest/most-central object in a reference image."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        fallback_if_missing: str = "passthrough",
        min_area_frac: float = 0.05,
        max_area_frac: float = 0.95,
        score_ratio_floor: float = 0.85,
        max_border_touch_frac: float = 0.02,
        **kwargs,
    ):
        self.weights_path = weights_path
        self.fallback_if_missing = fallback_if_missing
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.score_ratio_floor = score_ratio_floor
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
                import os
                ckpt = os.path.join(
                    os.path.expanduser("~"), ".cache", "mobile_sam", "mobile_sam.pt"
                )
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

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a binary mask (HxW bool) for the primary foreground object.

        Falls back to all-ones if MobileSAM is unavailable.
        """
        h, w = image_bgr.shape[:2]
        fallback_mask = np.ones((h, w), dtype=bool)

        if not self._available or self._predictor is None:
            return fallback_mask

        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(image_rgb)

            # Inset box prompt 5% from edges + central point prompt
            box_prompt = np.array([0.05 * w, 0.05 * h, 0.95 * w, 0.95 * h])
            cx, cy = w // 2, h // 2
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            try:
                masks, scores, _ = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_prompt[None, :],
                    multimask_output=True,
                )
            except Exception:
                # Fallback to point prompt only if box prompt is unsupported
                masks, scores, _ = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

            if masks is None or len(masks) == 0:
                return fallback_mask

            best_score = float(np.max(scores))
            border_total = float(2 * (h + w))
            valid_candidates = []

            for m, s in zip(masks, scores):
                m_bool = m.astype(bool)
                area_frac = float(m_bool.mean())

                # Area plausibility filter
                if not (self.min_area_frac <= area_frac <= self.max_area_frac):
                    continue

                # Score floor filter relative to best scoring candidate
                if s < (best_score * self.score_ratio_floor):
                    continue

                # Border touch filter
                border_touch_px = float(
                    np.sum(m_bool[0, :])
                    + np.sum(m_bool[-1, :])
                    + np.sum(m_bool[:, 0])
                    + np.sum(m_bool[:, -1])
                )
                if (border_touch_px / border_total) > self.max_border_touch_frac:
                    continue

                valid_candidates.append((m_bool, area_frac))

            if not valid_candidates:
                # Fallback: Pick highest-scoring mask that satisfies area bounds
                best_idx = int(np.argmax(scores))
                mask = masks[best_idx].astype(bool)
                area_frac = float(mask.mean())
                if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                    log.warning(
                        "MobileSAM mask area implausible (%.1f%% of frame), using passthrough mask.",
                        area_frac * 100,
                    )
                    return fallback_mask
                return mask

            # Prefer the largest mask among valid candidates
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            return valid_candidates[0][0]

        except Exception as e:
            log.warning("MobileSAM inference failed (%s), using passthrough mask.", e)
            return fallback_mask
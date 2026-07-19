"""MobileSAM wrapper for reference foreground masking (Stage 1).

If weights are unavailable and fallback_if_missing == "passthrough",
returns an all-ones mask without crashing.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class MobileSAMSegmenter:
    """Segment the largest/most-central object in a reference image."""

    def __init__(self, weights_path: str | None = None, fallback_if_missing: str = "passthrough"):
        self.weights_path = weights_path
        self.fallback_if_missing = fallback_if_missing
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
        if not self._available:
            return np.ones((h, w), dtype=bool)

        try:
            import torch
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(image_rgb)

            # Use a central point prompt — good heuristic for reference images
            # where the subject is typically centred
            cx, cy = w // 2, h // 2
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            # Pick the highest-scoring mask
            best_idx = int(np.argmax(scores))
            return masks[best_idx].astype(bool)
        except Exception as e:
            log.warning("MobileSAM inference failed (%s), using passthrough mask.", e)
            return np.ones((h, w), dtype=bool)

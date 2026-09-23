"""DAVE's (arXiv:2404.16622) OWN verify-stage embedding: the ResNet50+SWaV
backbone + its learned `feat_comp` projection (weights from
verification.pth), wrapped to satisfy this project's feature-extractor
interface (.extract() / .extract_crops(), both returning L2-normalized
float32 arrays -- see aero_eyes/models/features.py's module docstring)
so it is a drop-in for every existing consumer of that interface.

Two independent config switches consume DaveVerificationExtractor -- see
DaveVerificationConfig's own docstring (aero_eyes/config.py) for the full
rationale of each:
  - stage1.feature_extractor.model="dave_verification": use it as the
    extractor for the WHOLE pipeline (prototype + matching/threshold +,
    by extension, the cluster-verify affinity matrix too).
  - stage3.cluster_verification.embedding_source="dave_verification" (also
    consumed by stage123_geco2.dynamic_prototype.cluster_verification, same
    ClusterVerificationConfig primitive): use it ONLY for the cluster-verify
    affinity matrix, side by side with whatever stage1.feature_extractor is
    doing elsewhere.

Does NOT need the full DAVE repo checked out (no git submodule, no
sys.path surgery) -- aero_eyes.models._dave_vendor copies (not clones) just
the 2 small classes actually needed (Backbone, Feature_Transform), verbatim,
under DAVE's own MIT license (see that module's own header for the full
notice). Everything else in the original DAVE repo (training scripts,
FSC147 data loaders, its own detection head, eval tooling) is irrelevant
here and deliberately not vendored.

NOT YET VALIDATED against this project's own footage -- feat_comp was
trained on FSC147 (natural-image counting), not this project's own domain.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

from aero_eyes.models._dave_vendor import Backbone, Feature_Transform
from aero_eyes.types import Box

log = logging.getLogger(__name__)

# Standard ImageNet normalization -- matches DAVE's OWN preprocessing
# exactly (DAVE/utils/data.py: T.Normalize(mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225]) at every one of its own dataset classes), since
# the SWaV-pretrained ResNet50 backbone was trained under this convention.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class DaveVerificationExtractor:
    """DAVE's own verify-stage embedding.

    Unlike every extractor in aero_eyes/models/features.py, extract_crops()
    does NOT crop-then-resize each box independently -- it matches DAVE's
    OWN reference implementation (models/dave.py::COTR.forward): a SINGLE
    backbone forward pass over the whole frame, then RoI-Align pulls each
    box's region directly out of that one shared feature map (cheaper, and
    what verification.pth was actually trained against). extract() (used
    for whole-image reference photos) is the degenerate case of the same
    mechanism: the "box" is the entire image, matching this project's own
    convention (see configs/config.yaml's stage123_geco2 section) that a
    close-up reference photo, not living inside any larger frame, is
    treated as its own exemplar box = the whole image.

    One documented deviation from DAVE's own preprocessing: images are
    resized directly to image_size x image_size (matching how every other
    extractor here preprocesses, e.g. DINOv2FeatureExtractor), not padded
    to preserve aspect ratio via DAVE's own utils/helpers.py::pad_image.
    Simpler, but not bit-exact with DAVE's own pipeline -- acceptable given
    this whole mechanism is NOT YET VALIDATED either way.
    """

    def __init__(
        self, weights_path: str, device: str = "auto",
        image_size: int = 1024, reduction: int = 8, kernel_dim: int = 3,
    ):
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"verification.pth not found at '{weights_path}'. Download it "
                "from the Google Drive link in DAVE's README "
                "(https://github.com/jerpelhan/DAVE, 'Download the models' step) "
                "and set dave_verification.verification_weights_path."
            )

        self.device = _resolve_device(device)
        self.image_size = image_size
        self.reduction = reduction
        self.kernel_dim = kernel_dim

        self.backbone = Backbone(
            "resnet50", pretrained=True, dilation=False,
            reduction=reduction, swav=True, requires_grad=False,
        ).to(self.device).eval()

        self.feat_comp = Feature_Transform().to(self.device).eval()
        checkpoint = torch.load(weights_path, map_location=self.device)
        state = checkpoint.get("model", checkpoint)
        # Same extraction DAVE/main.py itself does: verification.pth is a
        # full model checkpoint, only its feat_comp.* weights are used --
        # this project never uses DAVE's own detection head.
        feat_comp_state = {
            k.split("feat_comp.", 1)[1]: v for k, v in state.items() if "feat_comp." in k
        }
        if not feat_comp_state:
            raise ValueError(
                f"'{weights_path}' has no 'feat_comp.*' weights -- is this really "
                "DAVE's verification.pth?"
            )
        self.feat_comp.load_state_dict(feat_comp_state, strict=True)

        self._output_dim = self._probe_dim()
        log.info(
            "DaveVerificationExtractor: loaded verification.pth from %s "
            "(backbone=resnet50+swav, reduction=%d, kernel_dim=%d, dim=%d) on %s",
            weights_path, reduction, kernel_dim, self._output_dim, self.device,
        )

    @torch.no_grad()
    def _probe_dim(self) -> int:
        # feat_comp's output dim depends on its own conv arithmetic (not
        # hardcoded like other extractors' _DIMS dicts) -- determine it
        # once via a dry run instead of hand-deriving it.
        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        feat_map = self.backbone(dummy)
        roi = torch.tensor(
            [[0.0, 0.0, 0.0, float(self.image_size), float(self.image_size)]],
            dtype=torch.float32, device=self.device,
        )
        aligned = roi_align(
            feat_map, roi, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction, aligned=True,
        )
        return int(self.feat_comp(aligned).shape[-1])

    def _dim(self) -> int:
        return self._output_dim

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(img_pil, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
        return tensor.to(self.device).float()

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._output_dim), dtype=np.float32)
        out_chunks: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensor = torch.cat([self._preprocess(im) for im in batch_imgs], dim=0)
            feat_map = self.backbone(tensor)
            n = feat_map.shape[0]
            rois = torch.tensor(
                [[float(j), 0.0, 0.0, float(self.image_size), float(self.image_size)] for j in range(n)],
                dtype=torch.float32, device=self.device,
            )
            aligned = roi_align(
                feat_map, rois, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True,
            )
            emb = self.feat_comp(aligned)
            out_chunks.append(F.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    @torch.no_grad()
    def extract_crops(
        self, frame_bgr: np.ndarray, boxes: list[Box],
        pad_ratio: float = 0.10, batch_size: int = 16,
    ) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._output_dim), dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        feat_map = self.backbone(self._preprocess(frame_bgr))
        scale_x = self.image_size / w
        scale_y = self.image_size / h
        rois = []
        for b in boxes:
            bw, bh = b.x2 - b.x1, b.y2 - b.y1
            pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
            x1 = max(0.0, b.x1 - pad_x) * scale_x
            y1 = max(0.0, b.y1 - pad_y) * scale_y
            x2 = min(float(w), b.x2 + pad_x) * scale_x
            y2 = min(float(h), b.y2 + pad_y) * scale_y
            rois.append([0.0, x1, y1, x2, y2])  # batch idx 0 -- single shared feat_map
        roi_tensor = torch.tensor(rois, dtype=torch.float32, device=self.device)

        out_chunks: list[np.ndarray] = []
        for i in range(0, roi_tensor.shape[0], batch_size):
            chunk = roi_tensor[i:i + batch_size]
            aligned = roi_align(
                feat_map, chunk, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True,
            )
            emb = self.feat_comp(aligned)
            out_chunks.append(F.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32)


def build_dave_verification_extractor(cfg) -> DaveVerificationExtractor:
    """Builds a DaveVerificationExtractor from cfg.dave_verification, using
    cfg.device() for the device -- same convention every build_*
    factory in this project follows (see
    aero_eyes.models.features.build_feature_extractor).
    """
    dv = cfg.dave_verification
    return DaveVerificationExtractor(
        weights_path=dv.verification_weights_path,
        device=cfg.device(),
        image_size=dv.image_size,
        reduction=dv.reduction,
        kernel_dim=dv.kernel_dim,
    )

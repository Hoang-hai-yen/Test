"""Patch-token matching for Stage 3 (stage3.patch_matching, opt-in).

Scores a candidate crop against each reference image by comparing SETS of
DINOv3 patch tokens (Chamfer/MaxSim or Sinkhorn optimal transport) instead of
the single global CLS vector -- see PatchMatchingConfig (aero_eyes/config.py)
for the rationale and the meaning of every knob.

All scores are cosine-like (higher = more similar, roughly in [-1, 1]).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from aero_eyes.models.features import _DINO_MEAN, _DINO_STD, DINOv3FeatureExtractor
from aero_eyes.utils.geometry import crop_with_pad

log = logging.getLogger(__name__)


def chamfer_score(ref: torch.Tensor, cand: torch.Tensor, symmetric: bool = True) -> float:
    """MaxSim between two L2-normalised patch sets [Nr,D], [Nc,D]: each ref
    patch takes its best cosine in cand (mean over ref patches); symmetric
    also averages in the cand->ref direction."""
    sim = ref @ cand.T
    r2c = sim.max(dim=1).values.mean()
    if not symmetric:
        return float(r2c)
    c2r = sim.max(dim=0).values.mean()
    return float(0.5 * (r2c + c2r))


def sinkhorn_score(ref: torch.Tensor, cand: torch.Tensor, epsilon: float = 0.05, iters: int = 50) -> float:
    """Entropic OT between two L2-normalised patch sets with uniform weights
    and cost = 1 - cosine (log-domain Sinkhorn, stable for small epsilon).
    Returns the mean cosine under the transport plan."""
    sim = ref @ cand.T
    cost = 1.0 - sim
    n, m = cost.shape
    log_a = torch.full((n,), -np.log(n), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((m,), -np.log(m), device=cost.device, dtype=cost.dtype)
    f = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(m, device=cost.device, dtype=cost.dtype)
    for _ in range(iters):
        f = epsilon * (log_a - torch.logsumexp((g[None, :] - cost) / epsilon, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - cost) / epsilon, dim=0))
    plan = torch.exp((f[:, None] + g[None, :] - cost) / epsilon)
    return float((plan * sim).sum())


def _unwrap_dinov3(extractor) -> DINOv3FeatureExtractor:
    """Find the DINOv3FeatureExtractor inside the wrappers
    build_feature_extractor may add (background masking, projection head) or
    the ensemble extractor."""
    seen = extractor
    for _ in range(4):
        if isinstance(seen, DINOv3FeatureExtractor):
            return seen
        seen = getattr(seen, "base", None) or getattr(seen, "dino", None)
        if seen is None:
            break
    raise ValueError(
        "stage3.patch_matching needs stage1.feature_extractor.model='dinov3' (or an ensemble "
        "with ensemble_dino_model='dinov3') -- no DINOv3 backbone found."
    )


class PatchMatcher:
    """Encodes images to patch-token sets with a DINOv3 HuggingFace backbone
    and scores candidate crops against pre-encoded reference images."""

    def __init__(self, extractor: DINOv3FeatureExtractor, pm_cfg):
        if extractor.source != "huggingface":
            raise ValueError(
                "stage3.patch_matching needs dinov3_source='huggingface' -- the raw kaggle "
                "checkpoint path has no output_hidden_states API."
            )
        self.extractor = extractor
        self.model = extractor.model
        self.device = extractor.device
        self.patch = extractor._PATCH_SIZE
        self.cfg = pm_cfg
        self.n_layers = self.model.config.num_hidden_layers
        self._refs: list[torch.Tensor] = []

    @classmethod
    def from_cfg(cls, cfg) -> "PatchMatcher":
        from aero_eyes.models.features import build_feature_extractor

        return cls(_unwrap_dinov3(build_feature_extractor(cfg)), cfg.stage3.patch_matching)

    # -- encoding -----------------------------------------------------------

    def _to_tensor(self, img_bgr: np.ndarray, is_candidate: bool) -> torch.Tensor:
        if self.cfg.reuse_cls_preprocess:
            mode = (
                self.extractor.candidate_preprocess_mode if is_candidate
                else self.extractor.preprocess_mode
            )
            return self.extractor.pixel_values([img_bgr], mode)
        h, w = img_bgr.shape[:2]
        long_side = self.cfg.long_side
        if self.cfg.keep_aspect:
            scale = long_side / max(h, w)
            new_w = max(self.patch, round(w * scale / self.patch) * self.patch)
            new_h = max(self.patch, round(h * scale / self.patch) * self.patch)
        else:
            new_w = new_h = max(self.patch, round(long_side / self.patch) * self.patch)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        arr = (rgb.astype(np.float32) / 255.0 - np.array(_DINO_MEAN, np.float32)) / np.array(_DINO_STD, np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())[None]

    @torch.no_grad()
    def encode(self, img_bgr: np.ndarray, is_candidate: bool = True) -> torch.Tensor:
        """Patch tokens [N, D*len(layers)] (per-layer L2-normalised, concatenated,
        re-normalised -- so a dot product is the mean of per-layer cosines).
        is_candidate only matters under reuse_cls_preprocess (picks
        candidate_preprocess_mode vs preprocess_mode)."""
        pv = self._to_tensor(img_bgr, is_candidate).to(self.device)
        out = self.model(pixel_values=pv, output_hidden_states=True)
        n_patches = (pv.shape[-2] // self.patch) * (pv.shape[-1] // self.patch)
        per_layer = []
        for idx in self.cfg.layers:
            if idx == -1 or idx == self.n_layers:
                h = out.last_hidden_state  # after the model's final norm
            else:
                h = out.hidden_states[idx]
            tokens = h[0, h.shape[1] - n_patches:, :]  # drop CLS (+ register tokens)
            per_layer.append(F.normalize(tokens, dim=-1))
        return F.normalize(torch.cat(per_layer, dim=-1), dim=-1)

    def set_references(self, ref_imgs_bgr: list[np.ndarray]) -> None:
        self._refs = [self.encode(im, is_candidate=False) for im in ref_imgs_bgr]

    # -- scoring ------------------------------------------------------------

    def score_pair(self, ref_tokens: torch.Tensor, cand_tokens: torch.Tensor) -> float:
        if self.cfg.method == "ot":
            return sinkhorn_score(ref_tokens, cand_tokens, self.cfg.ot_epsilon, self.cfg.ot_iters)
        return chamfer_score(ref_tokens, cand_tokens, self.cfg.symmetric)

    def score_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """[N_crops, N_refs] score matrix against the references set by
        set_references()."""
        out = np.zeros((len(crops_bgr), len(self._refs)), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            cand = self.encode(crop)
            for r, ref in enumerate(self._refs):
                out[i, r] = self.score_pair(ref, cand)
        return out


def score_candidates(cfg, sample_id: str, video_path: Path, frame_idxs: list[int], dets: list) -> np.ndarray:
    """Patch score matrix [N, num_refs] for Stage 3's flat candidate list
    (frame_idxs[i]/dets[i] describe candidate i). Crops use the same padding
    as Stage 3's own feature crops (stage2.candidate.feature_crop_pad)."""
    from collections import defaultdict

    from aero_eyes.utils.video import read_frame

    matcher = PatchMatcher.from_cfg(cfg)
    refs_dir = Path(cfg.data.data_root) / sample_id / cfg.data.refs_subdir
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else []) if p.suffix.lower() in exts
    )[: cfg.data.num_references]
    if not ref_paths:
        raise FileNotFoundError(f"stage3.patch_matching: no reference images found in {refs_dir}")
    matcher.set_references([cv2.imread(str(p)) for p in ref_paths])

    by_frame: dict[int, list[int]] = defaultdict(list)
    for i, fi in enumerate(frame_idxs):
        by_frame[fi].append(i)

    scores = np.zeros((len(dets), len(ref_paths)), dtype=np.float32)
    for fi in sorted(by_frame):
        idxs = by_frame[fi]
        frame = read_frame(video_path, fi)
        crops = [crop_with_pad(frame, dets[i].box, cfg.stage2.candidate.feature_crop_pad) for i in idxs]
        scores[idxs] = matcher.score_crops(crops)
    log.info(
        "[Stage3] %s: patch_matching (%s, layers=%s, long_side=%d) scored %d candidate(s) vs %d ref(s)",
        sample_id, cfg.stage3.patch_matching.method, cfg.stage3.patch_matching.layers,
        cfg.stage3.patch_matching.long_side, len(dets), len(ref_paths),
    )
    return scores

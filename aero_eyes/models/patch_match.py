"""Patch-token matching for Stage 3 (stage3.patch_matching, opt-in).

Scores a candidate crop against each reference image by comparing SETS of
DINOv3 patch tokens (Chamfer/MaxSim or Sinkhorn optimal transport) instead of
the single global CLS vector -- see PatchMatchingConfig (aero_eyes/config.py)
for the rationale and the meaning of every knob.

All scores are cosine-like (higher = more similar, roughly in [-1, 1]).
"""
from __future__ import annotations

import logging
import math
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


def patch_pair_scores(
    cand: torch.Tensor, ref: torch.Tensor, method: str = "chamfer", symmetric: bool = True,
    epsilon: float = 0.05, iters: int = 50, chunk: int = 8,
    cand_mask: torch.Tensor | None = None, ref_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched, differentiable form of chamfer_score / sinkhorn_score:
    cand [C,N,D] x ref [R,M,D] (L2-normalised patch tokens) -> [C,R] scores.
    Gradients flow to the tokens. For "ot" the transport plan is computed
    without gradient and only the similarity it weights is differentiated
    (the plan is the OT cost's gradient, so this is the exact first-order
    gradient and keeps memory flat). chunk = crops processed at a time.

    cand_mask [C,N] / ref_mask [R,M] (bool, True = compare this patch; None =
    all): masked-out patches (e.g. pad_to_square padding) neither look for a
    match nor serve as one -- Chamfer averages and maximises over the valid
    patches only, and OT puts its uniform mass on the valid patches only."""
    if method not in ("chamfer", "ot"):
        raise ValueError(f"Unknown patch method {method!r}. Must be 'chamfer' or 'ot'.")
    C, N = cand.shape[:2]
    R, M = ref.shape[:2]
    cmask = torch.ones(C, N, dtype=torch.bool, device=cand.device) if cand_mask is None else cand_mask.to(cand.device)
    rmask = torch.ones(R, M, dtype=torch.bool, device=cand.device) if ref_mask is None else ref_mask.to(cand.device)
    # an image with NO valid patch has no score (max over an empty set = -inf): compare all its patches instead
    cmask = torch.where(cmask.any(-1, keepdim=True), cmask, torch.ones_like(cmask))
    rmask = torch.where(rmask.any(-1, keepdim=True), rmask, torch.ones_like(rmask))
    n_ref = rmask.sum(-1).clamp(min=1).to(cand.dtype)            # [R]
    neg_inf = float("-inf")
    big = 1e4                                                    # log-weight of a masked-out patch in OT
    outs = []
    for i in range(0, C, chunk):
        cm = cmask[i:i + chunk]                                  # [c,N]
        n_cand = cm.sum(-1).clamp(min=1).to(cand.dtype)          # [c]
        sim = torch.einsum("cnd,rmd->crnm", cand[i:i + chunk], ref)      # [c,R,N,M]
        if method == "chamfer":
            # ref patch -> best VALID cand patch, averaged over VALID ref patches
            r2c = sim.masked_fill(~cm[:, None, :, None], neg_inf).max(dim=2).values      # [c,R,M]
            r2c = torch.where(rmask[None], r2c, torch.zeros_like(r2c)).sum(-1) / n_ref[None]     # [c,R]
            score = r2c
            if symmetric:
                c2r = sim.masked_fill(~rmask[None, :, None, :], neg_inf).max(dim=3).values  # [c,R,N]
                c2r = torch.where(cm[:, None, :], c2r, torch.zeros_like(c2r)).sum(-1) / n_cand[:, None]  # [c,R]
                score = 0.5 * (r2c + c2r)
        else:
            with torch.no_grad():
                cost = 1.0 - sim.detach()
                c_, R_, N_, M_ = cost.shape
                log_a = torch.where(cm, (-torch.log(n_cand))[:, None], torch.full_like(cm, -big, dtype=cost.dtype))
                log_a = log_a[:, None, :].expand(c_, R_, N_)                                # [c,R,N]
                log_b = torch.where(rmask, (-torch.log(n_ref))[:, None], torch.full_like(rmask, -big, dtype=cost.dtype))
                log_b = log_b[None].expand(c_, R_, M_)                                      # [c,R,M]
                f = torch.zeros(c_, R_, N_, device=cost.device, dtype=cost.dtype)
                g = torch.zeros(c_, R_, M_, device=cost.device, dtype=cost.dtype)
                for _ in range(iters):
                    f = epsilon * (log_a - torch.logsumexp((g[:, :, None, :] - cost) / epsilon, dim=3))
                    g = epsilon * (log_b - torch.logsumexp((f[:, :, :, None] - cost) / epsilon, dim=2))
                plan = torch.exp((f[..., None] + g[:, :, None, :] - cost) / epsilon)
            score = (plan * sim).sum(dim=(2, 3))
        outs.append(score)
    return torch.cat(outs, dim=0)


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
        # MaskedCropFeatureExtractor to background-mask candidate crops with
        # (set by from_cfg when candidate_background_masking is enabled).
        self.candidate_masker = None
        self._warn_lora_mismatch(extractor)

    def _warn_lora_mismatch(self, extractor) -> None:
        """A LoRA trained with scripts/train_lora_dinov3.py --score patch|both
        was fit to a specific patch scoring setup; warn when this run's
        stage3.patch_matching differs from it."""
        tc = getattr(extractor, "lora_train_config", {}) or {}
        if tc.get("score") not in ("patch", "both"):
            return
        c = self.cfg
        running = {
            "patch_method": c.method, "patch_layers": list(c.layers),
            "patch_symmetric": c.symmetric, "reuse_cls_preprocess": c.reuse_cls_preprocess,
            "patch_mask_padding": c.mask_padding, "patch_pad_min_frac": c.pad_min_content_frac,
        }
        if tc["score"] == "both":
            running["cls_weight"] = c.cls_weight
        if c.method == "ot":
            running["patch_ot_epsilon"], running["patch_ot_iters"] = c.ot_epsilon, c.ot_iters
        # the LoRA saw ext.preprocess_mode-style inputs, never long_side/keep_aspect ones
        want = {**tc, "reuse_cls_preprocess": True}
        for k, v in running.items():
            if k in want and want[k] != v:
                log.warning(
                    "LoRA was trained with score=%s but stage3.patch_matching differs -- %s: trained with %r, "
                    "running with %r. The adapters were fit to that patch setup, so results may degrade.",
                    tc["score"], k, want[k], v,
                )

    @classmethod
    def from_cfg(cls, cfg) -> "PatchMatcher":
        from aero_eyes.models.features import MaskedCropFeatureExtractor, build_feature_extractor

        extractor = build_feature_extractor(cfg)
        matcher = cls(_unwrap_dinov3(extractor), cfg.stage3.patch_matching)
        # stage1.feature_extractor.candidate_background_masking wraps the
        # extractor; unwrapping it for the DINOv3 backbone above would
        # silently drop that masking, so carry the wrapper over and apply it
        # to the candidate crops we encode ourselves.
        seen = extractor
        while seen is not None and not isinstance(seen, MaskedCropFeatureExtractor):
            seen = getattr(seen, "base", None)
        matcher.candidate_masker = seen
        if seen is not None:
            log.info("patch_matching: candidate crops are background-masked (candidate_background_masking)")
        return matcher

    # -- encoding -----------------------------------------------------------

    def _to_tensor(self, img_bgr: np.ndarray, is_candidate: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(pixel_values [1,3,H,W], patch-validity mask [N] or None). The mask
        is only ever non-trivial under reuse_cls_preprocess with
        pad_to_square (patch_matching's own preprocessing never pads)."""
        if self.cfg.reuse_cls_preprocess:
            mode = (
                self.extractor.candidate_preprocess_mode if is_candidate
                else self.extractor.preprocess_mode
            )
            if self.cfg.mask_padding:
                pv, mask = self.extractor.pixel_values_and_mask([img_bgr], mode, self.cfg.pad_min_content_frac)
                return pv, mask[0]
            return self.extractor.pixel_values([img_bgr], mode), None
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
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())[None], None

    @torch.no_grad()
    def encode(self, img_bgr: np.ndarray, is_candidate: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(patch tokens [N, D*len(layers)] -- per-layer L2-normalised,
        concatenated, re-normalised, so a dot product is the mean of per-layer
        cosines --, validity mask [N] or None = all patches). is_candidate only
        matters under reuse_cls_preprocess (picks candidate_preprocess_mode vs
        preprocess_mode)."""
        pv, mask = self._to_tensor(img_bgr, is_candidate)
        tokens = self.extractor.forward_tokens(pv.to(self.device), self.cfg.layers)[1][0]
        return tokens, (mask.to(self.device) if mask is not None else None)

    def set_references(self, ref_imgs_bgr: list[np.ndarray]) -> None:
        self._refs = [self.encode(im, is_candidate=False) for im in ref_imgs_bgr]

    # -- scoring ------------------------------------------------------------

    def score_pair(self, ref, cand) -> float:
        """ref/cand: (tokens, mask-or-None) as returned by encode()."""
        (rt, rm), (ct, cm) = ref, cand
        return float(patch_pair_scores(
            ct[None], rt[None], self.cfg.method, self.cfg.symmetric, self.cfg.ot_epsilon, self.cfg.ot_iters,
            cand_mask=None if cm is None else cm[None], ref_mask=None if rm is None else rm[None],
        )[0, 0])

    def score_crops(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """[N_crops, N_refs] score matrix against the references set by
        set_references()."""
        out = np.zeros((len(crops_bgr), len(self._refs)), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            if self.candidate_masker is not None:
                crop = self.candidate_masker.mask_crop(crop)
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
    ref_imgs = [cv2.imread(str(p)) for p in ref_paths]
    if cfg.stage3.patch_matching.reuse_stage1_ref_processing:
        from aero_eyes.models.segmentation import build_segmenter
        from aero_eyes.stages.stage1 import prepare_reference_images

        seg_cfg = cfg.stage1.segmentation
        segmenter = build_segmenter(seg_cfg, cfg) if seg_cfg.enabled else None
        ref_imgs, _, _ = prepare_reference_images(cfg, sample_id, ref_imgs, segmenter)
        log.info(
            "[Stage3] %s: patch_matching applied stage1 ref processing (segmentation=%s, "
            "background_mode=%s, crop_to_object=%s)", sample_id, seg_cfg.enabled,
            seg_cfg.background_mode, cfg.stage1.crop_to_object,
        )
    matcher.set_references(ref_imgs)

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

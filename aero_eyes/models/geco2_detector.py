"""GeCo2 few-shot exemplar detector — replaces Stage 1+2+3 when
cfg.pipeline.detector == "geco2".

GeCo2 (vendored, unmodified, under ../../GECO2) is a single-image few-shot
counter/detector: exemplar boxes are drawn INSIDE the same image it detects
on (RoI-align against that image's own backbone features -- see
GECO2/models/counter_infer.py::CNT.forward). aero_eyes instead has 3
reference images captured separately from the drone video, so this module
splits CNT.forward into the two halves it was always architecturally built
from -- exemplar tokens are only ever consumed as K/V sequences inside
cross-attention (GECO2/models/query_generator.py::C_base.forward) and never
mixed with the query image's own conv features, so nothing stops the two
backbone passes running on different images:

  encode_exemplars(ref_images)  -> prototype tokens   (replaces Stage 1)
  detect_frame(frame, prototype) -> boxes + scores     (replaces Stage 2+3)

Each reference image is treated as its own exemplar box = the whole image
(refs are already close-up crops of the target), backbone-encoded
independently; the resulting tokens from all refs are concatenated along
the token/sequence dimension before being handed to the cross-attention
adapter (adapt_features only ever attends over these as a flat KV sequence,
so token count does not need to match any particular "num_objects").

SAM2-based mask refinement (CNT.sam_mask) is intentionally NOT used here --
aero_eyes only needs boxes, and skipping it avoids an extra heavy pass per
keyframe.

Requires the GECO2 repo's own dependencies (hydra-core, omegaconf, iopath,
its vendored sam2 package) to be installed, and pretrained weights
downloaded -- see stage123_geco2.repo_path / weights_path in config.yaml.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from aero_eyes.types import Box

log = logging.getLogger(__name__)

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_geco2_repo_on_path: str | None = None


def _ensure_geco2_on_path(repo_path: str) -> None:
    """Prepend the vendored GECO2 repo to sys.path so its own bare
    `import models...` / `import utils...` (unqualified, not `aero_eyes.*`)
    resolve to GECO2/models and GECO2/utils rather than aero_eyes' own
    same-named packages. Prepending (not appending) makes GECO2's copies
    win the lookup regardless of where aero_eyes' project root sits on
    sys.path.
    """
    global _geco2_repo_on_path
    repo = str(Path(repo_path).resolve())
    if _geco2_repo_on_path == repo:
        return
    if not Path(repo).is_dir():
        raise FileNotFoundError(
            f"GECO2 repo not found at '{repo}'. Set stage123_geco2.repo_path "
            "to the GECO2 checkout directory."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _geco2_repo_on_path = repo


class GeCo2Detector:
    """Loads the GeCo2 CNT model once; exposes encode_exemplars()/detect_frame()."""

    def __init__(self, cfg):
        g = cfg.stage123_geco2
        _ensure_geco2_on_path(g.repo_path)

        from models.counter_infer import build_model  # GECO2/models/counter_infer.py

        if not Path(g.weights_path).exists():
            raise FileNotFoundError(
                f"GeCo2 weights not found at '{g.weights_path}'. Download "
                "CNTQG_multitrain_ca44.pth (see GECO2/README.md) and set "
                "stage123_geco2.weights_path."
            )

        self.device = torch.device(cfg.device())
        self.image_size = float(g.image_size)
        self.score_threshold_ratio = g.score_threshold_ratio
        self.score_threshold_abs = g.score_threshold_abs
        self.nms_iou = g.nms_iou
        self.topk_per_keyframe = g.topk_per_keyframe
        self.use_shape_token = g.use_shape_token

        args = _GeCo2Args(
            image_size=g.image_size,
            num_objects=1,
            zero_shot=True,
            emb_dim=g.emb_dim,
            kernel_dim=g.kernel_dim,
            reduction=g.reduction,
        )
        self.model = build_model(args).to(self.device)
        state_dict = torch.load(g.weights_path, map_location=self.device, weights_only=True)
        state_dict = state_dict.get("model", state_dict)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            # Group by top-level submodule (e.g. "sam_mask.*") so it's obvious
            # at a glance whether the gap is confined to sam_mask -- the mask
            # refinement submodule this wrapper never calls (detect_frame/
            # encode_exemplars only use boxes+scores) -- or spills into
            # something actually load-bearing like backbone/adapt_features.
            prefixes = sorted({k.split(".")[0] for k in missing})
            log.warning("GeCo2 checkpoint missing %d params (random init for those), "
                        "grouped by submodule: %s", len(missing), prefixes)
        if unexpected:
            log.warning("GeCo2 checkpoint has %d unused params not in the model "
                        "(ignored): %s", len(unexpected),
                        sorted({k.split(".")[0] for k in unexpected}))
        self.model.eval()
        log.info("GeCo2 loaded from %s on %s", g.weights_path, self.device)

    # ------------------------------------------------------------------
    # Preprocessing (mirrors GECO2/utils/data.py::resize_and_pad, but we
    # only ever need the "whole image is the box" case here)
    # ------------------------------------------------------------------

    def _load_and_pad(self, img_bgr: np.ndarray) -> tuple[torch.Tensor, float]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD

        from utils.data import resize_and_pad  # GECO2/utils/data.py

        h, w = img_bgr.shape[:2]
        whole_box = torch.tensor([[0.0, 0.0, float(w), float(h)]])
        padded, _, scale = resize_and_pad(t, whole_box, size=self.image_size, zero_shot=True)
        return padded, float(scale)

    # ------------------------------------------------------------------
    # Stage 1 replacement: exemplar token extraction from reference images
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_exemplars(
        self,
        ref_images_bgr: list[np.ndarray],
        ref_boxes: list[tuple[float, float, float, float] | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run backbone + RoI-align on each reference image independently,
        concatenating tokens across refs.

        ref_boxes: optional per-image exemplar box (x1,y1,x2,y2) in the
        ORIGINAL (pre-pad) pixel coords of the corresponding ref_images_bgr
        entry -- e.g. a tight MobileSAM mask bbox, so RoI-align pools only
        the actual object instead of also averaging in the mean-color-filled
        background and any resize_and_pad zero-padding (empirically
        confirmed to matter: a manual same-image test with a tight mask box
        detected correctly, while the whole-image box this defaults to did
        not). None (or a None entry) falls back to the whole image, as before.

        Note: this same box also drives the RoI-Align pooling REGION for
        the appearance tokens below -- self.use_shape_token=false removes
        ONLY the explicit (w,h) -> shape_or_objectness signal, it does NOT
        change what region gets pooled for `exemplar`/`exemplar_l1`/
        `exemplar_l2`. A wrong-scaled `ref_boxes` still produces a
        wrong-scaled appearance token either way (see
        stage123_geco2.scale_calibration for the fix that DOES change the
        pooling region).

        Returns a dict of the 3 token sets CNT.adapt_features needs as its
        `prototype_embeddings` / `hq_prototypes` arguments -- CPU tensors,
        safe to cache to disk via torch.save. Each ref image contributes 2
        tokens per scale ([exemplar, shape]) normally, or 1 token
        ([exemplar] only) when self.use_shape_token is False -- see
        GeCo2Detector.calibrate_prototype's `tokens_per_ref` for why
        callers that index into this layout need to know which.
        """
        m = self.model
        from torchvision.ops import roi_align

        main_tokens, l1_tokens, l2_tokens = [], [], []
        for i, img in enumerate(ref_images_bgr):
            padded, scale = self._load_and_pad(img)
            x = padded.unsqueeze(0).to(self.device)

            given_box = ref_boxes[i] if ref_boxes is not None else None
            if given_box is not None:
                bx1, by1, bx2, by2 = given_box
            else:
                h_img, w_img = img.shape[:2]
                bx1, by1, bx2, by2 = 0.0, 0.0, float(w_img), float(h_img)
            # Scale from original ref-image pixel coords into the padded canvas
            # (same uniform scale resize_and_pad used above, zero_shot=True).
            px1, py1, px2, py2 = bx1 * scale, by1 * scale, bx2 * scale, by2 * scale

            feats = m.backbone(x)
            src = feats["vision_features"]
            l1 = feats["backbone_fpn"][0]
            l2 = feats["backbone_fpn"][1]
            bs, _, w, h = src.shape
            reduction = self.image_size / w

            box = torch.tensor([[0.0, px1, py1, px2, py2]], device=self.device)  # [batch_idx, x1, y1, x2, y2]

            exemplar = roi_align(src, boxes=box, output_size=1,
                                  spatial_scale=1.0 / reduction, aligned=True)
            exemplar = exemplar.permute(0, 2, 3, 1).reshape(bs, 1, m.emb_dim)

            exemplar_l1 = roi_align(l1, boxes=box, output_size=1,
                                     spatial_scale=1.0 / reduction * 2 * 2, aligned=True)
            exemplar_l1 = exemplar_l1.permute(0, 2, 3, 1).reshape(bs, 1, m.emb_dim)

            exemplar_l2 = roi_align(l2, boxes=box, output_size=1,
                                     spatial_scale=1.0 / reduction * 2, aligned=True)
            exemplar_l2 = exemplar_l2.permute(0, 2, 3, 1).reshape(bs, 1, m.emb_dim)

            if self.use_shape_token:
                box_hw = torch.tensor([[[px2 - px1, py2 - py1]]], dtype=torch.float32, device=self.device)
                shape = m.shape_or_objectness(box_hw).reshape(bs, 1, m.emb_dim)
                main_tokens.append(torch.cat([exemplar, shape], dim=1).cpu())
                l1_tokens.append(torch.cat([exemplar_l1, shape], dim=1).cpu())
                l2_tokens.append(torch.cat([exemplar_l2, shape], dim=1).cpu())
            else:
                main_tokens.append(exemplar.cpu())
                l1_tokens.append(exemplar_l1.cpu())
                l2_tokens.append(exemplar_l2.cpu())

        return {
            "main": torch.cat(main_tokens, dim=1),
            "l1": torch.cat(l1_tokens, dim=1),
            "l2": torch.cat(l2_tokens, dim=1),
        }

    # ------------------------------------------------------------------
    # Stage 2+3 replacement: dense detection on one keyframe
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_scores(self, frame_bgr: np.ndarray, prototype: dict[str, torch.Tensor]):
        """Shared by detect_frame() and raw_scores(): run the query-image
        half of CNT.forward on one frame, cross-attend with the precomputed
        exemplar tokens, and return the RAW (unfiltered) dense predictions.
        Returns (pred_boxes [N,4] normalized xyxy in padded canvas, box_v
        [N] raw score, scale).
        """
        from utils.box_ops import boxes_with_scores  # GECO2/utils/box_ops.py

        m = self.model
        padded, scale = self._load_and_pad(frame_bgr)
        x = padded.unsqueeze(0).to(self.device)

        feats = m.backbone(x)
        src = feats["vision_features"]

        prototype_embeddings = prototype["main"].to(self.device)
        hq_prototypes = [prototype["l1"].to(self.device), prototype["l2"].to(self.device)]

        adapted_f, _ = m.adapt_features(
            image_embeddings=src,
            image_pe=m.sam_prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=feats["backbone_fpn"],
            hq_prototypes=hq_prototypes,
            hq_pos=feats["vision_pos_enc"],
        )
        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, m.emb_dim, -1).permute(0, 2, 1)
        centerness = m.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = m.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, _ = boxes_with_scores(centerness, outputs_coord, sort=False, validate=True)
        return outputs[0]["pred_boxes"], outputs[0]["box_v"], scale

    @torch.no_grad()
    def raw_scores(self, frame_bgr: np.ndarray, prototype: dict[str, torch.Tensor]) -> np.ndarray:
        """Diagnostic only: the raw, unfiltered per-location score map for
        one frame (before score_threshold_ratio/NMS/top-K), as a flat
        numpy array. Use this to check whether GeCo2's score actually
        separates "target present" from "target absent" frames on your own
        data -- see scripts/check_geco2_score_separation.py. GeCo2 is
        trained/evaluated on FSC147, a counting benchmark where every image
        is guaranteed to contain >=1 instance of the counted class, so it
        may never have learned what a genuine "absent" frame should score
        like; our pipeline's threshold is RELATIVE to each frame's own max
        score (see detect_frame below), so on its own it cannot express
        "nothing here" -- it always keeps at least the single highest-
        scoring point.
        """
        _, box_v, _ = self._forward_scores(frame_bgr, prototype)
        return box_v.cpu().numpy()

    @torch.no_grad()
    def detect_frame(self, frame_bgr: np.ndarray, prototype: dict[str, torch.Tensor]) -> list[Box]:
        """Run the query-image half of CNT.forward on one frame, cross-attend
        with the precomputed exemplar tokens, threshold + NMS, and return
        boxes in absolute pixel coords of `frame_bgr`.

        score_threshold_ratio alone is RELATIVE to this frame's own max
        score, so on its own it always returns at least one box whenever
        box_v.max() > 0 -- it cannot express "target absent this frame".
        score_threshold_abs adds an ABSOLUTE floor on that per-frame max: if
        the frame's best score doesn't even clear this, the whole frame is
        treated as empty (returns []) regardless of the relative ratio.
        Calibrate it with scripts/check_geco2_score_separation.py on your
        own present/absent-labeled frames -- default 0.0 keeps the old
        always-detects-something behavior.
        """
        from torchvision.ops import nms as torch_nms

        pred_boxes, box_v, scale = self._forward_scores(frame_bgr, prototype)
        if pred_boxes.numel() == 0:
            return []

        max_score = box_v.max()
        if max_score < self.score_threshold_abs:
            return []

        keep_mask = box_v > (max_score * self.score_threshold_ratio)
        if not bool(keep_mask.any()):
            return []
        cand_boxes = torch.clamp(pred_boxes[keep_mask], 0, 1)
        cand_scores = box_v[keep_mask]

        keep_idx = torch_nms(cand_boxes, cand_scores, self.nms_iou)
        cand_boxes = cand_boxes[keep_idx]
        cand_scores = cand_scores[keep_idx]
        if cand_boxes.shape[0] > self.topk_per_keyframe:
            top = torch.topk(cand_scores, self.topk_per_keyframe).indices
            cand_boxes = cand_boxes[top]
            cand_scores = cand_scores[top]

        # Padded-canvas-normalized -> original frame pixel coords (matches
        # GECO2/demo_gradio.py::post_process's `pred_boxes / scale * img.shape[-1]`).
        px_boxes = (cand_boxes / scale * self.image_size).cpu().numpy()
        scores = cand_scores.cpu().numpy()

        h_frame, w_frame = frame_bgr.shape[:2]
        results: list[Box] = []
        for (x1, y1, x2, y2), s in zip(px_boxes, scores):
            box = Box(float(x1), float(y1), float(x2), float(y2), score=float(s)).clip(w_frame, h_frame)
            if box.area() > 0:
                results.append(box)
        return results

    # ------------------------------------------------------------------
    # Domain calibration (stage123_geco2.domain_calibration) -- shifts
    # exemplar appearance tokens toward this video's own feature-space
    # region, correcting the systematic gap between running the backbone on
    # an isolated reference photo vs. on a real video frame (two
    # independent forward passes -> two different self-attention contexts
    # -> different token statistics, independent of the object's true
    # appearance).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def frame_domain_embedding(self, frame_bgr: np.ndarray) -> dict[str, torch.Tensor]:
        """Global-average-pooled backbone feature per scale for one frame --
        a proxy for "what this video's own domain looks like in feature
        space", in the same (1, 1, emb_dim) shape as a single exemplar
        token so it can be compared/blended with prototype tokens directly.
        """
        padded, _ = self._load_and_pad(frame_bgr)
        x = padded.unsqueeze(0).to(self.device)
        feats = self.model.backbone(x)
        return {
            "main": feats["vision_features"].mean(dim=(2, 3)).unsqueeze(1).cpu(),
            "l1": feats["backbone_fpn"][0].mean(dim=(2, 3)).unsqueeze(1).cpu(),
            "l2": feats["backbone_fpn"][1].mean(dim=(2, 3)).unsqueeze(1).cpu(),
        }

    def estimate_domain_shift(self, sample_frames_bgr: list[np.ndarray]) -> dict[str, torch.Tensor]:
        """Average frame_domain_embedding() over a handful of (unlabeled,
        unpaired) frames sampled from the video -- an estimate of this
        video's own mean token per scale, used as the calibration target.
        """
        if not sample_frames_bgr:
            raise ValueError("estimate_domain_shift requires at least one sample frame")
        sums: dict[str, torch.Tensor] = {}
        for frame in sample_frames_bgr:
            emb = self.frame_domain_embedding(frame)
            for scale, v in emb.items():
                sums[scale] = v if scale not in sums else sums[scale] + v
        return {scale: v / len(sample_frames_bgr) for scale, v in sums.items()}

    @staticmethod
    def calibrate_prototype(
        prototype: dict[str, torch.Tensor],
        video_domain_means: dict[str, torch.Tensor],
        num_refs: int,
        strength: float,
        tokens_per_ref: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Mean-shift the APPEARANCE tokens of `prototype` toward
        `video_domain_means`, blended by `strength` (0 = no change, 1 =
        appearance tokens' own mean fully replaced by the video's mean).
        Shape tokens (if present) are left untouched -- they encode box
        (w,h), not appearance, so they're not subject to the same domain gap.

        Token layout per scale is [app_1, shape_1, app_2, shape_2, ...,
        app_N, shape_N] when GeCo2Detector.use_shape_token is True (each ref
        image contributes one [exemplar, shape] pair, concatenated in that
        order across refs -- appearance at even indices 0,2,4,..., shape at
        odd indices), or just [app_1, app_2, ..., app_N] (every index is
        appearance) when use_shape_token is False. Pass
        tokens_per_ref=1 in that case -- see encode_exemplars.
        """
        app_idx = list(range(0, tokens_per_ref * num_refs, tokens_per_ref))
        calibrated: dict[str, torch.Tensor] = {}
        for scale, tokens in prototype.items():
            tokens = tokens.clone()
            ref_mean = tokens[:, app_idx, :].mean(dim=1, keepdim=True)
            delta = strength * (video_domain_means[scale] - ref_mean)
            tokens[:, app_idx, :] = tokens[:, app_idx, :] + delta
            calibrated[scale] = tokens
        return calibrated

    # ------------------------------------------------------------------
    # Prototype cache (torch tensors, not the numpy prototype.npz format)
    # ------------------------------------------------------------------

    @staticmethod
    def save_prototype(prototype: dict[str, torch.Tensor], path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(prototype, str(path))

    @staticmethod
    def load_prototype(path: str | Path) -> dict[str, torch.Tensor]:
        return torch.load(str(path), map_location="cpu", weights_only=True)


class _GeCo2Args:
    """Minimal stand-in for the argparse.Namespace GECO2.build_model expects."""

    def __init__(self, image_size: int, num_objects: int, zero_shot: bool,
                 emb_dim: int, kernel_dim: int, reduction: int):
        self.image_size = image_size
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.emb_dim = emb_dim
        self.kernel_dim = kernel_dim
        self.reduction = reduction

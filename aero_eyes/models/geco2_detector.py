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

SAM2-based mask refinement (CNT.sam_mask) is skipped by detect_frame() by
default -- aero_eyes only needs boxes, and skipping it avoids an extra
heavy pass per keyframe. It's available opt-in as box_refine.method=
"sam2_dense" (see sam2_refine_boxes below and aero_eyes/utils/box_refine.py),
which reuses GeCo2's OWN dense backbone features (the SAME Hiera pass that
scored the boxes) to refine already-chosen boxes via GECO2's own
GECO2/models/sam_mask.py::MaskProcessor -- unlike this module's MobileSAM-
based box_refine methods ("sam"/"sam_dense"), no separate crop or
re-encode is needed, at the cost of one extra GeCo2 backbone forward pass
per refined frame (feats aren't cached across calls; see
sam2_refine_boxes's docstring).

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
        self.min_box_area_enabled = g.min_box_area_enabled
        self.min_box_area = g.min_box_area
        self.use_shape_token = g.use_shape_token
        self.emb_dim = g.emb_dim
        self.reduction = g.reduction
        # Lazily built by sam2_refine_boxes on first use (downloads GECO2's
        # own pretrained SAM2 checkpoint) -- False (not None) once a build
        # attempt has failed, so it isn't retried every call.
        self._mask_processor = None

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
        [N] raw score, scale, feats) -- feats is the raw backbone output
        dict (m.backbone(x): vision_features/backbone_fpn/vision_pos_enc),
        needed by sam2_refine_boxes to run GECO2's own SAM2 mask_decoder on
        this SAME forward pass without re-running the backbone a second
        time; every other caller ignores it.
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
        return outputs[0]["pred_boxes"], outputs[0]["box_v"], scale, feats

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
        _, box_v, _, _ = self._forward_scores(frame_bgr, prototype)
        return box_v.cpu().numpy()

    @torch.no_grad()
    def forward_scores(self, frame_bgr: np.ndarray, prototype: dict[str, torch.Tensor]):
        """Public wrapper around _forward_scores: the RAW, unfiltered dense
        predictions for one frame -- (pred_boxes [N,4] normalized xyxy in
        padded canvas, box_v [N] raw score, scale). Unlike detect_frame(),
        applies no threshold/NMS/top-K at all.

        Used by stage123_geco2.py's stage123_geco2.global_adaptive_threshold
        path to pool raw scores across every keyframe in a video BEFORE
        deciding what counts as a real detection (see
        filter_boxes_by_threshold below for the second half) -- the same
        two-pass shape as stage3.py's own adaptive_threshold, applied to
        GeCo2's score instead of DINOv2 cosine similarity.
        """
        pred_boxes, box_v, scale, _ = self._forward_scores(frame_bgr, prototype)
        return pred_boxes, box_v, scale

    def filter_boxes_by_threshold(
        self,
        pred_boxes: torch.Tensor,
        box_v: torch.Tensor,
        scale: float,
        frame_bgr: np.ndarray,
        threshold: float,
    ) -> list[Box]:
        """Threshold (by an EXPLICIT absolute score value, not a per-frame
        ratio) + NMS + top-K + coordinate conversion -- the second half of
        detect_frame(), factored out so it can also be driven by a threshold
        computed externally (e.g. stage123_geco2.py's global adaptive
        threshold, pooled across a whole video) instead of only
        detect_frame()'s own per-frame-relative one.
        """
        from torchvision.ops import nms as torch_nms

        if pred_boxes.numel() == 0:
            return []

        keep_mask = box_v > threshold
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
            if box.area() <= 0:
                continue
            # stage123_geco2.min_box_area_enabled: reject a degenerate,
            # near-zero-area box the regression head produced (nothing else
            # here guards against one) -- see that field's own docstring for
            # why this is an AREA floor, not a min-side-length one (a real
            # object's thinnest side can legitimately be a couple pixels at
            # the frame edge; area does not shrink that far for any real box
            # in this project's own GT).
            if self.min_box_area_enabled and box.area() < self.min_box_area:
                continue
            results.append(box)
        return results

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
        always-detects-something behavior. See
        stage123_geco2.global_adaptive_threshold for a whole-video
        alternative to this per-frame-relative decision.
        """
        pred_boxes, box_v, scale, _ = self._forward_scores(frame_bgr, prototype)
        if pred_boxes.numel() == 0:
            return []

        max_score = box_v.max()
        if max_score < self.score_threshold_abs:
            return []

        threshold = max_score * self.score_threshold_ratio
        return self.filter_boxes_by_threshold(pred_boxes, box_v, scale, frame_bgr, threshold)

    # ------------------------------------------------------------------
    # box_refine.method == "sam2_dense": GeCo2-native SAM2 mask refinement
    # ------------------------------------------------------------------

    def _get_mask_processor(self):
        """Lazily build GECO2's own MaskProcessor (GECO2/models/sam_mask.py)
        -- downloads Meta's public pretrained SAM2 checkpoint
        (sam2_hiera_base_plus.pt, ~300+MB, cached by torch.hub after the
        first call) the first time this is called on a given detector
        instance. Returns None (and remembers not to retry) if the import
        or build fails for any reason (e.g. no network access, sam2
        package unavailable) -- callers must treat that as "refinement
        unavailable" and fall back to the unrefined boxes.
        """
        if self._mask_processor is False:
            return None
        if self._mask_processor is not None:
            return self._mask_processor
        try:
            from models.sam_mask import MaskProcessor  # GECO2/models/sam_mask.py
            self._mask_processor = MaskProcessor(
                self.emb_dim, int(self.image_size), self.reduction,
            ).to(self.device)
            self._mask_processor.eval()
        except Exception:
            log.warning(
                "box_refine.method=sam2_dense: failed to build GECO2's "
                "MaskProcessor (SAM2 checkpoint download or import failed) "
                "-- refinement unavailable this run, boxes left unchanged.",
                exc_info=True,
            )
            self._mask_processor = False
            return None
        return self._mask_processor

    @torch.no_grad()
    def sam2_refine_boxes(
        self,
        frame_bgr: np.ndarray,
        prototype: dict[str, torch.Tensor],
        boxes: list[Box],
        context_margin: float = 0.0,
        adaptive_context_margin_cfg=None,
        sample_reference_size: float | None = None,
        use_center_point: bool = False,
        select_best_mask: bool = False,
    ) -> list[Box]:
        """box_refine.method == "sam2_dense": refine `boxes` (already-chosen
        boxes on this frame, e.g. Stage 3's cosine-matched detections or a
        Stage 4 tracked box) via GECO2's OWN SAM2-based mask_decoder
        (GECO2/models/sam_mask.py::MaskProcessor, the same submodule
        CNT.forward calls as `self.sam_mask(feats, outputs)` when its own
        `validate=True` -- see GECO2/models/counter_infer.py).

        Legacy path (context_margin==0, use_center_point=False,
        select_best_mask=False -- the defaults, and the only behavior this
        method had before these parameters existed): calls
        MaskProcessor.forward() as-is, unchanged. That vendored code always
        prompts with the box exactly as given (SAM2's box-as-2-corner-points
        convention, no margin/point) and always takes GECO2's own hard-coded
        mask index [2] (`masks[:, 2]` / `iou_predictions_[:, 2]` in
        GECO2/models/sam_mask.py -- matching CNT.forward's own choice for
        ITS training task, not necessarily the best choice for refining an
        arbitrary tracked box on this project's own footage).

        New path (any of the new parameters actually requested): reimplements
        that same encode/decode flow from OUTSIDE GECO2/ -- calling
        MaskProcessor's own public submodules (forward_feats,
        prompt_encoder_sam, mask_decoder) directly instead of its forward()
        wrapper -- so this project's OWN box_refine knobs (context_margin/
        adaptive_context_margin, use_center_point_prompt) work here the same
        way they already do for box_refine.method="sam_dense"
        (MobileSAMSegmenter.segment_box_cached), WITHOUT modifying a single
        line inside GECO2/. See _sam2_refine_boxes_custom's own docstring
        for exactly how.

        Unlike the MobileSAM-based "sam"/"sam_dense" box_refine methods,
        this needs a FRESH GeCo2 backbone forward pass on `frame_bgr`
        (feats aren't cached from whatever detection pass originally
        produced `boxes` -- by the time box_refine runs, that pass is long
        over) -- one pass per FRAME, shared across every box in `boxes`,
        not one per box.

        Falls back to leaving a box UNCHANGED (never raises) if
        MaskProcessor is unavailable, or if that box's refined mask comes
        back empty -- callers apply their own min_iou_with_original gate
        afterward via aero_eyes.utils.box_refine.apply_iou_gate, same as
        every other box_refine method.
        """
        if not boxes:
            return boxes
        mask_processor = self._get_mask_processor()
        if mask_processor is None:
            return boxes

        _, _, scale, feats = self._forward_scores(frame_bgr, prototype)

        if context_margin == 0.0 and adaptive_context_margin_cfg is None and not use_center_point and not select_best_mask:
            return self._sam2_refine_boxes_legacy(frame_bgr, boxes, scale, feats, mask_processor)
        return self._sam2_refine_boxes_custom(
            frame_bgr, boxes, scale, feats, mask_processor,
            context_margin, adaptive_context_margin_cfg, sample_reference_size,
            use_center_point, select_best_mask,
        )

    def _sam2_refine_boxes_legacy(self, frame_bgr, boxes, scale, feats, mask_processor) -> list[Box]:
        """Original box_refine.method=sam2_dense behavior, unchanged --
        see sam2_refine_boxes's own docstring."""
        norm_boxes = []
        for b in boxes:
            norm_boxes.append([
                b.x1 * scale / self.image_size, b.y1 * scale / self.image_size,
                b.x2 * scale / self.image_size, b.y2 * scale / self.image_size,
            ])
        norm_boxes_t = torch.tensor(norm_boxes, dtype=torch.float32, device=self.device).clamp(0.0, 1.0)
        outputs_wrapped = [{"pred_boxes": norm_boxes_t.unsqueeze(0)}]

        try:
            _, _, corrected_bboxes = mask_processor(feats, outputs_wrapped)
            corrected = corrected_bboxes[0]  # [N,4], padded-canvas PIXEL coords; zeros = empty mask
        except Exception:
            log.warning("box_refine.method=sam2_dense: MaskProcessor forward "
                        "pass failed -- boxes left unchanged this frame.", exc_info=True)
            return boxes

        h_frame, w_frame = frame_bgr.shape[:2]
        results: list[Box] = []
        for i, b in enumerate(boxes):
            cb = corrected[i]
            if bool(torch.all(cb == 0)):
                results.append(b)
                continue
            x1, y1, x2, y2 = (cb / scale).tolist()
            refined = Box(x1, y1, x2, y2, score=b.score).clip(w_frame, h_frame)
            results.append(refined if refined.area() > 0 else b)
        return results

    def _sam2_refine_boxes_custom(
        self, frame_bgr, boxes, scale, feats, mask_processor,
        context_margin, adaptive_context_margin_cfg, sample_reference_size,
        use_center_point, select_best_mask,
    ) -> list[Box]:
        """Reimplements GECO2/models/sam_mask.py::MaskProcessor.forward()'s
        encode/decode flow using only its PUBLIC submodules (forward_feats,
        prompt_encoder_sam, mask_decoder -- standard, un-customized SAM2
        components, not GECO2-specific logic) called directly from here,
        instead of going through forward() itself -- so nothing in GECO2/
        is read differently or modified.

        Differs from the legacy path in 3 independent, each-optional ways
        (mirrors box_refine.use_center_point_prompt/context_margin/
        adaptive_context_margin for "sam_dense" -- see
        MobileSAMSegmenter.segment_box_cached's own docstring for the same
        rationale applied there):
          1. context_margin/adaptive_context_margin_cfg: the box PROMPT
             itself is expanded before encoding (legacy always prompts with
             the box exactly as given -- no margin concept existed here).
          2. use_center_point: the ORIGINAL (pre-margin) box's own center is
             ALSO passed as a positive point (SAM2 label 1) alongside the
             box's 2 corner points (labels 2/3) in the SAME prompt call --
             SAM2's prompt encoder natively supports mixing box-corner
             points with extra foreground points, so this needs no change
             to how PromptEncoder itself is called, just what's fed to it.
             The resulting mask is then isolated to the connected component
             containing that point (aero_eyes.utils.geometry.
             isolate_component_at_point -- same helper MobileSAMSegmenter
             uses), discarding an unrelated blob elsewhere in the frame.
          3. select_best_mask: mask_decoder(multimask_output=True) always
             produces 4 mask channels (index 0 = single-mask-mode token,
             1-3 = three multimask ambiguity-resolving hypotheses) with
             their own predicted-IoU scores (iou_predictions) -- legacy
             hard-codes index 2 regardless of those scores (a choice tuned
             for GECO2's OWN counting task, not ours). True picks
             argmax(iou_predictions[1:4]) + 1 per box instead, the same
             "trust the model's own confidence" principle
             segment_box_cached's `best_idx = scores.argmax()` already uses
             for MobileSAM.
        """
        import torch.nn.functional as F
        from aero_eyes.utils.box_refine import scale_context_margin
        from aero_eyes.utils.geometry import isolate_component_at_point, mask_bbox

        processed_feats = mask_processor.forward_feats(feats)

        box_corners = []   # canvas-PIXEL [x1,y1,x2,y2], margin-expanded
        centers = []       # canvas-PIXEL [cx,cy], from the ORIGINAL (pre-margin) box
        for b in boxes:
            margin = scale_context_margin(b, context_margin, adaptive_context_margin_cfg, sample_reference_size)
            bw, bh = b.x2 - b.x1, b.y2 - b.y1
            mx, my = bw * margin, bh * margin
            # frame-px -> canvas-PIXEL is just `* scale` (the legacy path's
            # `* scale / image_size` normalizes, then sam_mask.py's own
            # forward() immediately multiplies back by image_size -- the
            # two cancel out; margin is applied here, in frame-px, first).
            x1 = (b.x1 - mx) * scale
            y1 = (b.y1 - my) * scale
            x2 = (b.x2 + mx) * scale
            y2 = (b.y2 + my) * scale
            box_corners.append([x1, y1, x2, y2])
            centers.append([
                (b.x1 + b.x2) / 2.0 * scale, (b.y1 + b.y2) / 2.0 * scale,
            ])

        box_t = torch.tensor(box_corners, dtype=torch.float32, device=self.device)
        box_t = box_t.clamp(0.0, float(self.image_size))
        box_coords = box_t.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device).repeat(box_t.size(0), 1)

        if use_center_point:
            center_t = torch.tensor(centers, dtype=torch.float32, device=self.device)
            center_t = center_t.clamp(0.0, float(self.image_size)).unsqueeze(1)  # [N,1,2]
            point_coords = torch.cat([box_coords, center_t], dim=1)  # [N,3,2]
            point_labels = torch.cat(
                [box_labels, torch.ones((box_t.size(0), 1), dtype=torch.int, device=self.device)], dim=1,
            )
        else:
            point_coords, point_labels = box_coords, box_labels

        try:
            sparse_embeddings, dense_embeddings = mask_processor.prompt_encoder_sam(
                points=(point_coords, point_labels), boxes=None, masks=None,
            )
            low_res_masks, iou_predictions, _, _ = mask_processor.mask_decoder(
                image_embeddings=processed_feats[-1],
                image_pe=mask_processor.prompt_encoder_sam.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=True,
                high_res_features=processed_feats[:-1],
            )
            masks = F.interpolate(
                low_res_masks, (int(self.image_size), int(self.image_size)),
                mode="bilinear", align_corners=False,
            ) > 0  # [N, 4, image_size, image_size]
        except Exception:
            log.warning(
                "box_refine.method=sam2_dense (custom path): mask_decoder "
                "forward pass failed -- boxes left unchanged this frame.", exc_info=True,
            )
            return boxes

        h_frame, w_frame = frame_bgr.shape[:2]
        results: list[Box] = []
        for i, b in enumerate(boxes):
            if select_best_mask:
                mask_idx = 1 + int(torch.argmax(iou_predictions[i, 1:4]).item())
            else:
                mask_idx = 2
            mask_np = masks[i, mask_idx].cpu().numpy()
            if use_center_point:
                cx, cy = centers[i]
                mask_np = isolate_component_at_point(mask_np, int(cx), int(cy))
            tight = mask_bbox(mask_np)
            if tight is None:
                results.append(b)
                continue
            x1, y1, x2, y2 = tight
            refined = Box(x1 / scale, y1 / scale, x2 / scale, y2 / scale, score=b.score).clip(w_frame, h_frame)
            results.append(refined if refined.area() > 0 else b)
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


def load_geco2_detector_and_prototype(cfg, work_dir: Path) -> tuple:
    """Lazily build a GeCo2Detector + load its cached exemplar prototype
    (from stage123_geco2.py's Stage 1+2 exemplar-encoding step), for
    callers that need on-demand GeCo2 forward passes OUTSIDE the main
    stage123_geco2 keyframe loop -- Stage 4 re-detection
    (aero_eyes/stages/stage4.py::_load_geco2) and box_refine.method=
    "sam2_dense" refinement (aero_eyes/stages/stage3.py and stage4.py) both
    need this same (detector, prototype) pair on demand rather than once
    per keyframe. Returns (None, None) if the exemplar cache is missing
    (Stage 1+2 not run yet, or pipeline.detector != "geco2").
    """
    proto_path = work_dir / cfg.stage123_geco2.prototype_cache_name
    if not proto_path.exists():
        return None, None
    detector = GeCo2Detector(cfg)
    prototype = GeCo2Detector.load_prototype(proto_path)
    return detector, prototype


class GeCo2DynamicPrototypeTracker:
    """stage123_geco2.dynamic_prototype's online/incremental state -- one
    instance per sample, built once before its keyframe loop starts and
    fed the accepted best box at every keyframe via offer(). See
    Geco2DynamicPrototypeConfig's own docstring (config.py) for the full
    design rationale: why this can't be stage3.dynamic_prototype's batch
    2-pass mechanism (no "whole video" score distribution exists yet
    while still processing it), and why GeCo2's own score alone isn't
    trusted to accept a candidate (consecutive-hit confirmation + an
    independent cosine cross-check are both required first).
    """

    def __init__(
        self, cfg, detector: "GeCo2Detector", base_prototype: dict, work_dir: Path, sample_id: str,
        cross_check_extractor=None, cross_check_prototype=None, cross_check_per_ref_features=None,
    ):
        """cross_check_extractor/cross_check_prototype/cross_check_per_ref_features:
        pass these when the CALLER already built a stage1.feature_extractor
        instance and read prototype.npz for its own purposes (e.g.
        run_stage12_geco2_candidates already does both, to embed every
        surviving candidate for Stage 3's later cosine matching) -- reuses
        them instead of this tracker lazily building a SECOND, redundant
        instance of the same model. Only relevant when
        dp_cfg.cross_check_source == "feature_extractor"; ignored (and
        lazily built on first use instead, from
        work_dir/stage1.prototype.cache_name) when left as None.

        cross_check_per_ref_features (list[np.ndarray] | None): the 3
        individual reference-image embeddings read_prototype() returns
        alongside the fused prototype. When
        accuracy.cheap_boosters.multi_reference_embedding is on, the cross
        check scores the candidate against EACH ref separately and pools
        with accuracy.cheap_boosters.multi_ref_pooling (mean|max) -- same
        pattern Stage 3's own matching and
        stage4._detect_on_frame_geco2's cosine filter use -- instead of
        collapsing to a single mean vector first, which dilutes a good
        match against one ref with two poor ones under "max" pooling.
        """
        from aero_eyes.utils.detection_confirm import DetectionConfirmer

        dp_cfg = cfg.stage123_geco2.dynamic_prototype
        self.cfg = cfg
        self.dp_cfg = dp_cfg
        self.detector = detector
        self.base_prototype = base_prototype
        self.sample_id = sample_id
        self._confirmer = DetectionConfirmer(dp_cfg.min_consecutive_hits, dp_cfg.consecutive_hits_iou)
        self._dynamic_tokens: list[dict] = []
        # Lazily built on first use if not given here -- avoids loading a
        # second model at all when dynamic_prototype is disabled or uses
        # "hiera" instead.
        self._cross_extractor = cross_check_extractor
        self._cross_prototype = cross_check_prototype
        self._cross_per_ref_features = cross_check_per_ref_features
        self._cross_prototype_path = work_dir / cfg.stage1.prototype.cache_name
        self._warned_cross_unavailable = False
        # Debug viz (gated by cfg.runtime.save_visualizations, same
        # opt-out convention every other stage's viz already uses): saves
        # the crop + full-frame context for every ACCEPTED token, so a
        # confuser that slipped past consecutive-hit + cross-check can be
        # spotted by eye instead of only inferred from downstream metrics.
        self._viz_dir = work_dir / "viz" / "dynamic_prototype"
        self._save_viz = cfg.runtime.save_visualizations
        # cross_check_threshold_self_calibrate / topk_fusion.
        # min_absolute_cosine_floor_self_calibrate (opt-in, share this same
        # cached value) -- see _ref_self_sim()'s own docstring.
        self._ref_self_sim: float | None = None
        self._warned_ref_self_sim_unavailable = False
        # Diagnostic counters ONLY (never read by offer()/effective_prototype()
        # themselves) -- log_summary() reports these at end of run so a run
        # producing ZERO dynamic tokens has a way to tell WHERE it stalled
        # (never offered a box at all vs. never got min_consecutive_hits in a
        # row vs. cross-check consistently rejecting) instead of the silent
        # "nothing happened, no idea why" every non-summary path here leaves
        # at default INFO log level (per-candidate detail is log.debug, easy
        # to miss -- same class of observability gap stage4's
        # keep_tracking_on_missed_keyframe had before detections.json's
        # empty-keyframe omission bug was found).
        self._n_offers = 0
        self._n_confirmed = 0
        self._n_cross_check_unavailable = 0
        self._n_cross_check_rejected = 0
        self._n_appended = 0
        # dynamic_prototype.topk_fusion (opt-in, offer_topk() only -- see
        # that method's own docstring): running Z-score baselines + its own
        # diagnostic counters, separate from the generic ones above so
        # log_summary() can report how often fusion actually changed the
        # selected candidate vs. just replayed boxes[0].
        from collections import deque
        tk_cfg = dp_cfg.topk_fusion
        self._topk_cosine_history: deque = deque(maxlen=tk_cfg.running_window)
        self._topk_geco2_history: deque = deque(maxlen=tk_cfg.running_window)
        self._n_topk_offers = 0
        self._n_topk_warmup = 0
        self._n_topk_fused_selected_non_top1 = 0
        self._n_topk_intra_frame_baseline = 0
        self._n_topk_floor_rejected = 0
        self._warned_topk_fusion_unsupported = False
        if tk_cfg.enabled and dp_cfg.cross_check_source != "feature_extractor":
            log.warning(
                "[Stage123-GeCo2] %s: dynamic_prototype.topk_fusion.enabled=true but "
                "cross_check_source=%s -- topk_fusion needs 'feature_extractor' (K "
                "encode_exemplars() GeCo2 backbone passes per keyframe for 'hiera' would "
                "defeat the whole 'keep this infrequent' premise) -- offer_topk() will fall "
                "back to plain boxes[0] selection for the rest of this run.",
                sample_id, dp_cfg.cross_check_source,
            )
            self._warned_topk_fusion_unsupported = True
        log.info(
            "[Stage123-GeCo2] %s: dynamic_prototype tracker built (max_tokens=%d, "
            "min_consecutive_hits=%d, consecutive_hits_iou=%.2f, cross_check_source=%s, "
            "cross_check_threshold=%.2f, topk_fusion.enabled=%s)", sample_id, dp_cfg.max_tokens,
            dp_cfg.min_consecutive_hits, dp_cfg.consecutive_hits_iou, dp_cfg.cross_check_source,
            dp_cfg.cross_check_threshold, tk_cfg.enabled,
        )

    def dynamic_token_count(self) -> int:
        """Number of dynamic tokens accepted so far (excludes the 3
        original reference-image tokens) -- callers use this to decide
        whether dynamic_prototype.second_pass has anything worth
        re-running with (0 means pass 2 would be identical to pass 1)."""
        return len(self._dynamic_tokens)

    def effective_prototype(self) -> dict[str, torch.Tensor]:
        """The prototype to pass into detect_frame()/filter_boxes_by_threshold()
        for THIS keyframe -- the original reference-image tokens plus
        whatever this tracker has accepted so far. Identical to
        base_prototype (same object, no copy) when nothing has been
        accepted yet, so callers pay no extra cost until this actually
        does something."""
        if not self._dynamic_tokens:
            return self.base_prototype
        return {
            key: torch.cat([self.base_prototype[key]] + [t[key] for t in self._dynamic_tokens], dim=1)
            for key in ("main", "l1", "l2")
        }

    def offer(
        self, frame_bgr: np.ndarray, box: Box, precomputed_feature: np.ndarray | None = None,
        frame_idx: int | None = None,
    ) -> None:
        """Call once per keyframe with the box detect_frame()/
        filter_boxes_by_threshold() already selected as this frame's best
        (i.e. it already cleared GeCo2's own score threshold) -- decides
        whether to also accept it into the dynamic exemplar buffer. No-op
        if dynamic_prototype is disabled.

        precomputed_feature: pass the stage1.feature_extractor embedding
        for THIS SAME box when the caller already computed one for its own
        purposes (e.g. run_stage12_geco2_candidates already embeds every
        surviving candidate for Stage 3's later cosine matching) -- skips
        a redundant re-embed inside the "feature_extractor" cross-check.
        Ignored when cross_check_source="hiera", or when None.

        frame_idx: purely for the debug viz filename/log below (see
        _save_debug_viz) -- optional, no effect on accept/reject logic.
        """
        if not self.dp_cfg.enabled:
            return
        self._n_offers += 1
        confirmed = self._confirmer.offer(box)
        if confirmed is None:
            return  # not yet min_consecutive_hits in a row -- keep waiting
        self._n_confirmed += 1

        # One extra backbone forward pass -- only for a box that ALREADY
        # cleared both GeCo2's own threshold and consecutive-hit
        # confirmation, so this is infrequent relative to the per-keyframe
        # detect_frame() cost already being paid regardless.
        new_tokens = self.detector.encode_exemplars(
            [frame_bgr], [(confirmed.x1, confirmed.y1, confirmed.x2, confirmed.y2)],
        )

        sim = self._cross_check_similarity(frame_bgr, confirmed, new_tokens, precomputed_feature)
        if sim is None:
            self._n_cross_check_unavailable += 1
            return  # cross-check unavailable this run -- refuse to add rather than trust GeCo2 alone
        threshold = self._effective_cross_check_threshold()
        if sim < threshold:
            self._n_cross_check_rejected += 1
            log.debug(
                "[Stage123-GeCo2] %s: dynamic_prototype candidate at frame region "
                "(%.0f,%.0f,%.0f,%.0f) rejected (cross_check sim=%.3f < %.3f)",
                self.sample_id, confirmed.x1, confirmed.y1, confirmed.x2, confirmed.y2,
                sim, threshold,
            )
            return

        self._n_appended += 1
        self._dynamic_tokens.append(new_tokens)
        if len(self._dynamic_tokens) > self.dp_cfg.max_tokens:
            self._dynamic_tokens.pop(0)  # FIFO: oldest APPENDED token only, originals never evicted
        self._save_debug_viz(frame_bgr, confirmed, frame_idx, f"cross_check_sim={sim:.3f}")
        log.info(
            "[Stage123-GeCo2] %s: dynamic_prototype appended a token (frame=%s, cross_check "
            "sim=%.3f, source=%s) -- %d/%d dynamic token(s) active",
            self.sample_id, frame_idx, sim, self.dp_cfg.cross_check_source,
            len(self._dynamic_tokens), self.dp_cfg.max_tokens,
        )

    def offer_topk(self, frame_bgr: np.ndarray, boxes: list[Box], feats: np.ndarray, frame_idx: int | None = None) -> None:
        """stage123_geco2.dynamic_prototype.topk_fusion (opt-in) -- ONLY
        wired into run_stage12_geco2_candidates. Considers EVERY box in
        `boxes` (GeCo2's own score-descending surviving candidates this
        keyframe -- boxes[0] is what plain offer() would use) with its
        ALREADY-EMBEDDED feature_extractor vector `feats[i]`, instead of
        just boxes[0], so a confuser that happens to outscore the real
        target THIS keyframe doesn't starve the confirmer of real hits --
        see Geco2DynamicPrototypeTopKFusionConfig's own docstring for the
        full fused_score rationale, cold-start behavior, and why this
        needs cross_check_source="feature_extractor" specifically.

        Three independent opt-in guards against the baseline
        self-poisoning failure mode (a confuser that keeps winning argmax
        bakes itself into the baseline as "normal" -- see
        Geco2DynamicPrototypeTopKFusionConfig's docstring):
          - intra_frame_baseline: Z-score against THIS FRAME's own
            candidates instead of/before the temporal history.
          - history_update_on_append_only: temporal history only records
            candidates that were actually appended, not every argmax pick.
          - min_absolute_cosine_floor_enabled: hard floor under fused_score
            acceptance, same role as stage3.dynamic_prototype's
            adaptive_min_floor.
        All default False/off -- behavior is unchanged from before these
        existed unless explicitly turned on.

        Falls back to plain offer(boxes[0], precomputed_feature=feats[0])
        (updating the SAME generic counters log_summary() reports) when
        topk_fusion is disabled, cross_check_source isn't
        "feature_extractor", `boxes` is empty, or prototype.npz isn't
        available -- callers can call this unconditionally whenever
        dynamic_prototype.enabled, regardless of topk_fusion's own setting.
        """
        if not self.dp_cfg.enabled or not boxes:
            return
        tk_cfg = self.dp_cfg.topk_fusion
        if not tk_cfg.enabled or self.dp_cfg.cross_check_source != "feature_extractor" or self._cross_prototype is None:
            self.offer(frame_bgr, boxes[0], precomputed_feature=feats[0], frame_idx=frame_idx)
            return

        self._n_offers += 1
        self._n_topk_offers += 1
        cosines = np.array([self._cosine_from_feature(feats[i]) for i in range(len(boxes))])
        geco2_scores = np.array([b.score for b in boxes], dtype=np.float64)

        # Baseline priority: intra_frame_baseline (this frame's OWN
        # candidates -- same domain, no accumulated history needed, immune
        # to a past frame's confuser poisoning it) when enabled and this
        # frame has enough candidates to trust a std estimate from; else
        # the temporal running_window history (today's original behavior);
        # else cold start (baseline stays None).
        baseline = None
        if tk_cfg.intra_frame_baseline and len(boxes) >= tk_cfg.intra_frame_min_boxes:
            baseline = (
                float(np.mean(cosines)), float(np.std(cosines)) + 1e-8,
                float(np.mean(geco2_scores)), float(np.std(geco2_scores)) + 1e-8,
            )
            self._n_topk_intra_frame_baseline += 1
        elif len(self._topk_cosine_history) >= tk_cfg.min_window_for_zscore:
            baseline = (
                float(np.mean(self._topk_cosine_history)), float(np.std(self._topk_cosine_history)) + 1e-8,
                float(np.mean(self._topk_geco2_history)), float(np.std(self._topk_geco2_history)) + 1e-8,
            )

        fused_chosen: float | None = None
        if baseline is None:
            self._n_topk_warmup += 1
            chosen_idx = 0
        else:
            cosine_mean, cosine_std, geco2_mean, geco2_std = baseline
            cosine_z = (cosines - cosine_mean) / cosine_std
            geco2_z = (geco2_scores - geco2_mean) / geco2_std
            fused = tk_cfg.cosine_weight * cosine_z + (1.0 - tk_cfg.cosine_weight) * geco2_z
            chosen_idx = int(np.argmax(fused))
            fused_chosen = float(fused[chosen_idx])
            if chosen_idx != 0:
                self._n_topk_fused_selected_non_top1 += 1

        # Temporal history update (feeds the running_window fallback baseline
        # above, and IS the sole baseline when intra_frame_baseline is off):
        # false (default) records the CHOSEN candidate's raw values on EVERY
        # offer -- unchanged original behavior. true defers this until the
        # candidate is actually appended (see the accept branch below), so a
        # repeatedly-argmax-winning confuser that never gets confirmed/
        # accepted can't shape the baseline.
        if not tk_cfg.history_update_on_append_only:
            self._topk_cosine_history.append(float(cosines[chosen_idx]))
            self._topk_geco2_history.append(float(geco2_scores[chosen_idx]))

        chosen_box = boxes[chosen_idx]
        confirmed = self._confirmer.offer(chosen_box)
        if confirmed is None:
            return  # not yet min_consecutive_hits in a row on the CHOSEN candidate
        self._n_confirmed += 1

        new_tokens = self.detector.encode_exemplars(
            [frame_bgr], [(confirmed.x1, confirmed.y1, confirmed.x2, confirmed.y2)],
        )

        if fused_chosen is None:
            # Cold start: no Z-score baseline yet -- fall back to the same
            # plain absolute-cosine gate offer() itself uses (self-calibrated
            # or hand-set, per cross_check_threshold_self_calibrate).
            sim, gate_name, gate_value = float(cosines[chosen_idx]), "cosine", self._effective_cross_check_threshold()
        else:
            sim, gate_name, gate_value = fused_chosen, "fused_score", tk_cfg.acceptance_z_threshold
        if sim < gate_value:
            self._n_cross_check_rejected += 1
            log.debug(
                "[Stage123-GeCo2] %s: dynamic_prototype (topk_fusion) candidate at frame "
                "region (%.0f,%.0f,%.0f,%.0f) [chosen_idx=%d/%d] rejected (%s=%.3f < %.3f)",
                self.sample_id, confirmed.x1, confirmed.y1, confirmed.x2, confirmed.y2,
                chosen_idx, len(boxes), gate_name, sim, gate_value,
            )
            return

        # Absolute backstop (opt-in): a candidate can still clear fused_score
        # acceptance purely because the baseline itself drifted down with a
        # run of poor picks -- this floor refuses to trust fused_score alone
        # once raw cosine drops below a hand-set sanity minimum, same role
        # as stage3.dynamic_prototype's adaptive_min_floor.
        if tk_cfg.min_absolute_cosine_floor_enabled:
            floor = self._effective_min_absolute_cosine_floor()
            if cosines[chosen_idx] < floor:
                self._n_topk_floor_rejected += 1
                self._n_cross_check_rejected += 1
                log.debug(
                    "[Stage123-GeCo2] %s: dynamic_prototype (topk_fusion) candidate at frame "
                    "region (%.0f,%.0f,%.0f,%.0f) [chosen_idx=%d/%d] rejected by absolute floor "
                    "(cosine=%.3f < %.3f, despite %s=%.3f >= %.3f)",
                    self.sample_id, confirmed.x1, confirmed.y1, confirmed.x2, confirmed.y2,
                    chosen_idx, len(boxes), cosines[chosen_idx], floor,
                    gate_name, sim, gate_value,
                )
                return

        if tk_cfg.history_update_on_append_only:
            self._topk_cosine_history.append(float(cosines[chosen_idx]))
            self._topk_geco2_history.append(float(geco2_scores[chosen_idx]))

        self._n_appended += 1
        self._dynamic_tokens.append(new_tokens)
        if len(self._dynamic_tokens) > self.dp_cfg.max_tokens:
            self._dynamic_tokens.pop(0)  # FIFO: oldest APPENDED token only, originals never evicted
        self._save_debug_viz(frame_bgr, confirmed, frame_idx, f"{gate_name}={sim:.3f}")
        log.info(
            "[Stage123-GeCo2] %s: dynamic_prototype (topk_fusion) appended a token "
            "(frame=%s, chosen_idx=%d/%d, %s=%.3f) -- %d/%d dynamic token(s) active",
            self.sample_id, frame_idx, chosen_idx, len(boxes), gate_name, sim,
            len(self._dynamic_tokens), self.dp_cfg.max_tokens,
        )

    def _save_debug_viz(self, frame_bgr: np.ndarray, box: Box, frame_idx: int | None, label: str) -> None:
        """Gated by cfg.runtime.save_visualizations (same convention every
        other stage's viz uses) -- saves the crop + full-frame context for
        an ACCEPTED token under <work_dir>/<sample_id>/viz/dynamic_prototype/
        so a confuser that slipped past consecutive-hit + cross-check can
        be spotted by eye. `label` matches the corresponding "appended a
        token" log line's own score, so the two can be cross-referenced by
        token index (this method is only ever called right before that log
        line, using len(self._dynamic_tokens) as the token's position)."""
        if not self._save_viz:
            return
        from aero_eyes.utils import viz as vizmod
        vizmod.save_dynamic_prototype_token(
            frame_bgr, box, frame_idx, len(self._dynamic_tokens), label, self._viz_dir,
        )

    def log_summary(self) -> None:
        """Call once after the keyframe loop finishes -- reports WHERE
        offers stalled if dynamic_token_count() ended up at 0, since every
        rejection path in offer() is either a silent early-return (no log
        at all) or log.debug (invisible at the default INFO level): was
        offer() never even called (0 offers -- check boxes/results are
        actually non-empty), never confirmed (0 confirmed -- the
        consecutive-hit gate never saw 2 spatially-agreeing keyframes in a
        row, e.g. box_iou keeps missing consecutive_hits_iou on a fast/
        jittery target), or confirmed but cross-check kept rejecting
        (n_confirmed > 0 but n_appended == 0 -- try a lower
        cross_check_threshold or cross_check_source="hiera")."""
        log.info(
            "[Stage123-GeCo2] %s: dynamic_prototype summary -- %d offer(s), %d passed "
            "consecutive-hit gate, %d cross-check unavailable, %d cross-check rejected, "
            "%d appended (%d/%d active at end)",
            self.sample_id, self._n_offers, self._n_confirmed, self._n_cross_check_unavailable,
            self._n_cross_check_rejected, self._n_appended, len(self._dynamic_tokens), self.dp_cfg.max_tokens,
        )
        if self.dp_cfg.topk_fusion.enabled and self._n_topk_offers > 0:
            log.info(
                "[Stage123-GeCo2] %s: dynamic_prototype topk_fusion summary -- %d offer_topk() "
                "call(s) (%d still in cold-start warm-up, %d used intra_frame_baseline), %d chose "
                "a candidate OTHER than boxes[0] (GeCo2's own top pick), %d rejected by the "
                "absolute cosine floor", self.sample_id, self._n_topk_offers,
                self._n_topk_warmup, self._n_topk_intra_frame_baseline,
                self._n_topk_fused_selected_non_top1, self._n_topk_floor_rejected,
            )

    def _get_ref_self_sim(self) -> float | None:
        """Cached MIN pairwise feature_extractor-space cosine among the 3
        ORIGINAL reference images' own per-ref embeddings -- an empirical,
        per-sample ceiling on "how similar do two genuinely-matching crops
        of THIS reference-photo-set's domain even look", used by
        cross_check_threshold_self_calibrate and topk_fusion.
        min_absolute_cosine_floor_self_calibrate in place of a hand-tuned
        absolute cosine number. MIN (not mean) is used deliberately: it's
        the worst-agreeing pair, so a threshold derived from it doesn't
        end up optimistic relative to how loosely this domain's own refs
        agree with EACH OTHER.

        Returns None (callers fall back to their own hand-set absolute
        value, logged once here) when fewer than 2 per-ref vectors are on
        hand -- needs accuracy.cheap_boosters.multi_reference_embedding=
        true at Stage 1 time to have saved them into prototype.npz at all.
        """
        if self._ref_self_sim is not None:
            return self._ref_self_sim
        per_ref = self._cross_per_ref_features
        if not per_ref and self._cross_prototype_path.exists():
            from aero_eyes.utils.io import read_prototype
            _, _, per_ref = read_prototype(self._cross_prototype_path)
        if not per_ref or len(per_ref) < 2:
            if not self._warned_ref_self_sim_unavailable:
                log.warning(
                    "[Stage123-GeCo2] %s: dynamic_prototype self-calibration needs >=2 "
                    "per-ref feature_extractor vectors (accuracy.cheap_boosters."
                    "multi_reference_embedding=true at Stage 1 time) -- none found, "
                    "falling back to the hand-set absolute threshold/floor value(s).",
                    self.sample_id,
                )
                self._warned_ref_self_sim_unavailable = True
            return None
        sims = [
            float(per_ref[i] @ per_ref[j])
            for i in range(len(per_ref)) for j in range(i + 1, len(per_ref))
        ]
        self._ref_self_sim = min(sims)
        log.info(
            "[Stage123-GeCo2] %s: dynamic_prototype self-calibration ref-vs-ref "
            "self-similarity ceiling = %.3f (min of %d pair(s))",
            self.sample_id, self._ref_self_sim, len(sims),
        )
        return self._ref_self_sim

    def _effective_cross_check_threshold(self) -> float:
        """The cross_check_threshold to actually gate on -- self-calibrated
        (ref_self_sim * cross_check_threshold_self_calibrate_ratio) when
        dp_cfg.cross_check_threshold_self_calibrate is on, source is
        "feature_extractor" (ref_self_sim lives in THAT embedding space,
        not Hiera's), and per-ref vectors are available; the plain hand-set
        dp_cfg.cross_check_threshold otherwise."""
        if self.dp_cfg.cross_check_threshold_self_calibrate and self.dp_cfg.cross_check_source == "feature_extractor":
            ref_sim = self._get_ref_self_sim()
            if ref_sim is not None:
                return ref_sim * self.dp_cfg.cross_check_threshold_self_calibrate_ratio
        return self.dp_cfg.cross_check_threshold

    def _effective_min_absolute_cosine_floor(self) -> float:
        """Same self-calibration idea as _effective_cross_check_threshold(),
        for topk_fusion's min_absolute_cosine_floor instead -- only called
        when min_absolute_cosine_floor_enabled is already true (offer_topk()
        guards that itself); cross_check_source is always "feature_extractor"
        here since topk_fusion falls back to plain offer() otherwise."""
        tk_cfg = self.dp_cfg.topk_fusion
        if tk_cfg.min_absolute_cosine_floor_self_calibrate:
            ref_sim = self._get_ref_self_sim()
            if ref_sim is not None:
                return ref_sim * tk_cfg.min_absolute_cosine_floor_self_calibrate_ratio
        return tk_cfg.min_absolute_cosine_floor

    def _cross_check_similarity(
        self, frame_bgr: np.ndarray, box: Box, new_tokens: dict, precomputed_feature: np.ndarray | None = None,
    ) -> float | None:
        if self.dp_cfg.cross_check_source == "hiera":
            return self._hiera_similarity(new_tokens)
        return self._feature_extractor_similarity(frame_bgr, box, precomputed_feature)

    def _hiera_similarity(self, new_tokens: dict) -> float:
        """cross_check_source="hiera": cosine between the candidate's OWN
        appearance token (GeCo2's Hiera backbone, same one detect_frame()
        already used) and the mean of the ORIGINAL reference images'
        appearance tokens -- see Geco2DynamicPrototypeConfig's own
        docstring for why this is NOT an independent signal from GeCo2's
        own score (same backbone, same feature space) and is offered for
        A/B testing rather than as the recommended default.
        """
        tokens_per_ref = 2 if self.detector.use_shape_token else 1
        candidate_vec = new_tokens["main"][0, 0].cpu().numpy().astype(np.float64)
        # Appearance-only sub-tokens: index 0, tokens_per_ref, 2*tokens_per_ref, ...
        # -- drops the interleaved shape token per ref when use_shape_token
        # is on (see encode_exemplars: [exemplar, shape] pairs per ref).
        ref_main = self.base_prototype["main"][0].cpu().numpy().astype(np.float64)  # [K, D]
        ref_vecs = ref_main[::tokens_per_ref]
        ref_vec = ref_vecs.mean(axis=0)
        cand_n = candidate_vec / (np.linalg.norm(candidate_vec) + 1e-8)
        ref_n = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
        return float(cand_n @ ref_n)

    def _feature_extractor_similarity(
        self, frame_bgr: np.ndarray, box: Box, precomputed_feature: np.ndarray | None = None,
    ) -> float | None:
        """cross_check_source="feature_extractor" (default): embeds the
        candidate crop with stage1.feature_extractor (an INDEPENDENT model
        from GeCo2's own Hiera backbone) and compares against Stage 1's
        own reference embedding(s) -- the SAME ones
        stage4.geco2_redetect_cosine_filter already cross-checks GeCo2
        re-detects against. Returns None (caller refuses to add) if
        prototype.npz isn't available.

        When accuracy.cheap_boosters.multi_reference_embedding is on and
        per-ref vectors are available, scores against each of the 3
        reference images separately and pools with
        accuracy.cheap_boosters.multi_ref_pooling (mean|max) instead of
        collapsing to the single fused prototype vector first -- see
        __init__'s cross_check_per_ref_features docstring for why.
        """
        if self._cross_extractor is None and precomputed_feature is None:
            if not self._cross_prototype_path.exists():
                if not self._warned_cross_unavailable:
                    log.warning(
                        "[Stage123-GeCo2] %s: dynamic_prototype.cross_check_source="
                        "'feature_extractor' but no prototype.npz found at %s -- "
                        "dynamic_prototype disabled this run (needs Stage 1 to have "
                        "built one).", self.sample_id, self._cross_prototype_path,
                    )
                    self._warned_cross_unavailable = True
                return None
            from aero_eyes.models.features import build_feature_extractor
            from aero_eyes.utils.io import read_prototype

            self._cross_extractor = build_feature_extractor(self.cfg)
            self._cross_prototype, _, self._cross_per_ref_features = read_prototype(self._cross_prototype_path)
        if precomputed_feature is not None and self._cross_prototype is None:
            return None  # prototype.npz unavailable and no lazy-load was attempted

        feat = (
            precomputed_feature if precomputed_feature is not None
            else self._cross_extractor.extract_crops(
                frame_bgr, [box],
                pad_ratio=self.cfg.stage2.candidate.feature_crop_pad,
                batch_size=self.cfg.runtime.batch_size,
            )[0]
        )
        return self._cosine_from_feature(feat)

    def _cosine_from_feature(self, feat: np.ndarray) -> float:
        """Raw feature_extractor-space cosine of an ALREADY-embedded crop
        against the reference prototype -- the multi-ref-pooling-aware part
        of _feature_extractor_similarity, factored out so offer_topk can
        score EVERY surviving candidate's precomputed feature the same way
        without duplicating the mean/max pooling logic."""
        use_multi_ref = (
            self.cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
            and self.cfg.accuracy.cheap_boosters.multi_reference_embedding
            and self._cross_per_ref_features
        )
        if use_multi_ref:
            sims = np.array([float(feat @ ref_feat) for ref_feat in self._cross_per_ref_features])
            pooling = self.cfg.accuracy.cheap_boosters.multi_ref_pooling
            return float(sims.max()) if pooling == "max" else float(sims.mean())
        return float(feat @ self._cross_prototype)


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

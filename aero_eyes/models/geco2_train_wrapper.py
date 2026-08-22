"""Training-mode forward wrapper for finetuning GeCo2.

GPU-only -- requires the compiled MultiScaleDeformableAttention CUDA
extension (see notebooks/train_geco2_aeroeyes_vastai.ipynb) that
GECO2/models/query_generator.py::C_base depends on for ANY forward pass
through adapt_features. Cannot be exercised on a machine without a built
extension (no CPU fallback exists) -- see docs/GECO2_FINETUNE_PLAN.md.

Mirrors aero_eyes/models/geco2_detector.py::GeCo2Detector.encode_exemplars()
/ _forward_scores() (gradients disabled there, @torch.no_grad()), but with
gradients enabled and built on GECO2/models/counter.py::CNT (the
TRAINING-variant class, with aux heads) instead of counter_infer.py::CNT
(inference-only, no aux heads -- would AttributeError if forced into a
training-like path).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aero_eyes.models.geco2_detector import _IMAGENET_MEAN, _IMAGENET_STD, _ensure_geco2_on_path

log = logging.getLogger(__name__)

# Only these submodules receive gradients -- everything else (backbone.*,
# shape_or_objectness.*, sam_prompt_encoder.*) stays frozen. backbone is
# frozen because 14 training videos is nowhere near enough to safely touch
# a large pretrained vision backbone; shape_or_objectness is frozen because
# it is never invoked at all (the box-size-dependent shape token is
# removed from this plan entirely, see docs/GECO2_FINETUNE_PLAN.md point 1)
# so it would receive zero gradient regardless -- excluding it from the
# optimizer here is belt-and-suspenders against a future refactor that
# switches to `model.parameters()` wholesale; sam_prompt_encoder stays
# frozen by default per the spec's "open question, decide empirically"
# (unfreeze later only if validation plateaus with a specific reason to
# suspect positional encoding is the bottleneck).
FINETUNE_PREFIXES = (
    "adapt_features.", "class_embed.", "class_embed_aux.", "bbox_embed.", "bbox_embed_aux.",
)


class _GeCo2TrainArgs:
    """Minimal stand-in for the argparse.Namespace GECO2.build_model expects
    (mirrors aero_eyes/models/geco2_detector.py::_GeCo2Args, plus the
    `training` flag that selects the aux-head-bearing code path)."""

    def __init__(self, image_size: int, num_objects: int, zero_shot: bool,
                 emb_dim: int, kernel_dim: int, reduction: int, training: bool):
        self.image_size = image_size
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.emb_dim = emb_dim
        self.kernel_dim = kernel_dim
        self.reduction = reduction
        self.training = training


def build_training_model(cfg, device: torch.device | str | None = None) -> torch.nn.Module:
    """Build models/counter.py::CNT(training=True), load the base GeCo2
    checkpoint, and apply the freeze/finetune split. Returns the model on
    `device` (or cfg.device() if not given).
    """
    g = cfg.stage123_geco2
    _ensure_geco2_on_path(g.repo_path)

    # CNT.__init__ unconditionally calls
    # torch.hub.set_dir('/d/hpc/projects/FRI/pelhanj/CNT_SAM2/models/') --
    # a hardcoded SLURM path (GECO2/models/counter.py:40, marked
    # `# TODO REMOVE!!` in the vendored code). Neutralize it BEFORE
    # importing models.counter -- setting TORCH_HOME first does NOT help,
    # since this call runs unconditionally inside __init__ and overrides
    # whatever hub dir was set before.
    torch.hub.set_dir = lambda *a, **k: None

    from models.counter import build_model  # GECO2/models/counter.py (training variant, has aux heads)

    device = torch.device(device) if device is not None else torch.device(cfg.device())

    args = _GeCo2TrainArgs(
        image_size=g.image_size, num_objects=1, zero_shot=True,
        emb_dim=g.emb_dim, kernel_dim=g.kernel_dim, reduction=g.reduction,
        training=True,
    )
    model = build_model(args).to(device)

    if not Path(g.weights_path).exists():
        raise FileNotFoundError(
            f"GeCo2 base checkpoint not found at '{g.weights_path}'. Download "
            "CNTQG_multitrain_ca44.pth (see GECO2/README.md) first."
        )
    state_dict = torch.load(g.weights_path, map_location=device, weights_only=True)
    state_dict = state_dict.get("model", state_dict)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Group by top-level submodule so it's obvious whether the gap is
        # confined to something expected (e.g. box_correction -- never
        # built when training=True) or spills into something load-bearing
        # like backbone/adapt_features.
        prefixes = sorted({k.split(".")[0] for k in missing})
        log.warning("GeCo2 base checkpoint missing %d param(s) (random init), "
                    "grouped by submodule: %s", len(missing), prefixes)
    if unexpected:
        log.warning("GeCo2 base checkpoint has %d unused param(s) (ignored), "
                    "grouped by submodule: %s", len(unexpected),
                    sorted({k.split(".")[0] for k in unexpected}))

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(FINETUNE_PREFIXES)

    trainable_frozen = [
        n for n, p in model.named_parameters()
        if p.requires_grad and n.startswith(("backbone.", "shape_or_objectness."))
    ]
    assert not trainable_frozen, (
        f"freeze recipe bug: these should be frozen but requires_grad=True: {trainable_frozen}"
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("GeCo2 training model built on %s: %d/%d params trainable", device, n_trainable, n_total)

    return model


def set_train_mode(model: torch.nn.Module, training: bool) -> None:
    """model.train()/eval() control for one epoch, keeping the (always
    frozen) backbone in eval() mode regardless. CNT's own forward wraps
    backbone calls in torch.no_grad() unconditionally (mirrored by this
    wrapper's encode_exemplars_grad/forward_train_step below), so this is
    not required for correctness of gradients -- but eval() additionally
    stops any BatchNorm/dropout inside the backbone from updating running
    stats, which train() alone would not.
    """
    model.train(training)
    model.backbone.eval()


def _load_and_pad_grad(
    img_bgr: np.ndarray, image_size: float, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Same preprocessing as GeCo2Detector._load_and_pad, kept as a free
    function here (not wrapped in @torch.no_grad()) since gradients must
    flow through everything downstream of this call in the training
    wrapper."""
    import cv2

    from utils.data import resize_and_pad  # GECO2/utils/data.py

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD

    h, w = img_bgr.shape[:2]
    whole_box = torch.tensor([[0.0, 0.0, float(w), float(h)]])
    padded, _, scale = resize_and_pad(t, whole_box, size=image_size, zero_shot=True)
    return padded.to(device), float(scale)


@dataclass
class ExemplarTokens:
    main: torch.Tensor
    l1: torch.Tensor
    l2: torch.Tensor


def encode_exemplars_grad(
    model: torch.nn.Module,
    ref_images_bgr: list[np.ndarray],
    ref_boxes: list[tuple[float, float, float, float] | None],
    image_size: float,
    device: torch.device,
) -> ExemplarTokens:
    """Gradient-enabled mirror of GeCo2Detector.encode_exemplars(), always
    taking the use_shape_token=False branch -- shape_or_objectness is NEVER
    called anywhere in this training wrapper (see
    docs/GECO2_FINETUNE_PLAN.md point 1: the box-size-dependent shape token
    is removed from the architecture entirely, not just disabled by a
    flag). Each reference image contributes exactly 1 token per scale
    (the bare RoI-Align appearance token), matching
    GeCo2Detector.encode_exemplars' `else` branch.
    """
    from torchvision.ops import roi_align

    main_tokens, l1_tokens, l2_tokens = [], [], []
    for img, given_box in zip(ref_images_bgr, ref_boxes):
        padded, scale = _load_and_pad_grad(img, image_size, device)
        x = padded.unsqueeze(0)

        if given_box is not None:
            bx1, by1, bx2, by2 = given_box
        else:
            h_img, w_img = img.shape[:2]
            bx1, by1, bx2, by2 = 0.0, 0.0, float(w_img), float(h_img)
        px1, py1, px2, py2 = bx1 * scale, by1 * scale, bx2 * scale, by2 * scale

        # Backbone stays frozen: no_grad here matches CNT.forward's own
        # convention (GECO2/models/counter.py) even though the OUTER
        # function has gradients enabled -- a no-grad tensor still lets
        # downstream requires_grad=True layers (adapt_features etc.)
        # receive gradients; this is standard autograd, not a special case.
        with torch.no_grad():
            feats = model.backbone(x)
        src = feats["vision_features"]
        l1 = feats["backbone_fpn"][0]
        l2 = feats["backbone_fpn"][1]
        bs, _, w, h = src.shape
        reduction = image_size / w

        box = torch.tensor([[0.0, px1, py1, px2, py2]], device=device)

        exemplar = roi_align(src, boxes=box, output_size=1, spatial_scale=1.0 / reduction, aligned=True)
        exemplar = exemplar.permute(0, 2, 3, 1).reshape(bs, 1, model.emb_dim)

        exemplar_l1 = roi_align(l1, boxes=box, output_size=1, spatial_scale=1.0 / reduction * 2 * 2, aligned=True)
        exemplar_l1 = exemplar_l1.permute(0, 2, 3, 1).reshape(bs, 1, model.emb_dim)

        exemplar_l2 = roi_align(l2, boxes=box, output_size=1, spatial_scale=1.0 / reduction * 2, aligned=True)
        exemplar_l2 = exemplar_l2.permute(0, 2, 3, 1).reshape(bs, 1, model.emb_dim)

        main_tokens.append(exemplar)
        l1_tokens.append(exemplar_l1)
        l2_tokens.append(exemplar_l2)

    return ExemplarTokens(
        main=torch.cat(main_tokens, dim=1),
        l1=torch.cat(l1_tokens, dim=1),
        l2=torch.cat(l2_tokens, dim=1),
    )


@dataclass
class TrainForwardOutput:
    main: dict            # {"pred_boxes": [1,N,4], "box_v": [1,N]}
    aux: dict              # {"pred_boxes": [1,M,4], "box_v": [1,M]}
    centerness: torch.Tensor       # [1,W,H]
    ref_points: torch.Tensor       # [2,N]
    centerness_aux: torch.Tensor   # [1,W,H]
    ref_points_aux: torch.Tensor   # [2,M]
    scale: float


def forward_train_step(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    prototype: ExemplarTokens,
    image_size: float,
    device: torch.device,
) -> TrainForwardOutput:
    """Gradient-enabled mirror of GeCo2Detector._forward_scores(), but ALSO
    computes the aux head (class_embed_aux/bbox_embed_aux on
    adapt_features' SECOND return value, `adapted_f_aux` -- which the
    inference wrapper discards via `_`) since SetCriterion's aux loss term
    needs it. Uses boxes_with_scores(..., validate=False) (median
    threshold -- training mode) for BOTH heads, not the inference max/8
    threshold.
    """
    from utils.box_ops import boxes_with_scores  # GECO2/utils/box_ops.py

    padded, scale = _load_and_pad_grad(frame_bgr, image_size, device)
    x = padded.unsqueeze(0)

    with torch.no_grad():
        feats = model.backbone(x)
    src = feats["vision_features"]

    adapted_f, adapted_f_aux = model.adapt_features(
        image_embeddings=src,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        prototype_embeddings=prototype.main,
        hq_features=feats["backbone_fpn"],
        hq_prototypes=[prototype.l1, prototype.l2],
        hq_pos=feats["vision_pos_enc"],
    )

    def _heads(adapted, class_embed, bbox_embed):
        bs, c, w, h = adapted.shape
        flat = adapted.view(bs, model.emb_dim, -1).permute(0, 2, 1)
        centerness = class_embed(flat).view(bs, w, h, 1).permute(0, 3, 1, 2)
        coord = bbox_embed(flat).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, ref_points = boxes_with_scores(centerness, coord, sort=False, validate=False)
        # bs == 1 (single sample forward) -- [0] gives the per-image slice
        # matching GECO2/train.py's `centerness[idx]`/`ref_points[idx]`
        # convention (which SetCriterion.forward expects).
        return outputs[0], centerness[0], ref_points[0]

    main_out, centerness, ref_points = _heads(adapted_f, model.class_embed, model.bbox_embed)
    aux_out, centerness_aux, ref_points_aux = _heads(adapted_f_aux, model.class_embed_aux, model.bbox_embed_aux)

    return TrainForwardOutput(
        main=main_out, aux=aux_out,
        centerness=centerness, ref_points=ref_points,
        centerness_aux=centerness_aux, ref_points_aux=ref_points_aux,
        scale=scale,
    )

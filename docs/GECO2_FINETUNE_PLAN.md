# Plan: Light Finetune of GeCo2 on the AERO EYES Dataset

**Status:** planning only — not implemented. This document is a handoff spec
for an implementing agent/engineer. Read it fully before writing code.

## Why

`pipeline.detector: geco2` runs the vendored few-shot exemplar detector
(`GECO2/`) as a drop-in replacement for Stage 1+2+3. It was trained on
FSC147, where the exemplar box and the query region always live in the
**same photo** at the **same scale**. AERO EYES instead pairs 3 close-up
reference photos (object fills most of the frame, high detail) against a
drone video where the object is tiny (~20–150px, low detail) — a domain
gap the model was never trained for.

The current pipeline patches this entirely at **inference time**, with
config knobs in `stage123_geco2.*` (see `configs/config.yaml` and
`aero_eyes/stages/stage123_geco2.py`): `scale_calibration` (needs an
*estimated* `expected_object_px` — see below for why that estimate isn't
reliably available), `background_mode`, `domain_calibration`,
`color_postfilter`. All of them are workarounds bolted onto a frozen,
unmodified checkpoint.

**Core constraint driving this plan**: at real deployment inference time
there is no ground truth, so **no reliable estimate of the target's
apparent size in the video exists ahead of time** (`scale_calibration`
currently requires guessing `expected_object_px`, which is fragile — see
the conversation history that produced this plan for the empirical trail
of tuning that heuristic). That means the plan must NOT rely on knowing
box size at inference. Two things in the current architecture depend on
box size and must both be addressed by training, not by a better guess:

1. **The explicit shape token** — `shape_or_objectness(box_hw)` in
   `GECO2/models/counter_infer.py`/`models/counter.py`. This must be
   **removed from the pipeline entirely**, both training and inference —
   `aero_eyes/config.py::Stage123Geco2Config.use_shape_token` already
   exists as a toggle for exactly this (added earlier this session as a
   diagnostic ablation; this plan promotes it from "diagnostic" to
   "required setting" for the finetuned checkpoint).
2. **The resolution/detail gap** — even with the shape token gone, the
   RoI-Align appearance token is still pooled from a reference photo that
   is *objectively higher-detail* than anything the object will ever show
   in the drone video (few real pixels of the object at 20–150px, vs. a
   crisp close-up). `scale_calibration`'s canvas-building was one way to
   narrow this gap, but it requires knowing the target scale in advance —
   which we just established is not reliably available. **This plan
   instead trains the model to be robust across a wide RANGE of detail
   levels** (domain randomization), rather than trying to precisely match
   one specific (unknown) target detail level.

## Data available

- **Ground truth**: `annotations (1).json` at repo root — schema documented
  in `aero_eyes/utils/io.py::load_gt` (list of `{video_id, annotations:
  [{bboxes: [{frame, x1, y1, x2, y2}]}]}`, absolute xyxy pixels, 0-based
  frames, absent frames simply omitted). GT is used here **only as the
  training loss target** (where is the object in the video frame) — never
  fed into building the reference-image exemplar, precisely because real
  inference will never have it either. Keeping training and inference
  symmetric on this point is the whole point of this redesign.
- **20 total videos**: **14 for training**, **6 held out for test** — the 6
  test videos are the ones already used throughout this project's tuning
  work (`BlackBox_0`, `BlackBox_1`, `CardboardBox_0`, `CardboardBox_1`,
  `LifeJacket_0`, `LifeJacket_1` — confirm this is still the exact set).
  **Do not train on these 6.**
- Per sample: 3 reference photos (`data/<sample>/object_images/`) + 1 drone
  video (`data/<sample>/*.mp4`) — same layout `_load_ref_images` /
  `_locate_video` in `stage123_geco2.py` already use.
- **Identify the 14 training video IDs** from `cfg.data.data_root` (all
  sample dirs minus the 6 test ones) before writing the dataset loader.

## Key architecture facts (established this session — verify before relying on them)

1. **Two model variants exist in `GECO2/models/`:**
   - `counter_infer.py::CNT` — inference-only, no `class_embed_aux`/
     `bbox_embed_aux`. This is what `aero_eyes/models/geco2_detector.py::
     GeCo2Detector` currently loads.
   - `models/counter.py::CNT` — the training variant, with aux heads.
     **Finetuning must use this class** (or a trimmed copy).
   - `GECO2/train.py` is the reference training entrypoint (`models/
     counter.py::build_model`, `models/matcher.py::build_matcher`,
     `utils/losses.py::SetCriterion`, `utils/data.py::FSC147DATASET` +
     `pad_collate`). Its per-batch loss recipe (Hungarian match → GIoU + L1
     box loss + centerness L2 loss, main + aux heads) is reusable as-is;
     only the **data feeding** needs to change.

2. **`adapt_features` (cross-attention) does not care whether K/V (exemplar
   tokens) came from the same forward pass as the image query.** This is
   the fact `aero_eyes/models/geco2_detector.py::encode_exemplars()` /
   `_forward_scores()` already exploit at **inference** time. Training
   needs the identical split (encode 3 reference photos independently,
   score the query frame separately), but with gradients enabled and using
   `models/counter.py::CNT`. Do not feed `models/counter.py::CNT.forward(x,
   bboxes)` directly — that signature assumes exemplar boxes are *inside*
   `x`.

3. **`use_shape_token=False` is mandatory for this plan, in BOTH training
   and inference.** `aero_eyes/models/geco2_detector.py::encode_exemplars()`
   already implements the branch (`if self.use_shape_token: ... else:
   main_tokens.append(exemplar.cpu())` — no `box_hw`/`shape_or_objectness`
   call at all in that branch). The training-mode forward wrapper (task 2
   below) must mirror this exactly: never compute or pass `box_hw` into
   `shape_or_objectness`. `shape_or_objectness`'s weights are therefore
   never touched by gradients in this plan — leave it frozen (equivalently:
   irrelevant, since it's simply not called).

4. **RoI-Align still needs *a* box for the reference photo — but only to
   crop the object from its background, not to encode a target scale.**
   MobileSAM already answers "where is the object in this reference photo"
   without needing to know how big it'll appear in any video
   (`stage123_geco2.segmentation`, unchanged). Use the **tight MobileSAM
   mask bbox as-is, in the reference photo's own native resolution** — do
   **not** run it through `_build_scale_calibrated_canvas` (that function
   requires `expected_object_px`, which this plan avoids entirely).

5. **Bridge the detail gap via training-time domain randomization, reusing
   the existing (previously "useless for scale") blur mechanism.** Recall
   from this session: `_apply_ref_downscale` / `ref_downscale_factor`
   (shrink-then-let-`resize_and_pad`-upscale-back) was proven to be a
   **no-op on final object size** on the model's canvas (the pre-shrink
   and `resize_and_pad`'s own re-normalization exactly cancel), but it
   **does** genuinely reduce image detail/sharpness (real information is
   lost in the downscale step, even though the canvas size doesn't
   change). That is *exactly* the primitive needed here: at each training
   step, apply this same shrink function to the (already tightly-cropped)
   reference image with a **randomly sampled** downscale factor from a
   wide range (e.g. uniform or log-uniform over roughly `[0.03, 1.0]`,
   covering everything from "barely any detail left" to "full crispness")
   — so the model sees the SAME object at many different detail levels
   across training and learns appearance matching that doesn't assume one
   specific detail level. Re-sample the factor independently per
   reference image, per training step (not fixed per-sample), to maximize
   effective augmentation diversity from only ~11–14 distinct objects.
   **This removes the need for `scale_calibration`/`expected_object_px` at
   inference entirely** — with the finetuned checkpoint, run inference
   with `scale_calibration.enabled: false`; the model has been trained to
   not need it.

6. **Target box format**: `train.py` divides GT boxes by `1024` before
   comparing to `outputs[idx]['pred_boxes']` (normalized xyxy from
   `boxes_with_scores`, see `GECO2/utils/box_ops.py`). AERO EYES GT boxes
   are absolute pixel xyxy in the *original* video frame — convert through
   the same `resize_and_pad` scale factor used for that frame's own canvas
   before normalizing by `image_size`. **Verify this conversion on a
   handful of known examples (plot predicted vs. GT box in pixel space)
   before trusting a training run** — a systematically-offset loss still
   produces plausible-looking decreasing loss curves.

7. **`aero_eyes/config.py::DataConfig.gt.one_object_per_video: true`** — at
   most one GT box per frame. Confirm `SetCriterion`/`models/matcher.py`
   handle the 1-target (and 0-target, for absent frames — decide whether
   to include absent frames in training at all, see Risks) case correctly.

## What to freeze / what to train

- **Frozen, always**: `model.backbone` (SAM2 Hiera). 14 training videos is
  nowhere near enough to safely touch a large pretrained vision backbone.
- **Frozen, unused**: `shape_or_objectness` — never invoked (fact 3 above).
  No need to explicitly freeze it (no gradient reaches it either way), but
  don't accidentally include it in the optimizer's parameter groups.
- **Finetune** (small learning rate, e.g. start from `GECO2/train.sh`'s
  `lr=1e-4`): `adapt_features` (`C_base` — prototype-attention +
  deformable self-attention + cross-scale merge), `class_embed` (+
  `class_embed_aux`), `bbox_embed` (+ `bbox_embed_aux`).
- **Open question, decide empirically**: `sam_prompt_encoder` (used for
  `image_pe` inside `adapt_features`). Default frozen; only unfreeze if
  validation metrics plateau with a specific reason to suspect positional
  encoding is the bottleneck.

## Concrete tasks

1. **New dataset loader** (e.g. `aero_eyes/models/geco2_finetune_data.py`
   or a standalone script under `scripts/`). Per training step, per
   sample: load the 3 reference photos, run MobileSAM (fact 4), crop to
   the tight mask bbox in **native resolution** (no scale-canvas), apply a
   **freshly-sampled** downscale/blur factor per reference image (fact 5),
   `use_shape_token=False` throughout (fact 3). Pair with one video frame
   + its GT box (converted per fact 6). Split the 14 training videos into
   an internal train/val split (e.g. ~11/3) — **do not** touch the 6
   held-out test videos during this phase.
2. **Training-mode forward wrapper**: mirror `GeCo2Detector.
   encode_exemplars()` (with `use_shape_token=False`, fact 3) +
   `_forward_scores()` (fact 2), built on `models/counter.py::CNT`,
   gradients enabled, returning `(outputs, ref_points, centerness,
   outputs_coord, aux)` in the shape `train.py`'s loop expects, so
   `utils.losses.SetCriterion` is reusable unmodified.
3. **Training loop**: single-GPU, non-distributed (the original `train.py`
   is SLURM/`DistributedDataParallel`-based — overkill here). Start from
   `GECO2/train.sh`'s hyperparameters (`lr=1e-4`, `backbone_lr=0`,
   `weight_decay=1e-5`, `max_grad_norm=0.1`) but drastically reduce epoch
   count (light finetune from an already-converged checkpoint, not 200
   epochs from scratch — start around 10–30 with early stopping on the
   internal validation split).
4. **Checkpoint handling**: load `CNTQG_multitrain_ca44.pth` as the
   starting point; **never overwrite it**. Save to a new file (e.g.
   `GECO2/CNTQG_aeroeyes_finetuned.pth`), selectable via an overridden
   `stage123_geco2.weights_path`.
5. **Inference config for the finetuned checkpoint** (this is the point of
   the whole redesign — write this down explicitly so it isn't lost):
   `stage123_geco2.use_shape_token: false`,
   `stage123_geco2.scale_calibration.enabled: false`. No
   `expected_object_px` estimate needed at all.
6. **Evaluation**: run the **existing, unmodified** AERO EYES pipeline
   (`aero_eyes.stages.run_all` → `aero_eyes.evaluate`) on the 6 held-out
   test videos with the config from task 5. Compare ST-IoU against the
   best heuristic-only result obtained on the same 6 videos this session
   (pull the actual current per-sample numbers from prior runs before
   treating any single number as "the bar to beat").

## Acceptance criterion

The finetuned checkpoint must **clearly exceed** the current heuristic
pipeline's ST-IoU on the 6 held-out test videos to justify the added
maintenance burden of a second checkpoint + training pipeline. Equally
important given the goal stated in "Why": it must do so **without needing
any per-deployment `expected_object_px` estimate** — a finetuned model
that ties or slightly beats the heuristic pipeline but needs no scale
guess at all is a meaningfully better outcome than a slightly higher
ST-IoU that still requires guessing a parameter you don't have in real
deployment. Report both dimensions, not just ST-IoU.

## Risks to actively monitor, not just note

- **Overfitting**: only ~11–14 distinct object identities. Watch
  train/val loss divergence every epoch; stop early on divergence. Also
  reuse `aero_eyes/utils/geometry.py::generate_synth_views` /
  `homography_warp` (already built for the legacy pipeline's
  `synthetic_viewpoint_aug`) as additional training-time augmentation on
  the reference photos (viewpoint, on top of the detail-level
  randomization in fact 5) to squeeze more effective diversity out of the
  limited object count.
- **Detail-randomization range is a real hyperparameter, not a formality**
  — too narrow a range and the model won't generalize to the actual
  deployment detail levels (defeats the purpose); too extreme (e.g.
  constantly training on near-unrecognizable blur) may slow convergence
  or degrade accuracy on the easier, higher-detail end. Sweep/validate the
  `[low, high]` downscale-factor range on the internal validation split
  rather than picking it once and assuming it's right.
- **Generalization loss to genuinely novel target categories**: a model
  finetuned on 14 objects may regress on categories unlike them. Flag this
  tradeoff back to the project owner explicitly — it's a product decision.
- **Coordinate-system bugs are the most likely silent failure mode** (fact
  6) — sanity-check visually (plot predicted vs. GT boxes on known
  training frames) before trusting loss-curve numbers alone.
- **Absent-frame handling**: decide explicitly whether training includes
  frames where the target is absent (0 GT boxes) to teach the "nothing
  here" case, or trains only on present frames (mirroring FSC147's
  always-present assumption, which is exactly what causes
  `score_threshold_abs`'s existing problems at inference — see
  `scripts/check_geco2_score_separation.py`). Given this plan's goal is
  reducing hand-tuned inference-time workarounds, deliberately including
  absent frames during training is likely worth the extra complexity —
  but this is a design choice to make explicitly, not default into.

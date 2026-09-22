# Current codebase state: how the "prototype" (fused few-shot reference embedding) is built

Read-only investigation notes. All citations are `file:line` against the working tree at the
time of writing (branch `test-geco2`). Purpose: ground a later prototype-construction
recommendation in exactly what exists today (including what's already been measured), not
generic advice or re-suggestions of things already tried.

## 0. Two structurally different "prototype" objects, one per pipeline

`cfg.pipeline.detector` selects between two pipelines that build and consume the exemplar
representation completely differently:

1. **Legacy Stage 1** (`aero_eyes/stages/stage1.py`) — one fused **1-D vector** per sample
   (`prototype.npz`), produced by a DINOv2/DINOv3/CLIP/SigLIP/ensemble encoder pooling each
   reference image to a single global CLS/pooled token, then fusing the 3 refs into one vector
   (mean/max/concat_then_pca). Consumed by Stage 3 (`stage3.py`) as a plain cosine dot product.
2. **GeCo2** (`aero_eyes/stages/stage123_geco2.py::build_exemplar_prototype` +
   `aero_eyes/models/geco2_detector.py::GeCo2Detector.encode_exemplars`) — a **token-set dict**
   (`geco2_prototype.pt`, keys `main`/`l1`/`l2`), one RoI-Align-pooled vector **per reference
   image per pyramid level** (never globally pooled across refs into one vector at build time).
   Consumed by GeCo2's own cross-attention (`adapt_features`), not cosine similarity, against a
   K/V sequence of all ref tokens at once.

A "how do we build a better prototype" recommendation needs to say which of these it targets —
the two have almost no shared code path below `apply_background_mode`/`crop_to_object`/
`mask_bbox` (both reuse the same `aero_eyes/utils/geometry.py` primitives).

---

## 1. Legacy Stage 1 pipeline (`aero_eyes/stages/stage1.py`)

**Takeaway:** the *only* steps enabled by default are MobileSAM segmentation + `mean_fill`
background replacement + a fixed 3-scale pyramid (1.0/0.75/0.5x) + mask-area-weighted mean
fusion + L2-normalize. `crop_to_object`, `aerial_sim`, `domain_calibration`, and
`multi_reference_embedding` are all **opt-in, off by default** — a config exactly as shipped
does none of the domain-gap-narrowing steps beyond segmentation itself.

### Findings — exact sequence (`run_stage1`, `stage1.py:44-307`)

1. **Load 3 ref images** (`stage1.py:62-78`).
2. **Segmentation** (`stage1.py:80-107`): `build_segmenter(seg_cfg, cfg)` if
   `stage1.segmentation.enabled` (default `True`, `config.py:83`); model defaults to
   `"mobilesam"` (`config.py:103`). Optional `center_crop_fallback` (default `False`,
   `config.py:168`) replaces an implausible-area mask with a center-box mask.
   `apply_background_mode(img, mask, seg_cfg.background_mode, seg_cfg.blur_sigma)`
   (`stage1.py:106`) is applied unconditionally whenever segmentation ran — default
   `background_mode="mean_fill"` (`config.py:155`, flat mean-color fill; own docstring at
   `config.py:144-146` flags this as "far outside what the backbone ever saw" vs. the
   `keep_real`/`blur` alternatives).
3. **`crop_to_object`** (`stage1.py:113-151`) — opt-in, default `False` (`config.py:401`).
   When on: crops each masked ref to its MobileSAM tight bbox + `crop_context_margin` (default
   `0.5`, `config.py:402`) *before* the eventual resize to `image_size` (224), so the object
   occupies a larger canvas fraction post-resize. Requires `segmentation.enabled`.
   `synth_masks` is kept separate from the ORIGINAL `masks` specifically so the later
   mask-area-weighted fusion (step 5) still reflects each ref's original segmentation
   confidence, not a post-crop ratio that would converge across refs (`stage1.py:123-131`
   comment).
4. **`aerial_sim`** (`stage1.py:153-159`, `_apply_aerial_sim` at `stage1.py:26-41`) — opt-in,
   default `enabled=False` (`config.py:351`), `downscale_factor=1.0`/`blur_ksize=0` (no-op even
   if enabled at defaults, `config.py:352-353`). When configured, shrinks-then-upscales
   (`INTER_AREA` down, `INTER_LINEAR` up) and optionally Gaussian-blurs, to destroy fine detail
   the drone's actual view wouldn't have.
5. **Multi-scale pyramid + feature extraction** (`stage1.py:161-201`, always on, not
   config-gated): each ref is resized to 1.0x/0.75x/0.5x (`stage1.py:174-180`), plus optional
   `synthetic_viewpoint_aug` views only under `accuracy.mode == "max_accuracy"`
   (`stage1.py:182-192`), then `extractor.extract(imgs)` and **averaged** across all these
   views into one `avg_feat` per ref (`stage1.py:196-201`) — this per-ref averaging happens
   *before* the cross-ref fusion in step 6.
6. **Fusion across the 3 refs** (`stage1.py:205-229`), `stage1.prototype.fusion` default
   `"mean"` (`config.py:342`):
   - `"mean"` — **mask-area-weighted** mean, not a plain average: each ref's weight is its own
     `mask.sum()/mask.size` (fraction of frame the segmenter thinks is foreground), normalized
     to sum to 1 (`stage1.py:208-217`). Rationale in the code comment: a mask covering more of
     the frame is treated as a more confident/reliable segmentation.
   - `"max"` — elementwise max across the 3 ref feature vectors (`stage1.py:218-219`).
   - `"concat_then_pca"` — flattens all 3 into one vector, PCA-fits on the 3 (as if 3 samples in
     feature-dim space), takes the first principal component (`stage1.py:220-227`).
7. **L2-normalize** (`stage1.py:231-239`), default `True` (`config.py:343`) — normalizes both
   the fused prototype and each per-ref vector.
8. **`domain_calibration`** (`stage1.py:241-283`, `DinoDomainCalibrationConfig`) — opt-in,
   default `enabled=False` (`config.py:376`). When on: samples `num_sample_frames` (default 5)
   raw video frames (evenly spaced via `np.linspace`, not GT-anchored), embeds them with the
   *same* extractor, averages to a `video_domain_mean`, then linearly blends
   `(1-strength)*prototype + strength*video_domain_mean` (default `strength=0.3`,
   `config.py:381`) and re-normalizes — applied to both the fused prototype and each per-ref
   vector. This shifts toward whole video **frames** (background-heavy, available before any
   matching), explicitly distinguished in its own docstring (`config.py:356-375`) from
   `stage3.dynamic_prototype` (shifts toward matched **candidate crops** instead — see §4).
9. **Write** `prototype.npz` (`stage1.py:285-307`) — `per_ref_features` are only saved when
   `accuracy.mode in ("cheap_boosters","max_accuracy")` **and**
   `accuracy.cheap_boosters.multi_reference_embedding` is true (`stage1.py:293-297`) — see §5.

### Defaults summary (what a stock run actually does)

| Knob | Default | Enabled-by-default? |
|---|---|---|
| `stage1.segmentation.enabled` (mobilesam) | `True` | Yes |
| `stage1.segmentation.background_mode` | `"mean_fill"` | Yes (flat color fill) |
| `stage1.segmentation.center_crop_fallback` | `False` | No |
| `stage1.crop_to_object` | `False` | No |
| `stage1.aerial_sim.enabled` | `False` | No |
| Multi-scale pyramid (1.0/0.75/0.5x) | — | **Always on**, not gated |
| `stage1.prototype.fusion` | `"mean"` (mask-weighted) | Yes |
| `stage1.prototype.l2_normalize` | `True` | Yes |
| `stage1.domain_calibration.enabled` | `False` | No |
| `accuracy.mode` (gates multi-ref saving) | `"baseline"` | multi-ref **off** by default |

---

## 2. GeCo2 pipeline (`stage123_geco2.py::build_exemplar_prototype`, `geco2_detector.py`)

**Takeaway:** GeCo2's prototype build is a **branch selector among 4 mutually-exclusive
strategies** (auto_scale_calibration > learned_scale_fusion > scale_calibration/crop_to_object/
ref_downscale > segmentation-only), all off/no-op by default except segmentation itself, and the
resulting tokens are consumed by **RoI-Align → cross-attention**, never global-average-pooled
and never cosine-matched inside GeCo2 itself (cosine only re-enters via the optional
`cosine_rescore` bridge back to the legacy Stage 3 machinery — not part of this investigation's
scope, but relevant context: `config.py:1949-1972`).

### 2a. `build_exemplar_prototype` control flow (`stage123_geco2.py:179-546`)

Branch order (first matching branch wins, each overrides the others with a warning if the
others' fields were also set non-default):

1. **`auto_scale_calibration.enabled`** (default `False`, `config.py:1779`) — Track A, see §2b.
2. **`learned_scale_fusion.enabled`** (default `False`) — Track B, needs a checkpoint with a
   trained `scale_fusion_gates` submodule (none trained yet per docs — see §3). Builds one
   exemplar entry per (ref, `candidate_factors`) and fuses via a learned, query-conditioned gate
   (`encode_exemplars_fused`, `geco2_detector.py:288-386`) instead of flat concatenation.
3. **`segmentation.enabled`** (default `True`, reuses `SegmentationConfig`, same defaults as
   Stage 1 — mobilesam, `mean_fill`) — the default path:
   - Tight `mask_bbox` per ref (`stage123_geco2.py:351`) — used as the RoI-Align pooling region
     regardless of what else runs; code comment explicitly says pooling the whole masked image
     instead "dilutes the exemplar token (empirically confirmed to matter)" (`stage123_geco2.py:
     346-350`).
   - **`scale_calibration.enabled`** (default `False`, `config.py:1662`) — if on, replaces the
     ref image with a synthetic canvas (`_build_scale_calibrated_canvas`, `stage123_geco2.py:
     62-124`) sized so the object occupies the same canvas fraction it's expected to occupy in
     the actual video frame (`expected_object_px`, no safe default — must be hand-estimated).
     This is the *only* mechanism that changes the object's apparent SIZE on the model's canvas;
     `ref_downscale_factor` provably cannot (see §2b/§3 — `resize_and_pad` always renormalizes
     the whole image's longer side, canceling out a uniform pre-shrink).
   - Else, **`crop_to_object`** (default `False`, `config.py:2473`) — same primitive as Stage 1's
     `crop_to_object`, crops to tight bbox + `crop_context_margin` (default `0.5`) before
     `resize_and_pad`.
   - Then **`ref_downscale_factor`** (default `1.0`, no-op, `config.py:2444`) or
     `ref_downscale_levels` (default `None`; when set, builds one exemplar entry per (ref,
     factor) — flat-concatenated into the K/V sequence, explicitly flagged as **never trained**
     this way — see §3's train/inference-mismatch finding) via `_apply_ref_downscale`
     (`stage123_geco2.py:37-59`) — this only changes blur/detail, not canvas-relative size.
4. **No segmentation** (`stage123_geco2.py:478-497`) — only `ref_downscale_factor`/
   `ref_downscale_levels` apply to the raw (unmasked) image; no crop/scale-calibration possible
   (needs a mask box).

Final step (skipped when auto_scale_calibration/learned_scale_fusion already built `prototype`
themselves): `detector.encode_exemplars(ref_imgs, ref_boxes=ref_boxes)`
(`stage123_geco2.py:525`).

Then, **`domain_calibration`** (`DomainCalibrationConfig`, default `enabled=False`,
`config.py:1738`) — same shape as Stage 1's version but operates on GeCo2's own per-scale
appearance tokens: samples video frames, computes `estimate_domain_shift` (mean
global-average-pooled backbone feature per scale, `geco2_detector.py:840-852`), then
`GeCo2Detector.calibrate_prototype` (`geco2_detector.py:854-884`) mean-shifts only the
*appearance* token indices (never shape-token indices) by `strength * (video_mean - ref_mean)`,
default `strength=1.0` (fully replace — much more aggressive default than Stage 1's DINOv2
analog, which defaults to `strength=0.3`).

### 2b. `auto_scale_calibration` (Track A) — 2-D grid search (`geco2_auto_scale.py`)

`AutoScaleCalibrationConfig` (`config.py:1743-1827`), default `enabled=False`. Orchestrated by
`build_auto_scaled_prototype` (`geco2_auto_scale.py:135-234`), called from
`stage123_geco2.py:229-265`:

1. **Candidate grid**: `candidate_crop_margins` (default `[0.5, 1.0, 2.0, 4.0]`, controls object
   size on canvas via `crop_to_object`) × `candidate_downscale_factors` (default
   `[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03]`, log-spaced, controls blur/detail via
   `_apply_ref_downscale`) — 24 combos by default, evenly downsampled to `max_candidates=12`
   if the product exceeds it (`build_candidate_grid`, `geco2_auto_scale.py:42-57`).
2. Per candidate: crop+downscale all 3 refs, `detector.encode_exemplars(...)` **once**
   (reused for scoring and the final blend — `geco2_auto_scale.py:182-194`).
3. **Quality score** per candidate: `quality_metric` default `"auto"` — uses `gt_iou` (mean IoU
   of top-1 predicted box vs GT over sampled present frames, `score_candidate_gt_iou`,
   `geco2_auto_scale.py:76-87`) when GT exists for this `sample_id`, else falls back to
   `self_supervised_margin` (`(max-mean)/std` peakiness of the raw score map over
   `num_probe_frames`=12 sampled frames, `score_candidate_self_supervised`,
   `geco2_auto_scale.py:90-106` — explicitly flagged as a heuristic that "a confidently-wrong
   high-scoring background patch can also score high" on).
4. **Blend**: `select_weights` (`geco2_auto_scale.py:109-122`) — `selection_mode="soft"`
   (default) softmaxes z-scored qualities at `temperature=0.5`; `"hard"` one-hot-selects the
   best. Appearance tokens are blended as a weighted sum per scale
   (`geco2_auto_scale.py:207-209`); **shape tokens are never blended** — the best-scoring
   candidate's own (w,h)-derived shape token is used unblended, because averaging multiple
   candidates' box sizes would synthesize a (w,h) that matches no real candidate
   (`geco2_auto_scale.py:211-222`).
5. Writes `geco2_auto_scale_calibration.json` debug file (candidates, qualities, weights,
   metric used, best candidate) — `stage123_geco2.py:258-265`.

Cost, per the guide (`docs/GECO2_auto_scale_calibration_guide.md:87-89`): up to ~144 extra
query-side + 36 extra ref-side backbone forward passes per sample at default grid/probe sizes,
**one-time**, folded into the cached `geco2_prototype.pt`.

**Explicitly flagged NOT YET VALIDATED** against a manually-tuned fixed baseline
(`config.py:1775-1777`, guide file throughout) — `scripts/compare_auto_scale_vs_fixed.py` is the
prescribed comparison script but its output wasn't found in the docs read for this investigation
(no results file for it exists alongside the scale-calibration results docs).

### 2c. `encode_exemplars` — how the prototype is actually consumed (`geco2_detector.py:189-278`)

- Runs the Hiera backbone independently on each ref image (`_load_and_pad`, uses
  `GECO2/utils/data.py::resize_and_pad` with `zero_shot=True` always — `geco2_detector.py:
  173-183`).
- For each ref, **`torchvision.ops.roi_align(..., output_size=1)`** pools the given box (tight
  mask bbox, or whole image if no box) to exactly **one vector per ref per pyramid level**
  (`main` from `vision_features`, `l1`/`l2` from `backbone_fpn[0]`/`[1]`,
  `geco2_detector.py:242-261`) — this is RoI-Align-to-1-vector, **not** a global-average-pool
  (contrast with the legacy DINOv2 path's CLS-token/global pooling in §1, and with
  `frame_domain_embedding`'s own explicit global-average-pool used only for domain-shift
  estimation, `geco2_detector.py:825-838`).
- Optional shape token: `use_shape_token` (default `True`, `config.py:2491`) appends a learned
  `shape_or_objectness(w,h)` token per ref (`geco2_detector.py:263-268`) — a second, independent
  place box size feeds the prototype (also feeds the RoI-Align pooling region itself; disabling
  `use_shape_token` does **not** fix a wrong-scaled box, per `config.py:2480-2490`'s own
  ablation-toggle docstring).
- Returns `{"main": [1,N,D], "l1": [1,N,D], "l2": [1,N,D]}` where `N` = num_refs ×
  (2 if shape token else 1) × (num scale-calibration/ref_downscale_levels entries if multi-scale
  is active) — a **token sequence**, never collapsed to one vector.
- Matching (`_forward_scores`, `geco2_detector.py:392-429`) feeds this whole token dict as K/V
  into `m.adapt_features` (cross-attention) against the query frame's own backbone features —
  the "prototype" is consumed structurally, not via a dot product against a pooled vector.

---

## 3. Existing internal findings on prototype quality (already-measured numbers)

**Takeaway:** the codebase already has direct, real-footage evidence that (a) no single fixed
reference size/detail level works across videos — optimal ratio to true object size ranges
~1.1x–2.9x depending on the object, motivating Track A/B rather than a better global constant;
(b) raw cosine similarity's TP/FP separation ceiling on this project's own footage is weak
(~0.335–0.38 regardless of identity); (c) `auto_scale_calibration` itself has no committed
`compare_auto_scale_vs_fixed.py` results yet, so its actual benefit over hand-tuning is
unverified as of this doc.

### Findings — scale/crop-margin sweeps (checkpoint NOT finetuned, `expected_object_px` axis)

- `docs/GECO2_baseline_scale_calibration_results.md` — 6 videos (BlackBox/CardboardBox/
  LifeJacket, ×2 each), `scale_calibration` swept across 6 `expected_object_px` values, 2
  confidence modes (default threshold vs. `conf=0.0` unfiltered):
  - **Non-monotonic**: peak Mean ST-IoU at `75×51` (**0.4397** default-conf, F1@IoU0.5=**0.690**,
    F1@IoU0.3=**0.710**), degrading at both smaller sizes (22×17 → 0.1460) and the largest tested
    size (90×113 → drops to 0.2391).
  - At the largest size (90×113): **precision peaks (0.735) but recall collapses (0.349)** — a
    qualitatively different failure mode (detection-collapse, not localization error) from small
    sizes, where recall stays high (0.74–0.87) but IoU-gated recall drops (0.38–0.52) —
    i.e. small ref sizes mostly cause **mislocalization**, oversized ref sizes cause **missed
    detections outright** (`GECO2_baseline_scale_calibration_results.md:49-76`).
  - Per-video optimal size **never equals the true object size** — always **1.1x–1.9x** larger
    (`GECO2_baseline_scale_calibration_results.md:114-133`); `BlackBox_0/1` (true size 52–78px,
    the two largest-object videos in this set) never reached their peak within the tested range
    at all — ST-IoU was still monotonically increasing at the largest tested point (90×113).
  - `conf=0.0` (unfiltered) is uniformly ~2-3x worse F1 than the default confidence threshold at
    every tested size — confirms the score itself carries real signal, independent of the
    scale-calibration question (`GECO2_baseline_scale_calibration_results.md:80-82`).
- `docs/GECO2_baseline_scale_calibration_results_set2.md` — a **disjoint** 10-video set
  (Helmet/IDCard/Motorbike/Person2/Wallet), same original checkpoint, `conf=0.0` only (no
  default-threshold run recorded, so absolute numbers aren't directly comparable to set 1):
  - Opposite trend shape from set 1: ST-IoU **monotonically increasing** through the whole
    tested range (24×31 → 90×113, peak **0.1188** at 90×113, still rising, not yet plateaued).
  - `Motorbike_0` fails at every tested size (ST-IoU 0.003–0.021) — attributed to an unusually
    tall/narrow bbox aspect ratio (55×100px, ~1:1.8) not represented by any of the 5 fixed
    `[w,h]` test points, distinct from `Motorbike_1` (58×64, near-square) which improves sharply
    with size (up to 0.3515 at 90×113) — flagged as an aspect-ratio mismatch problem, not a pure
    size mismatch (`GECO2_baseline_scale_calibration_results_set2.md:91-93`). Note
    `crop_to_object` (reused by `auto_scale_calibration`) preserves the *original* box aspect
    ratio when cropping (margin scaled independently by `bw`/`bh`), so this specific failure
    mode is already structurally avoided by the crop-based mechanisms, per
    `docs/GECO2_scale_domain_gap_plan.md:34`.
  - Combined conclusion across both sets (`docs/GECO2_scale_domain_gap_plan.md:32`): **no single
    fixed reference size wins across all 16 tested videos**; the optimal size-to-true-size ratio
    itself varies by object group (~1.1x–2.9x) — this is the direct empirical justification for
    building a per-sample-adaptive mechanism (Track A/B) instead of searching for a better
    global constant.

### Findings — finetuning already tried, did not fully solve the problem

- `docs/GECO2_scale_domain_gap_plan.md:30-31` — a prior finetune (`docs/GECO2_FINETUNE_PLAN.md`)
  already trained GeCo2 with per-ref, per-step random `ref_downscale_factor` in
  `[0.03, 1.0]` (domain randomization), intending to make the model **fully invariant** to
  reference detail level so `ref_downscale_factor` would never need per-deployment tuning.
  **Result per the user's own report**: the finetuned model still needs manual
  `ref_downscale_factor` tuning, and the optimal value still varies per example — domain
  randomization made the model "more robust across a wide range" but did **not** achieve full
  invariance. This is the stated motivation for Track B (`ScaleFusionGate`, learned per-query
  scale selection) rather than repeating more random-scale augmentation.
- `docs/GECO2_scale_domain_gap_plan.md:23` — `ref_downscale_levels` (multi-scale flat
  concatenation at inference) has a known **train/inference mismatch**: the finetune data
  sampler (`geco2_finetune_data.py::sample_ref_downscale_factor`) only ever samples **one**
  random factor per ref per training step, never multiple simultaneous scale variants of the
  same ref — so a checkpoint has never seen the token-count/pattern that
  `ref_downscale_levels>1` or `scale_calibration.multi_scale_mode="all"` produces at inference.
  This is exactly the gap Track B's `num_ref_scale_variants` training change targets.
- `docs/GECO2_scale_domain_gap_plan.md:39` — reading `GECO2/utils/data.py::resize_and_pad`
  found the original paper's own eval convention normalizes exemplar boxes to ~80px average on
  a 1024 canvas in non-`zero_shot`/non-`train` mode — but this project's `GeCo2Detector` always
  calls with `zero_shot=True`, so **neither reference images nor query frames in this pipeline
  are ever subject to that 80px convention**; it's confirmed not usable as a universal constant
  substitute for `scale_calibration` anyway, since the true optimal ratio is object-dependent
  (confirmed by the two results docs above), not a fixed convention.

### Findings — cosine similarity ceiling / precision bottleneck (legacy/cross-check path)

- `docs/GECO2_precision_improvements_plan.md:13` and referenced in
  `docs/GECO2_precision_techniques_reference.md:41` — on this project's own footage, **raw
  cosine similarity's ceiling sits around 0.335–0.38 regardless of TP/FP identity** — i.e. even
  genuine matches don't score much higher than confusers under the current DINOv2 setup at this
  domain gap. This is the single most concrete existing number bearing on "is the fused
  prototype's embedding itself the bottleneck" (as opposed to the downstream
  threshold/verification mechanism).
- `docs/GECO2_precision_improvements_plan.md:7-9` — best validated real-footage baseline on
  sample `IDCard_0`: `stage3.adaptive_threshold_online` (causal z-score) reaches **F1=0.706,
  P=0.646, R=0.778**, beating the whole-video batch adaptive threshold (F1=0.684). Standalone
  per-keyframe cluster verification (`verification_method="cluster"`) **massively
  underperforms** both (F1=0.266) because most keyframes in a single-object tracking video
  contain zero real target instances — a different assumption from DAVE's own FSC147 benchmark
  this technique was ported from. `cluster_secondary_filter`'s unguarded "trusted window" did
  **not** meaningfully improve precision, suspected root cause: it admits every
  threshold-passing candidate with no corroboration gate, so one borderline FP can poison the
  window as a "trusted anchor" for later frames.
- These numbers are all about **downstream matching/verification**, not prototype construction
  per se — but they establish the operating context: precision is capped well below what a
  cleaner embedding would presumably allow, and the open diagnostic question flagged in the plan
  doc (`docs/GECO2_precision_improvements_plan.md:13`) — whether FPs are a **small number of
  recurring confusers** (systematic, fixable downstream) or **diffuse/scattered** (evidence the
  embedding itself, i.e. what the prototype is built from, is the bottleneck) — was **not yet
  answered** as of the plan doc; `scripts/diagnose_verification_errors.py` was specified to
  answer it (Phase 0) but this investigation did not find a results writeup from running it.

### Gap: no committed A/B numbers found for `auto_scale_calibration` itself, or for
`crop_to_object`/`aerial_sim`/`domain_calibration` in isolation

None of the 5 docs read for this investigation contain an actual results table for
`compare_auto_scale_vs_fixed.py` (the auto-calibration guide only describes how to run it,
`docs/GECO2_auto_scale_calibration_guide.md:71-85`), nor any A/B numbers isolating
`stage1.crop_to_object`, `stage1.aerial_sim`, `stage1.domain_calibration`, or
`stage123_geco2.domain_calibration` on their own (all are described in `config.py` docstrings as
opt-in / "NOT YET VALIDATED" but this investigation found no matching results doc for them — the
only *quantified* sweep evidence in the docs is the `expected_object_px`/`scale_calibration`
sweeps in §3 above, which vary object **size**, not detail/blur or background-fill choice).

---

## 4. Online/dynamic prototype adaptation (context: an existing alternative to a better one-time build)

**Takeaway:** two independently-implemented "adapt after the fact" mechanisms already exist —
one batch/offline (legacy Stage 3, whole-video 2-pass), one causal/online (GeCo2's own) — both
disabled by default, both explicitly flagged as separate from and complementary to Stage 1/
GeCo2's one-time domain_calibration (which shifts toward raw video *frames*, not matched
*candidate crops*).

### Findings

- **`stage3.dynamic_prototype`** (`DynamicPrototypeConfig`, `config.py:459-489`, default
  `enabled=False`) — implemented in `run_dynamic_prototype_rounds`
  (`stage3.py:516-618`): batch, 2-pass — after an initial match, candidates scoring above an
  *adaptive percentile* of this sample's own score distribution (`high_conf_percentile`, default
  90th, with an absolute floor `high_conf_abs_floor=0.15`) are averaged and blended into the
  prototype (`alpha`-weighted mean-shift, default `alpha=0.3`) or appended as an extra per-ref
  vector when multi-ref pooling is active (`stage3.py:606-612`), then all candidates are
  re-scored — repeated for `rounds` (default 2) passes. Needs the *whole video's* candidate
  score distribution up front, so it's explicitly incompatible with (and auto-skipped under)
  `verification_method="cluster"` and `adaptive_threshold_online` (`stage3.py:944-954`).
  `require_diverse_picks` (default `False`) can additionally require the high-confidence picks
  span `min_frame_span` frames before trusting a round, to avoid updating from a few
  near-duplicate frames.
- **`stage123_geco2.dynamic_prototype`** (`Geco2DynamicPrototypeConfig`, `config.py:2198-2287`,
  default `enabled=False`) — implemented by `GeCo2DynamicPrototypeTracker`
  (`geco2_detector.py:919-1728`): **causal/online**, appends new exemplar tokens to the K/V
  sequence *as the video is processed* (GeCo2's cross-attention already treats the prototype as
  an arbitrary-length token sequence, so appending needs no architecture change), using a
  sliding window (`max_tokens=5`, FIFO eviction of oldest *appended* token — original ref tokens
  are never evicted) instead of a whole-video percentile cut, because the causal single-pass
  keyframe loop has no "whole video" distribution to compute a percentile from until too late.
  A candidate is only appended once **both** (1) `min_consecutive_hits` (default 2) consecutive
  keyframes' best box spatially agree (IoU ≥ `consecutive_hits_iou`, default 0.5) **and** (2) a
  cosine cross-check (`cross_check_source`: `"feature_extractor"` default, reusing
  `stage1.feature_extractor`/`prototype.npz` as an independent signal, or `"hiera"`, GeCo2's own
  backbone — explicitly flagged as *not independent*, correlated with GeCo2's own score) clears
  `cross_check_threshold`. This dual-gate exists because GeCo2's own score is relative-only
  (no absolute "this is definitely the target" guarantee) — see `config.py:2219-2234`.
- Both are explicitly distinguished in code comments from `domain_calibration`
  (§1/§2a): domain_calibration shifts toward the video's own **mean frame appearance**
  (background-heavy, available immediately, before any matching has happened);
  dynamic_prototype shifts toward **matched candidate crops** (object-focused, only available
  once some plausible hits exist). `DinoDomainCalibrationConfig`'s own docstring
  (`config.py:364-371`) states they "can be used together... or on its own."
- Both are flagged **NOT YET VALIDATED** in their own docstrings
  (`config.py:2236-2238` for the GeCo2 version) — no A/B numbers for either were found in the 5
  docs read for this investigation.

---

## 5. Multi-reference handling: fused-into-one-vector (default) vs. score-pooling (opt-in)

**Takeaway:** the default behavior in **both** pipelines is to collapse the 3 reference images
into a single fused prototype (weighted mean, §1/§2) and score candidates against that one
vector. A separate opt-in path — `multi_reference_embedding` — keeps the 3 per-ref embeddings
**separate through matching** and pools per-reference *similarity scores* instead of pooling
*embeddings* upstream. This only applies to the legacy DINOv2-family cosine path (Stage 3); it
has no equivalent inside GeCo2's own cross-attention matching (GeCo2 always attends over all ref
tokens as one K/V set regardless of this setting — that's a structurally different kind of
"multi-reference," §2c).

### Findings

- **Default is fused, single-vector**: `AccuracyConfig.mode` defaults to `"baseline"`
  (`config.py:1627`), under which `stage1.py:293-297`'s `save_per_ref` condition is `False` — no
  `per_ref_features` are even written to `prototype.npz`, and `stage3.py:851-855`'s
  `use_multi_ref` computes to `False` regardless of `CheapBoostersConfig` values (mode gate takes
  precedence). So out of the box, matching is always against the one fused prototype vector.
- **`accuracy.cheap_boosters.multi_reference_embedding`** (`CheapBoostersConfig`,
  `config.py:1602-1618`) defaults to `True` **within** the `CheapBoostersConfig` model itself —
  but only takes effect when `accuracy.mode` is explicitly set to `"cheap_boosters"` or
  `"max_accuracy"` (`config.py:1627`, `stage3.py:851-853`). When active:
  - Stage 1 saves all 3 per-ref feature vectors alongside the fused prototype
    (`stage1.py:293-303`, `write_prototype(..., per_ref_features=...)`).
  - Stage 3 scores each candidate against **each** of the 3 (or more, once dynamic_prototype has
    appended extras) per-ref vectors independently (`sims_per_ref`, `stage3.py:918-921`), then
    pools the resulting per-ref similarity arrays via `_pool_sims`
    (`stage3.py:91-105`) per `multi_ref_pooling` (`config.py:1618`):
    - `"mean"` (default) — averages the 3 per-ref scores; a candidate matching one ref view very
      well but the other two poorly gets diluted (`config.py:1609-1613`).
    - `"max"` — takes the single best-matching ref's score; a genuinely good match from one
      well-aligned viewing angle isn't dragged down by refs shot from a different angle/lighting
      (`config.py:1614-1618`).
  - This is consumed downstream by several other mechanisms keyed off the same
    `use_multi_ref`/`per_ref_features`: `run_dynamic_prototype_rounds` appends new vectors to
    `per_ref_features` instead of blending into one prototype when multi-ref is active
    (`stage3.py:606-612`); `stage4`'s `cosine_arbitration.pooling` (`config.py:1209-1218`) can
    match Stage 3's own pooling choice (`"match_stage3"`) for consistency at that later stage;
    `GeCo2DynamicPrototypeTracker`'s `cross_check_source="feature_extractor"` reuses these same
    per-ref vectors as its cosine cross-check signal (`geco2_detector.py:933-949`); and
    `cross_check_threshold_self_calibrate` (`config.py:2308-2327`) computes the **min pairwise
    cosine among the 3 ref images' own embeddings** as a per-sample-calibrated threshold floor,
    only available when per-ref vectors were saved.
  - `adaptive_threshold_anchor_to_original_refs` (`config.py:911-931`) exists specifically
    because multi-ref max-pooling + dynamic_prototype's appended narrow extra refs can inflate
    the mean/std used for adaptive-threshold computation, "moving the goalpost" for every
    candidate — an interaction bug between multi-ref pooling and adaptive thresholding that this
    flag (opt-in, off by default) works around by computing threshold statistics from the
    pre-dynamic-prototype distribution while still accepting on the full pooled distribution.
- **No default/recommended guidance found for `multi_reference_embedding` itself** — no results
  doc among the 5 read compares `multi_reference_embedding=True` (score-pooling) against the
  default fused-single-vector baseline; `multi_ref_pooling="mean"` is only described as
  "reproduces original (pre-arbitration) behavior, unchanged" (`config.py:1215-1218`), not as an
  empirically-preferred choice. This is a genuine gap for a report writer: the mechanism exists
  and is wired through several downstream consumers, but its own marginal value over a single
  fused prototype vector is unmeasured in the docs available to this investigation.
- **GeCo2 has no analog switch** — it always keeps refs (and, when multi-scale mechanisms are
  active, multiple entries per ref) as separate tokens in the K/V sequence by construction
  (§2c); there's no "collapse to 1 vector" vs. "keep N separate" config choice inside GeCo2's own
  path the way there is for the legacy DINOv2 cosine path.

---

## Appendix: file map for the report writer

| Concern | File |
|---|---|
| Legacy Stage 1 prototype build (full sequence) | `aero_eyes/stages/stage1.py` |
| Stage 1 config defaults | `aero_eyes/config.py:82-402` (`SegmentationConfig`, `PrototypeConfig`, `AerialSimConfig`, `DinoDomainCalibrationConfig`, `Stage1Config`) |
| GeCo2 prototype build (branch selector) | `aero_eyes/stages/stage123_geco2.py:179-546` (`build_exemplar_prototype`) |
| GeCo2 auto-scale-calibration grid search | `aero_eyes/models/geco2_auto_scale.py` |
| GeCo2 exemplar encoding (RoI-Align, consumption) | `aero_eyes/models/geco2_detector.py:189-278` (`encode_exemplars`), `:392-429` (`_forward_scores`, cross-attention consumption) |
| GeCo2 domain calibration (feature-space mean-shift) | `aero_eyes/models/geco2_detector.py:824-884` |
| GeCo2 config defaults | `aero_eyes/config.py:1647-2511` (`ScaleCalibrationConfig`, `DomainCalibrationConfig`, `AutoScaleCalibrationConfig`, `Geco2LearnedScaleFusionConfig`, `Geco2DynamicPrototypeConfig`, `Stage123Geco2Config`) |
| Legacy Stage 3 cosine matching + dynamic_prototype | `aero_eyes/stages/stage3.py` (`_score_against_ref` 57-88, `_pool_sims` 91-105, `run_dynamic_prototype_rounds` 516-618, main scoring 850-973) |
| GeCo2 online dynamic-prototype tracker | `aero_eyes/models/geco2_detector.py:919-1728` (`GeCo2DynamicPrototypeTracker`) |
| Multi-reference config surface | `aero_eyes/config.py:1602-1618` (`CheapBoostersConfig`), `:1626-1629` (`AccuracyConfig.mode` gate) |
| Shared geometry primitives (crop/mask/background) | `aero_eyes/utils/geometry.py` (`crop_to_object:254`, `apply_background_mode:219`, `mask_bbox:181`, `center_box_mask:387`) |
| Scale/size sweep results (16 videos, original checkpoint) | `docs/GECO2_baseline_scale_calibration_results.md`, `docs/GECO2_baseline_scale_calibration_results_set2.md` |
| Track A/B design rationale + train/inference-mismatch findings | `docs/GECO2_scale_domain_gap_plan.md` |
| `auto_scale_calibration` usage guide + cost/risk notes | `docs/GECO2_auto_scale_calibration_guide.md` |
| Precision bottleneck findings (cosine ceiling, F1/P/R numbers) | `docs/GECO2_precision_improvements_plan.md` |

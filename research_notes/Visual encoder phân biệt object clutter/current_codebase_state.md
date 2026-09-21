# Current codebase state: feature extractor + cosine-matching pipeline (aero_eyes)

Read-only investigation notes. All citations are `file:line` against the working tree at the
time of writing (branch `test-geco2`). Purpose: ground a later integration recommendation in
what actually exists today, not generic advice.

## 0. Two parallel pipelines exist — know which one a recommendation targets

The repo currently runs **two structurally different detector pipelines**, selected by
`cfg.pipeline.detector`:

1. **Legacy Stage 1/2/3/4** (`dinov2`/etc. as the *sole* detector+matcher): Stage 1 builds a
   DINOv2/DINOv3/CLIP/SigLIP/ensemble prototype from 3 reference photos
   (`aero_eyes/stages/stage1.py`), Stage 2 proposes class-agnostic boxes (YOLO/FastSAM + SAHI),
   Stage 3 (`aero_eyes/stages/stage3.py`) cosine-matches every candidate crop's embedding
   against the prototype.
2. **GeCo2** (`pipeline.detector == "geco2"`, `aero_eyes/models/geco2_detector.py`): a few-shot
   exemplar counter/detector (vendored GECO2 repo) whose own Hiera backbone + cross-attention
   *replaces* Stages 1+2+3 end-to-end (`GeCo2Detector.encode_exemplars` / `detect_frame`).
   Optionally, `stage123_geco2.cosine_rescore.enabled=true`
   (`aero_eyes/config.py:1949-1972`, `configs/config.yaml:802-808`) re-inserts the legacy
   DINOv2-family Stage 3 as an *independent second opinion* on top of GeCo2's own score — this
   is the only place the two pipelines' scores are combined.

A "swap the encoder" recommendation needs to say which of these it targets:
`stage1.feature_extractor.*` (affects both the legacy pipeline and, when `cosine_rescore` is on,
the GeCo2 pipeline's cross-check), or GeCo2's own Hiera backbone (`GeCo2Detector`, a vendored,
不-swappable third-party model — see §5).

---

## 1. Feature extractor implementation

**Takeaway:** the swappable DINOv2/DINOv3/CLIP/SigLIP/ensemble encoders live entirely in
`aero_eyes/models/features.py`, all behind one factory function and one common two-method
interface (`extract`, `extract_crops`). Pooling is **global CLS-token / pooled-output only** —
no patch-token, no multi-scale, no GeM/attention pooling anywhere in this file. GeCo2's own
backbone (a separate model, `geco2_detector.py`) uses RoI-Align pooling instead, described
separately in §1b.

### Findings

- `aero_eyes/models/features.py:1-22` — module docstring enumerating the 5 supported models and
  their output dims; states "All extractors return L2-normalized float32 feature vectors."
- **DINOv2** — `DINOv2FeatureExtractor` class, `features.py:66-136`.
  - Loads via `torch.hub.load("facebookresearch/dinov2", ...)` first, falls back to HF
    `AutoModel` (`features.py:84-106`).
  - Pooling: `.extract()` at `features.py:108-121` — for the HF path, takes
    `last_hidden_state[:, 0]` (the **CLS token**, line 117); for the torch.hub path, the raw
    model call already returns the pooled CLS-equivalent output (line 119). No patch tokens are
    ever read.
  - `dinov2_use_registers` (line 71/85) swaps to the "with registers" variant — same output dim,
    same CLS-token extraction path, purely a cleaner-features flag per Meta's own ablations.
  - Variant→dim table: `_DIMS = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}`
    (`features.py:129`).
- **DINOv3** — `DINOv3FeatureExtractor`, `features.py:143-280`.
  - `source="huggingface"` (default): `AutoImageProcessor` + `AutoModel`, pooling via
    `.pooler_output` (`features.py:266`) — HF's own pooled head, not a manual CLS-index slice.
  - `source="kaggle"`: raw Meta checkpoint via `kagglehub` + `torch.hub.load(..., weights=...)`
    (`features.py:218-244`), same manual-tensor-in/tensor-out convention as DINOv2's torch.hub
    path (`features.py:250-259`).
  - `dinov3_pretrain_dataset` (`"lvd1689m"` natural-image vs `"sat493m"` satellite-imagery) is a
    **weight-selection field, not an architecture change** — same class either way
    (`features.py:146-161`, `_ARCHS`/`_PRETRAIN_DATASETS` at `features.py:177-178`).
  - Dims: `_DIMS = {"vits16": 384, "vitb16": 768, "vitl16": 1024}` (`features.py:179`).
- **CLIP** — `CLIPFeatureExtractor`, `features.py:287-343`. Pooling: calls
  `model.vision_model(...).pooler_output` then `model.visual_projection(...)` directly
  (`features.py:328-329`) rather than `get_image_features()`, with a comment explaining this
  works around a `transformers` API regression that made `get_image_features()` sometimes
  return a `ModelOutput` instead of a plain tensor. Dims: 512 (`vit-b/32`) / 768 (`vit-l/14`)
  (`features.py:339-340`).
- **SigLIP** — `SiglipFeatureExtractor`, `features.py:350-400`. Vision-only
  (`SiglipVisionModel`, no text tower loaded — `features.py:373`), pooling via
  `.pooler_output` (`features.py:386`). Dims: 768/1024/1152 for base/large/so400m
  (`features.py:358`).
- **Ensemble** — `EnsembleFeatureExtractor`, `features.py:407-468`. Concatenates a DINO-family
  extractor (DINOv2 default, or DINOv3 via `ensemble_dino_model`) with CLIP, then re-normalizes
  the concatenated vector (`features.py:448-455`): `dim = dino_dim + clip_dim` (e.g.
  768+512=1280 for `vitb14`+`vit-b/32`). `dino_model="dinov3"` path is explicitly flagged
  "NOT YET VALIDATED" (`features.py:414-419`, `config.py:255-260`) — proposed specifically to
  test whether CLIP's semantic/categorical objective helps distinguish real objects from
  texturally-similar background clutter that DINO's pure self-supervised texture-clustering
  objective confuses with the target.
- **Wrapper decorators** (both optional, both transparently drop-in per the same
  `extract`/`extract_crops`/`_dim` interface):
  - `MaskedCropFeatureExtractor` (`features.py:476-529`) — runs the *candidate crop* through the
    same segmentation+background-fill primitive Stage 1 already applies to reference photos,
    before embedding. Off by default (`candidate_background_masking.enabled=false`); flagged
    "NOT YET VALIDATED", with a real cost warning (one segmentation inference call per candidate
    crop per keyframe, no batched path — `features.py:280-288` in `config.py`).
  - `ProjectedFeatureExtractor` (`features.py:536-583`) — applies a small trained linear/MLP
    head on top of the frozen backbone output (`scripts/train_projection_head.py` trains it).
    Off by default; dimension-checked at construction against the base extractor's own
    `_feature_dim()` (`features.py:552-557`), fails loudly on mismatch.
- **Factory**: `build_feature_extractor(cfg)`, `features.py:589-660` — single entry point every
  caller (Stage 1 prototype build, Stage 3 candidate matching, Stage 4 verify/re-detect,
  `GeCo2DynamicPrototypeTracker`'s cross-check) goes through. Dispatches on
  `cfg.stage1.feature_extractor.model` (`features.py:594-637`), then optionally wraps with
  `MaskedCropFeatureExtractor` (`features.py:639-648`) and/or `ProjectedFeatureExtractor`
  (`features.py:650-659`).
- Preprocessing: DINOv2/DINOv3-kaggle use a hand-rolled ImageNet-normalize + PIL bicubic resize
  to `image_size` (default 224) — `_preprocess_dino`, `features.py:48-55`. DINOv3-huggingface/
  CLIP/SigLIP use each model family's own HF `*Processor`/`*ImageProcessor`
  (`features.py:264-268`, `321-330`, `383-387`) — i.e. preprocessing is **not uniform** across
  encoders; each one bakes in its own official normalization/resize convention.

### 1b. GeCo2's own encoder (separate from the above, not swappable via config)

- `aero_eyes/models/geco2_detector.py:83-166` — `GeCo2Detector` loads the vendored GECO2 repo's
  `CNT` model (a Hiera backbone + cross-attention adapter), built via `build_model(args)`
  (`geco2_detector.py:90,123`) and a fixed checkpoint (`stage123_geco2.weights_path`).
- Pooling for exemplars: **RoI-Align**, not CLS/global-average — `encode_exemplars`
  (`geco2_detector.py:189-278`) runs the Hiera backbone on each reference image, then
  `torchvision.ops.roi_align` over the (given or whole-image) exemplar box at 3 pyramid levels
  (`main`/`l1`/`l2`, `geco2_detector.py:242-261`), optionally concatenated with a learned
  "shape" token from `(w,h)` (`use_shape_token`, `geco2_detector.py:263-268`).
  `frame_domain_embedding` (`geco2_detector.py:824-838`) is the one place this backbone *does*
  do plain global-average pooling (`.mean(dim=(2,3))`), but only for domain-shift estimation,
  not for the exemplar tokens matching uses.
  Matching itself is **cross-attention**, not cosine similarity — `_forward_scores`
  (`geco2_detector.py:392-429`) feeds the exemplar tokens as K/V into `m.adapt_features`, and the
  resulting per-location score comes from `m.class_embed` (`geco2_detector.py:426`), not a
  dot-product against a pooled prototype vector.
- This model is vendored/frozen (`_ensure_geco2_on_path`, `geco2_detector.py:61-80`) — swapping
  its backbone (e.g. for DINOv3) is not a config change; it would mean either retraining GECO2's
  own architecture or building a new adapter, out of scope for the encoder-swap in §1.

---

## 2. Config surface (`FeatureExtractorConfig` + `PrototypeConfig`)

**Takeaway:** every encoder choice is one Pydantic model
(`aero_eyes/config.py:197-288`), one YAML block (`configs/config.yaml:107-171`), and one factory
switch (`features.py:589-637`) — three places kept in lockstep, no other file needs touching to
add a *variant* of an already-listed model family. Adding a genuinely new model family (a 6th
option beyond dinov2/dinov3/clip/siglip/ensemble) means touching all three plus a new
`*FeatureExtractor` class.

### Findings

- `FeatureExtractorConfig` (`config.py:197-288`):
  - `model: Literal["dinov2","dinov3","clip","siglip","ensemble"] = "dinov2"` (`config.py:198`).
  - `dinov2_variant`, `dinov2_use_registers` (`config.py:199-211`).
  - `dinov3_variant`, `dinov3_pretrain_dataset` (`"lvd1689m"`|`"sat493m"`), `dinov3_source`
    (`"huggingface"`|`"kaggle"`), `dinov3_kaggle_model_id` (`config.py:212-247`).
  - `clip_variant: str = "vit-b/32"` (`config.py:248`, free-form string, not a `Literal`, so any
    HF CLIP repo id/short name the `_VARIANT_MAP` in `features.py:290-293` recognizes, or a raw
    HF path, works).
  - `siglip_variant: Literal["base","large","so400m"] = "base"` (`config.py:250`).
  - `ensemble_dino_model: Literal["dinov2","dinov3"] = "dinov2"` (`config.py:261`) — the *only*
    ensemble-specific field; reuses every `dinov2_*`/`dinov3_*` field above, no duplicate
    ensemble-only copies.
  - `weights: Optional[str] = None` (`config.py:262`) — declared but **not read anywhere** in
    `features.py`'s constructors (every `*FeatureExtractor.__init__` ignores it); effectively
    dead/unused today, not a real "point at custom weights" mechanism.
  - `image_size: int = 224` (`config.py:263`) — shared by all encoders, though each one's HF
    processor may itself override effective input size (e.g. SigLIP so400m is natively 384px;
    `image_size` here is a preprocessing knob only consumed by the manual `_preprocess_dino`
    path, not the HF-processor paths).
  - `projection_head: ProjectionHeadConfig` (`config.py:264`, defined `config.py:174-194`).
  - `candidate_background_masking: SegmentationConfig` (`config.py:288`, opt-in, off by default).
- `PrototypeConfig` (`config.py:291-294`): `fusion: Literal["mean","max","concat_then_pca"] =
  "mean"`, `l2_normalize: bool = True`, `cache_name: str = "prototype.npz"`.
- `configs/config.yaml:107-176` mirrors the above 1:1 with inline rationale comments — notably
  `configs/config.yaml:148-156` states the *reason* `ensemble_dino_model=dinov3` exists: testing
  whether CLIP's semantic objective complements DINO against background clutter (e.g. dry
  leaves) — the exact question a later encoder recommendation is likely being asked to answer.
  `configs/config.yaml:177-198` also documents two encoder-adjacent, opt-in domain-gap
  mitigations that operate independently of which encoder is chosen: `aerial_sim` (degrade
  reference photos to look more "drone-shot" before embedding) and `domain_calibration`
  (mean-shift the prototype toward this video's own frame statistics).

---

## 3. Where cosine-similarity matching happens

**Takeaway:** there is one core scoring primitive (`_score_against_ref`,
`aero_eyes/stages/stage3.py:57-88`), consumed by every downstream mechanism (threshold variants,
cluster verification, margin verification, identity-chain filter, negative-prototype filter,
dynamic prototype). `stage3.similarity` (default `"cosine"`) is the single switch that changes
what "similar" means for *all* of them simultaneously — a real integration point if a new
encoder changes what a good similarity metric/threshold looks like.

### Findings — core scoring

- `_cosine_sim` (`stage3.py:27-29`) — trivial dot product of two assumed-L2-normalized vectors;
  used only as a small helper, not the main scoring path.
- `_score_against_ref` (`stage3.py:57-88`) — the actual per-candidate scorer, dispatches on
  `stage3.similarity`: `"cosine"` → `feats @ ref` (line 72); `"l2"`/`"l1"` → negated distance
  (lines 73-76); `"rmd"` → Relative Mahalanobis Distance against a video-fit background
  Gaussian (lines 77-87, uses `_fit_rmd_background`, `stage3.py:32-45`, `sklearn.covariance.
  LedoitWolf`). All 4 return "higher = more similar" so every consumer is metric-agnostic.
- `_pool_sims` (`stage3.py:91-105`) — when `accuracy.cheap_boosters.multi_reference_embedding`
  is on, per-reference-image scores are pooled `mean` or `max` instead of first collapsing to
  one fused prototype vector — this is the "3 reference photos, not 1" pathway.
- Main call site: `run_stage3` (`stage3.py:737-1524`).
  - Prototype load: `read_prototype(proto_path)` (`stage3.py:787`) — reads `prototype.npz`
    written by Stage 1.
  - Candidate features load: `read_candidates_with_features(cand_path)` (`stage3.py:795`) —
    reads Stage 2's `candidates.feats.npz`.
  - Score computation: `stage3.py:915-924` — `_score_against_ref` (or `_pool_sims` over
    per-ref scores) called once for the whole video's candidate pool.
  - RMD background fit happens **once per video**, before any per-ref scoring
    (`stage3.py:907-913`), from `all_feats` (i.e. the video's *own* candidate pool, "mostly
    background/FP by construction").

### Findings — `dynamic_prototype` (2 independent implementations)

- **Stage 3's own** (legacy pipeline): `run_dynamic_prototype_rounds`
  (`stage3.py:516-618`) — batch, 2-pass: after an initial adaptive-threshold pass, takes
  high-confidence candidates (`all_sims >= percentile(...)`, `stage3.py:570-571`), averages
  their features, blends into the prototype (`alpha`-weighted, `stage3.py:614`) or appends as a
  new per-ref vector when multi-ref pooling is active (`stage3.py:606-612`), then re-scores —
  repeated `dp.rounds` times. Config: `DynamicPrototypeConfig` referenced at `config.py:979`.
  Incompatible with `verification_method="cluster"` and `adaptive_threshold_online` (both
  causal/per-keyframe; dynamic_prototype needs the whole video up front) — explicitly guarded at
  `stage3.py:944-954`.
- **GeCo2's own online variant**: `GeCo2DynamicPrototypeTracker`
  (`geco2_detector.py:919-1728`) — causal/incremental (`offer`/`offer_topk`/`_offer_topk_cluster`,
  `geco2_detector.py:1082-1481`), because a live/streaming GeCo2 run has no "whole video"
  distribution yet. Its cross-check against a *candidate* box can use either GeCo2's own Hiera
  appearance token (`cross_check_source="hiera"`, `_hiera_similarity`,
  `geco2_detector.py:1623-1642` — explicitly documented as *not independent* of GeCo2's own
  score, offered only for A/B testing) or `stage1.feature_extractor`
  (`cross_check_source="feature_extractor"`, default; `_feature_extractor_similarity`,
  `geco2_detector.py:1644-1706` — an independent embedding space, reuses the *same*
  `build_feature_extractor(cfg)` factory and `prototype.npz` as the legacy pipeline). This is the
  concrete place GeCo2's cross-attention score and a DINOv2-family cosine score are combined as
  two independent signals.

### Findings — identity/verification filters that also consume `all_feats`/cosine

- `apply_identity_chain_filter` (`stage3.py:621-734`) — KeepTrack-style: builds per-keyframe
  top-K candidate sets, Hungarian-matches (`scipy.optimize.linear_sum_assignment`) consecutive
  keyframes' candidates by `1 - cosine_similarity (+ spatial term)` (`stage3.py:704-709`),
  accepts only candidates whose identity chain reached `min_chain_length`.
- Cluster verification (`verification_method="cluster"`, `stage3.py:1000-1066`, and
  `cluster_secondary_filter`, `stage3.py:1226-1307`) — both call the shared
  `aero_eyes.utils.cluster_verify.cluster_verify_candidates` primitive (not read in this pass;
  referenced at `stage3.py:1011`, `1231`) against `all_feats` directly.
- `negative_prototype_filter` (`stage3.py:1177-1219`) — rejects a candidate if it's closer to a
  causal window of recently-*rejected* features than to the reference prototype
  (`max_neg_cosine - own_cosine >= tau_negative_margin`, `stage3.py:1200-1203`) — note this
  branch always uses **raw cosine** (`all_feats[i] @ ref_feats_base_np.T`, line 1200) regardless
  of `stage3.similarity`, explicitly to keep both sides of the margin comparable even under
  `similarity="rmd"`.
- `margin_verification` (`stage3.py:1340-1369`) — per-keyframe top1-vs-runner-up margin using
  whatever `all_sims` metric is active (not hardcoded to cosine).

### Findings — `stage123_geco2.cosine_rescore` (config.yaml)

- `Geco2CosineRescoreConfig` (`config.py:1949-1972`, `configs/config.yaml:802-808`) — opt-in,
  off by default. When enabled, GeCo2 first produces a *wider*, looser-thresholded candidate set
  per keyframe (`candidate_score_threshold_ratio=0.15`, `candidate_topk_per_keyframe=15`, both
  much looser than GeCo2's own default `score_threshold_ratio`/`topk_per_keyframe`), then the
  **entire legacy `run_stage3` cosine-matching machinery described above runs unmodified** on
  top of it — i.e. `cosine_rescore` doesn't duplicate matching logic, it just re-routes GeCo2's
  candidates through Stage 3's existing pipeline.
- `GlobalAdaptiveThresholdConfig` (`config.py:1975-2005`, off by default) is the GeCo2-score
  analog of `stage3.adaptive_threshold` — pools GeCo2's own raw scores across the whole video,
  not related to the embedding/cosine question but worth distinguishing from `cosine_rescore`
  since both are "alternative to GeCo2's default per-frame-relative threshold."

---

## 4. Existing documentation — prior precision-tuning work

**Takeaway:** this project has already run real-footage A/B testing and landed a specific,
quantified conclusion about where the current bottleneck is — a later recommendation should
engage with this, not re-derive it from scratch.

### Findings

- `docs/GECO2_precision_improvements_plan.md:7-13` (Context section) — states the real-footage
  finding directly: on sample `IDCard_0`, the best validated baseline is
  `adaptive_threshold_online` (causal z-score) at **F1=0.706, P=0.646, R=0.778**; precision
  64.6% was judged "still too low" because a wrong track is more costly than a missed frame
  (Stage 4 recovers from misses, not from bad tracks). The **open, unresolved question** as of
  that doc: whether FPs are a **small number of recurring confuser objects** (systematic,
  addressable with identity-chain-style filtering) or **diffuse/scattered** (evidence the
  underlying DINOv2 appearance signal is just weak at this domain gap) — this determines which
  fix family is worth investing in, and is exactly the question a "different encoder" proposal
  needs to answer for its own case.
- `docs/GECO2_precision_improvements_plan.md:13` and `docs/GECO2_precision_techniques_reference.
  md:41` (repeated) — a **concrete, load-bearing number**: on this project's own footage, raw
  cosine similarity's ceiling sits around **0.335–0.38 regardless of TP/FP identity** — i.e. even
  genuine matches don't score much higher than confusers under the current DINOv2 setup at this
  domain gap. This is the empirical motivation for RMD (§3) and is the single most concrete piece
  of evidence about "is the embedding itself the bottleneck."
- `docs/GECO2_precision_improvements_plan.md:8` — standalone per-keyframe cluster verification
  (`verification_method="cluster"`) **massively underperformed** the global adaptive threshold
  (F1 0.266 vs 0.684) specifically because most keyframes in this project's single-object
  tracking videos contain **zero** real target instances — unlike DAVE's own FSC147 benchmark
  where every image guarantees a real instance. Relevant caveat for any clustering-based
  encoder-quality diagnostic borrowed from a different benchmark's assumptions.
- `docs/GECO2_precision_improvements_plan.md:10` — `cluster_secondary_filter`'s rolling "trusted
  window" was found to **not meaningfully improve precision**, suspected cause: it admits every
  threshold-passing candidate unconditionally, so a single false positive can poison it as a
  "trusted anchor" — later fixed via `window_admission_min_consecutive_hits` gating (§3,
  `GECO2_precision_techniques_reference.md:78-98`).
- `docs/GECO2_precision_techniques_reference.md:1-14` — status header: **every technique in this
  file is opt-in, off by default, and "NOT YET VALIDATED" on real footage** (only synthetic unit
  tests exist for most of them as of this doc). It explicitly separates itself from the plan doc:
  this file is "how each technique works and how to enable it," not a recommendation to turn any
  of them on.
- `docs/GECO2_precision_techniques_reference.md:16-33` (`scripts/diagnose_verification_errors.
  py`) — an existing, ready-to-run diagnostic that answers exactly the "systematic confusers vs.
  diffuse noise" question above: clusters FP features together (few large clusters = systematic;
  many small = diffuse), checks spatial recurrence, and computes an "oracle separability" upper
  bound (cluster all TP+FP+exemplars together with full hindsight — if even that can't separate
  them, the fix must be at the embedding/score level, not the decision-mechanism level). This
  script is directly reusable to evaluate whether a *new* encoder actually improves separability
  before touching any downstream matching logic.
- `configs/config.yaml:148-156` and `features.py:414-419` — the project's own stated hypothesis
  for why a CLIP-inclusive ensemble might help: CLIP's semantic/categorical training objective
  vs. DINO's pure self-supervised texture-clustering objective, specifically flagged as a
  candidate fix for "DINO alone confuses the target with texturally-similar background clutter
  (e.g. dry leaves)" — this is the closest existing statement in the repo to the exact question
  "visual encoder to distinguish object from clutter."
- No `GECO2_scale_domain_gap_plan.md` mention of encoder choice beyond `domain_calibration`
  (`geco2_detector.py:824-884`, feature-space mean-shift, "different axis from blur/scale,"
  `docs/GECO2_scale_domain_gap_plan.md:24`) — that doc is about box/scale calibration, not
  encoder selection, and explicitly says it doesn't touch domain_calibration.

---

## 5. Practical integration constraints

**Takeaway:** adding a new encoder *variant within an existing family* (e.g. a new DINOv3
checkpoint/size) is cheap and well-precedented; adding a genuinely new model family follows a
clear, already-repeated pattern (new class + factory branch + config `Literal` + YAML block);
embedding dimensionality is **not hardcoded** in the prototype/storage layer, but there are two
concrete places a dimension change needs care.

### Findings — abstraction layer (favorable)

- Every encoder implements exactly two methods callers use: `extract(images, batch_size)` and
  `extract_crops(frame_bgr, boxes, pad_ratio, batch_size)`, plus `_dim()`/`_feature_dim()`
  (`features.py`, all 5 classes + both wrappers). `build_feature_extractor(cfg)`
  (`features.py:589-660`) is the **only** place that needs to know which concrete class to
  instantiate; every caller (`stage1.py:163`, `stage3.py:814` under
  `recompute_candidate_features`, `geco2_detector.py:1673-1676` under the cross-check) goes
  through this one factory. A new encoder class that implements the same two-method interface
  is a drop-in.
- `prototype.npz`'s schema does not hardcode a dimension: `write_prototype`/`read_prototype`
  (`aero_eyes/utils/io.py:206-235`) store `prototype` and `ref_i` arrays at whatever shape the
  extractor produced; nothing downstream assumes a specific `D`. `_score_against_ref` and every
  Stage 3 consumer operate on `feats @ ref` generically regardless of `D` — RMD's
  `LedoitWolf().fit(all_feats)` (`stage3.py:44`) also handles arbitrary `D` (it exists
  specifically to regularize when candidate count < `D`).

### Findings — where dimensionality *does* need care

- **Stale-cache dimension mismatch** (already a documented footgun, not new): switching
  `stage1.feature_extractor.model`/variant without invalidating `candidates.feats.npz`
  (written by Stage 2 with the *previous* extractor) crashes Stage 3's `feats @ ref` with a
  shape mismatch — documented at `config.py:787-804` and `configs/config.yaml:235-250`, with a
  purpose-built fix (`stage3.recompute_candidate_features`, re-embeds existing candidate boxes
  with the *current* extractor without re-running box proposal). Any encoder swap in this repo
  must either set this flag once or clear `candidates.json`/`.feats.npz` and Stage 1's
  `prototype.npz` caches; `cfg.project.use_cache` gates whether stale caches get reused at all
  (`stage3.py:750-752`, `stage1.py` — not read in this pass but referenced by the same
  `use_cache` convention throughout).
- **Hardcoded 768 in one edge case**: `aero_eyes/stages/stage2.py:234` —
  `np.savez_compressed(str(feat_path), features=np.zeros((0, 768), dtype=np.float32))`, the
  fallback written when a keyframe produced *zero* candidates with features. This assumes 768-d
  (DINOv2 `vitb14`'s dim) but only ever writes a **zero-row** array in that branch, so it's
  currently harmless (no downstream code reads `.shape[1]` off a 0-row array before concatenating
  with a real, correctly-shaped batch) — still worth flagging as a latent inconsistency if a
  future refactor ever asserts a fixed feature dimension somewhere in the candidates-loading path.
- **`ProjectedFeatureExtractor` dimension check** (`features.py:552-557`) fails loudly (not
  silently) if a trained projection head's expected input dim doesn't match the base extractor's
  output dim — this is the one place in the codebase that actively guards against a dimension
  mismatch at construction time, a useful pattern to point to for any new wrapper.
- **RMD / LedoitWolf cost scales with `D`**: `_fit_rmd_background` (`stage3.py:32-45`) fits a
  `D x D` covariance once per video from `all_feats`. A much higher-dimensional encoder (e.g.
  `vitg14` at 1536-d, or the ensemble concatenation at 1280-d) increases this fit's cost and its
  reliance on Ledoit-Wolf shrinkage (candidate count per video is typically well under a few
  hundred, likely `< D` for the larger variants) — worth flagging for any recommendation
  involving both a larger encoder *and* `similarity="rmd"`.
- **GeCo2's own backbone is not part of this swappable surface** (§1b) — a recommendation to
  change "the encoder" needs to be explicit about whether it means `stage1.feature_extractor`
  (swappable, this section) or GeCo2's vendored Hiera backbone (not swappable via config; GeCo2
  itself is a frozen, pretrained third-party checkpoint loaded via `build_model`/
  `load_state_dict` in `geco2_detector.py:90-165`).
- **Preprocessing is per-family, not centralized**: `image_size` (`FeatureExtractorConfig.
  image_size`, default 224) is honored by the manual `_preprocess_dino` path
  (DINOv2, DINOv3-kaggle) but HF-processor-based paths (DINOv3-huggingface, CLIP, SigLIP) use
  each model's own `AutoImageProcessor`/`*Processor` defaults instead — so a new encoder
  integrated via an HF processor will not automatically respect `image_size`; this needs to be
  handled explicitly the way the existing HF-based classes do (they simply don't read
  `image_size` at all, e.g. `CLIPFeatureExtractor.__init__` has no `image_size` parameter,
  `features.py:295`).
- **No test coverage read in this pass** for `features.py` itself beyond what's implied by
  `tests/test_features_ensemble.py` and `tests/test_features_masked_crop.py` (both present in
  git status as new/untracked files, not opened in this investigation) — a later integration
  should check these for the existing test pattern before adding a new encoder class.

---

## Appendix: file map for the report writer

| Concern | File |
|---|---|
| Swappable encoders (DINOv2/DINOv3/CLIP/SigLIP/ensemble) + factory | `aero_eyes/models/features.py` |
| Encoder config schema | `aero_eyes/config.py:174-294` (`ProjectionHeadConfig`, `FeatureExtractorConfig`, `PrototypeConfig`) |
| Encoder config defaults + rationale comments | `configs/config.yaml:107-198` |
| Legacy Stage 1 prototype build (calls the factory) | `aero_eyes/stages/stage1.py` (build_feature_extractor at line 47/163, `.extract` at 198/265) |
| Stage 2 candidate feature caching | `aero_eyes/stages/stage2.py` (`_write_candidates_with_features` ~194-234, `read_candidates_with_features` ~237+) |
| Cosine/RMD/L1/L2 scoring + every matching mechanism | `aero_eyes/stages/stage3.py` |
| Stage 3 config (similarity metric, thresholds, all secondary filters) | `aero_eyes/config.py:786-980`ish (`Stage3Config`) |
| GeCo2's own Hiera backbone + RoI-Align exemplar encoding | `aero_eyes/models/geco2_detector.py` |
| GeCo2 online dynamic-prototype + cross-check against `stage1.feature_extractor` | `aero_eyes/models/geco2_detector.py:919-1728` (`GeCo2DynamicPrototypeTracker`) |
| GeCo2 <-> legacy cosine bridge | `aero_eyes/config.py:1949-1972` (`Geco2CosineRescoreConfig`), `configs/config.yaml:802-808` |
| `prototype.npz` I/O (dimension-agnostic) | `aero_eyes/utils/io.py:203-235` |
| Prior real-footage precision findings (F1/P/R numbers, cosine ceiling) | `docs/GECO2_precision_improvements_plan.md` |
| Catalog of opt-in precision techniques (all off by default, unvalidated) | `docs/GECO2_precision_techniques_reference.md` |
| Ready-to-run FP-diagnosis script (systematic vs. diffuse confusers) | `scripts/diagnose_verification_errors.py` (referenced, not opened in this pass) |

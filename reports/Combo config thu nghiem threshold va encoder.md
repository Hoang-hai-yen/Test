# Combo config đáng thử: encoder, prototype, threshold, verification

Tài liệu tra cứu nhanh — liệt kê các tổ hợp config cụ thể để test, kèm ràng buộc (cái nào loại trừ nhau, cái nào cần rebuild cache, cái nào chỉ có tác dụng trong điều kiện nào). Tất cả kỹ thuật dưới đây đều `NOT YET VALIDATED` trên footage thật trừ khi ghi chú riêng — luôn đo P/R/F1 bằng `scripts/diagnose_verification_errors.py` (hoặc chạy full pipeline) so với baseline trước khi giữ lại.

**Baseline hiện có** (đã validate trên sample IDCard_0): `adaptive_threshold: true`, `adaptive_threshold_online: true`, `adaptive_threshold_online_method: window_stat` (tức z_score chạy trên cửa sổ trượt) → F1=0.706, P=0.646, R=0.778.

**Nguyên tắc chung**: bật **một kỹ thuật/lần**, đo, rồi mới chồng — tránh lặp lại sai lầm cũ của dự án là đổi nhiều thứ cùng lúc khiến không quy được kết quả cải thiện/tệ đi về đâu.

---

## 1. Feature extractor / encoder (`stage1.feature_extractor`)

Các lựa chọn cho `model`: `dinov2` (mặc định) | `dinov3` | `clip` | `siglip` | `siglip2` | `evaclip` | `dinotxt` | `ensemble` (dino+clip concat) | `fgclip` | `radio`.

**Đổi `model` hoặc kích thước embedding → cache cũ (`prototype.npz`, `candidates.feats.npz`) không còn khớp chiều** — phải xoá cache hoặc bật `stage3.recompute_candidate_features: true` mỗi lần đổi nhóm này.

```yaml
# E1a — chỉ đổi kiến trúc DINOv2 -> DINOv3, GIỮ NGUYÊN pretrain mặc định
# (lvd1689m) -- tách riêng "đổi kiến trúc" khỏi "đổi domain pretrain" ở E1b
stage1:
  feature_extractor:
    model: dinov3
    dinov3_pretrain_dataset: lvd1689m   # mặc định, chỉ để tường minh

# E1b — DINOv3 + domain vệ tinh (kiến trúc + domain pretrain cùng lúc)
stage1:
  feature_extractor:
    model: dinov3
    dinov3_pretrain_dataset: sat493m   # thay vì lvd1689m mặc định

# E2 — DINOv2 with registers (drop-in swap, cùng chiều embedding)
stage1:
  feature_extractor:
    dinov2_use_registers: true

# E3 — multiscale_attn pooling thay vì CLS token đơn (DINOv2 hoặc DINOv3)
# Nồng cốt: nối attention-weighted-pooled patch tokens từ ~3 tầng transformer
# (50%/75%/100%) thay vì chỉ 1 CLS token toàn cục -- tinh thần giống DAVE nối
# nhiều tầng ResNet conv feature. Bắt buộc backend huggingface.
stage1:
  feature_extractor:
    dinov2_pooling: multiscale_attn      # hoặc dinov3_pooling nếu model=dinov3
    dinov3_source: huggingface           # bắt buộc nếu dùng dinov3_pooling=multiscale_attn

# E4 — đổi hẳn encoder sang CLIP/SigLIP (thiên semantic/category hơn texture)
stage1:
  feature_extractor:
    model: clip        # hoặc siglip
    clip_variant: vit-l/14     # hoặc siglip_variant: large

# E5a — ensemble DINOv2 (mặc định gốc trước khi dự án tổng quát hoá sang
# dinov3 -- CHƯA từng được A/B test riêng, đáng có trong sweep làm đối chứng)
stage1:
  feature_extractor:
    model: ensemble
    ensemble_dino_model: dinov2

# E5b — ensemble DINOv3 (lvd1689m) + CLIP
stage1:
  feature_extractor:
    model: ensemble
    ensemble_dino_model: dinov3
    dinov3_pretrain_dataset: lvd1689m

# E5c — ensemble DINOv3 (sat493m) + CLIP (nặng nhất, ghép domain pretrain + ensemble)
stage1:
  feature_extractor:
    model: ensemble
    ensemble_dino_model: dinov3
    dinov3_pretrain_dataset: sat493m

# E6 — FG-CLIP (hard fine-grained negative pairs) -- dự án đã thấy CLIP/SigLIP
# thường thua DINOv2/v3 trên footage thật, đây là biến thể khác hẳn mục tiêu
# huấn luyện (phân biệt instance gần giống nhau, không phải category rộng)
stage1:
  feature_extractor:
    model: fgclip
    fgclip_variant: base

# E7 — NVIDIA RADIO/C-RADIO (backbone distill từ nhiều teacher VFM cùng lúc)
# Bằng chứng "hybrid tốt hơn single-teacher" trong paper gốc chỉ có ở
# segmentation/classification/VQA, KHÔNG có ở retrieval -- đây là cược thực
# nghiệm thuần tuý, chưa có cơ sở lý thuyết trực tiếp cho use-case này.
stage1:
  feature_extractor:
    model: radio
    radio_variant: c-radio_v3-b   # *-b/l/h/g variants; chỉ c-radio_*/v4-* được phép dùng thương mại

# E8 — SigLIP2 (Global-Local + Masked Prediction losses trên nền SigLIP) --
# bằng chứng fine-grained/near-duplicate mạnh thứ 2 sau FG-CLIP, nhưng dùng
# class transformers chuẩn (không trust_remote_code) -- rủi ro tích hợp thấp
# hơn hẳn fgclip
stage1:
  feature_extractor:
    model: siglip2
    siglip2_variant: base   # base | so400m (nặng hơn)

# E9 — EVA02-CLIP-B/16 -- CẦN dependency MỚI (open_clip_torch), bằng chứng
# fine-grained yếu hơn E8/E6 (chỉ có số zero-shot classification, không có
# benchmark kiểu FG-OVD)
stage1:
  feature_extractor:
    model: evaclip
    evaclip_variant: base   # chỉ có "base" được wire

# E10 — dino.txt: text encoder LiT-aligned trên nền DINOv2 ViT-L/14 đóng băng
# -- thêm ngữ nghĩa/text-grounding mà vẫn ở trong họ DINO, không dependency
# mới, nhưng CHƯA verify độc lập được chi tiết preprocessing/output (docstring
# DinoTxtFeatureExtractor ghi rõ lý do)
stage1:
  feature_extractor:
    model: dinotxt
```

**Thứ tự đáng thử**: E1a (chỉ đổi kiến trúc, rẻ nhất) → E1b (thêm domain pretrain, so trực tiếp với E1a để biết phần nào thực sự đóng góp) → E2/E3 (vẫn DINO, không đổi chiều embedding nếu chỉ đổi pooling — kiểm tra kỹ vì multiscale_attn có thể đổi chiều output, cần xoá cache để chắc) → E4/E8 (CLIP/SigLIP2, so trực tiếp) → E10 (dinotxt, vẫn họ DINO, không dependency mới) → E5a/E5b/E5c (ghép, chỉ thử sau khi biết CLIP/SigLIP và DINOv2/v3 một mình có giúp gì không — E5a là baseline ensemble gốc chưa từng test) → E6/E9/E7 (ít có cơ sở lý thuyết trực tiếp hơn và/hoặc cần dependency/quyền truy cập thêm, thử sau cùng).

---

## 2. Prototype construction & tiền xử lý ảnh reference (`stage1.prototype` / `stage1.domain_calibration` / `stage1.segmentation` / `stage1.crop_to_object`)

Độc lập với nhau và với mục 1 — không cần đổi cache khi chỉ đổi 2 field này (không đổi chiều embedding).

```yaml
# P1a — fusion=max thay vì mean mặc định (baseline đã dùng "mean" ngầm, không
# cần combo riêng cho nó)
stage1:
  prototype:
    fusion: max

# P1b — fusion=concat_then_pca -- văn liệu ủng hộ hướng này cho các view BỔ
# SUNG NHAU, nhưng dự án tự ghi nhận: fit yếu + kém ổn định về mặt thống kê
# với 3 ảnh cận cảnh gần-trùng-lặp (không phải complementary views)
stage1:
  prototype:
    fusion: concat_then_pca

# P1 — agreement_weighted fusion (BD-CSPN-style, tự hạ trọng số ref lệch)
stage1:
  prototype:
    fusion: agreement_weighted
    agreement_weighted_epsilon: 10.0   # cao hơn = downweight outlier mạnh hơn, dễ overfit vì chỉ có 3 ref

# P2 — filter_target_like_frames: pool frame lớn hơn cho domain_calibration,
# giữ N frame ÍT giống target nhất thay vì random/đều -- tránh contamination
stage1:
  domain_calibration:
    filter_target_like_frames: true

# P3 — combo P1+P2
stage1:
  prototype: { fusion: agreement_weighted }
  domain_calibration: { filter_target_like_frames: true }

# P4 — aerial_sim: giả lập domain gap (ảnh ref cận cảnh -> giống góc nhìn xa
# của drone hơn) bằng cách shrink-rồi-upscale ảnh reference trước khi encode.
# downscale_factor=1.0 là no-op (tương đương enabled=false); càng nhỏ càng
# "xa"/mờ hơn.
stage1:
  aerial_sim:
    enabled: true
    downscale_factor: 0.5   # thử thêm 1.0 (no-op, đối chứng) và 0.1 (rất mạnh)

# P5 — background_mode: giữ nguyên nền thật của ảnh ref thay vì flat-fill mặc
# định -- đối chứng xem việc "làm sạch" nền ref có thực sự giúp hay chỉ tạo
# thêm 1 dạng domain gap khác (ref nền phẳng vs. candidate video nền thật)
stage1:
  segmentation:
    background_mode: keep_real   # mean_fill (mặc định) | keep_real | blur

# P6/P7 — crop_to_object: crop ảnh ref về đúng tight mask box (mở rộng theo
# crop_context_margin) TRƯỚC khi resize -- object chiếm phần lớn canvas hơn.
# Yêu cầu segmentation.enabled (mặc định đã true). Thử 2 margin khác nhau.
stage1:
  crop_to_object: true
  crop_context_margin: 0.0   # P6: bó sát nhất có thể

stage1:
  crop_to_object: true
  crop_context_margin: 0.2   # P7: chừa thêm chút ngữ cảnh quanh object
```

---

## 3. Cách tính threshold — `adaptive_threshold_method` (batch, z_score/otsu/gmm)

Chỉ có tác dụng khi `stage3.adaptive_threshold: true`. z_score là baseline **duy nhất đã validate** (empirically swept: z=1.0→0.269, 1.5→0.312, **2.0→0.338 (best)**, 2.5→0.332, 3.0→0.268 mean ST-IoU trên PublicTest). otsu/gmm chưa có A/B test thật nào trong repo.

```yaml
# T1 — z_score mặc định (đã tune z=2.0), so sánh baseline
stage3:
  adaptive_threshold: true
  adaptive_threshold_method: z_score
  adaptive_z_score: 2.0

# T2 — z_score robust (median/MAD thay mean/std, ít bị outlier kéo lệch)
stage3:
  adaptive_threshold_method: z_score
  adaptive_threshold_robust: true

# T3 — otsu (tự thích nghi hình dạng phân phối, không cần tay chỉnh z)
# Rủi ro: KHÔNG kiểm tra tính bimodal trước khi cắt -- nếu phân phối thực tế
# unimodal (video sạch, ít clutter), vẫn chẻ đôi noise gần như ngẫu nhiên.
stage3:
  adaptive_threshold_method: otsu
  adaptive_otsu_bins: 256

# T4 — gmm (có kiểm tra bimodal qua BIC + min_separation_std, an toàn hơn otsu)
stage3:
  adaptive_threshold_method: gmm
  adaptive_gmm_min_separation_std: 1.5
  adaptive_gmm_fallback_percentile: 20.0   # dùng khi fit ra unimodal
```

Cả otsu/gmm tự fallback về z_score nếu số sample < `adaptive_threshold_min_samples` (20) — không cần lo cold-start riêng.

**Combo với `adaptive_threshold_anchor_to_original_refs: true`**: đáng thử cùng bất kỳ T-nào ở trên nếu đang dùng `dynamic_prototype` (whole-video, mục 5 config) — giữ mean/std/otsu/gmm ổn định, không bị chính dynamic_prototype tự kéo lệch ngưỡng.

---

## 4. Cách cập nhật threshold theo thời gian thực — `adaptive_threshold_online_method`

Chỉ có tác dụng khi `adaptive_threshold: true` **và** `adaptive_threshold_online: true`. 4 lựa chọn **loại trừ lẫn nhau**, chỉ 1 chạy tại một thời điểm. `window_stat` là lựa chọn **duy nhất đã validate** (F1=0.706, thắng cả batch z_score). 3 cái còn lại là FAITHFUL port đúng công thức paper gốc nhưng chưa test trên footage thật.

```yaml
# O1 — window_stat (mặc định, đã validate) -- tái dùng T1-T4 ở mục 3 trên cửa sổ trượt
stage3:
  adaptive_threshold: true
  adaptive_threshold_online: true
  adaptive_threshold_online_window: 200
  adaptive_threshold_online_method: window_stat
  adaptive_threshold_method: z_score   # hoặc otsu/gmm -- áp dụng lên cửa sổ thay vì cả video

# O2 — ACI (Adaptive Conformal Inference, Gibbs & Candès 2021)
# percentile chạy, cập nhật gradient step theo tỉ lệ accept mỗi keyframe
stage3:
  adaptive_threshold_online_method: aci
  aci_target_error_rate: 0.1   # tỉ lệ chấp nhận mục tiêu mỗi keyframe
  aci_step_size: 0.05          # tốc độ cập nhật percentile

# O3 — SAFFRON (online FDR control, nhắm thẳng (false accepts)/(total accepts))
# Khác 3 cái còn lại: test TỪNG CANDIDATE riêng lẻ, không phải 1 threshold/keyframe
stage3:
  adaptive_threshold_online_method: saffron
  online_fdr:
    target_fdr: 0.1
    initial_wealth_fraction: 0.5
    lam: 0.5
    gamma_exponent: 2.0
    p_value_window: 200

# O4 — corruption_compensated (F-ROCP, bọc quanh ACI)
# Không có tham số riêng -- dùng lại aci_target_error_rate/aci_step_size ở O2
stage3:
  adaptive_threshold_online_method: corruption_compensated
```

**Lưu ý quan trọng khi target xuất hiện muộn trong video** (window đầy toàn background/clutter trước khi target thật xuất hiện lần đầu): cả 4 cơ chế đều không có khái niệm "chưa từng thấy positive thật" — chúng tính threshold từ đúng những gì có trong window/lịch sử, kể cả khi đó toàn là nhiễu. `otsu` (mục 3) rủi ro cao nhất trong tình huống này vì không có bimodal check.

---

## 5. Bộ lọc phụ Stage 3 (chỉ loại bớt, không thêm lại recall)

4 cái độc lập, có thể bật cùng lúc — áp dụng nối tiếp trong `run_stage3` theo thứ tự: `negative_prototype_filter` → `cluster_secondary_filter` → `identity_chain_filter`; `margin_verification` áp dụng riêng ở bước gom `frame_groups`.

```yaml
# F1 — negative_prototype_filter (rẻ nhất, không cần biết trước confuser là gì)
stage3:
  negative_prototype_filter:
    enabled: true
    window_size: 200
    tau_negative_margin: 0.0

# F2 — identity_chain_filter (nhắm confuser lặp lại có cấu trúc)
# Thử CẢ HAI spatial_weight -- 0 nếu confuser là 1 vật thể di chuyển,
# >0 (0.3-0.5) nếu confuser là 1 lớp texture rải rác nhiều vị trí (vd lá khô)
stage3:
  identity_chain_filter:
    enabled: true
    top_k_per_keyframe: 5
    min_chain_length: 2
    spatial_weight: 0.0   # rồi lặp lại với 0.3-0.5

# F3 — similarity=rmd (thay cosine ở tầng score chính, ảnh hưởng MỌI cơ chế downstream)
stage3:
  similarity: rmd
  adaptive_threshold: true   # bắt buộc -- match_threshold cố định sẽ sai hoàn toàn thang đo

# F3a/F3b — similarity=l1/l2: 2 lựa chọn built-in còn lại ngoài cosine/rmd --
# là KHOẢNG CÁCH bị âm dấu (không giới hạn, thường âm), không phải similarity
# có ý nghĩa/rationale riêng như rmd -- adaptive_min_floor tự động bỏ qua với
# mọi metric khác cosine (thang đo khác hẳn [-1,1])
stage3:
  similarity: l1   # hoặc l2
  adaptive_threshold: true

# F4 — margin_verification + cluster_secondary_filter (ít tốn kém, ít nhắm trúng hơn F1-F3)
stage3:
  margin_verification:
    enabled: true
    tau_margin: 0.05
  cluster_secondary_filter:
    enabled: true
    window_size: 50
    window_admission_min_consecutive_hits: 2
    window_admission_iou_threshold: 0.5

# F4c — cluster_secondary_filter.accumulate_new_anchors: false -- tắt tích luỹ
# anchor mới, MỌI keyframe cả video chỉ cluster với đúng 3 exemplar gốc
# (không bao giờ đổi). Cô lập xem chính việc TÍCH LUỸ có giúp hay hại --
# nghi vấn hiện tại: 1 FP lọt qua sớm có thể "đầu độc" window và tự hút thêm
# confuser giống nó suốt phần còn lại của video, đây là cách kiểm chứng.
stage3:
  cluster_secondary_filter:
    enabled: true
    accumulate_new_anchors: false   # true = mặc định (đã test, không cải thiện precision rõ rệt)

# F5 — chồng cả 4 (chỉ sau khi đã đo riêng từng cái ở F1-F4)
stage3:
  negative_prototype_filter: { enabled: true }
  cluster_secondary_filter: { enabled: true, window_admission_min_consecutive_hits: 2 }
  identity_chain_filter: { enabled: true, spatial_weight: 0.3 }
  margin_verification: { enabled: true, tau_margin: 0.05 }
```

**Cấm kết hợp**: `verification_method: cluster` với bất kỳ gì ở mục 3-5 (F1-F5, T1-T4, O1-O4) — 2 cơ chế quyết định chính khác nhau, không cùng lúc. Xem mục 7 để test riêng `verification_method=cluster`.

---

## 6. Multi-reference pooling (`accuracy.cheap_boosters.multi_ref_pooling`)

Cách gộp 3 similarity score per-reference thành 1 số, khi `accuracy.cheap_boosters.multi_reference_embedding: true` (mặc định đã bật). Độc lập với mục 3-5 — không đổi cache, chỉ đổi công thức gộp ở Stage 3 (`_pool_sims` trong `stage3.py`). Giá trị đang set trong config.yaml hiện tại là `max`.

```yaml
# M1 — mean (gộp trung bình 3 ref -- ý nghĩa gốc trong docstring dùng làm default)
accuracy:
  cheap_boosters:
    multi_ref_pooling: mean

# M2 — max (mặc định hiện tại của config.yaml) -- "OR" qua 3 ref, thiên recall,
# rủi ro: 1 ref nào đó tình cờ giống 1 loại clutter sẽ làm lọt clutter đó qua
# cho MỌI candidate.

# M3 — min (NOT YET VALIDATED) -- "AND" qua 3 ref, thiên precision, đáng thử
# nếu nghi ngờ max đang để lọt confuser qua 1 ref quá dễ dãi.
accuracy:
  cheap_boosters:
    multi_ref_pooling: min

# M4 — agreement_weighted (NOT YET VALIDATED) -- trọng số mỗi ref theo mức
# đồng thuận BD-CSPN-style với 2 ref còn lại, thay vì coi 3 ref ngang nhau
# (mean) hoặc để 1 ref một mình quyết (max/min).
accuracy:
  cheap_boosters:
    multi_ref_pooling: agreement_weighted
    agreement_weighted_epsilon: 10.0
```

---

## 7. Cơ chế verification thay thế — `stage3.verification_method: cluster`

DAVE-style: bỏ hẳn ngưỡng scalar, mỗi keyframe cluster candidate của chính nó cùng 3 exemplar, giữ lại candidate nào cluster chung với 1 exemplar. `verification_method` (không phải `cluster_verification.enabled`) mới là công tắc — khác với `stage123_geco2.dynamic_prototype.cluster_verification` ở mục 7 cũ, nơi `enabled` chính là công tắc.

**Đã có A/B test thật**: standalone cluster **THUA CẢ** batch lẫn online adaptive_threshold (TP/FP retention margin +10.5pp vs +61.4pp/+74.5pp) — lý do: phần lớn keyframe trong video tracking 1 object không hề chứa instance thật nào, mà quyết định per-keyframe thuần tuý không có cách nào biết "cả keyframe này là clutter nền" như 1 threshold tính trên nhiều keyframe làm được. Test lại đây chủ yếu để xem `cluster_method`/`pairwise_metric` có đổi được kết luận đó không, kỳ vọng thấp.

```yaml
# V1 — hdbscan + cosine (mặc định, DAVE-fidelity-matched) -- điểm đối chứng
stage3:
  verification_method: cluster

# V2 — hdbscan + l1 (Manhattan, cấu trúc cluster khác cosine trên vector đã L2-normalize)
stage3:
  verification_method: cluster
  cluster_verification:
    pairwise_metric: l1

# V3 — hdbscan + mahalanobis (whiten bằng CHÍNH inverse-covariance mà similarity=rmd dùng)
stage3:
  verification_method: cluster
  cluster_verification:
    pairwise_metric: mahalanobis

# V4 — spectral + cosine (khớp đúng reference implementation của DAVE, tự ước
# lượng số cluster per-keyframe qua eigengap thay vì hdbscan's density-based)
stage3:
  verification_method: cluster
  cluster_verification:
    cluster_method: spectral
```

**Cấm kết hợp**: không dùng chung với bất kỳ gì ở mục 3-6 (đường `verification_method=threshold` hoàn toàn khác).

---

## 8. GeCo2 online tracker (`stage123_geco2.dynamic_prototype`)

Chỉ áp dụng khi `pipeline.detector: geco2`. `interval_window_enabled` chỉ có tác dụng ở đường `offer()` mặc định — **vô hiệu nếu** `topk_fusion.enabled: true` hoặc `cluster_verification.enabled: true` đang bật (khi đó `offer_topk()` đi đường khác).

```yaml
# G1 — interval_window (buffer N offer, chỉ commit ứng viên điểm cao nhất mỗi 8 offer)
pipeline: { detector: geco2 }
stage123_geco2:
  dynamic_prototype:
    enabled: true
    interval_window_enabled: true
    interval_window_frames: 8
    topk_fusion: { enabled: false }        # bắt buộc false để G1 có tác dụng
    cluster_verification: { enabled: false } # bắt buộc false để G1 có tác dụng

# G2 — G1 + prototype tốt hơn từ mục 2 (ảnh hưởng qua cross_check nếu dùng)
pipeline: { detector: geco2 }
stage1:
  prototype: { fusion: agreement_weighted }
  domain_calibration: { filter_target_like_frames: true }
stage123_geco2:
  dynamic_prototype: { enabled: true, interval_window_enabled: true }
```

---

## 9. Thứ tự khuyến nghị tổng thể

1. **O1 (baseline, đã có)** → xác nhận lại số trên chính sample đang test.
2. **E1a** (1 dòng, rẻ) → đo, rồi **E1b** (thêm domain pretrain) → so trực tiếp với E1a.
3. **P3** (P1+P2, rẻ, độc lập) → đo. **P4** (aerial_sim), **P5** (background_mode), **P6/P7** (crop_to_object x margin) đo riêng từng cái, không chồng với P3 trong cùng 1 lần.
4. **F1** → **F2** (cả 2 spatial_weight) → **F3** → **F4** → **F4c** (ablation accumulate_new_anchors, chỉ có ý nghĩa nếu F4 cho kết quả khác biệt), mỗi bước đo riêng trước khi chồng thành **F5**.
5. **T2/T3/T4** — thử thay thế z_score mặc định trong O1, so trực tiếp với T1.
6. **O2/O3/O4** — thử thay window_stat, so trực tiếp với O1 (kỳ vọng thấp hơn vì chưa validate).
7. **M1/M3/M4** — thử thay `multi_ref_pooling: max` mặc định, so trực tiếp với baseline (= M2).
8. **V1** (đối chứng, đã biết thua threshold) → **V2/V3/V4** chỉ nếu tò mò `cluster_method`/`pairwise_metric` có đổi được kết luận đó không — kỳ vọng thấp, xếp cuối nhóm ưu tiên thấp.
9. **E2/E3** → **E4/E8** → **E10** → **E5a/E5b/E5c** → **E6/E9/E7** — nhánh đổi encoder, cần rebuild cache, thử sau cùng vì tốn compute nhất (E9 còn cần cài thêm `open_clip_torch`).
10. Nếu cân nhắc pipeline `geco2` hẳn: **G1** → **G2**, so sánh riêng (không so P/R/F1 trực tiếp với chuỗi trên vì khác cơ chế root).

Sau mỗi bước: `pytest tests/ -q` phải xanh toàn bộ trước khi coi thay đổi cấu hình là hợp lệ để thử tiếp bước sau.

---

## 10. Script tự động hoá

`scripts/sweep_config_combos.py` phủ combo mục 1-7 + 9 (bỏ mục 8 GeCo2 vì khác pipeline root), cho 1 hoặc nhiều sample, chỉ chạy Stage 1 (khi combo đổi encoder/prototype/aerial_sim) + Stage 3, dùng lại `candidates.json` đã có sẵn — không đụng Stage 2/4/5.

**Mặc định (FULL MATRIX)**: mọi combo "đụng Stage 1" (encoder E*, prototype P*, ~23 combo) được ghép chéo với MỌI combo "chỉ đụng Stage 3" (threshold T*, online O*, cluster-verification V*, filter F*, pooling M*, ~31 combo) → ~700 lần chạy Stage 3/sample. Lý do phải làm ma trận thay vì thuần OFAT: Stage 1 và Stage 3 chạy tách biệt trong script, nhưng config Stage 1 (encoder nào, fusion ra sao) quyết định KHÔNG GIAN EMBEDDING mà mọi quyết định Stage 3 (threshold/filter/cluster) vận hành trên đó — một filter đã validate dưới encoder mặc định không chắc còn đúng dưới 1 encoder khác. Để không tốn thêm chi phí: mỗi combo Stage 1 chỉ **rerun Stage 1 đúng 1 lần** (và recompute feature đúng 1 lần nếu đổi encoder), rồi TÁI SỬ DỤNG kết quả đó cho toàn bộ combo Stage 3 ghép với nó.

**`--ofat-only`**: quay lại hành vi rẻ cũ (~52 lần chạy) — mỗi combo Stage 1 chỉ test với threshold baseline, và mọi combo Stage 3 chỉ test với encoder mặc định (lát cắt "hàng 0 + cột 0" của cùng ma trận).

Đọc docstring đầu file script để biết cách dùng, danh sách combo mặc định, và cách nó backup/restore `prototype.npz`/`candidates.json`/`detections.json` của từng sample trước-sau khi sweep (không để lại trạng thái bẩn). Test riêng cho script: `tests/test_sweep_config_combos.py`.

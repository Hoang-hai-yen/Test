# Kết quả sweep combo — sample IDCard_1

Tổng hợp từ `docs/testcombo.log` — full matrix sweep (`scripts/sweep_config_combos.py`) trên sample `IDCard_1`.

**Cảnh báo quan trọng**: log này chạy bằng **code cũ**, trước khi 2 bug sau được tìm thấy và sửa trong cùng phiên làm việc:
1. `stage1_default` chưa force `recompute_candidate_features=true` → `candidates.json` có thể lệch dimension so với encoder hiện tại.
2. Nhóm combo `prototype` (P1a-P7) từng đứng SAU nhóm `encoder` trong danh sách → `candidates.json` bị nhóm encoder ghi đè dimension trước khi tới lượt prototype chạy.

Hệ quả: **toàn bộ 10 combo Stage-1 nhóm `prototype` (P1a-P7) lỗi 100% (31/31 mỗi combo)** — không có mặt trong bảng dưới đây. Chỉ 13 combo Stage-1 thuộc nhóm `encoder` (bao gồm `stage1_default`) chạy thành công đầy đủ 31/31 combo Stage-3, cho ra đúng **13 × 31 = 403 kết quả hợp lệ** — đây là toàn bộ nội dung bảng bên dưới. Cần chạy lại sweep với code mới để có kết quả nhóm `prototype`.

**Tổng quan**: 837 cặp chạy, 403 thành công, 434 lỗi (372 do 2 bug trên, 62 do `E8_siglip2` lỗi code thật + `E9_evaclip` thiếu dependency `open_clip_torch`).

---

## Chú giải config — combo Stage-1 (encoder, 13 cái chạy được)

| Combo | Config (`stage1.feature_extractor`) |
|---|---|
| `E10_dinotxt` | model=dinotxt (dino.txt -- DINOv2 ViT-L/14 đóng băng + text alignment) |
| `E1a_dinov3_lvd1689m` | model=dinov3, dinov3_variant=vitb16 (mặc định), dinov3_pretrain_dataset=lvd1689m |
| `E1b_dinov3_sat493m` | model=dinov3, dinov3_variant=vitl16, dinov3_pretrain_dataset=sat493m (domain vệ tinh) |
| `E2_dinov2_registers` | dinov2_use_registers=true (cùng dinov2 vitb14, thêm register token) |
| `E3_multiscale_attn` | dinov2_pooling=multiscale_attn (nối attention-pooled token từ nhiều tầng transformer) |
| `E4_clip` | model=clip, clip_variant=vit-l/14 |
| `E4b_siglip` | model=siglip, siglip_variant=base (mặc định) |
| `E5a_ensemble_dinov2_clip` | model=ensemble, ensemble_dino_model=dinov2 (DINOv2+CLIP, baseline ensemble gốc) |
| `E5b_ensemble_dinov3_lvd1689m_clip` | model=ensemble, ensemble_dino_model=dinov3, dinov3_pretrain_dataset=lvd1689m |
| `E5c_ensemble_dinov3_sat493m_clip` | model=ensemble, ensemble_dino_model=dinov3, dinov3_variant=vitl16, dinov3_pretrain_dataset=sat493m |
| `E6_fgclip` | model=fgclip, fgclip_variant=base (heavy) |
| `E7_radio` | model=radio, radio_variant=c-radio_v3-b (heavy) |
| `stage1_default` | DINOv2 vitb14, pooling=cls, fusion=mean (mặc định config.yaml, không override) |

## Chú giải config — combo Stage-3 (31 cái)

| Combo | Config (`stage3.*` / `accuracy.cheap_boosters.*`) |
|---|---|
| `baseline_z2.5` | adaptive_threshold_method=z_score (mean/std), online_method=window_stat, adaptive_z_score=2.5 |
| `baseline_z2.0` | adaptive_threshold_method=z_score (mean/std), online_method=window_stat, adaptive_z_score=2.0 |
| `baseline_z1.5` | adaptive_threshold_method=z_score (mean/std), online_method=window_stat, adaptive_z_score=1.5 |
| `baseline_z1.0` | adaptive_threshold_method=z_score (mean/std), online_method=window_stat, adaptive_z_score=1.0 |
| `baseline_z0.5` | adaptive_threshold_method=z_score (mean/std), online_method=window_stat, adaptive_z_score=0.5 |
| `T2_zscore_robust_z2.5` | adaptive_threshold_method=z_score, adaptive_threshold_robust=true (median/MAD), adaptive_z_score=2.5 |
| `T2_zscore_robust_z2.0` | adaptive_threshold_method=z_score, adaptive_threshold_robust=true (median/MAD), adaptive_z_score=2.0 |
| `T2_zscore_robust_z1.5` | adaptive_threshold_method=z_score, adaptive_threshold_robust=true (median/MAD), adaptive_z_score=1.5 |
| `T2_zscore_robust_z1.0` | adaptive_threshold_method=z_score, adaptive_threshold_robust=true (median/MAD), adaptive_z_score=1.0 |
| `T2_zscore_robust_z0.5` | adaptive_threshold_method=z_score, adaptive_threshold_robust=true (median/MAD), adaptive_z_score=0.5 |
| `T3_otsu` | adaptive_threshold_method=otsu (online_method vẫn window_stat, z_score baseline mặc định) |
| `T4_gmm` | adaptive_threshold_method=gmm |
| `O2_aci` | adaptive_threshold_online_method=aci (Adaptive Conformal Inference) |
| `O3_saffron` | adaptive_threshold_online_method=saffron (online FDR control) |
| `O4_corruption_compensated` | adaptive_threshold_online_method=corruption_compensated (F-ROCP) |
| `V1_cluster_hdbscan_cosine` | verification_method=cluster, cluster_method=hdbscan, pairwise_metric=cosine (mặc định) |
| `V2_cluster_hdbscan_l1` | verification_method=cluster, cluster_method=hdbscan, pairwise_metric=l1 |
| `V3_cluster_hdbscan_mahalanobis` | verification_method=cluster, cluster_method=hdbscan, pairwise_metric=mahalanobis |
| `V4_cluster_spectral_cosine` | verification_method=cluster, cluster_method=spectral, pairwise_metric=cosine |
| `F1_negative_prototype` | negative_prototype_filter.enabled=true |
| `F2_identity_chain_sw0` | identity_chain_filter.enabled=true, spatial_weight=0.0 |
| `F2b_identity_chain_sw03` | identity_chain_filter.enabled=true, spatial_weight=0.3 |
| `F3_rmd` | stage3.similarity=rmd (thay cosine) |
| `F3a_l1` | stage3.similarity=l1 |
| `F3b_l2` | stage3.similarity=l2 |
| `F4_margin_cluster` | margin_verification.enabled=true + cluster_secondary_filter.enabled=true |
| `F4c_cluster_secondary_no_accumulate` | cluster_secondary_filter.enabled=true, accumulate_new_anchors=false |
| `F5_stacked` | chồng cả 4: negative_prototype_filter + cluster_secondary_filter + identity_chain_filter(sw=0.3) + margin_verification |
| `M1_mean` | accuracy.cheap_boosters.multi_ref_pooling=mean |
| `M3_min` | accuracy.cheap_boosters.multi_ref_pooling=min |
| `M4_agreement_weighted` | accuracy.cheap_boosters.multi_ref_pooling=agreement_weighted |

---

## Bảng kết quả đầy đủ — 403 combo, sắp theo F1 giảm dần

| Rank | Stage-1 (encoder) | Stage-3 (kỹ thuật) | P | R | F1 |
|---|---|---|---|---|---|
| 1 | `E6_fgclip` | `F4_margin_cluster` | 0.758 | 0.893 | 0.820 |
| 2 | `E6_fgclip` | `M4_agreement_weighted` | 0.712 | 0.940 | 0.810 |
| 3 | `E1a_dinov3_lvd1689m` | `T2_zscore_robust_z2.5` | 0.723 | 0.869 | 0.789 |
| 4 | `E6_fgclip` | `T2_zscore_robust_z2.5` | 0.672 | 0.952 | 0.788 |
| 5 | `E7_radio` | `M1_mean` | 0.722 | 0.833 | 0.773 |
| 6 | `E7_radio` | `M4_agreement_weighted` | 0.703 | 0.845 | 0.768 |
| 7 | `E4b_siglip` | `M3_min` | 0.655 | 0.905 | 0.760 |
| 8 | `E6_fgclip` | `F2b_identity_chain_sw03` | 0.632 | 0.881 | 0.736 |
| 9 | `E7_radio` | `T2_zscore_robust_z2.5` | 0.667 | 0.810 | 0.731 |
| 10 | `E4b_siglip` | `M4_agreement_weighted` | 0.654 | 0.810 | 0.723 |
| 11 | `E6_fgclip` | `baseline_z2.0` | 0.600 | 0.893 | 0.718 |
| 12 | `E6_fgclip` | `F2_identity_chain_sw0` | 0.600 | 0.893 | 0.718 |
| 13 | `E1a_dinov3_lvd1689m` | `F2b_identity_chain_sw03` | 0.653 | 0.786 | 0.714 |
| 14 | `E6_fgclip` | `F4c_cluster_secondary_no_accumulate` | 0.597 | 0.881 | 0.712 |
| 15 | `E4b_siglip` | `M1_mean` | 0.632 | 0.798 | 0.705 |
| 16 | `E6_fgclip` | `F3a_l1` | 0.557 | 0.929 | 0.696 |
| 17 | `E5b_ensemble_dinov3_lvd1689m_clip` | `M4_agreement_weighted` | 0.674 | 0.714 | 0.694 |
| 18 | `E6_fgclip` | `F3b_l2` | 0.564 | 0.893 | 0.691 |
| 19 | `E1a_dinov3_lvd1689m` | `F4_margin_cluster` | 0.615 | 0.762 | 0.681 |
| 20 | `E1a_dinov3_lvd1689m` | `M4_agreement_weighted` | 0.589 | 0.786 | 0.673 |
| 21 | `E1a_dinov3_lvd1689m` | `F2_identity_chain_sw0` | 0.576 | 0.810 | 0.673 |
| 22 | `E1a_dinov3_lvd1689m` | `T2_zscore_robust_z2.0` | 0.520 | 0.940 | 0.669 |
| 23 | `E1a_dinov3_lvd1689m` | `F3b_l2` | 0.556 | 0.833 | 0.667 |
| 24 | `E6_fgclip` | `M1_mean` | 0.596 | 0.738 | 0.660 |
| 25 | `E1a_dinov3_lvd1689m` | `baseline_z2.0` | 0.553 | 0.810 | 0.657 |
| 26 | `E1a_dinov3_lvd1689m` | `F4c_cluster_secondary_no_accumulate` | 0.551 | 0.774 | 0.644 |
| 27 | `E7_radio` | `baseline_z2.5` | 0.671 | 0.607 | 0.637 |
| 28 | `E6_fgclip` | `T2_zscore_robust_z2.0` | 0.479 | 0.952 | 0.637 |
| 29 | `E7_radio` | `F4_margin_cluster` | 0.562 | 0.702 | 0.624 |
| 30 | `E1a_dinov3_lvd1689m` | `baseline_z2.5` | 0.738 | 0.536 | 0.621 |
| 31 | `E10_dinotxt` | `F4_margin_cluster` | 0.602 | 0.631 | 0.616 |
| 32 | `E7_radio` | `T2_zscore_robust_z2.0` | 0.474 | 0.869 | 0.613 |
| 33 | `E7_radio` | `F2b_identity_chain_sw03` | 0.530 | 0.726 | 0.613 |
| 34 | `E6_fgclip` | `baseline_z2.5` | 0.788 | 0.488 | 0.603 |
| 35 | `E7_radio` | `M3_min` | 0.523 | 0.690 | 0.595 |
| 36 | `E6_fgclip` | `baseline_z1.5` | 0.432 | 0.952 | 0.595 |
| 37 | `E1a_dinov3_lvd1689m` | `baseline_z1.5` | 0.423 | 0.952 | 0.586 |
| 38 | `E10_dinotxt` | `F3a_l1` | 0.541 | 0.631 | 0.582 |
| 39 | `E7_radio` | `F3a_l1` | 0.464 | 0.762 | 0.577 |
| 40 | `E10_dinotxt` | `F2b_identity_chain_sw03` | 0.505 | 0.667 | 0.574 |
| 41 | `E7_radio` | `baseline_z2.0` | 0.466 | 0.738 | 0.571 |
| 42 | `E7_radio` | `F2_identity_chain_sw0` | 0.466 | 0.738 | 0.571 |
| 43 | `E7_radio` | `F3b_l2` | 0.453 | 0.738 | 0.561 |
| 44 | `E7_radio` | `F4c_cluster_secondary_no_accumulate` | 0.465 | 0.702 | 0.559 |
| 45 | `E4_clip` | `M3_min` | 0.568 | 0.548 | 0.558 |
| 46 | `E6_fgclip` | `T2_zscore_robust_z1.5` | 0.385 | 0.976 | 0.552 |
| 47 | `E10_dinotxt` | `T2_zscore_robust_z2.5` | 0.457 | 0.690 | 0.550 |
| 48 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F2b_identity_chain_sw03` | 0.510 | 0.583 | 0.544 |
| 49 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F4_margin_cluster` | 0.511 | 0.571 | 0.539 |
| 50 | `E10_dinotxt` | `M4_agreement_weighted` | 0.529 | 0.548 | 0.538 |
| 51 | `E7_radio` | `baseline_z1.5` | 0.388 | 0.869 | 0.537 |
| 52 | `E1a_dinov3_lvd1689m` | `T2_zscore_robust_z1.5` | 0.372 | 0.964 | 0.536 |
| 53 | `E10_dinotxt` | `baseline_z2.0` | 0.444 | 0.667 | 0.533 |
| 54 | `E10_dinotxt` | `F2_identity_chain_sw0` | 0.444 | 0.667 | 0.533 |
| 55 | `E10_dinotxt` | `F3b_l2` | 0.441 | 0.667 | 0.531 |
| 56 | `E10_dinotxt` | `F4c_cluster_secondary_no_accumulate` | 0.453 | 0.631 | 0.527 |
| 57 | `E10_dinotxt` | `baseline_z2.5` | 0.524 | 0.524 | 0.524 |
| 58 | `E1a_dinov3_lvd1689m` | `M1_mean` | 0.452 | 0.619 | 0.523 |
| 59 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T2_zscore_robust_z2.5` | 0.459 | 0.595 | 0.518 |
| 60 | `E5b_ensemble_dinov3_lvd1689m_clip` | `baseline_z2.5` | 0.563 | 0.476 | 0.516 |
| 61 | `E7_radio` | `T2_zscore_robust_z1.5` | 0.353 | 0.929 | 0.511 |
| 62 | `E10_dinotxt` | `T2_zscore_robust_z2.0` | 0.374 | 0.762 | 0.502 |
| 63 | `E5b_ensemble_dinov3_lvd1689m_clip` | `M1_mean` | 0.438 | 0.583 | 0.500 |
| 64 | `E6_fgclip` | `baseline_z1.0` | 0.335 | 0.976 | 0.498 |
| 65 | `E5b_ensemble_dinov3_lvd1689m_clip` | `baseline_z2.0` | 0.421 | 0.607 | 0.498 |
| 66 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F2_identity_chain_sw0` | 0.421 | 0.607 | 0.498 |
| 67 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F3b_l2` | 0.415 | 0.607 | 0.493 |
| 68 | `E4_clip` | `M4_agreement_weighted` | 0.618 | 0.405 | 0.489 |
| 69 | `E10_dinotxt` | `M1_mean` | 0.457 | 0.500 | 0.477 |
| 70 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F4c_cluster_secondary_no_accumulate` | 0.407 | 0.571 | 0.475 |
| 71 | `E10_dinotxt` | `baseline_z1.5` | 0.351 | 0.726 | 0.473 |
| 72 | `E1a_dinov3_lvd1689m` | `baseline_z1.0` | 0.306 | 0.964 | 0.464 |
| 73 | `E6_fgclip` | `T2_zscore_robust_z1.0` | 0.295 | 0.988 | 0.455 |
| 74 | `E7_radio` | `baseline_z1.0` | 0.302 | 0.917 | 0.454 |
| 75 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T2_zscore_robust_z2.0` | 0.339 | 0.667 | 0.450 |
| 76 | `E1a_dinov3_lvd1689m` | `F3a_l1` | 0.374 | 0.548 | 0.444 |
| 77 | `E5a_ensemble_dinov2_clip` | `M4_agreement_weighted` | 0.500 | 0.393 | 0.440 |
| 78 | `E5b_ensemble_dinov3_lvd1689m_clip` | `baseline_z1.5` | 0.320 | 0.690 | 0.438 |
| 79 | `E10_dinotxt` | `T2_zscore_robust_z1.5` | 0.296 | 0.798 | 0.432 |
| 80 | `E10_dinotxt` | `baseline_z1.0` | 0.289 | 0.810 | 0.426 |
| 81 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F3a_l1` | 0.348 | 0.548 | 0.426 |
| 82 | `E1a_dinov3_lvd1689m` | `T2_zscore_robust_z1.0` | 0.271 | 0.976 | 0.424 |
| 83 | `E7_radio` | `T2_zscore_robust_z1.0` | 0.268 | 0.976 | 0.421 |
| 84 | `E7_radio` | `T4_gmm` | 0.263 | 0.988 | 0.415 |
| 85 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T2_zscore_robust_z1.5` | 0.278 | 0.738 | 0.404 |
| 86 | `E10_dinotxt` | `T2_zscore_robust_z1.0` | 0.255 | 0.881 | 0.396 |
| 87 | `E6_fgclip` | `baseline_z0.5` | 0.243 | 0.988 | 0.391 |
| 88 | `E5a_ensemble_dinov2_clip` | `T2_zscore_robust_z2.0` | 0.374 | 0.405 | 0.389 |
| 89 | `E7_radio` | `T3_otsu` | 0.241 | 0.988 | 0.388 |
| 90 | `E7_radio` | `baseline_z0.5` | 0.240 | 0.988 | 0.386 |
| 91 | `E5b_ensemble_dinov3_lvd1689m_clip` | `baseline_z1.0` | 0.256 | 0.774 | 0.385 |
| 92 | `E1a_dinov3_lvd1689m` | `baseline_z0.5` | 0.238 | 0.988 | 0.383 |
| 93 | `E10_dinotxt` | `T3_otsu` | 0.242 | 0.905 | 0.382 |
| 94 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T2_zscore_robust_z1.0` | 0.244 | 0.869 | 0.381 |
| 95 | `E10_dinotxt` | `baseline_z0.5` | 0.239 | 0.929 | 0.380 |
| 96 | `E6_fgclip` | `M3_min` | 0.350 | 0.417 | 0.380 |
| 97 | `E5b_ensemble_dinov3_lvd1689m_clip` | `baseline_z0.5` | 0.237 | 0.952 | 0.380 |
| 98 | `E1a_dinov3_lvd1689m` | `M3_min` | 0.317 | 0.464 | 0.377 |
| 99 | `E6_fgclip` | `V2_cluster_hdbscan_l1` | 0.246 | 0.774 | 0.374 |
| 100 | `E4b_siglip` | `F3a_l1` | 0.373 | 0.369 | 0.371 |
| 101 | `E5a_ensemble_dinov2_clip` | `F2b_identity_chain_sw03` | 0.435 | 0.321 | 0.370 |
| 102 | `E1a_dinov3_lvd1689m` | `T3_otsu` | 0.227 | 0.988 | 0.369 |
| 103 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T3_otsu` | 0.229 | 0.952 | 0.369 |
| 104 | `E6_fgclip` | `T3_otsu` | 0.227 | 0.976 | 0.368 |
| 105 | `E6_fgclip` | `T2_zscore_robust_z0.5` | 0.225 | 0.988 | 0.366 |
| 106 | `E1a_dinov3_lvd1689m` | `T2_zscore_robust_z0.5` | 0.222 | 0.988 | 0.362 |
| 107 | `E6_fgclip` | `V1_cluster_hdbscan_cosine` | 0.242 | 0.714 | 0.361 |
| 108 | `E10_dinotxt` | `T2_zscore_robust_z0.5` | 0.219 | 0.952 | 0.356 |
| 109 | `E5a_ensemble_dinov2_clip` | `T2_zscore_robust_z1.5` | 0.261 | 0.548 | 0.354 |
| 110 | `E7_radio` | `T2_zscore_robust_z0.5` | 0.214 | 0.988 | 0.352 |
| 111 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T4_gmm` | 0.216 | 0.940 | 0.352 |
| 112 | `E6_fgclip` | `V4_cluster_spectral_cosine` | 0.225 | 0.798 | 0.351 |
| 113 | `E1a_dinov3_lvd1689m` | `T4_gmm` | 0.215 | 0.940 | 0.350 |
| 114 | `E4b_siglip` | `T2_zscore_robust_z2.0` | 0.354 | 0.345 | 0.349 |
| 115 | `E5b_ensemble_dinov3_lvd1689m_clip` | `T2_zscore_robust_z0.5` | 0.214 | 0.952 | 0.349 |
| 116 | `E10_dinotxt` | `T4_gmm` | 0.216 | 0.845 | 0.344 |
| 117 | `E6_fgclip` | `T4_gmm` | 0.208 | 0.964 | 0.342 |
| 118 | `E5a_ensemble_dinov2_clip` | `baseline_z1.5` | 0.251 | 0.536 | 0.342 |
| 119 | `E5a_ensemble_dinov2_clip` | `F3a_l1` | 0.350 | 0.333 | 0.341 |
| 120 | `E5a_ensemble_dinov2_clip` | `baseline_z2.0` | 0.346 | 0.333 | 0.339 |
| 121 | `E5a_ensemble_dinov2_clip` | `F2_identity_chain_sw0` | 0.346 | 0.333 | 0.339 |
| 122 | `E10_dinotxt` | `O3_saffron` | 0.478 | 0.262 | 0.338 |
| 123 | `E4b_siglip` | `F2b_identity_chain_sw03` | 0.468 | 0.262 | 0.336 |
| 124 | `E5a_ensemble_dinov2_clip` | `F4_margin_cluster` | 0.379 | 0.298 | 0.333 |
| 125 | `E5a_ensemble_dinov2_clip` | `F3b_l2` | 0.326 | 0.333 | 0.329 |
| 126 | `E4b_siglip` | `T2_zscore_robust_z1.5` | 0.250 | 0.476 | 0.328 |
| 127 | `E4b_siglip` | `baseline_z1.5` | 0.255 | 0.440 | 0.323 |
| 128 | `E5a_ensemble_dinov2_clip` | `T2_zscore_robust_z2.5` | 0.500 | 0.238 | 0.323 |
| 129 | `E5a_ensemble_dinov2_clip` | `baseline_z1.0` | 0.212 | 0.667 | 0.322 |
| 130 | `E7_radio` | `O4_corruption_compensated` | 0.288 | 0.357 | 0.319 |
| 131 | `E3_multiscale_attn` | `T2_zscore_robust_z1.5` | 0.329 | 0.310 | 0.319 |
| 132 | `E7_radio` | `O2_aci` | 0.284 | 0.345 | 0.312 |
| 133 | `E6_fgclip` | `O2_aci` | 0.292 | 0.333 | 0.311 |
| 134 | `E6_fgclip` | `O4_corruption_compensated` | 0.292 | 0.333 | 0.311 |
| 135 | `E5a_ensemble_dinov2_clip` | `T2_zscore_robust_z1.0` | 0.201 | 0.690 | 0.311 |
| 136 | `E5a_ensemble_dinov2_clip` | `F4c_cluster_secondary_no_accumulate` | 0.321 | 0.298 | 0.309 |
| 137 | `E3_multiscale_attn` | `baseline_z1.5` | 0.545 | 0.214 | 0.308 |
| 138 | `E4b_siglip` | `baseline_z2.5` | 0.630 | 0.202 | 0.306 |
| 139 | `E4b_siglip` | `T2_zscore_robust_z2.5` | 0.529 | 0.214 | 0.305 |
| 140 | `E4b_siglip` | `baseline_z2.0` | 0.333 | 0.274 | 0.301 |
| 141 | `E4b_siglip` | `F2_identity_chain_sw0` | 0.333 | 0.274 | 0.301 |
| 142 | `E4b_siglip` | `F3b_l2` | 0.308 | 0.286 | 0.296 |
| 143 | `E3_multiscale_attn` | `baseline_z1.0` | 0.206 | 0.524 | 0.295 |
| 144 | `E5a_ensemble_dinov2_clip` | `baseline_z0.5` | 0.183 | 0.750 | 0.294 |
| 145 | `E10_dinotxt` | `O2_aci` | 0.277 | 0.310 | 0.292 |
| 146 | `E10_dinotxt` | `O4_corruption_compensated` | 0.277 | 0.310 | 0.292 |
| 147 | `E4b_siglip` | `F4_margin_cluster` | 0.328 | 0.262 | 0.291 |
| 148 | `E4b_siglip` | `F4c_cluster_secondary_no_accumulate` | 0.324 | 0.262 | 0.289 |
| 149 | `E5a_ensemble_dinov2_clip` | `T2_zscore_robust_z0.5` | 0.178 | 0.750 | 0.288 |
| 150 | `E1a_dinov3_lvd1689m` | `O2_aci` | 0.265 | 0.310 | 0.286 |
| 151 | `E1a_dinov3_lvd1689m` | `O4_corruption_compensated` | 0.265 | 0.310 | 0.286 |
| 152 | `E3_multiscale_attn` | `T2_zscore_robust_z1.0` | 0.195 | 0.524 | 0.284 |
| 153 | `E5a_ensemble_dinov2_clip` | `T3_otsu` | 0.169 | 0.786 | 0.278 |
| 154 | `E4b_siglip` | `T2_zscore_robust_z0.5` | 0.167 | 0.750 | 0.273 |
| 155 | `E4b_siglip` | `T2_zscore_robust_z1.0` | 0.181 | 0.560 | 0.273 |
| 156 | `E4b_siglip` | `baseline_z1.0` | 0.182 | 0.548 | 0.273 |
| 157 | `stage1_default` | `T3_otsu` | 0.164 | 0.798 | 0.272 |
| 158 | `E5a_ensemble_dinov2_clip` | `O2_aci` | 0.258 | 0.286 | 0.271 |
| 159 | `E5a_ensemble_dinov2_clip` | `O4_corruption_compensated` | 0.258 | 0.286 | 0.271 |
| 160 | `E5b_ensemble_dinov3_lvd1689m_clip` | `O2_aci` | 0.248 | 0.298 | 0.270 |
| 161 | `E1a_dinov3_lvd1689m` | `V3_cluster_hdbscan_mahalanobis` | 0.155 | 0.988 | 0.268 |
| 162 | `E5b_ensemble_dinov3_lvd1689m_clip` | `O4_corruption_compensated` | 0.243 | 0.298 | 0.267 |
| 163 | `stage1_default` | `T4_gmm` | 0.155 | 0.893 | 0.264 |
| 164 | `E5a_ensemble_dinov2_clip` | `T4_gmm` | 0.156 | 0.821 | 0.262 |
| 165 | `E4b_siglip` | `V4_cluster_spectral_cosine` | 0.150 | 0.929 | 0.258 |
| 166 | `E4b_siglip` | `baseline_z0.5` | 0.158 | 0.702 | 0.258 |
| 167 | `stage1_default` | `baseline_z0.5` | 0.159 | 0.631 | 0.254 |
| 168 | `E4b_siglip` | `T3_otsu` | 0.149 | 0.833 | 0.253 |
| 169 | `E4_clip` | `M1_mean` | 0.500 | 0.167 | 0.250 |
| 170 | `E3_multiscale_attn` | `T2_zscore_robust_z0.5` | 0.151 | 0.702 | 0.248 |
| 171 | `E5c_ensemble_dinov3_sat493m_clip` | `T4_gmm` | 0.146 | 0.833 | 0.248 |
| 172 | `E6_fgclip` | `V3_cluster_hdbscan_mahalanobis` | 0.143 | 0.917 | 0.248 |
| 173 | `E4b_siglip` | `V3_cluster_hdbscan_mahalanobis` | 0.142 | 0.929 | 0.246 |
| 174 | `E7_radio` | `O3_saffron` | 0.348 | 0.190 | 0.246 |
| 175 | `E5b_ensemble_dinov3_lvd1689m_clip` | `V3_cluster_hdbscan_mahalanobis` | 0.141 | 0.964 | 0.245 |
| 176 | `E7_radio` | `V3_cluster_hdbscan_mahalanobis` | 0.139 | 0.952 | 0.243 |
| 177 | `E4b_siglip` | `T4_gmm` | 0.139 | 0.893 | 0.241 |
| 178 | `E5c_ensemble_dinov3_sat493m_clip` | `baseline_z0.5` | 0.148 | 0.619 | 0.239 |
| 179 | `stage1_default` | `T2_zscore_robust_z0.5` | 0.150 | 0.571 | 0.237 |
| 180 | `E5b_ensemble_dinov3_lvd1689m_clip` | `M3_min` | 0.235 | 0.238 | 0.237 |
| 181 | `E2_dinov2_registers` | `T3_otsu` | 0.145 | 0.643 | 0.236 |
| 182 | `E3_multiscale_attn` | `baseline_z0.5` | 0.141 | 0.714 | 0.236 |
| 183 | `E5c_ensemble_dinov3_sat493m_clip` | `T2_zscore_robust_z0.5` | 0.144 | 0.631 | 0.235 |
| 184 | `E5a_ensemble_dinov2_clip` | `V3_cluster_hdbscan_mahalanobis` | 0.133 | 0.905 | 0.232 |
| 185 | `E5c_ensemble_dinov3_sat493m_clip` | `T3_otsu` | 0.137 | 0.690 | 0.229 |
| 186 | `E4b_siglip` | `O4_corruption_compensated` | 0.217 | 0.238 | 0.227 |
| 187 | `E2_dinov2_registers` | `baseline_z0.5` | 0.142 | 0.560 | 0.227 |
| 188 | `E2_dinov2_registers` | `T4_gmm` | 0.136 | 0.655 | 0.226 |
| 189 | `E5c_ensemble_dinov3_sat493m_clip` | `T2_zscore_robust_z1.0` | 0.150 | 0.452 | 0.225 |
| 190 | `E2_dinov2_registers` | `T2_zscore_robust_z1.0` | 0.146 | 0.476 | 0.223 |
| 191 | `E1b_dinov3_sat493m` | `T4_gmm` | 0.131 | 0.738 | 0.223 |
| 192 | `E3_multiscale_attn` | `V4_cluster_spectral_cosine` | 0.127 | 0.857 | 0.221 |
| 193 | `E1a_dinov3_lvd1689m` | `O3_saffron` | 0.371 | 0.155 | 0.218 |
| 194 | `E3_multiscale_attn` | `T3_otsu` | 0.126 | 0.821 | 0.218 |
| 195 | `E4b_siglip` | `O2_aci` | 0.209 | 0.226 | 0.217 |
| 196 | `E2_dinov2_registers` | `T2_zscore_robust_z0.5` | 0.135 | 0.560 | 0.217 |
| 197 | `E3_multiscale_attn` | `T4_gmm` | 0.125 | 0.821 | 0.217 |
| 198 | `E5c_ensemble_dinov3_sat493m_clip` | `baseline_z1.0` | 0.144 | 0.417 | 0.214 |
| 199 | `E2_dinov2_registers` | `baseline_z1.0` | 0.140 | 0.417 | 0.210 |
| 200 | `E4_clip` | `V4_cluster_spectral_cosine` | 0.111 | 0.726 | 0.192 |
| 201 | `E4_clip` | `T4_gmm` | 0.110 | 0.679 | 0.189 |
| 202 | `E6_fgclip` | `O3_saffron` | 0.241 | 0.155 | 0.188 |
| 203 | `E4_clip` | `V3_cluster_hdbscan_mahalanobis` | 0.107 | 0.679 | 0.185 |
| 204 | `E5a_ensemble_dinov2_clip` | `baseline_z2.5` | 0.417 | 0.119 | 0.185 |
| 205 | `E4_clip` | `T3_otsu` | 0.108 | 0.595 | 0.183 |
| 206 | `E5c_ensemble_dinov3_sat493m_clip` | `T2_zscore_robust_z1.5` | 0.138 | 0.274 | 0.183 |
| 207 | `E1b_dinov3_sat493m` | `T3_otsu` | 0.105 | 0.536 | 0.175 |
| 208 | `E2_dinov2_registers` | `V3_cluster_hdbscan_mahalanobis` | 0.104 | 0.536 | 0.175 |
| 209 | `stage1_default` | `V3_cluster_hdbscan_mahalanobis` | 0.099 | 0.631 | 0.172 |
| 210 | `stage1_default` | `baseline_z1.0` | 0.113 | 0.345 | 0.171 |
| 211 | `E5c_ensemble_dinov3_sat493m_clip` | `F3_rmd` | 0.161 | 0.179 | 0.169 |
| 212 | `E3_multiscale_attn` | `O2_aci` | 0.156 | 0.179 | 0.167 |
| 213 | `E3_multiscale_attn` | `O4_corruption_compensated` | 0.156 | 0.179 | 0.167 |
| 214 | `E5a_ensemble_dinov2_clip` | `O3_saffron` | 0.375 | 0.107 | 0.167 |
| 215 | `E3_multiscale_attn` | `O3_saffron` | 0.346 | 0.107 | 0.164 |
| 216 | `E7_radio` | `V1_cluster_hdbscan_cosine` | 0.118 | 0.262 | 0.163 |
| 217 | `E5c_ensemble_dinov3_sat493m_clip` | `baseline_z1.5` | 0.123 | 0.226 | 0.160 |
| 218 | `E4_clip` | `baseline_z0.5` | 0.097 | 0.405 | 0.157 |
| 219 | `E6_fgclip` | `F1_negative_prototype` | 1.000 | 0.083 | 0.154 |
| 220 | `E6_fgclip` | `F5_stacked` | 1.000 | 0.083 | 0.154 |
| 221 | `E7_radio` | `F1_negative_prototype` | 1.000 | 0.083 | 0.154 |
| 222 | `E4b_siglip` | `O3_saffron` | 0.250 | 0.107 | 0.150 |
| 223 | `E10_dinotxt` | `F3_rmd` | 0.154 | 0.143 | 0.148 |
| 224 | `stage1_default` | `T2_zscore_robust_z1.0` | 0.099 | 0.286 | 0.147 |
| 225 | `E5b_ensemble_dinov3_lvd1689m_clip` | `O3_saffron` | 0.225 | 0.107 | 0.145 |
| 226 | `E4b_siglip` | `V2_cluster_hdbscan_l1` | 0.082 | 0.512 | 0.142 |
| 227 | `E4b_siglip` | `V1_cluster_hdbscan_cosine` | 0.083 | 0.488 | 0.142 |
| 228 | `E2_dinov2_registers` | `T2_zscore_robust_z1.5` | 0.101 | 0.238 | 0.141 |
| 229 | `E4_clip` | `F2b_identity_chain_sw03` | 0.267 | 0.095 | 0.140 |
| 230 | `E4_clip` | `T2_zscore_robust_z0.5` | 0.087 | 0.357 | 0.140 |
| 231 | `E5c_ensemble_dinov3_sat493m_clip` | `T2_zscore_robust_z2.0` | 0.136 | 0.143 | 0.140 |
| 232 | `E4b_siglip` | `F3_rmd` | 0.141 | 0.131 | 0.136 |
| 233 | `E5c_ensemble_dinov3_sat493m_clip` | `O2_aci` | 0.128 | 0.143 | 0.135 |
| 234 | `E5c_ensemble_dinov3_sat493m_clip` | `O4_corruption_compensated` | 0.128 | 0.143 | 0.135 |
| 235 | `E2_dinov2_registers` | `baseline_z1.5` | 0.098 | 0.214 | 0.134 |
| 236 | `E5a_ensemble_dinov2_clip` | `F1_negative_prototype` | 1.000 | 0.071 | 0.133 |
| 237 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F1_negative_prototype` | 1.000 | 0.071 | 0.133 |
| 238 | `E10_dinotxt` | `F1_negative_prototype` | 1.000 | 0.071 | 0.133 |
| 239 | `E5c_ensemble_dinov3_sat493m_clip` | `T2_zscore_robust_z2.5` | 0.211 | 0.095 | 0.131 |
| 240 | `E4_clip` | `baseline_z2.0` | 0.195 | 0.095 | 0.128 |
| 241 | `E4_clip` | `F2_identity_chain_sw0` | 0.195 | 0.095 | 0.128 |
| 242 | `E3_multiscale_attn` | `T2_zscore_robust_z2.0` | 0.600 | 0.071 | 0.128 |
| 243 | `E3_multiscale_attn` | `F3_rmd` | 0.137 | 0.119 | 0.127 |
| 244 | `E4_clip` | `T2_zscore_robust_z1.0` | 0.087 | 0.238 | 0.127 |
| 245 | `E4_clip` | `V2_cluster_hdbscan_l1` | 0.072 | 0.452 | 0.125 |
| 246 | `E4_clip` | `O2_aci` | 0.118 | 0.131 | 0.124 |
| 247 | `E4_clip` | `O4_corruption_compensated` | 0.118 | 0.131 | 0.124 |
| 248 | `E5c_ensemble_dinov3_sat493m_clip` | `F3a_l1` | 0.116 | 0.131 | 0.123 |
| 249 | `E2_dinov2_registers` | `O2_aci` | 0.113 | 0.131 | 0.122 |
| 250 | `E7_radio` | `V4_cluster_spectral_cosine` | 0.087 | 0.202 | 0.121 |
| 251 | `E4_clip` | `baseline_z1.0` | 0.083 | 0.226 | 0.121 |
| 252 | `E2_dinov2_registers` | `F3b_l2` | 0.104 | 0.143 | 0.121 |
| 253 | `E4_clip` | `F3b_l2` | 0.163 | 0.095 | 0.120 |
| 254 | `E2_dinov2_registers` | `O4_corruption_compensated` | 0.111 | 0.131 | 0.120 |
| 255 | `E2_dinov2_registers` | `T2_zscore_robust_z2.0` | 0.096 | 0.155 | 0.119 |
| 256 | `E2_dinov2_registers` | `F2b_identity_chain_sw03` | 0.127 | 0.107 | 0.116 |
| 257 | `E2_dinov2_registers` | `baseline_z2.0` | 0.102 | 0.131 | 0.115 |
| 258 | `E2_dinov2_registers` | `F2_identity_chain_sw0` | 0.102 | 0.131 | 0.115 |
| 259 | `E1a_dinov3_lvd1689m` | `F1_negative_prototype` | 1.000 | 0.060 | 0.112 |
| 260 | `E4b_siglip` | `F1_negative_prototype` | 1.000 | 0.060 | 0.112 |
| 261 | `E7_radio` | `F3_rmd` | 0.115 | 0.107 | 0.111 |
| 262 | `E1b_dinov3_sat493m` | `baseline_z0.5` | 0.068 | 0.298 | 0.111 |
| 263 | `E5c_ensemble_dinov3_sat493m_clip` | `M4_agreement_weighted` | 0.127 | 0.095 | 0.109 |
| 264 | `E3_multiscale_attn` | `baseline_z2.0` | 0.625 | 0.060 | 0.109 |
| 265 | `E3_multiscale_attn` | `F2_identity_chain_sw0` | 0.625 | 0.060 | 0.109 |
| 266 | `E3_multiscale_attn` | `F3b_l2` | 0.625 | 0.060 | 0.109 |
| 267 | `E4_clip` | `V1_cluster_hdbscan_cosine` | 0.063 | 0.369 | 0.107 |
| 268 | `E1b_dinov3_sat493m` | `T2_zscore_robust_z0.5` | 0.065 | 0.298 | 0.107 |
| 269 | `E4_clip` | `T2_zscore_robust_z2.0` | 0.146 | 0.083 | 0.106 |
| 270 | `E1a_dinov3_lvd1689m` | `F3_rmd` | 0.111 | 0.095 | 0.103 |
| 271 | `E4_clip` | `baseline_z1.5` | 0.084 | 0.131 | 0.102 |
| 272 | `E4_clip` | `F4_margin_cluster` | 0.158 | 0.071 | 0.098 |
| 273 | `E2_dinov2_registers` | `F3_rmd` | 0.100 | 0.095 | 0.098 |
| 274 | `E4_clip` | `F4c_cluster_secondary_no_accumulate` | 0.154 | 0.071 | 0.098 |
| 275 | `E4_clip` | `T2_zscore_robust_z1.5` | 0.077 | 0.131 | 0.097 |
| 276 | `E2_dinov2_registers` | `F4_margin_cluster` | 0.098 | 0.095 | 0.096 |
| 277 | `E5b_ensemble_dinov3_lvd1689m_clip` | `V4_cluster_spectral_cosine` | 0.061 | 0.202 | 0.094 |
| 278 | `stage1_default` | `baseline_z1.5` | 0.073 | 0.131 | 0.094 |
| 279 | `stage1_default` | `O2_aci` | 0.092 | 0.095 | 0.094 |
| 280 | `stage1_default` | `O4_corruption_compensated` | 0.092 | 0.095 | 0.094 |
| 281 | `E5c_ensemble_dinov3_sat493m_clip` | `F3b_l2` | 0.089 | 0.095 | 0.092 |
| 282 | `E2_dinov2_registers` | `F1_negative_prototype` | 1.000 | 0.048 | 0.091 |
| 283 | `E4b_siglip` | `F5_stacked` | 1.000 | 0.048 | 0.091 |
| 284 | `E7_radio` | `F5_stacked` | 1.000 | 0.048 | 0.091 |
| 285 | `E3_multiscale_attn` | `F3a_l1` | 0.800 | 0.048 | 0.090 |
| 286 | `E2_dinov2_registers` | `baseline_z2.5` | 0.105 | 0.071 | 0.085 |
| 287 | `E2_dinov2_registers` | `F4c_cluster_secondary_no_accumulate` | 0.077 | 0.095 | 0.085 |
| 288 | `E5c_ensemble_dinov3_sat493m_clip` | `baseline_z2.0` | 0.086 | 0.083 | 0.085 |
| 289 | `E5c_ensemble_dinov3_sat493m_clip` | `F2_identity_chain_sw0` | 0.086 | 0.083 | 0.085 |
| 290 | `E2_dinov2_registers` | `F3a_l1` | 0.076 | 0.095 | 0.085 |
| 291 | `stage1_default` | `F3_rmd` | 0.080 | 0.083 | 0.081 |
| 292 | `E4_clip` | `T2_zscore_robust_z2.5` | 0.250 | 0.048 | 0.080 |
| 293 | `E5a_ensemble_dinov2_clip` | `V4_cluster_spectral_cosine` | 0.051 | 0.179 | 0.079 |
| 294 | `E2_dinov2_registers` | `T2_zscore_robust_z2.5` | 0.087 | 0.071 | 0.078 |
| 295 | `E5c_ensemble_dinov3_sat493m_clip` | `F2b_identity_chain_sw03` | 0.106 | 0.060 | 0.076 |
| 296 | `E5b_ensemble_dinov3_lvd1689m_clip` | `V2_cluster_hdbscan_l1` | 0.050 | 0.155 | 0.076 |
| 297 | `stage1_default` | `T2_zscore_robust_z1.5` | 0.062 | 0.095 | 0.075 |
| 298 | `E6_fgclip` | `F3_rmd` | 0.073 | 0.071 | 0.072 |
| 299 | `stage1_default` | `F1_negative_prototype` | 1.000 | 0.036 | 0.069 |
| 300 | `E1b_dinov3_sat493m` | `F1_negative_prototype` | 1.000 | 0.036 | 0.069 |
| 301 | `E3_multiscale_attn` | `baseline_z2.5` | 1.000 | 0.036 | 0.069 |
| 302 | `E3_multiscale_attn` | `F1_negative_prototype` | 1.000 | 0.036 | 0.069 |
| 303 | `E5a_ensemble_dinov2_clip` | `F5_stacked` | 1.000 | 0.036 | 0.069 |
| 304 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F5_stacked` | 1.000 | 0.036 | 0.069 |
| 305 | `E5c_ensemble_dinov3_sat493m_clip` | `F1_negative_prototype` | 1.000 | 0.036 | 0.069 |
| 306 | `E10_dinotxt` | `F5_stacked` | 1.000 | 0.036 | 0.069 |
| 307 | `E3_multiscale_attn` | `M4_agreement_weighted` | 0.750 | 0.036 | 0.068 |
| 308 | `E3_multiscale_attn` | `T2_zscore_robust_z2.5` | 0.600 | 0.036 | 0.067 |
| 309 | `E3_multiscale_attn` | `M1_mean` | 0.600 | 0.036 | 0.067 |
| 310 | `E3_multiscale_attn` | `F2b_identity_chain_sw03` | 0.500 | 0.036 | 0.067 |
| 311 | `E4_clip` | `baseline_z2.5` | 0.429 | 0.036 | 0.066 |
| 312 | `stage1_default` | `M3_min` | 0.072 | 0.060 | 0.065 |
| 313 | `E4_clip` | `F3a_l1` | 0.098 | 0.048 | 0.064 |
| 314 | `E5b_ensemble_dinov3_lvd1689m_clip` | `F3_rmd` | 0.066 | 0.060 | 0.062 |
| 315 | `E2_dinov2_registers` | `O3_saffron` | 0.087 | 0.048 | 0.062 |
| 316 | `E5a_ensemble_dinov2_clip` | `M1_mean` | 0.087 | 0.048 | 0.062 |
| 317 | `E5c_ensemble_dinov3_sat493m_clip` | `F4_margin_cluster` | 0.078 | 0.048 | 0.059 |
| 318 | `E1a_dinov3_lvd1689m` | `V4_cluster_spectral_cosine` | 0.045 | 0.083 | 0.059 |
| 319 | `stage1_default` | `T2_zscore_robust_z2.0` | 0.077 | 0.048 | 0.059 |
| 320 | `E2_dinov2_registers` | `M1_mean` | 0.057 | 0.060 | 0.058 |
| 321 | `E2_dinov2_registers` | `M4_agreement_weighted` | 0.057 | 0.060 | 0.058 |
| 322 | `E1b_dinov3_sat493m` | `T2_zscore_robust_z1.0` | 0.037 | 0.131 | 0.058 |
| 323 | `stage1_default` | `F2_identity_chain_sw0` | 0.074 | 0.048 | 0.058 |
| 324 | `stage1_default` | `baseline_z2.5` | 0.150 | 0.036 | 0.058 |
| 325 | `stage1_default` | `baseline_z2.0` | 0.073 | 0.048 | 0.058 |
| 326 | `stage1_default` | `T2_zscore_robust_z2.5` | 0.130 | 0.036 | 0.056 |
| 327 | `E4_clip` | `O3_saffron` | 0.130 | 0.036 | 0.056 |
| 328 | `stage1_default` | `M1_mean` | 0.067 | 0.048 | 0.056 |
| 329 | `E7_radio` | `V2_cluster_hdbscan_l1` | 0.041 | 0.083 | 0.055 |
| 330 | `stage1_default` | `F3b_l2` | 0.065 | 0.048 | 0.055 |
| 331 | `E5c_ensemble_dinov3_sat493m_clip` | `O3_saffron` | 0.115 | 0.036 | 0.055 |
| 332 | `E1b_dinov3_sat493m` | `baseline_z2.5` | 0.103 | 0.036 | 0.053 |
| 333 | `E5c_ensemble_dinov3_sat493m_clip` | `baseline_z2.5` | 0.103 | 0.036 | 0.053 |
| 334 | `stage1_default` | `F2b_identity_chain_sw03` | 0.100 | 0.036 | 0.053 |
| 335 | `E5a_ensemble_dinov2_clip` | `F3_rmd` | 0.053 | 0.048 | 0.050 |
| 336 | `E5c_ensemble_dinov3_sat493m_clip` | `F4c_cluster_secondary_no_accumulate` | 0.053 | 0.048 | 0.050 |
| 337 | `stage1_default` | `O3_saffron` | 0.083 | 0.036 | 0.050 |
| 338 | `E1b_dinov3_sat493m` | `F3_rmd` | 0.051 | 0.048 | 0.049 |
| 339 | `E1b_dinov3_sat493m` | `T2_zscore_robust_z2.5` | 0.075 | 0.036 | 0.048 |
| 340 | `E4_clip` | `F3_rmd` | 0.049 | 0.048 | 0.048 |
| 341 | `E3_multiscale_attn` | `M3_min` | 0.071 | 0.036 | 0.048 |
| 342 | `E1a_dinov3_lvd1689m` | `F5_stacked` | 1.000 | 0.024 | 0.047 |
| 343 | `E4_clip` | `F1_negative_prototype` | 1.000 | 0.024 | 0.047 |
| 344 | `stage1_default` | `V4_cluster_spectral_cosine` | 0.034 | 0.071 | 0.046 |
| 345 | `E3_multiscale_attn` | `F4_margin_cluster` | 0.400 | 0.024 | 0.045 |
| 346 | `E3_multiscale_attn` | `F4c_cluster_secondary_no_accumulate` | 0.400 | 0.024 | 0.045 |
| 347 | `stage1_default` | `M4_agreement_weighted` | 0.055 | 0.036 | 0.043 |
| 348 | `E1b_dinov3_sat493m` | `F2b_identity_chain_sw03` | 0.050 | 0.036 | 0.042 |
| 349 | `E5c_ensemble_dinov3_sat493m_clip` | `M1_mean` | 0.050 | 0.036 | 0.042 |
| 350 | `E1b_dinov3_sat493m` | `T2_zscore_robust_z1.5` | 0.029 | 0.071 | 0.041 |
| 351 | `stage1_default` | `F3a_l1` | 0.047 | 0.036 | 0.041 |
| 352 | `E1b_dinov3_sat493m` | `baseline_z1.0` | 0.025 | 0.083 | 0.039 |
| 353 | `E1b_dinov3_sat493m` | `M1_mean` | 0.043 | 0.036 | 0.039 |
| 354 | `E5a_ensemble_dinov2_clip` | `M3_min` | 0.043 | 0.036 | 0.039 |
| 355 | `E1b_dinov3_sat493m` | `M4_agreement_weighted` | 0.042 | 0.036 | 0.038 |
| 356 | `E1a_dinov3_lvd1689m` | `V1_cluster_hdbscan_cosine` | 0.032 | 0.048 | 0.038 |
| 357 | `E1a_dinov3_lvd1689m` | `V2_cluster_hdbscan_l1` | 0.032 | 0.048 | 0.038 |
| 358 | `E1b_dinov3_sat493m` | `M3_min` | 0.037 | 0.036 | 0.036 |
| 359 | `E1b_dinov3_sat493m` | `baseline_z1.5` | 0.026 | 0.060 | 0.036 |
| 360 | `E1b_dinov3_sat493m` | `F2_identity_chain_sw0` | 0.036 | 0.036 | 0.036 |
| 361 | `E1b_dinov3_sat493m` | `baseline_z2.0` | 0.034 | 0.036 | 0.035 |
| 362 | `E1b_dinov3_sat493m` | `O2_aci` | 0.034 | 0.036 | 0.035 |
| 363 | `E1b_dinov3_sat493m` | `O4_corruption_compensated` | 0.034 | 0.036 | 0.035 |
| 364 | `E1b_dinov3_sat493m` | `F3b_l2` | 0.033 | 0.036 | 0.034 |
| 365 | `E1b_dinov3_sat493m` | `T2_zscore_robust_z2.0` | 0.032 | 0.036 | 0.034 |
| 366 | `E1b_dinov3_sat493m` | `F3a_l1` | 0.031 | 0.036 | 0.033 |
| 367 | `E2_dinov2_registers` | `V1_cluster_hdbscan_cosine` | 0.025 | 0.048 | 0.033 |
| 368 | `E2_dinov2_registers` | `V2_cluster_hdbscan_l1` | 0.025 | 0.048 | 0.033 |
| 369 | `E2_dinov2_registers` | `V4_cluster_spectral_cosine` | 0.025 | 0.048 | 0.032 |
| 370 | `E3_multiscale_attn` | `V1_cluster_hdbscan_cosine` | 0.024 | 0.048 | 0.032 |
| 371 | `E3_multiscale_attn` | `V2_cluster_hdbscan_l1` | 0.024 | 0.048 | 0.032 |
| 372 | `E3_multiscale_attn` | `V3_cluster_hdbscan_mahalanobis` | 0.024 | 0.048 | 0.032 |
| 373 | `E10_dinotxt` | `V1_cluster_hdbscan_cosine` | 0.024 | 0.048 | 0.032 |
| 374 | `E10_dinotxt` | `V2_cluster_hdbscan_l1` | 0.024 | 0.048 | 0.032 |
| 375 | `E10_dinotxt` | `V3_cluster_hdbscan_mahalanobis` | 0.024 | 0.048 | 0.032 |
| 376 | `E10_dinotxt` | `V4_cluster_spectral_cosine` | 0.024 | 0.048 | 0.032 |
| 377 | `E5c_ensemble_dinov3_sat493m_clip` | `V4_cluster_spectral_cosine` | 0.020 | 0.060 | 0.030 |
| 378 | `E5b_ensemble_dinov3_lvd1689m_clip` | `V1_cluster_hdbscan_cosine` | 0.021 | 0.048 | 0.029 |
| 379 | `E2_dinov2_registers` | `M3_min` | 0.029 | 0.024 | 0.026 |
| 380 | `stage1_default` | `V1_cluster_hdbscan_cosine` | 0.019 | 0.036 | 0.024 |
| 381 | `stage1_default` | `V2_cluster_hdbscan_l1` | 0.019 | 0.036 | 0.024 |
| 382 | `E5a_ensemble_dinov2_clip` | `V1_cluster_hdbscan_cosine` | 0.018 | 0.036 | 0.024 |
| 383 | `E5c_ensemble_dinov3_sat493m_clip` | `V1_cluster_hdbscan_cosine` | 0.018 | 0.036 | 0.024 |
| 384 | `E5c_ensemble_dinov3_sat493m_clip` | `V2_cluster_hdbscan_l1` | 0.018 | 0.036 | 0.024 |
| 385 | `E5c_ensemble_dinov3_sat493m_clip` | `V3_cluster_hdbscan_mahalanobis` | 0.018 | 0.036 | 0.024 |
| 386 | `E5a_ensemble_dinov2_clip` | `V2_cluster_hdbscan_l1` | 0.018 | 0.036 | 0.024 |
| 387 | `E5c_ensemble_dinov3_sat493m_clip` | `M3_min` | 0.023 | 0.024 | 0.023 |
| 388 | `stage1_default` | `F4_margin_cluster` | 0.025 | 0.012 | 0.016 |
| 389 | `E1b_dinov3_sat493m` | `V1_cluster_hdbscan_cosine` | 0.012 | 0.024 | 0.016 |
| 390 | `E1b_dinov3_sat493m` | `V2_cluster_hdbscan_l1` | 0.012 | 0.024 | 0.016 |
| 391 | `E1b_dinov3_sat493m` | `V3_cluster_hdbscan_mahalanobis` | 0.012 | 0.024 | 0.016 |
| 392 | `E1b_dinov3_sat493m` | `V4_cluster_spectral_cosine` | 0.012 | 0.024 | 0.016 |
| 393 | `E10_dinotxt` | `M3_min` | 0.022 | 0.012 | 0.016 |
| 394 | `stage1_default` | `F4c_cluster_secondary_no_accumulate` | 0.020 | 0.012 | 0.015 |
| 395 | `stage1_default` | `F5_stacked` | 0.000 | 0.000 | 0.000 |
| 396 | `E1b_dinov3_sat493m` | `O3_saffron` | 0.000 | 0.000 | 0.000 |
| 397 | `E1b_dinov3_sat493m` | `F4_margin_cluster` | 0.000 | 0.000 | 0.000 |
| 398 | `E1b_dinov3_sat493m` | `F4c_cluster_secondary_no_accumulate` | 0.000 | 0.000 | 0.000 |
| 399 | `E1b_dinov3_sat493m` | `F5_stacked` | 0.000 | 0.000 | 0.000 |
| 400 | `E2_dinov2_registers` | `F5_stacked` | 0.000 | 0.000 | 0.000 |
| 401 | `E3_multiscale_attn` | `F5_stacked` | 0.000 | 0.000 | 0.000 |
| 402 | `E4_clip` | `F5_stacked` | 0.000 | 0.000 | 0.000 |
| 403 | `E5c_ensemble_dinov3_sat493m_clip` | `F5_stacked` | 0.000 | 0.000 | 0.000 |

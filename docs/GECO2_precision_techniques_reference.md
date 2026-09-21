# Tổng hợp các kỹ thuật cải thiện precision cho Stage 3 (GeCo2 candidate verification)

**Status:** tất cả các kỹ thuật trong file này đều **tắt mặc định (opt-in)** và **NOT YET VALIDATED** trên dữ liệu thật của dự án — mới chỉ có unit test/integration test tổng hợp (synthetic). Đây là tài liệu tra cứu (reference), không phải khuyến nghị bật mặc định.

Bối cảnh đầy đủ (lý do ra đời, các thử nghiệm A/B đã chạy, deep-research report) nằm ở [GECO2_precision_improvements_plan.md](GECO2_precision_improvements_plan.md). File này chỉ tập trung: **mỗi kỹ thuật hoạt động thế nào và dùng ra sao**. Xem thêm [GECO2_cluster_verification_guide.md](GECO2_cluster_verification_guide.md) cho 2 cơ chế cluster-verification gốc (`verification_method=cluster` và `dynamic_prototype.cluster_verification`) — không nhắc lại ở đây, chỉ nhắc tới khi liên quan trực tiếp (`cluster_secondary_filter` dùng lại chung primitive).

## 0. Vì sao có các kỹ thuật này

Thử nghiệm thật (sample IDCard_0) đã xác nhận: ngưỡng `adaptive_threshold_online` (z-score chạy trên cửa sổ, causal) là baseline tốt nhất hiện có (F1=0.706, P=0.646, R=0.778) — nhưng precision 64.6% vẫn còn thấp: một track sai do false positive gây hại nhiều hơn một frame miss (Stage 4 có cơ chế phục hồi cho miss, không có cho track sai). Toàn bộ các kỹ thuật dưới đây nhắm vào **cải thiện precision** của đúng bước quyết định TP/FP đó, theo 2 hướng:

- **Bộ lọc phụ (secondary filter)** — chỉ có thể **loại bớt** candidate mà `match_threshold`/`adaptive_threshold` đã chấp nhận, không bao giờ thêm lại recall: margin-over-runner-up, corroboration-gated window, identity chain filter.
- **Thay thế cơ chế tính threshold/score chính** — thay z-score/otsu/gmm hoặc thay cosine similarity: RMD score, ACI, SAFFRON, F-ROCP.

Trước khi chọn kỹ thuật nào để đầu tư tiếp, hãy chạy script chẩn đoán ở mục 1.

## 1. Script chẩn đoán (Phase 0) — chạy trước khi quyết định dùng kỹ thuật nào

File: `scripts/diagnose_verification_errors.py`

```bash
python -m scripts.diagnose_verification_errors --config configs/config.yaml --sample IDCard_0
python -m scripts.diagnose_verification_errors --config configs/config.yaml --sample IDCard_0 --iou-threshold 0.3
```

Đọc `candidates.json`/`detections.json`/`prototype.npz` đã có sẵn (không chạy lại pipeline), in 5 báo cáo:

1. **FP tự cluster với nhau** — một vài "confuser" lặp lại (systematic, đáng đầu tư `identity_chain_filter`) hay rải rác/diffuse (đáng đầu tư `similarity=rmd` hơn)?
2. **FP có tụ theo vùng không gian cố định** trong khung hình không (vật thể nền tĩnh gây nhiễu lặp lại).
3. **Oracle separability** — cluster TOÀN BỘ TP+FP+exemplar cùng lúc (cheat, có hindsight cả video) để biết trần khả năng tách được của chính embedding — nếu ngay cả oracle cũng không tách nổi, phải sửa ở tầng score (RMD), không phải tầng quyết định (threshold/cluster).
4. **Phân phối similarity TP vs FP** cạnh nhau.
5. **Mẫu đại diện** của các cụm FP lớn nhất để xem lại video bằng mắt.

Kết luận từ đây quyết định nên ưu tiên `similarity=rmd` (nếu FP rải rác/diffuse) hay `identity_chain_filter` (nếu FP là vài confuser lặp lại có cấu trúc).

## 2. Bảng tổng hợp nhanh

| # | Kỹ thuật | Bật bằng | Loại | Độ tin cậy port |
|---|---|---|---|---|
| 1 | Margin-over-runner-up | `stage3.margin_verification.enabled` / `stage123_geco2.dynamic_prototype.margin_verification.enabled` | Bộ lọc phụ | Ý tưởng đơn giản, tự viết theo mô tả WildFusion (không cần port công thức phức tạp) |
| 2 | Corroboration-gated trusted window | `stage3.cluster_secondary_filter.window_admission_*` | Bộ lọc phụ (cần bật `cluster_secondary_filter` trước) | Tái dùng nguyên `DetectionConfirmer` đã có sẵn, không phải kỹ thuật mới |
| 3 | RMD (Relative Mahalanobis Distance) | `stage3.similarity: rmd` | Thay thế score | Công thức RMD chuẩn (Mahalanobis khác biệt), không có bài báo cụ thể để "port sai" |
| 4 | ACI (Adaptive Conformal Inference) | `stage3.adaptive_threshold_online_method: aci` | Thay thế threshold | **FAITHFUL** — đúng công thức Gibbs & Candès (đã đọc trực tiếp), có điều chỉnh dấu (sign) cho đúng chiều bài toán |
| 5 | SAFFRON | `stage3.adaptive_threshold_online_method: saffron` | Thay thế threshold | **FAITHFUL** — đã đọc trực tiếp PDF arXiv:1802.09098, port đúng công thức Section 2.3 |
| 6 | F-ROCP (corruption_compensated) | `stage3.adaptive_threshold_online_method: corruption_compensated` | Thay thế threshold | **FAITHFUL** cho phần F-ROCP (đã đọc trực tiếp PDF arXiv:2605.20515) — phần AC-ROCP nâng cao hơn **không port** (xem mục 6.4) |
| 7 | KeepTrack identity chain filter | `stage3.identity_chain_filter.enabled` | Bộ lọc phụ | Dùng thuật toán Hungarian chuẩn (`scipy.optimize.linear_sum_assignment`), lấy ý tưởng cấu trúc từ KeepTrack, không phải port nguyên bài |
| 8 | Negative prototype / hard-negative anchor | `stage3.negative_prototype_filter.enabled` | Bộ lọc phụ | Ý tưởng riêng của dự án (không từ paper nào), sinh ra từ chính phân tích lỗi thật (xem mục 8) |
| 9 | DINOv3 + CLIP ensemble | `stage1.feature_extractor.model: ensemble` + `ensemble_dino_model: dinov3` | Thay encoder | Ghép 2 encoder có sẵn, không có gì để "port sai" — cần A/B test thật |
| 10 | Background masking cho candidate crop | `stage1.feature_extractor.candidate_background_masking.enabled` | Tiền xử lý encoder | Tái dùng nguyên segmenter + `apply_background_mode` đã có cho reference photo |
| 11 | Grounding DINO backbone làm encoder | — | Thay encoder | **CHƯA IMPLEMENT** — xem mục 11, cần thêm class mới trong `features.py` |
| 12 | VA-Count Noise Suppression Module | — | — | **CHƯA IMPLEMENT** (xem mục 12) — cần pipeline train riêng, ưu tiên thấp nhất theo plan |

## 3. Margin-over-runner-up

**Bài toán:** một keyframe có ≥2 candidate đều vượt threshold — chỉ 1 cái là thật, cái còn lại là confuser điểm số gần bằng. Threshold đơn thuần không phân biệt được 2 trường hợp "chỉ có 1 candidate rõ ràng" và "2 candidate ngang điểm nhau, không chắc cái nào đúng".

**Cách hoạt động:** trong mỗi keyframe, so **candidate điểm cao nhất** với **candidate điểm cao nhì**. Nếu `top_score - runner_up_score < tau_margin` → coi cả keyframe là **mơ hồ (ambiguous)**, bỏ toàn bộ (báo cáo "absent") thay vì đoán bừa. Nếu chỉ có 1 candidate trong keyframe, hoặc margin đủ rộng, giữ nguyên như cũ.

- **Stage 3 (đường threshold):** `aero_eyes/stages/stage3.py::run_stage3` — áp dụng ngay tại bước gom `frame_groups`, trước NMS/topk.
- **GeCo2 online (`dynamic_prototype`):** `aero_eyes/models/geco2_detector.py::offer_topk` (đường cold-start/topk_fusion, so trên `cosines`/`fused` score) và `_offer_topk_cluster` (so trên cosine giữa các candidate đã qua cluster-verify).

**Cách bật:**

```yaml
stage3:
  margin_verification:
    enabled: true
    tau_margin: 0.05   # candidate top phải hơn runner-up ít nhất 0.05 (đơn vị của stage3.similarity)

stage123_geco2:
  dynamic_prototype:
    margin_verification:
      enabled: true
      tau_margin: 0.05
```

Hai chỗ bật độc lập nhau — bật riêng Stage 3 hoặc riêng GeCo2 dynamic_prototype đều được, tuỳ đường pipeline nào đang dùng.

**Test:** `tests/test_stage3_margin_verification.py`, `tests/test_geco2_dynamic_prototype_margin_verification.py`.

**Lưu ý:** đây là tradeoff precision-vs-recall thuần tuý — chỉ loại, không bao giờ thêm lại candidate. `tau_margin` quá lớn sẽ loại oan nhiều keyframe có 2 candidate hợp lệ do trùng lặp góc nhìn.

## 4. Corroboration-gated trusted window

**Bài toán:** `stage3.cluster_secondary_filter` (đã có từ trước, xem [GECO2_cluster_verification_guide.md](GECO2_cluster_verification_guide.md) mục 3.1 cho primitive dùng chung) test thật cho thấy **không cải thiện precision đáng kể**. Nghi ngờ nguyên nhân: cửa sổ "candidate đã được chấp nhận gần đây" (`trusted_window`) nhận MỌI candidate vượt threshold vô điều kiện — chỉ cần 1 false positive lọt vào là nó trở thành "mỏ neo tin cậy" (trusted anchor) cho các keyframe sau, tự lan truyền sai số.

**Cách hoạt động:** thêm một `DetectionConfirmer` (chính là utility `stage4.confirm_detections` và `GeCo2DynamicPrototypeTracker` đã dùng — không phải cơ chế mới) đứng trước cửa `trusted_window`. Một candidate vượt qua cluster-check per-keyframe **chưa được thêm ngay** vào window — phải đồng thuận về mặt không gian (IoU ≥ `window_admission_iou_threshold`) qua `window_admission_min_consecutive_hits` keyframe liên tiếp trước khi thực sự được coi là "trusted anchor".

**Cách bật** (phải bật `cluster_secondary_filter.enabled` trước, đây chỉ là gate bổ sung bên trong nó):

```yaml
stage3:
  cluster_secondary_filter:
    enabled: true
    window_size: 50
    cluster_verification:
      enabled: false   # field này không phải công tắc ở đây, chỉ dùng các field cluster_method/min_cluster_size/...
      cluster_method: hdbscan
    window_admission_min_consecutive_hits: 2   # >1 để bật gate; <=1 = hành vi cũ (nhận vô điều kiện)
    window_admission_iou_threshold: 0.5
```

**Test:** `tests/test_stage3_cluster_secondary_filter.py` (test `test_window_admission_requires_corroboration` so sánh `min_consecutive_hits=1` vs `=2`).

## 5. RMD — Relative Mahalanobis Distance

**Bài toán:** cosine similarity thuần không tách được "gần giống exemplar do đúng identity" khỏi "gần giống exemplar do cùng domain/background chung" khi domain gap lớn (ảnh ref cận cảnh vs. crop video nén/mờ) — mọi điểm trong video đều bị kéo về một vùng similarity thấp chung, không phân biệt rõ TP/FP.

**Cách hoạt động:** thay vì "candidate giống exemplar bao nhiêu" (cosine), hỏi "candidate giống exemplar HƠN giống background của chính video này bao nhiêu" (tương đối, whitened theo hiệp phương sai chung):

```
score(x) = MahalanobisDistance(x, μ_background, Σ) − MahalanobisDistance(x, μ_exemplar, Σ)
```

- `μ_background, Σ` fit **một lần cho cả video** từ TẤT CẢ candidate feature (`sklearn.covariance.LedoitWolf` — có shrinkage vì số candidate có thể ít hơn chiều embedding).
- Số càng lớn = càng gần exemplar hơn background (điểm càng cao = càng có khả năng là TP), đúng chiều với cosine cũ (điểm cao = tốt).
- Là **thay thế trực tiếp** cho `all_sims` — mọi cơ chế downstream (threshold, cluster, margin, dynamic_prototype) dùng nguyên không cần sửa gì thêm.

**Cách bật:**

```yaml
stage3:
  similarity: rmd   # thay cho cosine/l1/l2
  # QUAN TRỌNG: thang đo của RMD khác hẳn cosine [-1,1] -- match_threshold/
  # adaptive_min_floor mặc định (tune cho cosine) sẽ SAI hoàn toàn nếu giữ
  # nguyên. Dùng adaptive_threshold=true (tự suy ra ngưỡng theo phân phối
  # của chính video) thay vì match_threshold cố định khi đổi sang rmd.
  adaptive_threshold: true
```

**Test:** `tests/test_stage3_rmd_score.py`.

**Lưu ý:** vì thang đo khác cosine hoàn toàn, **không dùng `match_threshold` cố định** khi bật `similarity=rmd` — phải dùng `adaptive_threshold` (batch hoặc online), nếu không mọi candidate có thể bị loại/giữ toàn bộ do ngưỡng sai thang.

## 6. Họ threshold online: window_stat / ACI / SAFFRON / F-ROCP

Cả 4 là các lựa chọn cho `stage3.adaptive_threshold_online_method`, chỉ có tác dụng khi `stage3.adaptive_threshold=true` **và** `stage3.adaptive_threshold_online=true` (threshold tính causal — chỉ dùng dữ liệu các keyframe TRƯỚC đó, phù hợp live feed).

```yaml
stage3:
  adaptive_threshold: true
  adaptive_threshold_online: true
  adaptive_threshold_online_window: 200
  adaptive_threshold_online_method: window_stat   # window_stat | aci | saffron | corruption_compensated
```

### 6.1. `window_stat` (mặc định, đã có từ trước)

z_score/otsu/gmm tính trên cửa sổ trượt (`adaptive_threshold_online_window` mẫu gần nhất) thay vì cả video — đây là baseline đã validate tốt nhất trên footage thật (F1=0.706).

### 6.2. `aci` — Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021)

**FAITHFUL PORT** — đã đọc trực tiếp công thức gốc `alpha_{t+1} = alpha_t + step*(target - err_t)`.

**Cách hoạt động:** thay vì tính lại thống kê (mean/std hay otsu) mỗi lần, giữ MỘT con số `percentile` (đóng vai trò `alpha_t` của bài báo), cập nhật dần bằng gradient step mỗi keyframe: nếu tỉ lệ chấp nhận của keyframe đó vượt `aci_target_error_rate` (quá permissive) → tăng percentile (ngưỡng chặt hơn); ngược lại → giảm. Có **đảo dấu (sign flip)** so với công thức gốc của bài báo — vì hệ mình chấp nhận candidate khi `similarity >= threshold` (điểm cao là tốt), ngược cực với "conformal regression" gốc chấp nhận khi `nonconformity <= threshold` (điểm cao là xấu) — đã ghi rõ lý do trong docstring `ACIOnlineThreshold`.

```yaml
stage3:
  adaptive_threshold_online_method: aci
  aci_target_error_rate: 0.1   # tỉ lệ chấp nhận mục tiêu mỗi keyframe
  aci_step_size: 0.05          # tốc độ cập nhật percentile mỗi bước
```

**Test:** `tests/test_stage3_online_fdr_methods.py` (`test_aci_*`).

### 6.3. `saffron` — SAFFRON (Ramdas, Zrnic, Wainwright, Jordan, PMLR v80/ICML 2018)

**FAITHFUL PORT** — đã đọc trực tiếp PDF `docs/1802.09098v2.pdf`, port đúng thuật toán Section 2.3 (không phải bản xấp xỉ như lần thử đầu tiên).

**Cách hoạt động:** khác hẳn 3 cơ chế còn lại — SAFFRON test **từng candidate riêng lẻ** (không tính 1 ngưỡng chung cho cả keyframe), nhắm thẳng vào chỉ số `(false accepts)/(total accepts)` (chính là precision của tập được chấp nhận). Cơ chế "wealth" (ngân sách sai số): mỗi lần chấp nhận một candidate mở ra một "epoch" mới, được cấp một phần ngân sách (`target_fdr`, hoặc `initial_wealth_fraction*target_fdr` cho epoch đầu tiên) giải ngân dần theo dãy suy giảm `gamma_j ∝ j^-gamma_exponent`; mức ý nghĩa (`alpha_t`) cho phép chấp nhận candidate hiện tại = tổng đóng góp từ **mọi epoch trong quá khứ**, giới hạn trên bởi `lam`.

Vì không có mô hình thống kê thật để tính p-value cho mỗi candidate (yêu cầu của SAFFRON), dự án dùng một suy diễn riêng (không phải từ bài báo): p-value = tỉ lệ các candidate gần đây (trong `p_value_window` mẫu) có similarity **cao hơn hoặc bằng** candidate hiện tại (càng hiếm/càng cao so với nền → p-value càng thấp → càng có khả năng được chấp nhận). Đây là phần có độ tin cậy thấp hơn phần thuật toán chính (đã port đúng công thức).

```yaml
stage3:
  adaptive_threshold_online_method: saffron
  online_fdr:
    target_fdr: 0.1               # mục tiêu (false accepts)/(total accepts)
    initial_wealth_fraction: 0.5  # W0 = target_fdr * cái này
    lam: 0.5                      # ngưỡng p-value để coi là "ứng viên" -- default của chính bài báo
    gamma_exponent: 2.0           # dãy suy giảm gamma_j ∝ j^-gamma_exponent
    p_value_window: 200
```

**Test:** `tests/test_stage3_online_fdr_methods.py` (`test_saffron_*`).

**Lưu ý hiệu năng:** SAFFRON giữ 1 danh sách "epoch" tăng dần theo số lần CHẤP NHẬN (không phải theo tổng số candidate) — chi phí tính `alpha_t` mỗi candidate là O(số epoch đã có). Với video dài có rất nhiều candidate được chấp nhận, chi phí này tăng dần; chưa cần tối ưu ở quy mô video hiện tại của dự án.

### 6.4. `corruption_compensated` — F-ROCP (Wang, Zecchin, Simeone, 2026)

**FAITHFUL PORT cho phần F-ROCP** — đã đọc trực tiếp PDF `docs/2605.20515v1.pdf`, port đúng Algorithm 1 (Section IV), bọc quanh `aci` ở trên.

**Cách hoạt động:** F-ROCP dựa trên một quan sát đơn giản — nếu ngưỡng đã ra ngoài khoảng hợp lệ [0, B) (ở đây: `percentile` chạm 0 hoặc 100), thì kết quả của keyframe đó là **chắc chắn về mặt toán học** (percentile=0 → ngưỡng = điểm thấp nhất trong window → gần như mọi candidate đều "pass", chắc chắn là quá permissive; percentile=100 → ngược lại), **không cần tin vào tín hiệu quan sát được của chính keyframe đó** (vốn có thể nhiễu vì chỉ dựa vào vài candidate). Ở giữa khoảng (percentile trong (0, 100)) thì tin thẳng tín hiệu quan sát được, giống hệt cơ chế `aci`.

```yaml
stage3:
  adaptive_threshold_online_method: corruption_compensated
  # Không có tham số riêng -- dùng lại nguyên aci_target_error_rate/aci_step_size ở trên.
```

**Test:** `tests/test_stage3_online_fdr_methods.py` (`test_corruption_compensated_*`).

**HONESTY NOTE quan trọng — vì sao chỉ có F-ROCP, không có AC-ROCP:** bài báo gốc có thêm cơ chế **AC-ROCP** (active compensation) mạnh hơn F-ROCP — chủ động "probe" định kỳ để ước lượng TỈ LỆ tín hiệu feedback bị nhiễu (so sánh feedback THẬT vs feedback QUAN SÁT ĐƯỢC bị nhiễu). Cơ chế này đòi hỏi phải có **2 tín hiệu riêng biệt tồn tại song song** (một thật, một bị nhiễu) để so sánh và học tỉ lệ nhiễu. Trong bài toán của dự án, **không có ground truth thật ở bất kỳ đâu khi chạy inference** — chỉ có DUY NHẤT một proxy tự suy ra (tỉ lệ chấp nhận của mỗi keyframe), không phải một CẶP thật/nhiễu. Vì vậy AC-ROCP **không có cách ánh xạ trung thực** vào bài toán này — cố ép implement sẽ là giả tạo (bịa ra một "tỉ lệ corruption" không tương ứng với gì thực tế), không phải port đúng, nên **chủ động không implement phần này**.

## 7. KeepTrack identity chain filter

**Bài toán:** hiện tại quyết định TP/FP chỉ dựa vào appearance score của **1 keyframe riêng lẻ** — một confuser xuất hiện đúng 1 lần với điểm số cao vẫn có thể được chấp nhận, dù nó không hề "tái xuất hiện" một cách nhất quán như mục tiêu thật thường làm.

**Cách hoạt động:** giữ lại top-K candidate + feature mỗi keyframe (`top_k_per_keyframe`), giải bài toán ghép cặp (bipartite matching, thuật toán Hungarian chuẩn — `scipy.optimize.linear_sum_assignment`, không phải xấp xỉ) giữa candidate của 2 keyframe liên tiếp, với chi phí `1 - cosine_similarity` (+ tuỳ chọn cộng thêm khoảng cách không gian đã chuẩn hoá, `spatial_weight`). Theo dõi các "chuỗi danh tính" (identity chain) — một candidate ở frame t nối với 1 candidate ở frame t+1, nối tiếp t+2, ... Candidate được chấp nhận ở mỗi keyframe là candidate thuộc **chuỗi dài nhất đang hoạt động** (≥ `min_chain_length`), không phải candidate điểm cao nhất frame đó — một confuser điểm cao nhưng xuất hiện đúng 1 lần (chuỗi độ dài 1) sẽ bị loại dù điểm số cao hơn.

```yaml
stage3:
  identity_chain_filter:
    enabled: true
    top_k_per_keyframe: 5
    min_chain_length: 2
    spatial_weight: 0.0    # 0 = chỉ dùng appearance; >0 cộng thêm khoảng cách không gian chuẩn hoá
    max_match_cost: 0.5    # chỉ nối chuỗi nếu cost < 0.5 (tương đương cosine > 0.5 khi spatial_weight=0)
```

**Test:** `tests/test_stage3_identity_chain_filter.py`.

**Khi nào nên thử:** theo script chẩn đoán ở mục 1 — nếu FP là **vài confuser lặp lại có cấu trúc** (không phải rải rác/diffuse), kỹ thuật này trực tiếp giải quyết đúng kiểu lỗi đó. Nếu FP rải rác, ưu tiên `similarity=rmd` (mục 5) trước.

**⚠️ Lưu ý quan trọng nếu confuser là loại "texture lặp lại nhưng KHÔNG phải 1 vật thể duy nhất"** (case thực tế: FP chủ yếu là các đám lá khô rải rác khắp video, không phải 1 vật thể di chuyển liên tục) — nếu để `spatial_weight=0` (mặc định, chỉ so appearance), thuật toán ghép cặp Hungarian có thể **nối nhầm 2 đám lá ở 2 vị trí hoàn toàn khác nhau thành 1 "chuỗi giả"**, chỉ vì chúng tình cờ giống nhau về texture — chuỗi giả này vẫn đủ dài để "qua mặt" `min_chain_length`. Đặt `spatial_weight > 0` (ví dụ 0.3–0.5) buộc 2 candidate liên tiếp phải GẦN NHAU về không gian mới được nối chuỗi — một đám lá xuất hiện lẻ tẻ ở các vị trí khác nhau sẽ không ghép được thành chuỗi dài (mục tiêu thật di chuyển mượt, vị trí frame sau dự đoán được từ frame trước), nên bị loại đúng. Luôn thử cả `spatial_weight=0` lẫn `spatial_weight>0` khi confuser có tính chất "1 lớp texture xuất hiện ở nhiều vị trí" thay vì "1 vật thể cụ thể di chuyển".

## 8. Negative prototype / hard-negative anchor

**Bài toán:** phát hiện từ chính script chẩn đoán (mục 1) trên dữ liệu thật — trên 1 sample, **một loại confuser DUY NHẤT** (một đám lá khô có embedding DINO tình cờ gần với target) chiếm tới 47.8% tổng số FP. `similarity=rmd` (mục 5) sửa vấn đề bằng cách so với "nền chung khuếch tán" (trung bình TOÀN BỘ candidate trong video) — nhưng nếu 1 loại confuser cụ thể áp đảo như vậy, một tham chiếu KHUẾCH TÁN có thể bị pha loãng, không đủ sắc để bắt đúng loại lỗi đó.

**Cách hoạt động:** thay vì chỉ hỏi "candidate này giống exemplar (positive) bao nhiêu?", hỏi thêm "candidate này giống một mẫu ĐÃ BIẾT LÀ SAI bao nhiêu?", rồi so sánh margin:

```
reject nếu: max_k cosine(x, negative_window_k) − cosine(x, exemplar) >= tau_negative_margin
```

- **`negative_window`**: một deque FIFO thuần causal, tích luỹ từ chính các candidate ĐÃ BỊ REJECT bởi bước quyết định trước đó (threshold chính, hoặc 1 bộ lọc phụ khác đã chạy trước) — KHÔNG cần biết trước "lá khô" là gì, không cần GT, không cần curate thủ công.
- Một confuser LẶP LẠI (như lá khô) tự nhiên tích luỹ nhiều thành viên giống nhau trong window → candidate tương lai cùng loại dễ tìm thấy 1 match rất gần. Nhiễu ngẫu nhiên (one-off) chỉ đóng góp các điểm cô lập, hiếm khi khớp lại — sự phân biệt này **tự nhiên nổi lên**, không cần clustering/curate tường minh.
- Tự tính cosine similarity RIÊNG từ feature thô cho cả 2 vế so sánh (độc lập với `stage3.similarity`) — để margin luôn có cùng đơn vị đo, kể cả khi `similarity=rmd` đang bật (thang đo khác hẳn cosine).

**Cách bật:**

```yaml
stage3:
  negative_prototype_filter:
    enabled: true
    window_size: 200            # số feature bị reject gần nhất giữ lại làm "anchor tiêu cực"
    min_window_for_check: 20    # dưới ngưỡng này, bỏ qua check (chưa đủ bằng chứng)
    tau_negative_margin: 0.0    # 0.0 = reject ngay khi giống negative BẰNG HOẶC HƠN giống positive
```

**Test:** `tests/test_stage3_negative_prototype_filter.py`.

**Lưu ý:** cũng chỉ có thể REJECT, không thêm lại recall. Nằm TRƯỚC `cluster_secondary_filter`/`identity_chain_filter` trong pipeline (`run_stage3`) để negative window có cơ hội tích luỹ bằng chứng sớm nhất từ ngay bước threshold chính.

## 9. Đổi/kết hợp encoder

### 9.1. DINOv3 + CLIP ensemble

**Bài toán:** DINOv2/DINOv3 (self-supervised) học để phân cụm visual structure/texture chung — không phân biệt "đây là 1 vật thể" khỏi "đây là clutter nền" (lá khô, đất, đá...). CLIP (contrastive image-text) thiên về nhận diện KHÁI NIỆM/CATEGORY hơn — có thể phân biệt "một vật thể" khỏi "một đám lá" tốt hơn vì bản chất mục tiêu huấn luyện khác.

**Cách hoạt động:** `model="ensemble"` đã có sẵn từ trước nhưng **hardcode DINOv2+CLIP** — giờ tổng quát hoá để chọn được DINOv3 thay DINOv2 (`EnsembleFeatureExtractor` trong `aero_eyes/models/features.py`, tái dùng nguyên `dinov3_variant`/`dinov3_source`/`dinov3_pretrain_dataset`/`dinov3_kaggle_model_id` đã có, không thêm field trùng lặp). Nối 2 vector rồi L2-normalize, giống hệt cơ chế ensemble cũ.

**Cách bật:**

```yaml
stage1:
  feature_extractor:
    model: ensemble
    ensemble_dino_model: dinov3   # dinov2 (cũ, mặc định) | dinov3
    dinov3_variant: vitb16
    dinov3_pretrain_dataset: lvd1689m   # hoặc sat493m nếu muốn domain vệ tinh
    clip_variant: vit-b/32
```

**Test:** `tests/test_features_ensemble.py` (dùng fake extractor, không tải model thật).

**Lưu ý:** đổi `model`/kích thước embedding sẽ làm `prototype.npz`/`candidates.feats.npz` đã cache CŨ không còn khớp chiều — cần xoá cache hoặc bật `stage3.recompute_candidate_features=true` (xem comment đầu `stage3:` trong config.yaml).

### 9.2. Encoder khác (tham khảo, không cần code mới)

Đã có sẵn, chỉ cần đổi `stage1.feature_extractor.model`:
- **`dinov3` với `dinov3_pretrain_dataset: sat493m`** — bản DINOv3 pretrain trên ảnh vệ tinh thay vì ảnh web chung, gần domain aerial hơn hẳn — **rẻ nhất để thử** (chỉ đổi 1 dòng, không code mới).
- **`clip`/`siglip` thay hẳn DINOv3** — so sánh trực tiếp xem encoder theo hướng "semantic/category" có tốt hơn "texture-clustering" thuần cho đúng loại lỗi confuser hiện tại không.

## 10. Background masking cho candidate crop

**Bài toán:** phát hiện bất đối xứng khi đọc code — `stage1.segmentation` CHỈ che nền (mean-fill/blur/giữ nguyên) cho ẢNH EXEMPLAR/REFERENCE trước khi encode, nhưng candidate crop lấy từ video (`crop_with_pad` trong `aero_eyes/models/features.py`) **không hề được che nền** — luôn chứa background thật xung quanh box (đất, lá, clutter). Exemplar "sạch" so với candidate "lẫn cả nền" là một bất đối xứng có thể góp phần gây nhiễu embedding.

**Cách hoạt động:** thêm 1 wrapper mới `MaskedCropFeatureExtractor` (`aero_eyes/models/features.py`) bọc quanh BẤT KỲ encoder nào — khi `extract_crops()` được gọi (đường lấy candidate từ video, KHÔNG áp dụng cho `extract()` vốn nhận ảnh đã chuẩn bị sẵn như reference photo), mỗi crop được chạy qua CHÍNH segmenter (MobileSAM/FastSAM/SAM2) và CHÍNH cơ chế `apply_background_mode` mà `stage1.segmentation` đã dùng cho reference, trước khi đưa vào encoder. Lỗi segmentation trên 1 crop tự động fallback về crop gốc (không mask), không làm hỏng cả batch.

**Cách bật:**

```yaml
stage1:
  feature_extractor:
    candidate_background_masking:
      enabled: true
      model: mobilesam            # mobilesam | fastsam | sam2
      background_mode: mean_fill  # mean_fill | keep_real | blur
```

**Test:** `tests/test_features_masked_crop.py` (dùng fake segmenter/encoder, không tải model thật).

**⚠️ Giới hạn quan trọng cần hiểu trước khi thử cho đúng case "lá khô":** kỹ thuật này chỉ hữu ích khi vấn đề là **background LẪN VÀO xung quanh 1 box đúng** (ví dụ box đúng có viền là đất/lá). Nếu confuser là 1 candidate box mà NỘI DUNG BÊN TRONG box đó chính là lá khô (segmenter coi cả box là "foreground" hợp lệ vì nó chiếm gần hết khung crop) thì che nền sẽ **không thay đổi gì** — vấn đề không nằm ở background bị lẫn, mà ở chỗ bản thân candidate đó chưa từng là ứng viên hợp lệ ngay từ khâu đề xuất box. Trường hợp này nên ưu tiên `negative_prototype_filter` (mục 8) hoặc `identity_chain_filter` với `spatial_weight>0` (mục 7) hơn.

**Chi phí:** chạy segmentation cho MỖI candidate crop mỗi keyframe (không có batch inference) — có thể chậm đáng kể nếu nhiều candidate/keyframe, cần đo lại tốc độ trước khi dùng production.

## 11. Grounding DINO backbone làm encoder — chưa implement

**Ý tưởng:** Grounding DINO = backbone thị giác (Swin-Transformer) + backbone text (BERT) + các lớp fusion 2 luồng + detection head theo prompt. Backbone Swin đứng TRƯỚC bước fusion với text, hoàn toàn độc lập với prompt — trích multi-scale feature map thuần thị giác y hệt vai trò DINOv2/CLIP hiện tại. Backbone này được train (cùng lúc với detection) trên dữ liệu detection/grounding thật (Objects365, GoldG, COCO...) — học ngầm "cái gì đáng là 1 object có tên gọi, cái gì là nền/clutter cần bỏ qua" — khác hẳn DINO (self-supervised, chỉ phân cụm texture chung).

**Vì sao chưa implement:** cần 1 class mới (`GroundingDinoBackboneFeatureExtractor` trong `aero_eyes/models/features.py`, theo đúng khuôn mẫu các extractor hiện có) load `transformers.AutoModel.from_pretrained("IDEA-Research/grounding-dino-tiny")` nhưng **chỉ lấy submodule backbone**, bỏ qua phần fusion/text/detection head, rồi tự quyết định cách pooling (Swin không có sẵn 1 CLS token tổng hợp như DINO/CLIP — cần global-average-pool tầng feature map cuối, hoặc ghép nhiều tầng, là quyết định thiết kế cần thử nghiệm). Backbone Swin-B/Swin-L cũng nặng hơn ViT-B đang dùng — cần đo lại tốc độ trích embedding cho mỗi candidate mỗi frame.

**Ưu tiên:** thử SAU các phương án rẻ hơn ở mục 8-10 — đây là hướng có cơ sở lý luận hợp lý nhưng chưa có gì đảm bảo, và tốn công implement nhất trong các kỹ thuật thay-encoder.

## 12. VA-Count Noise Suppression Module — chưa implement

Kỹ thuật cuối trong plan gốc (ECCV 2024) — một head học bằng contrastive loss để phân biệt TP/FP thay vì clustering/threshold thuần suy luận. Đòi hỏi một pipeline train riêng (dữ liệu synthetic ghép nhiều class, GPU training time) — chi phí đầu tư lớn hơn hẳn mọi kỹ thuật khác trong tài liệu này (toàn bộ các kỹ thuật khác chỉ là thay đổi thuật toán ở bước inference, dùng code/model đã có). Theo đúng plan gốc: **chỉ nên cân nhắc nếu mọi kỹ thuật khác trong tài liệu này không đủ** sau khi đã A/B test trên footage thật.

## 13. Cách kết hợp nhiều kỹ thuật & lưu ý xung đột

- `identity_chain_filter`, `margin_verification`, `cluster_secondary_filter`, `negative_prototype_filter` là **4 bộ lọc phụ độc lập** — có thể bật cùng lúc (áp dụng nối tiếp nhau trong `run_stage3`, theo đúng thứ tự: `negative_prototype_filter` → `cluster_secondary_filter` → `identity_chain_filter`; `margin_verification` áp dụng riêng ở bước gom `frame_groups`), nhưng càng bật nhiều càng dễ **mất recall** (mỗi cái chỉ có thể loại thêm, không thêm lại) — nên bật và đo từng cái một trước khi bật chồng.
- `similarity=rmd` là thay thế cho cosine ở TẦNG SCORE CHÍNH — ảnh hưởng đến **mọi** cơ chế khác dùng `all_sims` (threshold, cluster, margin, identity_chain). Riêng `negative_prototype_filter` và `identity_chain_filter` tự tính cosine RIÊNG từ feature thô cho phần so sánh nội bộ của chúng (độc lập với `similarity=rmd`) — xem lưu ý trong docstring từng cái.
- 4 lựa chọn của `adaptive_threshold_online_method` (`window_stat`/`aci`/`saffron`/`corruption_compensated`) là **loại trừ lẫn nhau** — chỉ 1 cái chạy tại một thời điểm, chọn qua field đó.
- `verification_method=cluster` (xem guide cluster riêng) và các kỹ thuật `stage3.*` trong file này đều nhắm vào đường `verification_method=threshold` — **không** kết hợp với `verification_method=cluster` (2 cơ chế quyết định chính khác nhau).
- Đổi `stage1.feature_extractor.model`/`ensemble_dino_model` (đổi chiều embedding) làm cache cũ (`prototype.npz`, `candidates.feats.npz`) không còn khớp — xoá cache hoặc bật `stage3.recompute_candidate_features=true`.
- `candidate_background_masking` và mọi cơ chế Stage 3 khác trong file này **độc lập, không xung đột** (masking xảy ra ở tầng encoder, trước khi `all_sims` được tính) — có thể bật cùng lúc với bất kỳ tổ hợp nào ở trên.

## 14. Trình tự khuyến nghị khi thử nghiệm

Danh sách này được viết lại để KHÔNG cần nhớ hết mọi kỹ thuật — mỗi bước chỉ cần đúng 1 dòng config, đo P/R/F1, rồi quyết định đi tiếp hay dừng. Xếp theo **chi phí thử tăng dần** (rẻ/nhanh trước, tốn công implement/compute sau), kết hợp với phát hiện thực tế từ script chẩn đoán (ví dụ: FP chủ yếu là 1 loại confuser lặp lại như lá khô → một số bước dưới đây được ưu tiên hơn các bước còn lại).

0. **Luôn làm trước tiên:** chạy `scripts/diagnose_verification_errors.py` (mục 1), xác nhận baseline hiện có (`adaptive_threshold_online=true`, `adaptive_threshold_online_method=window_stat`) trên chính sample đang test — đây là con số để so sánh mọi bước sau.
1. **`dinov3_pretrain_dataset: sat493m`** (nếu đang dùng dinov3) — 1 dòng config, không code mới, không tốn thêm compute đáng kể. (mục 9.2)
2. **`negative_prototype_filter.enabled: true`** — cấu hình mặc định là đủ để thử, không cần biết trước confuser là gì. Đặc biệt đáng thử nếu chẩn đoán cho thấy 1 loại confuser áp đảo (một cụm chiếm phần lớn tổng FP). (mục 8)
3. **`identity_chain_filter.enabled: true` + thử cả `spatial_weight=0` VÀ `spatial_weight>0` (0.3–0.5)** — nếu confuser là "1 lớp texture rải rác nhiều vị trí" (như lá khô), `spatial_weight>0` mới lọc đúng; nếu confuser là "1 vật thể cụ thể di chuyển", `spatial_weight=0` là đủ. (mục 7)
4. **`similarity: rmd`** (nhớ đổi `adaptive_threshold: true` cùng lúc, xem lưu ý về thang đo ở mục 5) — vẫn đáng thử dù confuser tập trung vào 1 loại cụ thể, nhưng kỳ vọng thấp hơn bước 2 cho đúng trường hợp đó (RMD dùng nền khuếch tán, không nhắm riêng 1 loại confuser). (mục 5)
5. **`margin_verification.enabled: true`** và/hoặc **`cluster_secondary_filter.window_admission_min_consecutive_hits: 2`** — ít tốn kém, nhưng ít nhắm trúng đúng loại lỗi cụ thể hơn các bước 2-4, nên xếp sau. (mục 3, 4)
6. **`stage1.feature_extractor.model: clip` hoặc `siglip`** (thử riêng, so sánh trực tiếp với DINOv3) — đổi hẳn encoder, cần xoá cache/`recompute_candidate_features=true`. (mục 9.2)
7. **`ensemble_dino_model: dinov3`** (ghép DINOv3+CLIP) — nặng hơn (2 backbone), thử sau khi đã biết CLIP/SigLIP một mình có giúp gì không. (mục 9.1)
8. **`candidate_background_masking.enabled: true`** — chỉ đáng thử nếu chẩn đoán cho thấy vấn đề là "nền lẫn quanh box đúng", KHÔNG phải "cả nội dung box là confuser" (đọc kỹ giới hạn ở mục 10 trước khi thử, tránh tốn compute mà không đúng loại lỗi). (mục 10)
9. **Grounding DINO backbone** — cần code mới, thử sau cùng trong nhóm "đổi encoder" nếu bước 6-8 không đủ. (mục 11)
10. **VA-Count NSM** — chỉ cân nhắc nếu KHÔNG kỹ thuật nào ở trên đủ, cần cả pipeline train riêng. (mục 12)

Ở mỗi bước: bật **một mình nó** (không chồng với bước chưa đo), so P/R/F1 với baseline bước 0, rồi mới quyết định giữ/bỏ trước khi sang bước tiếp theo — tránh lặp lại sai lầm trước đó của dự án là bật chồng nhiều thay đổi cùng lúc khiến không quy được kết quả về đâu. `pytest tests/ -q` phải xanh toàn bộ trước và sau mỗi thay đổi cấu hình mặc định.

## 15. File liên quan (code map)

| File | Vai trò |
|---|---|
| `aero_eyes/config.py` | `MarginVerificationConfig`, `ClusterSecondaryFilterConfig` (+ `window_admission_*`), `OnlineFDRConfig`, `CorruptionCompensatedThresholdConfig`, `IdentityChainFilterConfig`, `NegativePrototypeFilterConfig`, `FeatureExtractorConfig.ensemble_dino_model`/`candidate_background_masking`, `Stage3Config.similarity`/`adaptive_threshold_online_method`/`aci_*` |
| `aero_eyes/stages/stage3.py` | `_fit_rmd_background`/`_score_against_ref` (RMD), `ACIOnlineThreshold`, `SaffronInspiredOnlineFDR`, `CorruptionCompensatedThreshold`, `apply_identity_chain_filter`, khối margin-over-runner-up, `cluster_secondary_filter` và `negative_prototype_filter` trong `run_stage3` |
| `aero_eyes/models/geco2_detector.py` | Margin-over-runner-up trong `offer_topk`/`_offer_topk_cluster` (`GeCo2DynamicPrototypeTracker`) |
| `aero_eyes/models/features.py` | `EnsembleFeatureExtractor` (đã tổng quát hoá cho `dino_model="dinov3"`), `MaskedCropFeatureExtractor` (candidate background masking) |
| `configs/config.yaml` | Toàn bộ field mặc định (tắt), comment giải thích từng cái |
| `scripts/diagnose_verification_errors.py` | Script chẩn đoán Phase 0 |
| `docs/GECO2_precision_improvements_plan.md` | Kế hoạch gốc, bối cảnh đầy đủ, deep-research report liên quan |
| `docs/1802.09098v2.pdf` | Bài báo gốc SAFFRON (đã đọc trực tiếp để port) |
| `docs/2605.20515v1.pdf` | Bài báo gốc F-ROCP/AC-ROCP (đã đọc trực tiếp để port) |
| `tests/test_stage3_margin_verification.py`, `tests/test_geco2_dynamic_prototype_margin_verification.py` | Test margin-over-runner-up |
| `tests/test_stage3_cluster_secondary_filter.py` | Test corroboration-gated window |
| `tests/test_stage3_rmd_score.py` | Test RMD |
| `tests/test_stage3_online_fdr_methods.py` | Test ACI/SAFFRON/F-ROCP |
| `tests/test_stage3_identity_chain_filter.py` | Test identity chain filter |
| `tests/test_stage3_negative_prototype_filter.py` | Test negative prototype filter |
| `tests/test_features_ensemble.py` | Test DINOv2/DINOv3 + CLIP ensemble dispatch |
| `tests/test_features_masked_crop.py` | Test candidate background masking wrapper |

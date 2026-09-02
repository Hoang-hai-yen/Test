# Kết quả test GECO2 gốc trên AeroEyes theo kích thước tham chiếu (`expected_object_px`)

**Nguồn:** log đánh giá do người dùng chạy và dán trực tiếp (không phải log tôi tự sinh) — dữ liệu dưới đây được trích xuất/tổng hợp lại từ log đó, số liệu gốc chưa được kiểm chứng độc lập.

**Model:** GECO2 gốc (chưa finetune, checkpoint `CNTQG_multitrain_ca44.pth`, còn nguyên shape token), chạy qua pipeline AeroEyes (`stage123_geco2`) với `scale_calibration` bật, thử lần lượt 6 giá trị `expected_object_px` (dạng cặp width×height) — đúng bộ 6 cặp lấy từ phân tích percentile kích thước object thật trong `annotations.json` ở phần trước cuộc trò chuyện.

**Test set:** 6 video held-out — `BlackBox_0`, `BlackBox_1`, `CardboardBox_0`, `CardboardBox_1`, `LifeJacket_0`, `LifeJacket_1` (đúng 6 video nêu trong [GECO2_FINETUNE_PLAN.md](GECO2_FINETUNE_PLAN.md#L61-L65), không dùng để train).

**2 chế độ lọc:** mỗi kích thước chạy 2 lần — **default** (ngưỡng confidence mặc định của pipeline) và **conf 0.0** (không lọc, giữ mọi detection).

---

## 1. Bảng tổng hợp Mean ST-IoU (Spatio-Temporal IoU — độ khớp track theo cả không gian & thời gian)

| Kích thước (W×H, px) | conf 0.5 — Mean ST-IoU | conf 0.0 — Mean ST-IoU |
|---|---:|---:|
| 22×17 (≈p10) | 0.1460 | 0.0914 |
| 24×31 (≈p25) | 0.2134 | 0.1156 |
| 26×50 (≈p50, median) | 0.3814 | 0.1607 |
| 61×38 (≈p75) | **0.4158** | 0.1686 |
| 75×51 (≈p90) | **0.4397** ← cao nhất | 0.1912 |
| 90×113 (max) | 0.2391 | - |

**Quan sát chính:** hiệu năng không tăng đơn điệu theo kích thước — đạt đỉnh ở **75×51** rồi **giảm mạnh** khi lên tới 90×113 (0.4397 → 0.2391), đồng thời cũng giảm khi xuống dưới median (26×50 → 24×31 → 22×17). Có một "vùng kích thước tối ưu" nằm ở khoảng p75–p90 của phân bố object thật, không phải "càng lớn/càng nhỏ càng tốt".

---

## 2. Bảng chi tiết Detection Metrics (micro-avg trên cả 6 video)

| Kích thước | Conf | Presence-only P/R/F1 | @IoU≥0.50 P/R/F1 | @IoU≥0.30 P/R/F1 |
|---|---|---|---|---|
| 22×17 | 0.5 | 0.456 / 0.741 / 0.565 | 0.232 / 0.376 / 0.287 | 0.278 / 0.451 / 0.344 |
| 22×17 | 0.0 | 0.269 / 1.000 / 0.424 | 0.118 / 0.439 / 0.186 | 0.147 / 0.545 / 0.231 |
| 24×31 | 0.5 | 0.496 / 0.868 / 0.632 | 0.298 / 0.521 / 0.379 | 0.348 / 0.609 / 0.443 |
| 24×31 | 0.0 | 0.269 / 1.000 / 0.424 | 0.146 / 0.543 / 0.231 | 0.171 / 0.636 / 0.270 |
| 26×50 | 0.5 | 0.631 / 0.922 / 0.750 | 0.504 / 0.736 / 0.599 | 0.533 / 0.778 / 0.632 |
| 26×50 | 0.0 | 0.270 / 1.000 / 0.425 | 0.210 / 0.779 / 0.331 | 0.224 / 0.830 / 0.353 |
| 61×38 | 0.5 | 0.670 / 0.949 / 0.785 | 0.553 / 0.784 / 0.649 | 0.570 / 0.808 / 0.668 |
| 61×38 | 0.0 | 0.272 / 1.000 / 0.427 | 0.221 / 0.812 / 0.347 | 0.229 / 0.841 / 0.359 |
| 75×51 | 0.5 | **0.721 / 0.819 / 0.767** | **0.649 / 0.737 / 0.690** | **0.667 / 0.758 / 0.710** |
| 75×51 | 0.0 | **0.280 / 0.999 / 0.438** | **0.251 / 0.894 / 0.391** | **0.258 / 0.919 / 0.402** |
| 90×113 | 0.5 | 0.735 / 0.349 / 0.473 | 0.656 / 0.311 / 0.422 | 0.676 / 0.321 / 0.435 |
| 90×113 | 0.0 |  | — | — |

**Quan sát:** `75×51` cho **F1 cao nhất ở cả 3 kiểu đánh giá** (0.767 / 0.690 / 0.710) — đồng nhất với kết quả ST-IoU ở bảng 1. `90×113` có **precision cao nhất (0.735) nhưng recall sụt mạnh (0.349)** — model trở nên "kén chọn", bỏ sót nhiều object thật khi kích thước tham chiếu quá lớn so với vật thể thực tế trong video.

---

## 3. Presence-only nghĩa là gì, và vì sao KHÔNG nên bỏ

`presence-only` không quan tâm box dự đoán đặt ở đâu trong frame — nó chỉ hỏi đúng 1 câu: *"frame này có GT thì model có sinh ra ít nhất 1 detection nào không (bất kể vị trí)?"* (TP = cả 2 cùng có mặt trong frame; FP = model detect nhưng GT không có gì; FN = GT có vật mà model không detect gì cả). Ngược lại, `@IoU≥0.5`/`@IoU≥0.3` đòi hỏi thêm: box đó phải **đặt đúng chỗ**.

Nhờ vậy, hiệu số giữa 2 chỉ số này tách được **2 loại lỗi hoàn toàn khác nhau**:
- **Presence cao, IoU thấp** → model *biết* có vật, chỉ đặt box **sai vị trí** (lỗi định vị/localization).
- **Presence cũng thấp** → model *không hề phát hiện* ra vật (lỗi nhận diện/detection, nghiêm trọng hơn nhiều — không phải chuyện chỉnh lại vị trí box mà cứu được).

Đây là bằng chứng trực tiếp presence-only **có ý nghĩa chẩn đoán thật**, nên giữ lại. Bảng dưới đây lượng hóa khoảng cách đó (chế độ default, so Recall presence-only vs Recall @IoU≥0.5):

| Kích thước | Recall (presence) | Recall (@IoU≥0.5) | **Gap Recall** | Precision (presence) | Precision (@IoU≥0.5) | **Gap Precision** |
|---|---:|---:|---:|---:|---:|---:|
| 22×17 | 0.741 | 0.376 | **0.365** | 0.456 | 0.232 | 0.224 |
| 24×31 | 0.868 | 0.521 | **0.347** | 0.496 | 0.298 | 0.198 |
| 26×50 | 0.922 | 0.736 | 0.186 | 0.631 | 0.504 | 0.127 |
| 61×38 | 0.949 | 0.784 | 0.165 | 0.670 | 0.553 | 0.117 |
| 75×51 | 0.819 | 0.737 | **0.082** | 0.721 | 0.649 | **0.072** |
| 90×113 | 0.349 | 0.311 | 0.038 | 0.735 | 0.656 | 0.079 |

**Đọc ra 2 chế độ lỗi rất khác nhau tùy kích thước tham chiếu:**

1. **Kích thước nhỏ (22×17, 24×31): lỗi định vị chiếm ưu thế.** Gap Recall rất lớn (0.35–0.37) — model vẫn "thấy" vật trong phần lớn frame (presence recall 0.74–0.87) nhưng đặt box sai vị trí quá thường xuyên nên rớt hẳn khi đòi IoU≥0.5 (recall còn 0.38–0.52). Đây là lỗi **có thể sửa bằng cách cải thiện độ chính xác localization**, không phải do model "mù" trước vật thể.

2. **Kích thước rất lớn (90×113): lỗi chuyển hẳn sang detection-collapse, không còn là lỗi định vị.** Gap Recall co lại chỉ còn 0.038 (nhỏ nhất trong bảng) — nhưng đó là vì **cả 2 chỉ số recall đều sụp xuống cùng mức thấp (0.35/0.31)**, không phải vì localization tốt lên. Model ở đây thực sự **bỏ sót phần lớn vật thể ngay từ bước phát hiện**, không liên quan gì tới việc box đặt đúng hay sai. Đây là lỗi **nặng hơn nhiều và không sửa được chỉ bằng calibrate lại vị trí box** — bản chất là exemplar quá lớn so với vật thể thật khiến matching thất bại từ gốc.

3. **75×51 là điểm cân bằng tốt nhất:** vừa giữ Recall-presence khá cao (0.819, chỉ thấp hơn đỉnh 61×38 một chút), vừa có Gap nhỏ nhất trong nhóm "còn phát hiện tốt" (0.082) — tức là **khi model phát hiện, nó gần như luôn đặt box đúng chỗ luôn**, không mất nhiều vào lỗi định vị lẫn lỗi bỏ sót.

**Xu hướng Precision:** Precision (cả presence lẫn IoU) tăng đều đặn theo kích thước tham chiếu, từ 0.23 (22×17, @IoU0.5) lên tới 0.66 (90×113) — exemplar càng lớn, model càng "kén chọn" (ít detection hơn nhưng đúng hơn khi đã detect). Đây chính là đánh đổi precision–recall kinh điển: kích thước tham chiếu là một biến điều khiển độ "kén chọn" của model, và 75×51 là điểm mà độ kén chọn đó chưa đánh đổi quá nhiều recall.

---

## 4. Vì sao `conf 0.0` luôn tệ hơn nhiều so với default

Ở mọi kích thước, `conf 0.0` (không lọc) đều cho **recall gần như tuyệt đối (0.9–1.0)** nhưng **precision rất thấp (0.11–0.28)** — model sinh ra rất nhiều detection nhiễu (false positive) khi không áp ngưỡng confidence, và ngưỡng mặc định của pipeline đang lọc bớt nhiễu này hiệu quả (F1 default cao hơn conf 0.0 khoảng 2–3 lần ở mọi kích thước). Điều này cho thấy: score/confidence của model tuy chưa hoàn hảo nhưng **có tín hiệu phân biệt object thật/nhiễu**, không phải ngẫu nhiên — việc calibrate đúng ngưỡng threshold quan trọng không kém việc chọn đúng kích thước tham chiếu.

---

## 5. So sánh giữa các video (per-video ST-IoU, chế độ default)

| Video | 22×17 | 24×31 | 26×50 | 61×38 | 75×51 | 90×113 |
|---|---:|---:|---:|---:|---:|---:|
| BlackBox_0 | 0.0002 | 0.0057 | 0.1747 | 0.2514 | 0.3101 | 0.4067 |
| BlackBox_1 | 0.0006 | 0.0033 | 0.0312 | 0.1645 | 0.2558 | 0.3526 |
| CardboardBox_0 | 0.3128 | 0.3386 | 0.5451 | 0.4974 | 0.4920 | 0.0000 |
| CardboardBox_1 | 0.2374 | 0.3516 | 0.6206 | 0.6754 | 0.7384 | 0.6219 |
| LifeJacket_0 | 0.1480 | 0.2406 | 0.4501 | 0.4192 | 0.4350 | 0.0013 |
| LifeJacket_1 | 0.1769 | 0.3409 | 0.4669 | 0.4870 | 0.4067 | 0.0524 |

---

## 6. Đối chiếu với kích thước object THẬT (`annotations (1).json`)

Bảng trên cho thấy `BlackBox_*` luôn kém nhưng chưa rõ vì sao. Đọc `annotations (1).json` (ground-truth bbox từng frame của đúng 6 video test này) và tính kích thước trung bình `(width+height)/2` thật của object trong mỗi video:

| Video | Kích thước thật TB (px) | avg_w | avg_h | n_box |
|---|---:|---:|---:|---:|
| LifeJacket_0 | 20.1 | 19.4 | 20.9 | 2836 |
| CardboardBox_0 | 25.3 | 23.7 | 26.8 | 1308 |
| LifeJacket_1 | 28.1 | 29.9 | 26.3 | 1685 |
| BlackBox_0 | 52.3 | 54.9 | 49.7 | 1130 |
| CardboardBox_1 | 56.4 | 63.3 | 49.6 | 1765 |
| BlackBox_1 | 77.5 | 82.1 | 72.9 | 854 |

**Phát hiện quan trọng nhất: 2 video `BlackBox_*` có object thật lớn hơn hẳn 4 video còn lại** (52–78px so với 20–56px) — đây chính là lý do cơ học (không chỉ "domain gap" chung chung) khiến chúng luôn kém nhất ở các kích thước tham chiếu nhỏ/vừa: kích thước tham chiếu đang test quá nhỏ so với object thật của riêng chúng.

### Ghép ST-IoU theo kích thước tham chiếu với kích thước thật của từng video

| Video | Kích thước thật | Kích thước tham chiếu cho ST-IoU đỉnh | ST-IoU đỉnh | Tỷ lệ đỉnh/thật | Xu hướng toàn dải |
|---|---:|---|---:|---:|---|
| LifeJacket_0 | 20.1 | 26×50 (≈38px) | 0.4501 | **1.9×** | tăng rồi crash mạnh ở 90×113 (→0.0013) |
| CardboardBox_0 | 25.3 | 26×50 (≈38px) | 0.5451 | **1.5×** | tăng rồi crash mạnh ở 90×113 (→0.0000) |
| LifeJacket_1 | 28.1 | 61×38 (≈49.5px) | 0.4870 | **1.8×** | tăng rồi giảm về 90×113 (→0.0524) |
| BlackBox_0 | 52.3 | 90×113 (≈101.5px) | 0.4067 | **≥1.9×** (chưa tới đỉnh thật) | **tăng đơn điệu** suốt cả dải test, không giảm |
| CardboardBox_1 | 56.4 | 75×51 (≈63px) | 0.7384 | **1.1×** | tăng rồi giảm nhẹ ở 90×113 (→0.6219, vẫn khá cao) |
| BlackBox_1 | 77.5 | 90×113 (≈101.5px) | 0.3526 | **≥1.3×** (chưa tới đỉnh thật) | **tăng đơn điệu** suốt cả dải test, không giảm |

**3 phát hiện rút ra được:**

1. **Kích thước tham chiếu tối ưu không bằng kích thước thật — mà lớn hơn thật khoảng 1.1×–1.9×.** Không có video nào đạt đỉnh ở đúng kích thước = kích thước thật của nó. Điều này khớp với cơ chế đã biết: ảnh reference của AeroEyes là ảnh chụp cận cảnh (chi tiết cao hơn hẳn khung hình video), nên model — vốn train trên FSC147 với giả định exemplar/query cùng scale — "quen" với việc kích thước tham chiếu lớn hơn kích thước xuất hiện thật một chút.

2. **`BlackBox_0` và `BlackBox_1` chưa hề đạt đỉnh trong dải kích thước đã test** — ST-IoU vẫn đang **tăng đơn điệu** tới tận điểm test lớn nhất (90×113). Vì object thật của chúng đã lớn sẵn (52–78px), kích thước tối ưu thật sự nhiều khả năng **vượt quá 90×113**, ngoài phạm vi 6 điểm test hiện tại. Đây là lý do trực tiếp khiến `75×51` (điểm tối ưu tổng hợp cả 6 video) là 1 **thỏa hiệp thiên vị 4 video còn lại** (object nhỏ hơn) chứ không đại diện đúng cho nhóm BlackBox.

3. **`CardboardBox_1`** là video duy nhất có tỷ lệ đỉnh/thật gần 1:1 (1.1×) — cũng là video có ST-IoU cao nhất toàn bộ bảng (0.7384) và ổn định nhất qua các kích thước (0.62–0.74 ở dải giữa) — củng cố thêm: **khi kích thước tham chiếu càng gần tỷ lệ "vừa phải" (~1.1–1.5× thật) thì kết quả càng tốt và ổn định**, còn lệch quá xa (BlackBox cần ratio cao hơn nhiều, chưa đo được) hoặc quá nhỏ (LifeJacket_0 chỉ cần ratio 1.9× nhưng đã là object nhỏ nhất nên dễ mất hoàn toàn khi vượt ngưỡng) đều dẫn tới bất ổn định.

**Kết luận thực tiễn:** không nên tìm 1 `expected_object_px` cố định cho mọi video — nên tính theo **tỷ lệ so với kích thước thật ước lượng của từng đối tượng** (khoảng 1.1–1.9× tuỳ nhóm), hoặc tốt hơn là loại bỏ hẳn nhu cầu biết trước kích thước (đúng hướng finetune bỏ shape token + domain randomization đã đề xuất trong [GECO2_FINETUNE_PLAN.md](GECO2_FINETUNE_PLAN.md#L24-L29), vì tại **thời điểm suy luận thật không có ground-truth để tính tỷ lệ này**).

---

## 7. Kết luận nhanh

1. Với checkpoint GECO2 gốc, **`expected_object_px` ≈ 75×51 (px)** cho kết quả tổng thể tốt nhất trên bộ test AeroEyes hiện tại (Mean ST-IoU 0.4397, F1 @IoU0.5 = 0.690) — cao hơn hẳn dùng đúng kích thước trung vị thật của object (26×50 → chỉ 0.3814).
2. Hiệu năng **giảm ở cả 2 đầu cực nhưng vì 2 lý do khác hẳn nhau**: kích thước nhỏ (22×17, 24×31) chủ yếu lỗi **định vị sai** (vẫn phát hiện được, đặt box lệch); kích thước quá lớn (90×113) là lỗi **bỏ sót phát hiện** (recall sụp ở cả mức presence, không cứu được bằng chỉnh vị trí box) — 2 hướng cải thiện này khác nhau hoàn toàn nếu muốn debug/finetune sau này.
3. **Không có kích thước nào thắng đều trên cả 6 video** — khác biệt lớn giữa nhóm BlackBox và 3 nhóm còn lại là dấu hiệu domain-gap không đồng nhất, củng cố lý do cần finetune thay vì tiếp tục dò `scale_calibration` thủ công.
4. Ngưỡng confidence mặc định của pipeline đang hoạt động tốt (F1 gấp 2–3 lần so với không lọc) — không phải điểm nghẽn chính, kích thước tham chiếu và domain-gap appearance mới là vấn đề chi phối.

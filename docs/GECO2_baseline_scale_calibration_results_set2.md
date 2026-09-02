# Kết quả test GECO2 gốc trên tập test thứ 2 (Helmet/IDCard/Motorbike/Person2/Wallet) theo kích thước tham chiếu

**Nguồn:** log đánh giá do người dùng chạy và dán trực tiếp — số liệu dưới đây trích xuất/tổng hợp lại, chưa kiểm chứng độc lập.

**Model:** GECO2 gốc (chưa finetune, còn nguyên shape token), pipeline AeroEyes, `scale_calibration` bật, thử 5 giá trị `expected_object_px` (thiếu điểm nhỏ nhất 22×17 so với tập test trước).

**Test set:** 10 video — `Helmet_0/1`, `IDCard_0/1`, `Motorbike_0/1`, `Person2_0/1`, `Wallet_0/1` — **khác hoàn toàn** 6 video ở [báo cáo tập test 1](GECO2_baseline_scale_calibration_results.md) (`BlackBox/CardboardBox/LifeJacket`). GT: `annotations_converted.json`.

**⚠️ Khác biệt quan trọng so với báo cáo tập test 1:** toàn bộ log ở đây chỉ chạy ở **`conf 0.0`** (không lọc confidence) — **không có** đợt chạy "default" (có ngưỡng confidence) để đối chiếu. Vì vậy các con số Precision/F1 tuyệt đối trong file này **thấp hơn hẳn** báo cáo tập 1 một cách hệ thống (do thiếu bước lọc nhiễu), **không nên so sánh trực tiếp** 2 file để kết luận "tập test này khó hơn/dễ hơn" — chỉ nên so sánh nội bộ giữa các kích thước trong cùng file này, hoặc so với riêng phần `conf 0.0` của báo cáo tập 1.

Theo yêu cầu, bỏ bảng `IoU≥0.10` (chỉ giữ presence-only, `IoU≥0.50`, `IoU≥0.30`).

---

## 1. Mean ST-IoU theo kích thước (conf 0.0)

| Kích thước (W×H, px) | Mean ST-IoU |
|---|---:|
| 24×31 | 0.0498 |
| 26×50 | 0.0866 |
| 61×38 | 0.0878 |
| 75×51 | 0.0963 |
| 90×113 | **0.1188** ← cao nhất trong dải test |

**Khác biệt lớn với tập test 1:** ở đây ST-IoU **tăng đơn điệu** theo kích thước suốt cả dải test, **chưa có dấu hiệu đạt đỉnh** — khác hẳn tập test 1, nơi hiệu năng đạt đỉnh ở 75×51 rồi giảm mạnh tại 90×113. Nhiều khả năng kích thước tối ưu thật sự của tập test này **lớn hơn 90×113** (ngoài phạm vi đã test).

---

## 2. Chi tiết Detection Metrics (micro-avg trên 10 video, conf 0.0)

| Kích thước | Presence-only P/R/F1 | @IoU≥0.50 P/R/F1 | @IoU≥0.30 P/R/F1 |
|---|---|---|---|
| 24×31 | 0.265 / 1.000 / 0.419 | 0.056 / 0.212 / 0.089 | 0.078 / 0.296 / 0.124 |
| 26×50 | 0.265 / 1.000 / 0.419 | 0.099 / 0.373 / 0.156 | 0.134 / 0.505 / 0.212 |
| 61×38 | 0.269 / 1.000 / 0.423 | 0.101 / 0.374 / 0.159 | 0.137 / 0.510 / 0.216 |
| 75×51 | 0.272 / 1.000 / 0.428 | 0.113 / 0.414 / 0.177 | 0.153 / 0.561 / 0.240 |
| 90×113 | 0.288 / 0.999 / 0.448 | **0.148 / 0.514 / 0.230** | **0.186 / 0.644 / 0.288** |

`90×113` cho F1 cao nhất ở cả 3 cách đánh giá — nhất quán với bảng Mean ST-IoU.

---

## 3. Phân tích Recall — vì sao gap Presence↔IoU ở đây đo được gần như thuần túy lỗi định vị

Vì **toàn bộ log đều chạy ở `conf 0.0`**, Recall (presence) luôn bão hòa gần 1.000 ở mọi kích thước (0.999–1.000) — không có bước lọc confidence nào làm giảm nó. Nghĩa là khoảng cách Recall(presence) − Recall(IoU) ở đây **không còn lẫn lộn giữa "lỗi bỏ sót phát hiện" và "lỗi định vị sai"** như báo cáo tập test 1 (nơi có so sánh conf mặc định vs conf 0.0) — ở đây gap gần như **thuần túy phản ánh chất lượng định vị**, vì phần phát hiện đã bão hòa.

| Kích thước | Recall (presence) | Recall (@IoU≥0.50) | Gap Recall (≈ lỗi định vị thuần) |
|---|---:|---:|---:|
| 24×31 | 1.000 | 0.212 | **0.788** |
| 26×50 | 1.000 | 0.373 | 0.627 |
| 61×38 | 1.000 | 0.374 | 0.626 |
| 75×51 | 1.000 | 0.414 | 0.586 |
| 90×113 | 0.999 | 0.514 | **0.485** |

Gap giảm dần đều theo kích thước (0.788 → 0.485) — **định vị chính xác hơn khi kích thước tham chiếu lớn hơn**, nhất quán với tập test 1, nhưng ở đây thấy rõ hơn vì không bị nhiễu bởi hiệu ứng lọc confidence.

---

## 4. So sánh giữa các video (per-video ST-IoU)

| Video | 24×31 | 26×50 | 61×38 | 75×51 | 90×113 |
|---|---:|---:|---:|---:|---:|
| Helmet_0 | 0.1029 | 0.1407 | 0.1397 | 0.1394 | 0.1239 |
| Helmet_1 | 0.0367 | 0.0496 | 0.0531 | 0.0591 | 0.0869 |
| IDCard_0 | 0.0220 | 0.0512 | 0.0488 | 0.0376 | 0.0186 |
| IDCard_1 | 0.0447 | 0.0850 | 0.0812 | 0.0769 | 0.0346 |
| Motorbike_0 | 0.0035 | 0.0083 | 0.0072 | 0.0104 | 0.0214 |
| Motorbike_1 | 0.0058 | 0.0232 | 0.0519 | 0.1464 | **0.3515** |
| Person2_0 | 0.1016 | 0.1453 | 0.1209 | 0.0982 | 0.1030 |
| Person2_1 | 0.0689 | 0.1033 | 0.1114 | 0.1315 | 0.1895 |
| Wallet_0 | 0.1012 | 0.1274 | 0.1211 | 0.1221 | 0.1170 |
| Wallet_1 | 0.0108 | 0.1324 | 0.1422 | 0.1417 | 0.1417 |

**Quan sát nổi bật:** `Motorbike_1` có mức tăng mạnh nhất toàn bảng — từ gần 0 (0.0058 ở 24×31) vọt lên **0.3515** ở 90×113 (cao nhất trong toàn bộ tập test này), vẫn đang tăng dốc, chưa có dấu hiệu chững lại. Ngược lại `Motorbike_0` gần như thất bại hoàn toàn ở **mọi** kích thước (0.003–0.021, không bao giờ vượt 0.03) — 2 video cùng loại đối tượng nhưng hành vi trái ngược hẳn nhau, cần xem riêng (khả năng do góc quay/tư thế xe khác biệt lớn, xem mục 5).

---

## 5. Đối chiếu với kích thước object THẬT (`annotations_converted.json`)

| Video | Kích thước thật TB (px) | avg_w | avg_h | Ghi chú |
|---|---:|---:|---:|---|
| Helmet_0 | 13.2 | 14.8 | 11.6 | nhỏ nhất |
| IDCard_0 | 23.4 | 29.0 | 17.8 | |
| Person2_0 | 26.8 | 22.3 | 31.4 | |
| Helmet_1 | 28.1 | 29.2 | 27.0 | |
| Wallet_0 | 31.4 | 42.0 | 20.8 | |
| IDCard_1 | 34.0 | 42.5 | 25.6 | |
| Person2_1 | 50.9 | 38.4 | 63.4 | |
| Motorbike_1 | 61.3 | 58.1 | 64.5 | |
| Wallet_1 | 64.8 | 87.1 | 42.6 | |
| Motorbike_0 | 77.6 | 55.3 | **99.9** | **bbox rất cao/hẹp** (w=55 nhưng h=100) |

**Phát hiện quan trọng: `Motorbike_0` có tỷ lệ khung hình (aspect ratio) object bất thường** — width=55px nhưng height=99.9px (tỷ lệ ~1:1.8, rất cao/hẹp), khác hẳn `Motorbike_1` (58×64, gần vuông). Cả 6 kích thước tham chiếu đang test đều lấy từ phân tích percentile của 1 tập dữ liệu khác (không có cặp nào cao/hẹp cỡ này) — nghĩa là **không có kích thước nào trong 5 điểm test khớp đúng tỷ lệ khung hình thật của `Motorbike_0`**, có thể là lý do chính khiến nó thất bại ở mọi kích thước (0.003–0.021) — đây là vấn đề **lệch tỷ lệ khung hình (aspect ratio)**, không đơn thuần là lệch kích thước tổng thể.

**Tỷ lệ đỉnh/thật (trong các video đã có dấu hiệu đạt đỉnh, không tính 2 video vẫn đang tăng ở 90×113):**

| Video | Kích thước thật | Kích thước cho ST-IoU đỉnh (trong dải test) | Tỷ lệ đỉnh/thật |
|---|---:|---|---:|
| Helmet_0 | 13.2 | 26×50 (≈38px) | **2.9×** |
| IDCard_0 | 23.4 | 26×50 (≈38px) | 1.6× |
| Person2_0 | 26.8 | 26×50 (≈38px) | 1.4× |
| Wallet_0 | 31.4 | 26×50 (≈38px) | 1.2× |
| IDCard_1 | 34.0 | 26×50 (≈38px) | 1.1× |
| Wallet_1 | 64.8 | 61×38 (≈49.5px) | 0.76× (nhỏ hơn thật) |

Còn **`Helmet_1`, `Person2_1`, `Motorbike_1`, `Motorbike_0`** đều **chưa đạt đỉnh** trong dải test (vẫn tăng tới tận 90×113) — object thật của chúng (28–78px) đã tương đối lớn, khớp với phát hiện ở báo cáo tập test 1 rằng **video có object thật lớn cần kích thước tham chiếu lớn hơn nhiều mới tới điểm tối ưu**, thường vượt ngoài phạm vi các điểm đã test.

---

## 6. Kết luận nhanh

1. Trong dải 5 kích thước đã test (24×31 → 90×113), **90×113 luôn tốt nhất** cho tập test này — nhưng xu hướng vẫn đang tăng, nên **90×113 nhiều khả năng chưa phải điểm tối ưu thật sự**, cần test thêm các kích thước lớn hơn để tìm đỉnh thật.
2. Vì tất cả log đều ở `conf 0.0`, **không đánh giá được vai trò của ngưỡng confidence** trên tập test này — cần chạy thêm ở ngưỡng mặc định (như tập test 1) để so sánh công bằng.
3. Gap Recall (presence − IoU0.5) giảm đều theo kích thước (0.788 → 0.485) — nhất quán với tập test 1: **định vị chính xác hơn khi kích thước tham chiếu lớn hơn**.
4. `Motorbike_0` là trường hợp thất bại toàn diện, khả năng cao do **lệch tỷ lệ khung hình** (bbox cao/hẹp bất thường) chứ không chỉ lệch kích thước — gợi ý hướng cải thiện: nên thử kích thước tham chiếu theo đúng **tỷ lệ w:h thật của từng object**, không chỉ theo giá trị percentile độ lớn trung bình.
5. Cùng loại đối tượng (`Motorbike_0` vs `Motorbike_1`) có thể cho kết quả cực kỳ trái ngược tùy hình dạng/góc quay cụ thể — không nên gộp chung kết luận theo "loại object", cần xét từng video riêng.

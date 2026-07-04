# Hướng Dẫn Chạy AERO EYES trên Colab & Kaggle

## Mục Lục
1. [Upload project lên GitHub](#1-upload-project-lên-github)
2. [Google Colab](#2-google-colab)
3. [Kaggle](#3-kaggle)
4. [So sánh hai nền tảng](#4-so-sánh-hai-nền-tảng)
5. [Mẹo tiết kiệm thời gian GPU](#5-mẹo-tiết-kiệm-thời-gian-gpu)

---

## 1. Upload Project Lên GitHub

### Bước 1 — Khởi tạo git repo
```powershell
cd c:\Projects\aero_eyes_starter
git init
git add .
git commit -m "Initial commit: AERO EYES pipeline"
```

> **File `.gitignore` đã được cấu hình sẵn** để loại trừ:
> - `data/` — video và ảnh dataset (upload riêng lên Drive/Kaggle)
> - `runs/` — kết quả pipeline
> - `*.pt`, `*.onnx` — model weights (tự download khi chạy)
> - `.venv/` — môi trường Python local

### Bước 2 — Tạo repo trên GitHub
1. Vào [github.com/new](https://github.com/new)
2. Đặt tên repo, ví dụ: `aero_eyes_starter`
3. Chọn **Private** nếu là project thi đấu
4. **Không** tick "Add README" (đã có sẵn)

### Bước 3 — Push lên GitHub
```powershell
git remote add origin https://github.com/YOUR_USERNAME/aero_eyes_starter.git
git branch -M main
git push -u origin main
```

### Những gì được push lên GitHub
```
✅ aero_eyes/          (toàn bộ source code)
✅ configs/config.yaml
✅ scripts/
✅ tests/
✅ docs/
✅ notebooks/aero_eyes_colab.ipynb
✅ requirements.txt
✅ annotations (1).json   ← file GT nhỏ, push luôn được
❌ data/                  (upload riêng lên Drive hoặc Kaggle)
❌ runs/                  (kết quả runtime)
❌ .venv/                 (môi trường local)
```

---

## 2. Google Colab

### Thông số kỹ thuật (2025)
| | Free | Pro ($10/tháng) |
|--|--|--|
| GPU | T4 (15 GB VRAM) | T4 / A100 (40 GB) |
| RAM | 12–13 GB | 26–52 GB |
| Disk | ~100 GB | ~200 GB |
| Thời gian tối đa | ~12h session | ~24h session |
| Idle timeout | ~90 phút | ~3 giờ |

### Cách 1 — Dùng notebook có sẵn (khuyến nghị)

1. **Upload data lên Google Drive trước:**
   ```
   MyDrive/
   └── aero_eyes_data/
       ├── annotations (1).json
       ├── Backpack_0/
       │   ├── refs/ref_0.jpg, ref_1.jpg, ref_2.jpg
       │   └── video.mp4
       └── Person_1/ ...
   ```

2. **Mở notebook từ GitHub:**
   - Vào [colab.research.google.com](https://colab.research.google.com)
   - `File → Open notebook → GitHub`
   - Nhập URL repo của bạn
   - Chọn `notebooks/aero_eyes_colab.ipynb`

3. **Bật GPU:**
   - `Runtime → Change runtime type → T4 GPU → Save`

4. **Chỉnh 2 dòng trong notebook rồi chạy từ trên xuống:**
   ```python
   GITHUB_REPO = 'https://github.com/YOUR_USERNAME/aero_eyes_starter.git'
   DRIVE_DATA_DIR = '/content/drive/MyDrive/aero_eyes_data'
   ```

### Cách 2 — Chạy thủ công bằng lệnh

Tạo notebook mới trên Colab và chạy từng cell:

**Cell 1 — Kiểm tra GPU:**
```python
!nvidia-smi
```

**Cell 2 — Clone repo:**
```bash
!git clone https://github.com/YOUR_USERNAME/aero_eyes_starter.git
%cd aero_eyes_starter
```

**Cell 3 — Cài dependencies:**
```bash
!pip install -r requirements.txt -q
!pip install -e . -q
```

**Cell 4 — Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 5 — Tạo symlink data:**
```python
import os
os.symlink('/content/drive/MyDrive/aero_eyes_data', 'data')
os.symlink('/content/drive/MyDrive/aero_eyes_data/annotations (1).json',
           'annotations (1).json')
```

**Cell 6 — Chạy pipeline:**
```bash
!python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --sample Backpack_0 \
    --set project.work_dir=/content/drive/MyDrive/aero_eyes_runs/exp001
```

**Cell 7 — Đánh giá:**
```bash
!python -m aero_eyes.evaluate \
    --pred /content/drive/MyDrive/aero_eyes_runs/exp001/Backpack_0/submission.json \
    --gt "annotations (1).json" \
    --config configs/config.yaml
```

### Lưu kết quả khỏi mất khi session hết

> ⚠️ **Colab xoá toàn bộ `/content/` khi session kết thúc.**

Giải pháp: **luôn trỏ `work_dir` vào Google Drive:**
```python
--set project.work_dir=/content/drive/MyDrive/aero_eyes_runs/exp001
```

Kết quả sẽ được lưu ở `MyDrive/aero_eyes_runs/exp001/<sample_id>/` và **tồn tại vĩnh viễn** ngay cả khi session Colab kết thúc.

### Tiếp tục sau khi session hết

```bash
# Chạy lại từ stage bị ngắt (dùng --from-stage)
!python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --sample Backpack_0 \
    --from-stage 3 \
    --set project.work_dir=/content/drive/MyDrive/aero_eyes_runs/exp001
```

---

## 3. Kaggle

### Thông số kỹ thuật (2025)
| | Free |
|--|--|
| GPU | 2× T4 (16 GB mỗi card) hoặc P100 (16 GB) |
| RAM | 30 GB |
| Disk | 20 GB (notebook) + dataset riêng |
| GPU quota | 30h/tuần |
| Thời gian tối đa | 12h mỗi run |

### Bước 1 — Upload dataset lên Kaggle

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**

2. Upload theo cấu trúc:
   ```
   aero-eyes-dataset/
   ├── annotations (1).json
   ├── Backpack_0/
   │   ├── refs/ref_0.jpg, ref_1.jpg, ref_2.jpg
   │   └── video.mp4
   └── Person_1/ ...
   ```

3. Đặt tên dataset (ví dụ: `aero-eyes-dataset`), chọn **Private**

4. Lưu lại **Dataset path**, ví dụ: `your_username/aero-eyes-dataset`

### Bước 2 — Tạo notebook Kaggle

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**

2. Bật GPU: `Settings (bên phải) → Accelerator → GPU T4 x2`

3. Thêm dataset: `Settings → Add data → Your datasets → aero-eyes-dataset`

4. Dán các cell sau:

**Cell 1 — Clone repo:**
```bash
%%bash
git clone https://github.com/YOUR_USERNAME/aero_eyes_starter.git /kaggle/working/aero_eyes
cd /kaggle/working/aero_eyes
pip install -r requirements.txt -q
pip install -e . -q
echo "Done!"
```

**Cell 2 — Cấu hình đường dẫn:**
```python
import os

REPO_DIR = '/kaggle/working/aero_eyes'
# Kaggle mount dataset vào /kaggle/input/<dataset-name>/
DATA_DIR = '/kaggle/input/aero-eyes-dataset'   # ← đổi theo tên dataset của bạn
WORK_DIR = '/kaggle/working/runs/exp001'

os.chdir(REPO_DIR)

# Kiểm tra data
samples = [d for d in os.listdir(DATA_DIR) if os.path.isdir(f'{DATA_DIR}/{d}')]
print(f'Tìm thấy {len(samples)} sample(s):', samples[:5])
```

**Cell 3 — Chạy pipeline:**
```bash
%%bash
cd /kaggle/working/aero_eyes

python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --set data.data_root=/kaggle/input/aero-eyes-dataset \
    --set project.work_dir=/kaggle/working/runs/exp001 \
    --set stage1.feature_extractor.dinov2_variant=vitb14 \
    --set accuracy.mode=cheap_boosters
```

**Cell 4 — Đánh giá:**
```bash
%%bash
cd /kaggle/working/aero_eyes

# Gom tất cả submission thành 1 file
python -c "
import json, os, glob
preds = []
for f in glob.glob('/kaggle/working/runs/exp001/*/submission.json'):
    preds.extend(json.load(open(f)))
json.dump(preds, open('/kaggle/working/all_submissions.json','w'))
print(f'Gom {len(preds)} video')
"

python -m aero_eyes.evaluate \
    --pred /kaggle/working/all_submissions.json \
    --gt '/kaggle/input/aero-eyes-dataset/annotations (1).json' \
    --config configs/config.yaml
```

**Cell 5 — Lưu output để download:**
```python
# Kaggle tự động lưu /kaggle/working/ vào output của notebook
import shutil
shutil.copy('/kaggle/working/all_submissions.json',
            '/kaggle/working/submission.json')
print('File submission.json đã sẵn sàng ở Output tab!')
```

### Download kết quả từ Kaggle
- Sau khi notebook chạy xong: **Output tab (bên phải)** → `submission.json` → Download

---

## 4. So Sánh Hai Nền Tảng

| | Google Colab Free | Kaggle Free |
|--|--|--|
| **GPU** | T4 (15 GB) | T4 x2 hoặc P100 (16 GB) |
| **RAM** | 12 GB | 30 GB |
| **Thời gian** | ~12h (có thể ngắt sớm) | 12h (ổn định hơn) |
| **Quota** | Không rõ, có thể bị hạn chế | 30h GPU/tuần rõ ràng |
| **Lưu file** | Cần mount Google Drive | Tự động lưu `/kaggle/working/` |
| **Dataset lớn** | Dùng Google Drive | Upload lên Kaggle Dataset |
| **Dễ dùng** | Rất dễ, giao diện quen | Cần upload dataset trước |
| **Internet** | Có | Có (cần bật trong Settings) |
| **Khuyến nghị cho** | Debug nhanh, dataset trên Drive | Chạy dài, dataset cố định |

### Khuyến nghị theo use case

| Tình huống | Nền tảng |
|--|--|
| Test nhanh, đang phát triển | **Colab** |
| Chạy toàn bộ dataset, nộp leaderboard | **Kaggle** |
| Dataset đã có trên Google Drive | **Colab** |
| Muốn version dataset rõ ràng | **Kaggle** |
| Cần nhiều RAM (>15 GB) | **Kaggle** (30 GB RAM) |

---

## 5. Mẹo Tiết Kiệm Thời Gian GPU

### Dùng cache — chạy lại không tốn thời gian

Pipeline có cơ chế cache (`project.use_cache: true`). Nếu artifact đã tồn tại, stage đó bị bỏ qua:

```bash
# Stage 1 (prototype) chỉ chạy 1 lần dù restart session
# → chỉ tốn thời gian lần đầu
```

Để **buộc chạy lại** (ví dụ sau khi đổi config):
```bash
--set project.use_cache=false
```

### Chạy song song nhiều sample (Kaggle có 2x T4)

```python
import subprocess, concurrent.futures

samples = ['Backpack_0', 'Person_1', 'Car_2']
base_cmd = [
    'python', '-m', 'aero_eyes.stages.run_all',
    '--config', 'configs/config.yaml',
    '--set', 'data.data_root=/kaggle/input/aero-eyes-dataset',
    '--set', 'project.work_dir=/kaggle/working/runs/exp001',
]

def run_sample(sid):
    cmd = base_cmd + ['--sample', sid]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return sid, result.returncode

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    for sid, code in ex.map(run_sample, samples):
        print(f'{sid}: {"OK" if code == 0 else "FAILED"}')
```

### Tắt visualizations để nhanh hơn

```bash
--set runtime.save_visualizations=false
```

### Giảm keyframe interval nếu video dài

```bash
--set stage2.keyframe_interval=4   # dày hơn, chậm hơn nhưng recall cao hơn
--set stage2.keyframe_interval=16  # thưa hơn, nhanh hơn
```

### Kiểm tra tiến trình không bị ngắt (Colab)

Thêm cell này để Colab không idle timeout:
```javascript
// Chạy trong Console của browser (F12)
function keepAlive() {
  document.querySelector('#run-all-btn').click();
  setTimeout(keepAlive, 60000);
}
// Không dùng cách này — thay vào đó hãy mua Colab Pro hoặc dùng Kaggle
```

> **Cách tốt hơn:** Dùng **Kaggle** nếu cần chạy >4h, vì Kaggle không có idle timeout như Colab Free.

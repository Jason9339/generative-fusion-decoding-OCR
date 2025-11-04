# TrOCR + LLM Fusion OCR 測試指南

本指南說明如何測試 TrOCR + LLM Fusion 架構的 OCR 效果。

## 目錄

1. [環境設定](#環境設定)
2. [測試腳本說明](#測試腳本說明)
3. [快速測試](#快速測試)
4. [完整評估測試](#完整評估測試)
5. [使用單檔測試](#使用單檔測試)
6. [測試結果解讀](#測試結果解讀)

---

## 環境設定

### 1. 啟動虛擬環境

本專案使用虛擬環境，確保先啟動：

```bash
source .venv/bin/activate
```

或使用 conda：

```bash
conda activate your_env_name
```

### 2. 檢查依賴套件

確認已安裝必要套件：

```bash
pip list | grep -E "(torch|transformers|Pillow)"
```

應該看到：
- `torch` (2.2.1 或更高)
- `transformers` (4.40.1 或更高)
- `Pillow` (10.3.0 或更高)

---

## 測試腳本說明

專案提供三個測試腳本：

### 1. `test_trocr_setup.py` - TrOCR 基礎測試

**用途**: 驗證 TrOCR 模型載入、special tokens 和基本識別功能

**測試項目**:
- TrOCR 模型載入 (microsoft/trocr-base-printed)
- Tokenizer 和 special tokens 驗證
- Byte tokenizer 功能測試（convert_ids_to_bytes、tokenize_from_byte）
- 圖片識別測試（使用程式生成的測試圖片）

**執行方式**:
```bash
PYTHONPATH=. python test_trocr_setup.py
```

**預期輸出**:
- ✓ 模型載入成功
- ✓ Tokenizer 功能正常
- ✓ 圖片識別結果正確

---

### 2. `test_trocr_fusion.py` - 完整 Fusion 測試

**用途**: 測試 TrOCR + LLM 的 fusion 流程

**測試項目**:
- 使用配置檔案載入完整 fusion 架構
- 測試多個文字案例（HELLO WORLD、GENERATIVE FUSION 等）
- 驗證 beam search + LLM fusion 的正確性
- 檢查識別準確率

**執行方式**:
```bash
PYTHONPATH=. python test_trocr_fusion.py
```

**預期輸出**:
```
測試案例 1: 'HELLO WORLD'
識別結果: 'HELLO WORLD'
匹配: ✓

...

測試總結
總測試數: 4
成功匹配: 4
匹配率: 100.0%
```

---

### 3. `test_ocr_evaluation.py` - 完整效果評估

**用途**: 全面評估 OCR 在不同條件下的表現

**測試項目**:

#### 測試 1: 基礎文字識別
- 測試 5 種不同的文字內容
- 評估基本識別準確率

#### 測試 2: 不同字體大小
- 測試字體大小：20, 30, 40, 50, 60
- 評估模型對不同字體大小的適應性

#### 測試 3: 噪音干擾測試
- 測試噪音等級：0.0, 0.05, 0.1, 0.15, 0.2
- 評估抗噪能力

#### 測試 4: Fusion 開關對比
- 比較 with fusion (fusing_r=0.2) vs without fusion (fusing_r=0.0)
- 評估 LLM fusion 的實際效益

**執行方式**:
```bash
PYTHONPATH=. python test_ocr_evaluation.py
```

**注意**: 此測試會載入兩個模型（with/without fusion），需要較多時間和記憶體。

**輸出檔案**: `/tmp/ocr_evaluation_results.json`

---

## 快速測試

如果只想快速驗證系統是否正常運作：

```bash
# 1. 啟動虛擬環境
source .venv/bin/activate

# 2. 執行基礎測試
export PYTHONPATH=.
python test_trocr_setup.py
```

預期 3-5 分鐘完成，應看到所有測試通過。

---

## 完整評估測試

如果要進行完整的效果評估：

```bash
# 1. 啟動虛擬環境
source .venv/bin/activate

# 2. 執行完整評估
export PYTHONPATH=.
python test_ocr_evaluation.py

# 3. 查看結果
cat /tmp/ocr_evaluation_results.json
```

預期 15-30 分鐘完成（視 CPU 速度而定）。

---

## 使用單檔測試

如果要測試自己的圖片：

### 1. 使用命令行工具

```bash
python benchmarks/run_single_file.py \
  --model_name gfd \
  --setting ocr-en \
  --image_file_path /path/to/your/image.png \
  --result_output_path output.json
```

### 2. 使用 Python 腳本

```python
from PIL import Image
from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

# 載入配置
model_config = process_config('config_files/model/gfd-ocr-en.yaml')
prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')
config = combine_config(prompt_config, model_config)

# 創建模型
model = Breezper(config)

# 載入圖片
image = Image.open('/path/to/your/image.png')

# 執行識別
result = model.get_transcription(
    image,
    num_beams=5,
    ocr_prompt=config.ocr_prompt,
    llm_prompt=config.llm_prompt
)

print(f"識別結果: {result}")
```

---

## 測試結果解讀

### 1. 基礎測試結果

**成功的指標**:
- 所有測試項目顯示 ✓
- 生成的測試圖片能正確識別
- Bytes roundtrip 成功
- 沒有錯誤訊息

**可能的警告**:
```
Some weights of VisionEncoderDecoderModel were not initialized...
```
這是正常的，因為使用預訓練的 encoder 和 decoder 組合。

```
The tokenizer class you load from this checkpoint is not the same type...
```
這也是正常的，因為我們使用了自定義的 `TrOCRByteTokenizer`。

### 2. Fusion 測試結果

**評估指標**:

- **匹配率**: 應該達到 90% 以上
- **識別速度**: CPU 上每張圖片約 2-10 秒
- **Debug 輸出**: 顯示每個 beam 的 score，可觀察 fusion 過程

**預期行為**:
```
[0] recognizer_score=-0.002, llm_score=-5.133, fuse_score=-1.028,
HELLO
```

- `recognizer_score`: TrOCR 的分數（越高越好）
- `llm_score`: LLM 的分數（越高越好）
- `fuse_score`: 融合後的分數 = (1-r)*recognizer + r*llm

### 3. 評估測試結果

**關鍵指標**:

1. **準確率**
   - 基礎識別: 應 > 95%
   - 字體大小: 20-60 範圍內應保持穩定
   - 噪音測試: 低噪音 (< 0.1) 應保持高準確率

2. **Fusion 效益**
   - With Fusion 的準確率應 ≥ Without Fusion
   - 時間成本增加合理（約 20-50%）

3. **穩定性**
   - 相同輸入應產生相同輸出
   - 不應有崩潰或異常錯誤

---

## 配置調整

### 修改 OCR 模型

編輯 `config_files/model/gfd-ocr-en.yaml`:

```yaml
ocr_model_path: 'microsoft/trocr-base-printed'  # 可改為其他 TrOCR 模型
```

可用的 TrOCR 模型：
- `microsoft/trocr-base-printed`: 印刷體（推薦）
- `microsoft/trocr-base-handwritten`: 手寫體
- `microsoft/trocr-large-printed`: 大型印刷體模型（更準確但更慢）

### 修改 LLM 模型

編輯 `config_files/model/gfd-ocr-en.yaml`:

```yaml
llm_model_path: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # 可改為其他 Llama 模型
```

### 調整 Fusion 比例

編輯 `config_files/model/gfd-ocr-en.yaml`:

```yaml
fusing_r: 0.2  # 範圍 0.0-1.0，越高 LLM 影響越大
```

- `fusing_r = 0.0`: 純 TrOCR（無 fusion）
- `fusing_r = 0.2`: 推薦值（輕度 fusion）
- `fusing_r = 0.5`: 中度 fusion
- `fusing_r = 1.0`: 純 LLM（僅用 TrOCR 的 byte output）

### 修改 Prompt

編輯 `config_files/prompt/ocr-default-prompt.yaml`:

```yaml
ocr_prompt: ''  # TrOCR 的 prompt（通常留空）
llm_prompt: 'The following is a transcription of the text contained in the image:'
```

---

## 常見問題

### Q1: 測試時出現 CUDA out of memory

**解決方案**:
```yaml
# 在 config_files/model/gfd-ocr-en.yaml 中
ocr_device: 'cpu'
llm_device: 'cpu'
```

### Q2: 識別結果不理想

**檢查清單**:
1. 圖片是否清晰？
2. 文字是否為英文印刷體？
3. 是否使用了正確的 TrOCR 模型？
4. 嘗試調整 `num_beams` 參數（3-5）
5. 嘗試調整 `fusing_r` 比例

### Q3: 測試速度太慢

**優化建議**:
1. 使用更小的 LLM 模型（如 TinyLlama）
2. 降低 `num_beams` 數量
3. 使用 GPU（如有）
4. 設定 `use_cache: 'dynamic'`（預設已開啟）

### Q4: 警告訊息很多

**說明**: 以下警告可以忽略
- `Some weights of VisionEncoderDecoderModel were not initialized...`
- `The tokenizer class you load from this checkpoint is not the same type...`
- `FutureWarning: resume_download is deprecated...`

---

## 進階測試

### 測試實際文件圖片

如果要測試實際掃描的文件：

```python
from PIL import Image
from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

# 載入配置
model_config = process_config('config_files/model/gfd-ocr-en.yaml')
prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')
config = combine_config(prompt_config, model_config)

# 創建模型
model = Breezper(config)

# 測試多個文件
image_paths = [
    'scanned_doc_1.png',
    'scanned_doc_2.jpg',
    'receipt.png',
]

for img_path in image_paths:
    print(f"\n處理: {img_path}")
    image = Image.open(img_path)

    # 可能需要預處理
    # image = image.convert('RGB')
    # image = image.resize((width, height))

    result = model.get_transcription(image, num_beams=5)
    print(f"結果: {result}")
```

### Benchmark 不同模型

```python
import time
from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

models_to_test = [
    'microsoft/trocr-base-printed',
    'microsoft/trocr-large-printed',
]

for model_path in models_to_test:
    # 更新配置
    config.ocr_model_path = model_path

    # 重新載入模型
    model = Breezper(config)

    # 測試
    start = time.time()
    result = model.get_transcription(test_image, num_beams=5)
    elapsed = time.time() - start

    print(f"{model_path}: {result} ({elapsed:.2f}s)")
```

---

## 總結

完整測試流程：

1. **快速驗證**: 執行 `test_trocr_setup.py`
2. **Fusion 測試**: 執行 `test_trocr_fusion.py`
3. **完整評估**: 執行 `test_ocr_evaluation.py`
4. **實際應用**: 使用 `run_single_file.py` 處理真實圖片

如有問題，請檢查：
- 虛擬環境是否啟動
- 依賴套件是否完整
- 網路連接是否正常（首次需下載模型）
- 記憶體是否足夠（建議 8GB+）

---

最後更新: 2025-11-04

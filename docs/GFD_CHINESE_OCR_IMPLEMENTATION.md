# GFD 繁體中文 OCR 實現文檔

## 📋 概述

成功將 **Generative Fusion Decoding (GFD)** 架構適配到你的繁體中文 TrOCR 模型上！

### 核心技術突破

**問題**：GFD 原生需要 byte-level tokenizer（如 RobertaTokenizer），但你的模型使用字符級 `PreTrainedTokenizerFast`

**解決方案**：創建了一個 **byte-level 兼容層**，將字符級 tokenizer 包裝為 GFD 所需的接口

---

## 🏗️ 實現架構

### 1. Tokenizer 兼容層

**文件**: `generative-fusion-decoding-OCR/gfd/tokenizer_chinese_adapter.py`

**核心類**: `ChineseCharTokenizerAdapter`

**功能**：
- 繼承自 `PreTrainedTokenizerFast` 和 `ByteTokenizer`
- 實現 `byte_encoder` 和 `byte_decoder` 映射
- 實現 `convert_ids_to_bytes()` 方法
- 實現 `tokenize_from_byte()` 方法
- 完全兼容 GFD 的 byte-space 操作

**測試結果** ✅：
```
測試 1: 基本 tokenization - ✓
測試 2: convert_ids_to_bytes - ✓
測試 3: tokenize_from_byte - ✓
```

---

### 2. 中文 GFD 核心

**文件**: `generative-fusion-decoding-OCR/gfd/gfd_chinese.py`

**核心類**: `ChineseOCRBreezper`

**改進點**：
- 使用 `ChineseCharTokenizerAdapter` 替代 `TrOCRByteTokenizer`
- 支援中文字符級 tokenization
- 保留完整的 GFD 融合解碼算法
- 支援動態 KV cache 優化

**GFD 工作原理**：
```
1. OCR 模型生成候選 tokens
2. 將 tokens 轉換為 bytes
3. LLM 在 byte-space 中評估語言合理性
4. 融合 OCR score 和 LLM score
5. Beam search 選擇最佳路徑
```

---

### 3. 配置文件

**模型配置**: `config_files/model/gfd-ocr-multilmdb-zhtw.yaml`

```yaml
ocr_model_path: '/mnt/whliao/experiment/output_multi_lmdb'
ocr_tokenizer_path: '/mnt/whliao/experiment/tokenizer'
llm_model_path: 'MediaTek-Research/Breeze-7B-32k-Base-v1_0'

ocr_device: 'cuda:0'
llm_device: 'cuda:0'
ocr_torch_dtype: 'float16'
llm_torch_dtype: 'float16'

# Fusion 設定
fuse_strategy: 'simple'
fusing_r: 0.3  # 30% LLM, 70% OCR
```

**提示詞配置**: `config_files/prompt/ocr-zhtw-prompt.yaml`

```yaml
ocr_prompt: ''
llm_prompt: '以下是圖片中繁體中文文字的辨識結果：'
```

---

## 🧪 測試腳本

**文件**: `test_gfd_inference_example.py`

**用法**：
```bash
# 完整測試 (使用 LLM)
python test_gfd_inference_example.py \
  --max_samples 10 \
  --num_beams 5 \
  --output gfd_results.json

# 不使用 LLM (僅 OCR baseline)
python test_gfd_inference_example.py \
  --max_samples 10 \
  --num_beams 5 \
  --no_llm \
  --output gfd_baseline_results.json
```

**參數說明**：
- `--max_samples`: 測試的最大樣本數
- `--num_beams`: Beam search 寬度（越大越準確但越慢）
- `--no_llm`: 停用 LLM，僅使用 OCR（用於對比）
- `--output`: 結果保存路徑

---

## 📊 預期效果

### 原始 TrOCR（無 GFD）
- **CER**: 42.09%
- **完全匹配**: 2.38%
- **問題**: 對實際掃描圖的泛化能力不足

### GFD 預期改善

**理論優勢**：
1. **語言模型糾錯**: Breeze-7B LLM 具有強大的中文語言理解能力
2. **上下文理解**: LLM 可以根據前後文糾正 OCR 錯誤
3. **字形相似性**: LLM 可以區分形似字（如"刊"vs"列"）

**預期改善領域**：
- 形似字錯誤（刊→列、叢→議）
- 上下文不合理的字（防→所）
- 標點符號處理

---

## 🎯 GFD 關鍵參數調整

### fusing_r（融合比例）

```yaml
fusing_r: 0.0   # 完全使用 OCR（baseline）
fusing_r: 0.3   # 30% LLM, 70% OCR（推薦起點）
fusing_r: 0.5   # 50%-50% 平衡
fusing_r: 0.7   # 更依賴 LLM
```

**調整建議**：
- OCR 很準確（如合成資料）→ 使用較低 fusing_r (0.2-0.3)
- OCR 不準確（如實際掃描圖）→ 使用較高 fusing_r (0.4-0.6)
- 需要實驗找出最佳值

### num_beams（搜索寬度）

```bash
num_beams=1   # 最快，但準確度最低
num_beams=3   # 平衡
num_beams=5   # 推薦
num_beams=10  # 最準確，但最慢
```

### llm_temp（LLM 溫度）

```yaml
llm_temp: 1.0  # 保守（更依賴訓練數據）
llm_temp: 1.5  # 平衡
llm_temp: 1.7  # 當前設定
llm_temp: 2.0  # 創造性（更多樣化）
```

---

## 📁 文件結構

```
experiment/
├── generative-fusion-decoding-OCR/
│   ├── gfd/
│   │   ├── tokenizer_chinese_adapter.py  [新增] 兼容層
│   │   ├── gfd_chinese.py                [新增] 中文 GFD
│   │   ├── gfd.py                        [原始] 英文 GFD
│   │   ├── model.py                      LLM 模型
│   │   └── beam.py                       Beam search
│   └── config_files/
│       ├── model/
│       │   └── gfd-ocr-multilmdb-zhtw.yaml  [新增]
│       └── prompt/
│           └── ocr-zhtw-prompt.yaml         [新增]
├── test_gfd_inference_example.py         [新增] 測試腳本
├── test_gfd_compatibility.py             [新增] 兼容性測試
└── output_multi_lmdb/                    你的 TrOCR 模型
```

---

## 🔧 故障排除

### 問題 1: CUDA Out of Memory

**解決方案**：
```yaml
# 降低模型精度
ocr_torch_dtype: 'float16'
llm_torch_dtype: 'float16'

# 或使用 CPU
llm_device: 'cpu'
```

### 問題 2: LLM 下載慢

**解決方案**：
```bash
# 預先下載模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("MediaTek-Research/Breeze-7B-32k-Base-v1_0")
```

### 問題 3: Tokenizer 兼容性錯誤

**檢查**：
```python
# 運行兼容性測試
python test_gfd_compatibility.py
```

---

## 🚀 進階優化

### 1. 使用更輕量的 LLM

如果 Breeze-7B 太大：
```yaml
llm_model_path: 'ckip-joint/bloom-1b1-zh'  # 更小的中文 LLM
```

### 2. 多階段融合

```python
# 在不同長度使用不同融合比例
if len(sequence) < 5:
    fusing_r = 0.2  # 短序列更依賴 OCR
else:
    fusing_r = 0.4  # 長序列更依賴 LLM 的上下文理解
```

### 3. 自適應融合

根據 OCR 的 confidence 動態調整：
```python
if ocr_confidence < 0.8:
    fusing_r = 0.5  # OCR 不確定時更依賴 LLM
else:
    fusing_r = 0.2  # OCR 確定時保持原結果
```

---

## 📈 實驗建議

### 對比實驗

1. **Baseline**: 純 OCR（`--no_llm`）
2. **GFD-0.2**: `fusing_r=0.2`
3. **GFD-0.3**: `fusing_r=0.3`（當前設定）
4. **GFD-0.5**: `fusing_r=0.5`
5. **GFD-0.7**: `fusing_r=0.7`

### 測試數據集

- **inference_example**: 實際掃描文件（困難）
- **合成驗證集**: 你的 LMDB 驗證集（簡單）
- **混合集**: 包含各種難度的樣本

### 評估指標

- **CER (Character Error Rate)**: 字符錯誤率
- **完全匹配率**: 整句完全正確的比例
- **形似字準確率**: 特定關注形似字的辨識
- **推理速度**: Samples/sec

---

## ✅ 當前狀態

- ✅ Tokenizer 兼容層創建完成
- ✅ 中文 GFD 核心代碼完成
- ✅ 配置文件設定完成
- ✅ 測試腳本準備就緒
- 🔄 正在運行初始測試...

---

## 📞 使用方法總結

**快速開始**：
```bash
# 測試 2 個樣本（快速驗證）
python test_gfd_inference_example.py --max_samples 2 --num_beams 3

# 完整測試
python test_gfd_inference_example.py --max_samples 42 --num_beams 5

# Baseline 對比
python test_gfd_inference_example.py --max_samples 42 --no_llm
```

**查看結果**：
```bash
cat gfd_results.json | jq '.summary'
```

---

**創建時間**: 2025-11-05
**狀態**: ✅ 實現完成，測試進行中
**預期完成時間**: 取決於 LLM 下載和推理速度

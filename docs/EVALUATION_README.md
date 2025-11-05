# 完整 OCR 評估流程說明

## 📋 概述

完整的評估系統，整合 **Pure OCR**（無LLM）和 **GFD with LLM** 的推理結果，自動生成 CSV 比較報告。

## 🗂️ 測試資料

### 1. inference_example (42個文字區域)
- **來源**: 實際掃描文件
- **格式**: 1張大圖 + JSON 標註
- **路徑**: `inference_example/rec_1_id_2226.jpg`

### 2. test_samples_40 (40個單字樣本)
- **來源**: TCSynth-VAL 驗證集隨機抽取
- **格式**: 40張單字圖片 + labels.json
- **路徑**: `test_samples_40/`

**總計**: 82個測試樣本

## 🚀 使用方式

### 基本用法

```bash
# 使用預設參數運行完整評估
python run_complete_evaluation.py
```

### 進階參數

```bash
# 自訂 beam search 寬度和輸出目錄
python run_complete_evaluation.py \
  --num_beams 5 \
  --output_dir my_evaluation_results
```

### 參數說明

- `--num_beams`: Beam search 寬度（預設: 5）
  - 1: 最快，但準確度較低
  - 3: 平衡
  - 5: 推薦（預設）
  - 10: 最準確，但較慢

- `--output_dir`: 輸出目錄（預設: `results_YYYYMMDD_HHMMSS`）

## 📊 輸出結構

```
results_YYYYMMDD_HHMMSS/
├── raw_results/
│   ├── pure_ocr_results.json       # Pure OCR 原始結果
│   └── gfd_llm_results.json        # GFD with LLM 原始結果
├── comparison.csv                   # 逐筆比較結果 ⭐
├── summary.csv                      # 統計摘要 ⭐
└── analysis_report.md               # 詳細分析報告
```

### comparison.csv 欄位說明

| 欄位 | 說明 |
|------|------|
| `source` | 資料來源 (inference_example / test_samples_40) |
| `image_name` | 圖片檔名 |
| `region_id` | 區域ID |
| `bbox` | Bounding box 座標 |
| `ground_truth` | 原始 Ground Truth |
| `ground_truth_no_punct` | 移除標點符號後的 GT |
| `pure_ocr_pred` | Pure OCR 預測結果 |
| `pure_ocr_pred_no_punct` | Pure OCR 預測（無標點） |
| `pure_ocr_cer` | Pure OCR 的 CER |
| `gfd_llm_pred` | GFD with LLM 預測結果 |
| `gfd_llm_pred_no_punct` | GFD with LLM 預測（無標點） |
| `gfd_llm_cer` | GFD with LLM 的 CER |
| `cer_improvement` | CER 改善值 (pure - gfd) |
| `match_pure_ocr` | Pure OCR 是否完全匹配 |
| `match_gfd_llm` | GFD with LLM 是否完全匹配 |

### summary.csv 欄位說明

| 欄位 | 說明 |
|------|------|
| `mode` | 模式 (Pure OCR / GFD with LLM / Improvement) |
| `total_samples` | 總樣本數 |
| `average_cer` | 平均 CER |
| `perfect_matches` | 完全匹配數量 |
| `match_rate` | 匹配率 |

## 🔧 執行流程

腳本會自動執行以下階段：

### 階段 1: 載入測試資料
- 載入 inference_example (42個區域)
- 載入 test_samples_40 (40個樣本)
- **總計**: 82個測試樣本

### 階段 2: Pure OCR 推理
- 使用 TrOCR 模型（無LLM）
- `fusing_r = 0.0`
- 計算 CER（移除標點符號）

### 階段 3: GFD with LLM 推理
- 使用 GFD + Breeze-7B LLM
- `fusing_r = 0.3` (30% LLM, 70% OCR)
- 計算 CER（移除標點符號）

### 階段 4: 生成比較 CSV
- 整合兩種模式的結果
- 計算 CER 改善值
- 標記完全匹配

### 階段 5: 生成統計摘要
- 計算平均 CER
- 計算完全匹配率
- 比較兩種模式的改善

### 階段 6: 生成分析報告
- Top 10 改善案例
- Top 10 惡化案例
- 詳細統計分析

## ⏱️ 預估執行時間

- **Pure OCR**: ~10-15分鐘（82個樣本）
- **GFD with LLM**: ~30-40分鐘（82個樣本，需載入LLM）
- **總計**: ~45-60分鐘

## 📈 預期效果

### 標點符號移除的影響
- 移除標點符號後，CER 應該降低 10-25%
- 更公平地評估文字內容識別能力

### GFD 改善
- 理論上，GFD with LLM 應該在以下方面改善：
  - 形似字錯誤（刊→列、叢→議）
  - 上下文不合理的字（防→所）
  - 整體 CER 降低 5-15%

## 🔍 結果分析建議

### 1. 查看 summary.csv
快速了解整體表現：
```bash
cat results_*/summary.csv
```

### 2. 查看 comparison.csv
詳細分析每個樣本：
```bash
# 找出改善最多的案例
sort -t',' -k13 -rn results_*/comparison.csv | head -20

# 找出惡化的案例
sort -t',' -k13 -n results_*/comparison.csv | head -20
```

### 3. 查看 analysis_report.md
了解改善和惡化的具體案例

## 🛠️ 故障排除

### 問題 1: CUDA Out of Memory

**解決方案**:
- 降低 num_beams (例如 --num_beams 3)
- 修改配置文件使用 CPU (llm_device: 'cpu')

### 問題 2: 找不到測試資料

**檢查**:
```bash
# 確認 inference_example 存在
ls -l inference_example/

# 確認 test_samples_40 存在
ls -l test_samples_40/
```

如果 test_samples_40 不存在，重新提取：
```bash
python extract_test_samples.py
```

### 問題 3: 模型載入失敗

**檢查**:
```bash
# 確認模型路徑
ls -l output_multi_lmdb/

# 確認 tokenizer 路徑
ls -l tokenizer/
```

## 📁 相關檔案

```
experiment/
├── run_complete_evaluation.py      [主腳本] 完整評估流程
├── extract_test_samples.py        [工具] 從驗證集提取樣本
├── test_gfd_inference_example.py  [舊版] 單獨測試 GFD
├── inference_multi_lmdb.py        [舊版] 單獨測試 TrOCR
├── test_punctuation_removal.py    [工具] 測試標點符號移除
├── EVALUATION_README.md           [本文件] 使用說明
├── PUNCTUATION_REMOVAL_README.md  [參考] 標點符號移除說明
├── GFD_CHINESE_OCR_IMPLEMENTATION.md [參考] GFD 實現文檔
├── inference_example/             [測試資料] 42個區域
└── test_samples_40/               [測試資料] 40個樣本
```

## ✅ 完整流程檢查清單

- [x] 清理舊結果檔案
- [x] 提取40個驗證集樣本
- [x] 創建完整評估腳本
- [x] 腳本語法檢查通過
- [ ] 運行完整評估
- [ ] 檢查結果檔案
- [ ] 分析評估結果

## 🎯 下一步

運行完整評估：

```bash
# 開始評估（預計45-60分鐘）
python run_complete_evaluation.py --num_beams 5

# 或者使用 nohup 在背景執行
nohup python run_complete_evaluation.py --num_beams 5 > evaluation.log 2>&1 &

# 監控進度
tail -f evaluation.log
```

---

**建立時間**: 2025-11-06
**狀態**: ✅ 準備就緒，等待執行

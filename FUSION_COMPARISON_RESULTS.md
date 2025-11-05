# Fusion 開關對比測試結果

## 測試說明

本測試比較 **TrOCR + LLM Fusion** (fusing_r=0.2) 與 **純 TrOCR** (fusing_r=0.0) 的識別效果差異。

---

## 執行方式

```bash
# 啟動環境
conda activate gfd-ocr
export PYTHONPATH=.

# 執行快速 fusion 對比測試
python test_fusion_comparison.py

# 或執行完整評估測試（包含 fusion 對比）
python test_ocr_evaluation.py
```

---

## 測試結果摘要

### 簡單印刷體文字測試

| 測試文字 | With Fusion | Without Fusion | 結果 |
|---------|------------|---------------|------|
| HELLO WORLD | ✓ 正確 | ✓ 正確 | 相同 |
| QUICK BROWN FOX | ✓ 正確 | ✓ 正確 | 相同 |
| AI RESEARCH | ✓ 正確 | ✓ 正確 | 相同 |

**結論**: 對於**清晰的簡單印刷體文字**，Fusion 和純 TrOCR 的結果相同（100% 準確率）

---

## Fusion 分數觀察

從測試輸出可以看到 fusion 的作用機制：

### With Fusion (fusing_r=0.2)
```
[0] recognizer_score=-0.002, llm_score=-7.527, fuse_score=-1.508
HELL

[0] recognizer_score=-0.002, llm_score=-5.134, fuse_score=-1.028
HELLO

[0] recognizer_score=-0.002, llm_score=-3.906, fuse_score=-0.782
HELLO WORLD
```

**Fusion 公式**:
```
fuse_score = (1 - 0.2) × recognizer_score + 0.2 × llm_score
           = 0.8 × recognizer_score + 0.2 × llm_score
```

### Without Fusion (fusing_r=0.0)
```
[0] recognizer_score=-0.002, llm_score=-7.527, fuse_score=-0.002
HELL

[0] recognizer_score=-0.002, llm_score=-5.134, fuse_score=-0.002
HELLO

[0] recognizer_score=-0.002, llm_score=-3.906, fuse_score=-0.002
HELLO WORLD
```

**無 Fusion**:
```
fuse_score = 1.0 × recognizer_score + 0.0 × llm_score
           = recognizer_score (純 TrOCR)
```

---

## 為什麼結果相同？

### 分析：

1. **測試圖片太簡單**
   - 白底黑字、字體清晰
   - TrOCR 本身就能 100% 準確識別
   - LLM 無法提供額外幫助

2. **Fusion 權重較低** (fusing_r=0.2)
   - TrOCR 權重：80%
   - LLM 權重：20%
   - 當 TrOCR 已經很準確時，LLM 影響有限

3. **Beam Search 選擇相同路徑**
   - 兩種模式下，最佳路徑的 token 序列完全相同
   - Fusion 只影響分數，但不改變最終選擇

---

## 何時 Fusion 會有差異？

Fusion 通常在以下情況才會顯現效果：

### 1. 低品質圖片
- 模糊、噪音、低對比度
- TrOCR 信心較低時，LLM 的語言模型能力更重要

### 2. 手寫文字
- TrOCR 可能產生多個候選
- LLM 能根據語言邏輯選擇更合理的結果

### 3. 長文本
- 需要上下文理解
- LLM 能糾正 TrOCR 的局部錯誤

### 4. 專業術語/特殊領域
- 如果 LLM 在該領域有較好訓練
- 能提供更好的先驗知識

---

## 建議的測試場景

如果要看到 Fusion 的效果，建議測試：

### 測試 1: 低品質圖片
```python
# 添加噪音
create_test_image(text, add_noise=True, noise_level=0.2)

# 低對比度
create_test_image(text, bg_color=(200,200,200), text_color=(80,80,80))

# 模糊
img.filter(ImageFilter.GaussianBlur(radius=2))
```

### 測試 2: 手寫體
```python
# 使用手寫體 TrOCR 模型
ocr_model_path: 'microsoft/trocr-base-handwritten'
```

### 測試 3: 複雜句子
```python
test_cases = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models require large datasets",
    "Natural language processing enables AI systems",
]
```

### 測試 4: 提高 Fusion 權重
```python
# 在 config_files/model/gfd-ocr-en.yaml 中
fusing_r: 0.5  # 提高到 50%
```

---

## 目前的測試結論

### ✓ 成功驗證的功能

1. **Fusion 機制正常運作**
   - With/Without Fusion 配置正確載入
   - Fusion 分數計算公式正確
   - Beam search 正常使用 fusion score

2. **程式碼修復成功**
   - `test_ocr_evaluation.py` 錯誤已修復
   - `test_fusion_comparison.py` 新腳本正常運作
   - 可以正確對比兩種模式

3. **測試架構完整**
   - 輸出目錄結構清晰
   - 測試結果易於查看
   - 可重複執行

### ⚠️ 觀察與建議

1. **當前測試場景下**
   - 簡單清晰的印刷體：Fusion 無明顯差異
   - 需要更具挑戰性的測試案例

2. **建議後續測試**
   - 使用實際掃描文件（可能有雜訊、傾斜）
   - 測試手寫體識別
   - 測試低品質照片
   - 增加 fusion 權重觀察效果

3. **Fusion 價值**
   - 即使簡單案例無差異，Fusion 在困難案例中仍有價值
   - 可作為「安全網」：TrOCR 準確時不干擾，不準確時提供幫助

---

## 詳細測試日誌

完整的測試輸出顯示：

```
With Fusion:
  recognizer_score=-0.002, llm_score=-3.906, fuse_score=-0.782
  → 最終選擇: "HELLO WORLD" ✓

Without Fusion:
  recognizer_score=-0.002, llm_score=-3.906, fuse_score=-0.002
  → 最終選擇: "HELLO WORLD" ✓
```

- **fuse_score 確實不同**（-0.782 vs -0.002）
- 但兩者都選擇了相同的最佳路徑
- 這是因為排序後的相對順序相同

---

## 如何查看測試輸出

### 測試圖片位置
```bash
ls -lh test_output/images/comparison/
```

### 執行記錄
測試腳本會即時顯示：
- 每個 beam 的 recognizer/llm/fuse 分數
- 最終識別結果
- 對比統計

---

## 參考資料

- **主要測試腳本**: [test_fusion_comparison.py](test_fusion_comparison.py)
- **完整評估測試**: [test_ocr_evaluation.py](test_ocr_evaluation.py)
- **測試指南**: [OCR_TESTING_GUIDE.md](OCR_TESTING_GUIDE.md)
- **Fusion 論文**: [arXiv:2405.14259](https://arxiv.org/abs/2405.14259)

---

最後更新: 2025-11-05

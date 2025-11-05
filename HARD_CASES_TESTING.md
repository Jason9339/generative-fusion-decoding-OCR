# 困難場景 Fusion 測試指南

本指南說明如何使用困難場景來測試 TrOCR + LLM Fusion 的效果。

---

## 🎯 測試目的

透過更長、更困難的文字，觀察 Fusion 是否能在挑戰性場景中發揮作用：

1. **長句子** - 測試語言模型的上下文理解能力
2. **專業術語** - 測試 LLM 的詞彙知識
3. **噪音干擾** - 測試在低品質圖片下的魯棒性
4. **低對比度** - 測試在視覺困難條件下的表現
5. **模糊圖片** - 測試圖片退化的處理能力
6. **混合條件** - 測試多重困難因素的綜合處理

---

## 📋 測試腳本

### 1. 快速測試（推薦開始使用）

```bash
# 啟動環境
conda activate gfd-ocr
export PYTHONPATH=.

# 執行快速測試（5 個代表性案例）
python test_fusion_quick_hard.py
```

**特點**：
- ✅ 快速（約 5-10 分鐘）
- ✅ 代表性案例
- ✅ 清晰的對比輸出
- ✅ fusing_r = 0.3（較高權重）

**測試內容**：
1. 長句子："The quick brown fox jumps over the lazy dog"
2. 專業術語："Convolutional Neural Networks for Image Classification"
3. 噪音干擾：noise_level=0.15
4. 低對比度：灰色背景 + 深灰文字
5. 模糊：blur_radius=1.5

### 2. 完整測試（深入分析）

```bash
# 執行完整測試（6 個測試場景，共 18+ 案例）
python test_fusion_hard_cases.py
```

**特點**：
- 🔬 詳細（約 30-60 分鐘）
- 📊 完整統計分析
- 📈 多種困難條件
- 💡 深入對比

**測試場景**：
- 測試 1：長句子（4 個案例）
- 測試 2：專業術語（4 個案例）
- 測試 3：噪音干擾（3 個案例）
- 測試 4：低對比度（3 個案例）
- 測試 5：模糊圖片（2 個案例）
- 測試 6：混合困難條件（2 個案例）

---

## 📊 輸出解讀

### 測試過程輸出

```
================================================================================
測試 1/5: 長句子
================================================================================
文字: 'The quick brown fox jumps over the lazy dog'
圖片: test_output/images/quick_hard/test_1_長句子.png

[With Fusion (r=0.3)]
ocr prompt:
llm prompt: The following is a transcription of the text contained in the image:
[0] recognizer_score=-0.002, llm_score=-5.134, fuse_score=-1.541
...

[Without Fusion (r=0.0)]
ocr prompt:
llm prompt: The following is a transcription of the text contained in the image:
[0] recognizer_score=-0.002, llm_score=-5.134, fuse_score=-0.002
...

────────────────────────────────────────────────────────────────────────────────
結果:
  原文:    'The quick brown fox jumps over the lazy dog'
  With:    'The quick brown fox jumps over the lazy dog' ✓
  Without: 'The quick brown fox jumps over the lazy dog' ✓
  → 結果相同
────────────────────────────────────────────────────────────────────────────────
```

### 總結輸出

```
================================================================================
測試總結
================================================================================

總測試數: 5
With Fusion 正確: 4/5 (80.0%)
Without Fusion 正確: 3/5 (60.0%)
結果不同: 2/5 (40.0%)

詳細:
  長句子         : With ✓ vs Without ✓ =
  專業術語       : With ✓ vs Without ✗ ≠  💡 Fusion 修正了錯誤！
  噪音干擾       : With ✓ vs Without ✗ ≠  💡 Fusion 修正了錯誤！
  低對比度       : With ✗ vs Without ✗ =
  模糊           : With ✓ vs Without ✓ =

================================================================================
✅ Fusion 提升準確率 20.0%
💡 Fusion 在 2 個案例中產生不同結果
================================================================================
```

**符號說明**：
- ✓ = 識別正確
- ✗ = 識別錯誤
- = = 結果相同
- ≠ = 結果不同
- 💡 = Fusion 修正錯誤
- ⚠️ = Fusion 產生錯誤

---

## 🔧 調整測試參數

### 修改 Fusion 權重

編輯腳本中的 `fusing_r_with` 參數：

```python
# test_fusion_quick_hard.py 第 52 行
override_with = argparse.Namespace(fusing_r=0.3)  # 改為 0.5 測試更高權重
```

或直接在配置檔案修改：
```yaml
# config_files/model/gfd-ocr-en.yaml
fusing_r: 0.5
```

### 調整困難程度

在 `create_test_image()` 中修改參數：

```python
# 增加噪音
'noise_level': 0.2  # 原本 0.15

# 增加模糊
'blur_radius': 2.0  # 原本 1.5

# 更低對比度
bg_color = (220, 220, 220)  # 更淺的背景
text_color = (60, 60, 60)   # 更深的文字
```

### 添加自己的測試案例

在 `test_cases` 列表中添加：

```python
test_cases = [
    # 原有案例...
    {
        'name': '我的測試',
        'text': 'Your custom text here',
        'params': {
            'add_noise': True,
            'noise_level': 0.2,
            'blur': True,
            'blur_radius': 1.5
        }
    },
]
```

---

## 🎨 測試案例範例

### 範例 1: 長句子

```python
{
    'text': 'The quick brown fox jumps over the lazy dog',
    'params': {'size': (900, 150), 'font_size': 28}
}
```

**目的**：測試 LLM 的語言模型能力，看是否能利用語境糾正 TrOCR 的錯誤。

### 範例 2: 專業術語

```python
{
    'text': 'Convolutional Neural Networks for Image Classification',
    'params': {'size': (950, 150), 'font_size': 26}
}
```

**目的**：測試 LLM 對技術術語的理解，如 "Convolutional"、"Neural Networks" 等。

### 範例 3: 噪音干擾

```python
{
    'text': 'This text has moderate noise interference',
    'params': {
        'size': (850, 150),
        'font_size': 28,
        'add_noise': True,
        'noise_level': 0.15
    }
}
```

**目的**：測試在低品質圖片下，LLM 是否能幫助 TrOCR 做出更好的判斷。

### 範例 4: 低對比度

```python
{
    'text': 'Low contrast makes text harder to read',
    'params': {
        'size': (850, 150),
        'font_size': 28,
        'low_contrast': True
    }
}
```

**目的**：測試在視覺困難條件下的魯棒性。

### 範例 5: 模糊

```python
{
    'text': 'Blurred images challenge OCR systems',
    'params': {
        'size': (850, 150),
        'font_size': 28,
        'blur': True,
        'blur_radius': 1.5
    }
}
```

**目的**：測試圖片模糊時的處理能力。

### 範例 6: 混合困難

```python
{
    'text': 'Multiple degradation factors affect accuracy',
    'params': {
        'size': (950, 150),
        'font_size': 26,
        'add_noise': True,
        'noise_level': 0.12,
        'blur': True,
        'blur_radius': 1.0,
    }
}
```

**目的**：測試多重困難因素下的綜合表現。

---

## 📈 預期結果

### 簡單場景（之前的測試）

- With Fusion: 100% 正確
- Without Fusion: 100% 正確
- **結論**：無差異（圖片太簡單）

### 困難場景（本測試）

**可能的結果**：

#### 情況 1：Fusion 有正向效果
```
With Fusion: 85% 正確
Without Fusion: 70% 正確
結論: ✅ Fusion 提升 15%
```

#### 情況 2：Fusion 無明顯效果
```
With Fusion: 75% 正確
Without Fusion: 73% 正確
結論: → Fusion 輕微提升 2%
```

#### 情況 3：Fusion 有負面效果
```
With Fusion: 70% 正確
Without Fusion: 80% 正確
結論: ⚠️ Fusion 降低 10%（可能 LLM 引入錯誤）
```

---

## 💡 分析建議

### 如果 Fusion 無效果

可能原因：
1. **fusing_r 太低** → 提高到 0.5 或更高
2. **LLM 模型太小** → 使用更大的 LLM
3. **測試仍不夠困難** → 增加噪音、模糊程度

### 如果 Fusion 有負面效果

可能原因：
1. **LLM 過度自信** → 降低 fusing_r
2. **LLM 對該領域不熟** → 換用該領域訓練的 LLM
3. **Prompt 不適當** → 調整 llm_prompt

### 如果 Fusion 有正向效果

**恭喜！** 這證明：
- ✅ Fusion 機制有效
- ✅ LLM 能提供有價值的語言知識
- ✅ 系統在困難場景下更魯棒

---

## 🔍 進階測試建議

### 1. 測試不同 LLM

```yaml
# 小型 LLM（快速）
llm_model_path: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# 中型 LLM（平衡）
llm_model_path: 'meta-llama/Llama-2-7b-hf'

# 大型 LLM（準確）
llm_model_path: 'meta-llama/Llama-2-13b-hf'
```

### 2. 測試不同 fusing_r

```python
fusing_r_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
# 繪製準確率 vs fusing_r 曲線
```

### 3. 測試真實文件

```python
# 使用實際掃描的文件、發票、收據等
test_cases = [
    {'text': '...', 'image_file': 'scanned_invoice.png'},
    {'text': '...', 'image_file': 'receipt.jpg'},
]
```

---

## 📂 輸出位置

### 快速測試
- 圖片：`test_output/images/quick_hard/`
- 命名：`test_1_長句子.png`, `test_2_專業術語.png` 等

### 完整測試
- 圖片：`test_output/images/hard_cases/`
- 命名：`測試_1_長句子_1.png`, `測試_2_專業術語_1.png` 等

### 查看圖片

```bash
# 列出所有測試圖片
ls -lh test_output/images/quick_hard/
ls -lh test_output/images/hard_cases/

# 用圖片查看器打開
xdg-open test_output/images/quick_hard/test_1_長句子.png
```

---

## 🎯 快速開始

```bash
# 1. 啟動環境
conda activate gfd-ocr
export PYTHONPATH=.

# 2. 執行快速測試（推薦）
python test_fusion_quick_hard.py

# 3. 查看結果
# 輸出會即時顯示在終端
# 圖片保存在 test_output/images/quick_hard/
```

**預計時間**：5-10 分鐘

---

## 📚 相關文件

- **快速測試腳本**: [test_fusion_quick_hard.py](test_fusion_quick_hard.py)
- **完整測試腳本**: [test_fusion_hard_cases.py](test_fusion_hard_cases.py)
- **Fusion 對比結果**: [FUSION_COMPARISON_RESULTS.md](FUSION_COMPARISON_RESULTS.md)
- **測試指南**: [OCR_TESTING_GUIDE.md](OCR_TESTING_GUIDE.md)

---

最後更新: 2025-11-05

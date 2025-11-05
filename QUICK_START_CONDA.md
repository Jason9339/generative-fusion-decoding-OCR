# 🚀 快速開始 - Conda 版本

最簡單的方式開始使用 TrOCR + LLM Fusion OCR 系統！

---

## 📋 前置需求

- **Anaconda** 或 **Miniconda** 已安裝
  - 下載 Anaconda: https://www.anaconda.com/download
  - 下載 Miniconda: https://docs.conda.io/en/latest/miniconda.html

---

## ⚡ 方法 1: 一鍵自動安裝（推薦）

```bash
# 1. 進入專案目錄
cd /home/vipl/generative-fusion-decoding-OCR

# 2. 執行自動安裝腳本
./setup_conda_env.sh
```

腳本會自動完成：
- ✅ 創建 conda 環境 (gfd-ocr)
- ✅ 安裝 PyTorch (CPU/GPU 可選)
- ✅ 安裝所有依賴套件
- ✅ 安裝專案
- ✅ 驗證安裝
- ✅ 設定環境變數

**預計時間**: 5-10 分鐘（視網速而定）

---

## 🔧 方法 2: 手動安裝

### 步驟 1: 創建環境

```bash
conda create -n gfd-ocr python=3.9 -y
conda activate gfd-ocr
```

### 步驟 2: 安裝 PyTorch

**CPU 版本**:
```bash
conda install pytorch==2.2.1 cpuonly -c pytorch -y
```

**GPU 版本（如有 NVIDIA GPU）**:
```bash
conda install pytorch==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 步驟 3: 安裝依賴

```bash
# Conda 套件
conda install -y numpy scipy pandas pillow pyyaml requests

# Pip 套件
pip install transformers==4.40.1 accelerate sentencepiece
```

### 步驟 4: 安裝專案

```bash
cd /home/vipl/generative-fusion-decoding-OCR
python setup.py install
```

---

## ✅ 驗證安裝

### 檢查套件

```bash
conda activate gfd-ocr

# 檢查 PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"

# 檢查 Transformers
python -c "import transformers; print('Transformers:', transformers.__version__)"

# 檢查專案
python -c "import gfd; print('GFD: OK')"
```

應該全部正常顯示版本號。

---

## 🧪 執行測試

### 測試 1: 基礎測試（必做）

```bash
conda activate gfd-ocr
cd /home/vipl/generative-fusion-decoding-OCR
export PYTHONPATH=.

python test_trocr_setup.py
```

**預期結果**: 看到 `✓ 所有測試通過！`

### 測試 2: Fusion 測試

```bash
python test_trocr_fusion.py
```

**預期結果**: 看到 `匹配率: 100.0%`

### 測試 3: 測試自己的圖片

```bash
python benchmarks/run_single_file.py \
  --model_name gfd \
  --setting ocr-en \
  --image_file_path /path/to/your/image.png \
  --result_output_path result.json

# 查看結果
cat result.json
```

---

## 📝 常用命令

### 啟動環境
```bash
conda activate gfd-ocr
```

### 退出環境
```bash
conda deactivate
```

### 列出所有環境
```bash
conda env list
```

### 更新套件
```bash
conda activate gfd-ocr
pip install --upgrade transformers
```

### 刪除環境（如需重建）
```bash
conda env remove -n gfd-ocr
```

---

## 🎯 快速使用範例

### Python 腳本範例

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
image = Image.open('your_image.png')

# 執行 OCR
result = model.get_transcription(image, num_beams=3)
print(f"識別結果: {result}")
```

### 命令行範例

```bash
# 基本使用
python benchmarks/run_single_file.py \
  --model_name gfd \
  --setting ocr-en \
  --image_file_path test.png \
  --result_output_path output.json
```

---

## 🔧 配置調整

### 使用 GPU

編輯 `config_files/model/gfd-ocr-en.yaml`:

```yaml
ocr_device: 'cuda:0'  # 改為 cuda
llm_device: 'cuda:0'
ocr_torch_dtype: 'float16'  # 改為 float16
llm_torch_dtype: 'float16'
```

### 調整 Fusion 比例

```yaml
fusing_r: 0.2  # 範圍 0.0-1.0
# 0.0 = 純 TrOCR，無 LLM
# 0.2 = 推薦值
# 1.0 = 純 LLM
```

### 更換 TrOCR 模型

```yaml
ocr_model_path: 'microsoft/trocr-base-handwritten'  # 手寫體
# 或
ocr_model_path: 'microsoft/trocr-large-printed'  # 大型印刷體
```

---

## ❓ 常見問題

### Q: conda: command not found

**A**: 安裝 Anaconda 或 Miniconda，然後執行：
```bash
~/anaconda3/bin/conda init bash
source ~/.bashrc
```

### Q: 測試時出現 "No module named 'gfd'"

**A**: 設定 PYTHONPATH：
```bash
export PYTHONPATH=/home/vipl/generative-fusion-decoding-OCR:$PYTHONPATH
```

或重新安裝專案：
```bash
python setup.py install
```

### Q: 識別效果不好

**A**: 檢查：
1. 圖片是否清晰？
2. 是否為英文印刷體？
3. 嘗試調整 `num_beams` (3-5)
4. 嘗試調整 `fusing_r` (0.1-0.3)

### Q: 速度太慢

**A**: 優化建議：
1. 使用 GPU（如有）
2. 降低 `num_beams` 至 3
3. 使用更小的 LLM 模型

---

## 📚 更多資源

- **詳細測試指南**: [OCR_TESTING_GUIDE.md](OCR_TESTING_GUIDE.md)
- **Conda 完整設定**: [CONDA_SETUP_GUIDE.md](CONDA_SETUP_GUIDE.md)
- **專案 README**: [README.md](README.md)

---

## 🎉 完成！

現在你已經完成設定，可以開始使用 TrOCR + LLM Fusion 進行 OCR 識別了！

**建議測試流程**:
1. ✅ 執行 `test_trocr_setup.py` 驗證環境
2. ✅ 執行 `test_trocr_fusion.py` 驗證 fusion
3. ✅ 使用自己的圖片測試實際效果

祝使用愉快！🚀

---

最後更新: 2025-11-04

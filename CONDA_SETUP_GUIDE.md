# TrOCR + LLM Fusion - Conda 環境設定指南

本指南說明如何使用 Conda 設定 TrOCR + LLM Fusion 的開發環境。

---

## 📋 目錄

1. [建立 Conda 環境](#建立-conda-環境)
2. [安裝依賴套件](#安裝依賴套件)
3. [驗證安裝](#驗證安裝)
4. [執行測試](#執行測試)
5. [常見問題](#常見問題)

---

## 🚀 建立 Conda 環境

### 方法 1: 使用指定的 Python 版本（推薦）

```bash
# 創建名為 gfd-ocr 的環境，使用 Python 3.9
conda create -n gfd-ocr python=3.9 -y

# 啟動環境
conda activate gfd-ocr
```

### 方法 2: 使用其他 Python 版本

```bash
# Python 3.8
conda create -n gfd-ocr python=3.8 -y

# Python 3.10
conda create -n gfd-ocr python=3.10 -y

# 啟動環境
conda activate gfd-ocr
```

### 驗證環境創建成功

```bash
# 檢查 Python 版本
python --version
# 應顯示: Python 3.9.x

# 檢查是否在正確的環境中
which python
# 應顯示: /path/to/anaconda3/envs/gfd-ocr/bin/python
```

---

## 📦 安裝依賴套件

### 步驟 1: 安裝 PyTorch（CPU 版本）

```bash
# 確保在 gfd-ocr 環境中
conda activate gfd-ocr

# 安裝 PyTorch CPU 版本
conda install pytorch==2.2.1 torchvision torchaudio cpuonly -c pytorch -y
```

**如果有 CUDA GPU，可以安裝 GPU 版本**:

```bash
# CUDA 11.8
conda install pytorch==2.2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CUDA 12.1
conda install pytorch==2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 步驟 2: 安裝其他必要套件

```bash
# 使用 conda 安裝基礎套件
conda install -y numpy scipy scikit-learn pandas pillow pyyaml requests

# 使用 pip 安裝其餘套件
pip install transformers==4.40.1 accelerate datasets sentencepiece
pip install jiwer opencc-python-reimplemented
```

### 步驟 3: 安裝專案

```bash
# 在專案根目錄下
cd /home/vipl/generative-fusion-decoding-OCR

# 安裝專案
python setup.py install
```

---

## ✅ 驗證安裝

### 檢查關鍵套件

```bash
# 啟動環境
conda activate gfd-ocr

# 檢查 PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"
# 應顯示: PyTorch: 2.2.1

# 檢查 Transformers
python -c "import transformers; print('Transformers:', transformers.__version__)"
# 應顯示: Transformers: 4.40.1

# 檢查 Pillow
python -c "import PIL; print('Pillow:', PIL.__version__)"
# 應顯示: Pillow: 10.3.0 或類似版本

# 檢查 CUDA 是否可用（如果安裝了 GPU 版本）
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 完整套件檢查

```bash
# 列出所有已安裝的套件
conda list

# 檢查特定套件
conda list | grep -E "(torch|transformers|pillow)"
```

---

## 🧪 執行測試

### 測試 1: 基礎環境測試

```bash
# 確保在專案根目錄
cd /home/vipl/generative-fusion-decoding-OCR

# 啟動環境
conda activate gfd-ocr

# 設定 PYTHONPATH
export PYTHONPATH=.

# 執行基礎測試
python test_trocr_setup.py
```

**預期輸出**:
```
================================================================================
測試 1: TrOCR 模型載入
================================================================================
載入模型: microsoft/trocr-base-printed
✓ 模型載入成功
...
✓ 所有測試通過！
```

### 測試 2: Fusion 測試

```bash
# 在 gfd-ocr 環境中
conda activate gfd-ocr
export PYTHONPATH=.

# 執行 fusion 測試
python test_trocr_fusion.py
```

**預期輸出**:
```
測試總結
總測試數: 4
成功匹配: 4
匹配率: 100.0%
✓ 所有測試完成！
```

### 測試 3: 完整評估

```bash
conda activate gfd-ocr
export PYTHONPATH=.

# 執行完整評估（需要較長時間）
python test_ocr_evaluation.py
```

---

## 🔧 Conda 環境管理

### 常用命令

```bash
# 列出所有 conda 環境
conda env list

# 啟動環境
conda activate gfd-ocr

# 退出環境
conda deactivate

# 刪除環境（如需重建）
conda env remove -n gfd-ocr

# 匯出環境配置
conda env export > environment.yml

# 從配置檔案建立環境
conda env create -f environment.yml
```

### 更新套件

```bash
# 啟動環境
conda activate gfd-ocr

# 更新特定套件
conda update numpy
pip install --upgrade transformers

# 更新所有 conda 套件
conda update --all
```

---

## 📝 完整安裝腳本

如果你想要一鍵安裝，可以創建一個腳本：

### `setup_conda_env.sh`

```bash
#!/bin/bash

# TrOCR + LLM Fusion - Conda 環境設定腳本

set -e  # 遇到錯誤立即停止

echo "======================================"
echo "TrOCR + LLM Fusion 環境設定"
echo "======================================"

# 1. 創建環境
echo ""
echo "步驟 1/5: 創建 Conda 環境..."
conda create -n gfd-ocr python=3.9 -y

# 2. 啟動環境
echo ""
echo "步驟 2/5: 啟動環境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gfd-ocr

# 3. 安裝 PyTorch
echo ""
echo "步驟 3/5: 安裝 PyTorch (CPU 版本)..."
conda install pytorch==2.2.1 torchvision torchaudio cpuonly -c pytorch -y

# 4. 安裝其他依賴
echo ""
echo "步驟 4/5: 安裝依賴套件..."
conda install -y numpy scipy scikit-learn pandas pillow pyyaml requests
pip install transformers==4.40.1 accelerate datasets sentencepiece
pip install jiwer opencc-python-reimplemented

# 5. 安裝專案
echo ""
echo "步驟 5/5: 安裝專案..."
python setup.py install

# 驗證安裝
echo ""
echo "======================================"
echo "驗證安裝..."
echo "======================================"
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
python -c "import PIL; print('✓ Pillow:', PIL.__version__)"

echo ""
echo "======================================"
echo "✓ 環境設定完成！"
echo "======================================"
echo ""
echo "使用以下命令啟動環境："
echo "  conda activate gfd-ocr"
echo ""
echo "執行測試："
echo "  export PYTHONPATH=."
echo "  python test_trocr_setup.py"
echo ""
```

**使用方式**:

```bash
# 賦予執行權限
chmod +x setup_conda_env.sh

# 執行腳本
./setup_conda_env.sh
```

---

## 🐛 常見問題

### Q1: conda: command not found

**原因**: Conda 未安裝或未加入 PATH

**解決方案**:

```bash
# 檢查 conda 是否安裝
which conda

# 如果找不到，初始化 conda
~/anaconda3/bin/conda init bash
# 或
~/miniconda3/bin/conda init bash

# 重新載入 shell
source ~/.bashrc
```

### Q2: PackagesNotFoundError: The following packages are not available

**解決方案**: 使用 pip 作為替代

```bash
conda activate gfd-ocr
pip install <package-name>
```

### Q3: 環境啟動後 python 版本不對

**解決方案**: 確保完全啟動環境

```bash
# 退出當前環境
conda deactivate

# 重新啟動
conda activate gfd-ocr

# 驗證
which python
python --version
```

### Q4: PyTorch 安裝失敗或版本不對

**解決方案**: 手動指定來源

```bash
# 移除現有的 PyTorch
pip uninstall torch torchvision torchaudio -y

# 重新安裝
conda install pytorch==2.2.1 torchvision torchaudio cpuonly -c pytorch -y
```

### Q5: ImportError: No module named 'gfd'

**解決方案**: 確保設定 PYTHONPATH 或安裝專案

```bash
# 方法 1: 設定 PYTHONPATH
export PYTHONPATH=/home/vipl/generative-fusion-decoding-OCR:$PYTHONPATH

# 方法 2: 安裝專案（推薦）
cd /home/vipl/generative-fusion-decoding-OCR
python setup.py develop  # 開發模式
# 或
python setup.py install  # 正式安裝
```

### Q6: CUDA out of memory（如使用 GPU）

**解決方案**: 修改配置使用 CPU

編輯 `config_files/model/gfd-ocr-en.yaml`:

```yaml
ocr_device: 'cpu'
llm_device: 'cpu'
ocr_torch_dtype: 'float32'
llm_torch_dtype: 'float32'
```

---

## 📊 環境配置建議

### 最小配置（CPU）

```bash
# 環境
conda create -n gfd-ocr python=3.9 -y
conda activate gfd-ocr

# PyTorch CPU
conda install pytorch==2.2.1 cpuonly -c pytorch -y

# 基礎套件
pip install transformers pillow pyyaml
python setup.py install
```

### 推薦配置（完整）

```bash
# 環境
conda create -n gfd-ocr python=3.9 -y
conda activate gfd-ocr

# PyTorch
conda install pytorch==2.2.1 torchvision torchaudio cpuonly -c pytorch -y

# 完整依賴
conda install -y numpy scipy scikit-learn pandas pillow pyyaml requests
pip install transformers==4.40.1 accelerate datasets sentencepiece
pip install jiwer opencc-python-reimplemented

# 專案
python setup.py install
```

### GPU 配置（如有 NVIDIA GPU）

```bash
# 環境
conda create -n gfd-ocr python=3.9 -y
conda activate gfd-ocr

# PyTorch GPU (CUDA 11.8)
conda install pytorch==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 完整依賴（同上）
conda install -y numpy scipy scikit-learn pandas pillow pyyaml requests
pip install transformers==4.40.1 accelerate datasets sentencepiece
pip install jiwer opencc-python-reimplemented

# 專案
python setup.py install
```

**修改配置使用 GPU**:

編輯 `config_files/model/gfd-ocr-en.yaml`:

```yaml
ocr_device: 'cuda:0'
llm_device: 'cuda:0'
ocr_torch_dtype: 'float16'
llm_torch_dtype: 'float16'
```

---

## 🎯 快速開始（完整流程）

```bash
# 1. 創建並啟動環境
conda create -n gfd-ocr python=3.9 -y
conda activate gfd-ocr

# 2. 安裝 PyTorch (CPU)
conda install pytorch==2.2.1 cpuonly -c pytorch -y

# 3. 安裝依賴
conda install -y numpy scipy pandas pillow pyyaml requests
pip install transformers==4.40.1 accelerate sentencepiece

# 4. 進入專案目錄
cd /home/vipl/generative-fusion-decoding-OCR

# 5. 安裝專案
python setup.py install

# 6. 執行測試
export PYTHONPATH=.
python test_trocr_setup.py

# 7. 如果測試通過，執行 fusion 測試
python test_trocr_fusion.py
```

---

## 📚 相關文件

- **主要測試指南**: [OCR_TESTING_GUIDE.md](OCR_TESTING_GUIDE.md)
- **專案 README**: [README.md](README.md)

---

## 💡 小技巧

### 1. 自動啟動環境

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
# 自動啟動 gfd-ocr 環境
# conda activate gfd-ocr  # 取消註解以自動啟動
```

### 2. 設定永久的 PYTHONPATH

在 conda 環境中設定：

```bash
conda activate gfd-ocr

# 創建環境變數目錄
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# 創建啟動腳本
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << EOF
#!/bin/bash
export PYTHONPATH=/home/vipl/generative-fusion-decoding-OCR:\$PYTHONPATH
EOF

# 創建停用腳本
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << EOF
#!/bin/bash
unset PYTHONPATH
EOF

# 賦予執行權限
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# 重新啟動環境
conda deactivate
conda activate gfd-ocr

# 驗證
echo $PYTHONPATH
```

### 3. 創建 Conda 環境配置檔案

```bash
# 匯出當前環境
conda activate gfd-ocr
conda env export > environment.yml

# 在其他機器上重建環境
conda env create -f environment.yml
```

---

## ✅ 驗證檢查表

安裝完成後，確認以下項目：

- [ ] Conda 環境已創建並可啟動
- [ ] Python 版本正確（3.9）
- [ ] PyTorch 已安裝（2.2.1+）
- [ ] Transformers 已安裝（4.40.1+）
- [ ] Pillow 已安裝
- [ ] 專案已安裝（可以 import gfd）
- [ ] test_trocr_setup.py 測試通過
- [ ] test_trocr_fusion.py 測試通過

---

## 📞 取得協助

如果遇到問題：

1. 檢查 [常見問題](#常見問題) 章節
2. 查看 [OCR_TESTING_GUIDE.md](OCR_TESTING_GUIDE.md)
3. 檢查 conda 和 pip 的安裝日誌
4. 確認網路連接正常（首次需下載模型）

---

最後更新: 2025-11-04

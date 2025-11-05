#!/bin/bash
#
# TrOCR + LLM Fusion - Conda 環境自動設定腳本
#
# 使用方式:
#   chmod +x setup_conda_env.sh
#   ./setup_conda_env.sh
#

set -e  # 遇到錯誤立即停止

# 顏色設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 環境名稱
ENV_NAME="gfd-ocr"
PYTHON_VERSION="3.9"

# 函數：打印彩色訊息
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 函數：檢查 conda 是否安裝
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda 未安裝或未在 PATH 中"
        echo ""
        echo "請先安裝 Anaconda 或 Miniconda:"
        echo "  Anaconda: https://www.anaconda.com/download"
        echo "  Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda 已安裝: $(conda --version)"
}

# 函數：檢查環境是否已存在
check_env_exists() {
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "環境 '${ENV_NAME}' 已存在"
        read -p "是否要刪除並重建？ (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "刪除現有環境..."
            conda env remove -n ${ENV_NAME} -y
            print_success "環境已刪除"
            return 0
        else
            print_info "保留現有環境，繼續安裝套件..."
            return 1
        fi
    fi
    return 0
}

# 函數：創建環境
create_env() {
    print_info "創建 Conda 環境 '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    print_success "環境創建完成"
}

# 函數：啟動環境
activate_env() {
    print_info "啟動環境..."
    # 初始化 conda（如果需要）
    if [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]; then
        source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"
    elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate ${ENV_NAME}
    print_success "環境已啟動"
}

# 函數：安裝 PyTorch
install_pytorch() {
    print_info "安裝 PyTorch (CPU 版本)..."

    # 檢查是否要安裝 GPU 版本
    read -p "是否有 NVIDIA GPU 並想使用 GPU 加速？ (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "安裝 PyTorch GPU 版本 (CUDA 11.8)..."
        conda install pytorch==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        print_info "安裝 PyTorch CPU 版本..."
        conda install pytorch==2.2.1 cpuonly -c pytorch -y
    fi

    print_success "PyTorch 安裝完成"
}

# 函數：安裝依賴套件
install_dependencies() {
    print_info "安裝依賴套件..."

    # Conda 套件
    print_info "安裝 conda 套件..."
    conda install -y numpy scipy scikit-learn pandas pillow pyyaml requests

    # Pip 套件
    print_info "安裝 pip 套件..."
    pip install transformers==4.40.1 accelerate datasets sentencepiece
    pip install jiwer opencc-python-reimplemented

    print_success "依賴套件安裝完成"
}

# 函數：安裝專案
install_project() {
    print_info "安裝專案..."

    # 檢查是否在專案目錄
    if [ ! -f "setup.py" ]; then
        print_error "找不到 setup.py，請確保在專案根目錄執行此腳本"
        exit 1
    fi

    python setup.py install
    print_success "專案安裝完成"
}

# 函數：驗證安裝
verify_installation() {
    print_info "驗證安裝..."

    # 檢查 PyTorch
    if python -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null; then
        print_success "PyTorch 正常"
    else
        print_error "PyTorch 驗證失敗"
        return 1
    fi

    # 檢查 Transformers
    if python -c "import transformers; print('Transformers:', transformers.__version__)" 2>/dev/null; then
        print_success "Transformers 正常"
    else
        print_error "Transformers 驗證失敗"
        return 1
    fi

    # 檢查 Pillow
    if python -c "import PIL; print('Pillow:', PIL.__version__)" 2>/dev/null; then
        print_success "Pillow 正常"
    else
        print_error "Pillow 驗證失敗"
        return 1
    fi

    # 檢查專案
    if python -c "import gfd" 2>/dev/null; then
        print_success "GFD 專案正常"
    else
        print_error "GFD 專案驗證失敗"
        return 1
    fi

    print_success "所有套件驗證通過"
}

# 函數：執行測試
run_tests() {
    print_info "是否要執行測試？"
    read -p "執行基礎測試？ (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "執行 test_trocr_setup.py..."
        export PYTHONPATH=.
        if python test_trocr_setup.py; then
            print_success "基礎測試通過"
        else
            print_error "基礎測試失敗"
            return 1
        fi

        read -p "執行 fusion 測試？ (y/N): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "執行 test_trocr_fusion.py..."
            if python test_trocr_fusion.py; then
                print_success "Fusion 測試通過"
            else
                print_error "Fusion 測試失敗"
                return 1
            fi
        fi
    fi
}

# 函數：設定環境變數
setup_env_vars() {
    print_info "設定環境變數..."

    local ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
    local DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
    local PROJECT_DIR="$(pwd)"

    # 創建目錄
    mkdir -p "${ACTIVATE_DIR}"
    mkdir -p "${DEACTIVATE_DIR}"

    # 創建啟動腳本
    cat > "${ACTIVATE_DIR}/env_vars.sh" << EOF
#!/bin/bash
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH}"
EOF

    # 創建停用腳本
    cat > "${DEACTIVATE_DIR}/env_vars.sh" << EOF
#!/bin/bash
unset PYTHONPATH
EOF

    # 賦予執行權限
    chmod +x "${ACTIVATE_DIR}/env_vars.sh"
    chmod +x "${DEACTIVATE_DIR}/env_vars.sh"

    print_success "環境變數設定完成"
    print_info "PYTHONPATH 將自動設定為: ${PROJECT_DIR}"
}

# 主函數
main() {
    print_header "TrOCR + LLM Fusion 環境設定"

    # 檢查 conda
    check_conda

    # 檢查並創建環境
    if check_env_exists; then
        create_env
    fi

    # 啟動環境
    activate_env

    # 安裝 PyTorch
    install_pytorch

    # 安裝依賴
    install_dependencies

    # 安裝專案
    install_project

    # 設定環境變數
    setup_env_vars

    # 驗證安裝
    verify_installation

    print_header "安裝完成！"

    print_success "環境 '${ENV_NAME}' 已成功設定"
    echo ""
    echo "使用以下命令啟動環境："
    echo -e "  ${GREEN}conda activate ${ENV_NAME}${NC}"
    echo ""
    echo "執行測試："
    echo -e "  ${GREEN}export PYTHONPATH=.${NC}"
    echo -e "  ${GREEN}python test_trocr_setup.py${NC}"
    echo ""
    echo "更多資訊請參考："
    echo "  - OCR_TESTING_GUIDE.md (測試指南)"
    echo "  - CONDA_SETUP_GUIDE.md (Conda 設定詳解)"
    echo ""

    # 詢問是否執行測試
    run_tests

    print_header "感謝使用！"
}

# 執行主函數
main "$@"

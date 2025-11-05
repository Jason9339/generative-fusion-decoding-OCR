#!/usr/bin/env python3
"""
測試 GFD 與自定義 TrOCR tokenizer 的兼容性
"""

import sys
import os

# 將 repo 根目錄加入 Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast
from gfd.tokenizer import TrOCRByteTokenizer

print("="*80)
print("測試 1: 載入用戶的自定義 TrOCR 模型")
print("="*80)

model_dir = os.path.join(REPO_ROOT, "models/ocr_model")
tokenizer_dir = os.path.join(REPO_ROOT, "models/tokenizer")

try:
    # 嘗試用標準方式載入
    print(f"\n載入模型: {model_dir}")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, use_safetensors=True)
    print("✓ 模型載入成功")

    print(f"\n載入 tokenizer (PreTrainedTokenizerFast): {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    print("✓ Tokenizer 載入成功")
    print(f"  - Vocab size: {len(tokenizer)}")
    print(f"  - Special tokens: {tokenizer.all_special_tokens}")

except Exception as e:
    print(f"✗ 載入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("測試 2: 檢查 tokenizer 是否有 byte_encoder/byte_decoder")
print("="*80)

has_byte_encoder = hasattr(tokenizer, 'byte_encoder')
has_byte_decoder = hasattr(tokenizer, 'byte_decoder')

print(f"has byte_encoder: {has_byte_encoder}")
print(f"has byte_decoder: {has_byte_decoder}")

if not has_byte_encoder or not has_byte_decoder:
    print("\n⚠️  用戶的 tokenizer 不是 byte-level tokenizer")
    print("   GFD 需要 byte_encoder 和 byte_decoder")
    print("   需要創建兼容層或使用標準 TrOCR tokenizer")

print("\n" + "="*80)
print("測試 3: 嘗試用 TrOCRByteTokenizer 載入")
print("="*80)

try:
    print(f"嘗試載入為 TrOCRByteTokenizer...")
    byte_tokenizer = TrOCRByteTokenizer.from_pretrained(model_dir)
    print("✓ TrOCRByteTokenizer 載入成功")
    print(f"  - Vocab size: {byte_tokenizer.vocab_size}")
except Exception as e:
    print(f"✗ TrOCRByteTokenizer 載入失敗: {e}")
    print("\n這是預期的結果，因為用戶的 tokenizer 不是 RobertaTokenizer 格式")

print("\n" + "="*80)
print("測試總結")
print("="*80)

if has_byte_encoder and has_byte_decoder:
    print("✓ 用戶的 tokenizer 兼容 GFD")
    print("  可以直接使用 GFD 進行推理")
else:
    print("⚠️  用戶的 tokenizer 不完全兼容 GFD")
    print("  選項 1: 使用標準的 microsoft/trocr-base-* tokenizer")
    print("  選項 2: 創建兼容層包裝用戶的 tokenizer")
    print("  選項 3: 修改 GFD 代碼以支援字符級 tokenizer")
    print("\n建議: 選項 1 是最簡單的方案，因為 GFD 主要改善的是解碼過程，")
    print("       使用標準 tokenizer 不會影響模型權重")

print("="*80)

#!/usr/bin/env python3
"""
測試腳本：完整測試 TrOCR + LLM Fusion 流程
"""

import sys
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from types import SimpleNamespace

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "images", "fusion")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_image(text="HELLO WORLD", size=(400, 100), font_size=40):
    """創建測試圖片"""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2

    draw.text((x, y), text, fill="black", font=font)
    return img

def test_fusion_with_config_files():
    """使用配置檔案測試 Fusion"""
    print("=" * 80)
    print("測試 1: 使用配置檔案的 TrOCR + LLM Fusion")
    print("=" * 80)

    # 載入配置
    model_config = process_config('config_files/model/gfd-ocr-en.yaml')
    prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')
    combined_config = combine_config(prompt_config, model_config)

    print("✓ 配置載入成功")
    print(f"  - OCR model: {combined_config.ocr_model_path}")
    print(f"  - LLM model: {combined_config.llm_model_path}")
    print(f"  - OCR device: {combined_config.ocr_device}")
    print(f"  - LLM device: {combined_config.llm_device}")
    print(f"  - Fusing ratio: {combined_config.fusing_r}")
    print(f"  - OCR prompt: '{combined_config.ocr_prompt}'")
    print(f"  - LLM prompt: '{combined_config.llm_prompt}'")

    # 創建模型
    print("\n載入 Breezper 模型...")
    model = Breezper(combined_config)
    print("✓ Breezper 模型載入成功")

    # 測試圖片
    test_cases = [
        ("HELLO WORLD", (400, 100)),
        ("GENERATIVE FUSION", (500, 120)),
        ("OCR TEST 2024", (450, 100)),
        ("AI RESEARCH", (400, 100)),
    ]

    results = []
    for i, (text, size) in enumerate(test_cases, 1):
        print(f"\n{'-' * 80}")
        print(f"測試案例 {i}: '{text}'")
        print(f"{'-' * 80}")

        # 創建圖片
        img = create_test_image(text, size=size)
        img_path = os.path.join(OUTPUT_DIR, f"test_fusion_{i}.png")
        img.save(img_path)
        print(f"圖片已保存: {img_path}")

        # 執行識別
        print(f"開始識別...")
        result = model.get_transcription(
            img,
            num_beams=3,
            ocr_prompt=combined_config.ocr_prompt,
            llm_prompt=combined_config.llm_prompt
        )

        print(f"\n結果:")
        print(f"  原始文字: '{text}'")
        print(f"  識別結果: '{result}'")
        print(f"  匹配: {'✓' if result.upper().strip() == text.upper().strip() else '✗'}")

        results.append({
            'original': text,
            'recognized': result,
            'match': result.upper().strip() == text.upper().strip()
        })

    return results

def test_fusion_with_simple_config():
    """使用簡單配置測試 Fusion（快速測試）"""
    print("\n" + "=" * 80)
    print("測試 2: 使用簡單配置的快速測試")
    print("=" * 80)

    # 創建簡單配置
    config = SimpleNamespace(
        ocr_model_path='microsoft/trocr-base-printed',
        llm_model_path='hf-internal-testing/tiny-random-LlamaForCausalLM',
        ocr_device='cpu',
        llm_device='cpu',
        ocr_torch_dtype='float32',
        llm_torch_dtype='float32',
        use_cache='dynamic',
        fuse_strategy='simple',
        fusing_r=0.2,
        llm_attn_implementation=None,
        llm_temp=1.5,
        repetition_penalty=1.0,
        repetition_penalty_last=50,
        repetition_penalty_window=50,
        repetition_penalty_threshold=1.0,
        beam_terminated_strategy='when_all_end',
        beam_select_strategy='best',
        beam_max_decode_len=128,
        beam_max_len_diff=10,
        beam_max_len=-1,
        beam_min_len=9999,
        logprob_min=-100000
    )

    print("✓ 配置建立成功")
    print("  (使用 tiny-random-LlamaForCausalLM 作為快速測試)")

    # 創建模型
    print("\n載入 Breezper 模型...")
    model = Breezper(config)
    print("✓ Breezper 模型載入成功")

    # 簡單測試
    text = "QUICK TEST"
    print(f"\n測試文字: '{text}'")
    img = create_test_image(text)
    quick_img_path = os.path.join(OUTPUT_DIR, "test_quick.png")
    img.save(quick_img_path)
    print(f"圖片已保存: {quick_img_path}")

    result = model.get_transcription(img, num_beams=2)
    print(f"識別結果: '{result}'")

    return result

def print_summary(results):
    """打印測試總結"""
    print("\n" + "=" * 80)
    print("測試總結")
    print("=" * 80)

    total = len(results)
    matched = sum(1 for r in results if r['match'])

    print(f"\n總測試數: {total}")
    print(f"成功匹配: {matched}")
    print(f"匹配率: {matched/total*100:.1f}%")

    print("\n詳細結果:")
    for i, r in enumerate(results, 1):
        status = "✓" if r['match'] else "✗"
        print(f"  {i}. {status} '{r['original']}' → '{r['recognized']}'")

def main():
    try:
        print("TrOCR + LLM Fusion 完整測試")
        print("=" * 80)

        # 測試 1: 完整配置
        results = test_fusion_with_config_files()

        # 測試 2: 快速測試
        # quick_result = test_fusion_with_simple_config()

        # 總結
        print_summary(results)

        print("\n" + "=" * 80)
        print("✓ 所有測試完成！")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
快速測試 Fusion 開關對比
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import argparse

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "images", "comparison")
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

def test_fusion_comparison():
    """測試 Fusion 開關對比"""
    print("=" * 80)
    print("Fusion 開關對比測試")
    print("=" * 80)

    test_cases = [
        ("HELLO WORLD", (400, 100)),
        ("QUICK BROWN FOX", (450, 100)),
        ("AI RESEARCH", (400, 100)),
    ]

    # 載入兩個配置
    print("\n準備兩個模型配置...")

    # With Fusion (fusing_r = 0.2)
    prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')
    model_config_with = process_config('config_files/model/gfd-ocr-en.yaml')
    config_with = combine_config(prompt_config, model_config_with)
    print(f"✓ With Fusion: fusing_r = {config_with.fusing_r}")

    # Without Fusion (fusing_r = 0.0)
    override_args = argparse.Namespace(fusing_r=0.0)
    model_config_without = process_config('config_files/model/gfd-ocr-en.yaml', args=override_args)
    config_without = combine_config(prompt_config, model_config_without)
    print(f"✓ Without Fusion: fusing_r = {config_without.fusing_r}")

    # 載入模型
    print("\n載入模型...")
    print("  [1/2] 載入 With Fusion 模型...")
    model_with = Breezper(config_with)
    print("  ✓ With Fusion 模型載入完成")

    print("  [2/2] 載入 Without Fusion 模型...")
    model_without = Breezper(config_without)
    print("  ✓ Without Fusion 模型載入完成")

    # 測試
    results = []
    print("\n" + "=" * 80)
    print("開始測試")
    print("=" * 80)

    for i, (text, size) in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"測試案例 {i}/{len(test_cases)}: '{text}'")
        print(f"{'─' * 80}")

        # 創建圖片
        img = create_test_image(text, size=size)
        img_path = os.path.join(OUTPUT_DIR, f"test_comparison_{i}.png")
        img.save(img_path)
        print(f"圖片已保存: {img_path}")

        # With Fusion
        print(f"\n[With Fusion]")
        result_with = model_with.get_transcription(
            img,
            num_beams=3,
            ocr_prompt=config_with.ocr_prompt,
            llm_prompt=config_with.llm_prompt
        )
        match_with = result_with.upper().strip() == text.upper().strip()

        # Without Fusion
        print(f"\n[Without Fusion (Pure TrOCR)]")
        result_without = model_without.get_transcription(
            img,
            num_beams=3,
            ocr_prompt=config_without.ocr_prompt,
            llm_prompt=config_without.llm_prompt
        )
        match_without = result_without.upper().strip() == text.upper().strip()

        # 結果比較
        print(f"\n{'─' * 80}")
        print(f"結果比較:")
        print(f"  原始文字:      '{text}'")
        print(f"  With Fusion:   '{result_with}' {'✓' if match_with else '✗'}")
        print(f"  Without Fusion: '{result_without}' {'✓' if match_without else '✗'}")

        if result_with == result_without:
            print(f"  → 結果相同")
        else:
            print(f"  → 結果不同！Fusion 有影響")
        print(f"{'─' * 80}")

        results.append({
            'text': text,
            'with_fusion': result_with,
            'without_fusion': result_without,
            'match_with': match_with,
            'match_without': match_without,
            'same': result_with == result_without
        })

    # 總結
    print("\n" + "=" * 80)
    print("測試總結")
    print("=" * 80)

    total = len(results)
    with_correct = sum(1 for r in results if r['match_with'])
    without_correct = sum(1 for r in results if r['match_without'])
    same_results = sum(1 for r in results if r['same'])

    print(f"\n總測試數: {total}")
    print(f"With Fusion 正確: {with_correct}/{total} ({with_correct/total*100:.1f}%)")
    print(f"Without Fusion 正確: {without_correct}/{total} ({without_correct/total*100:.1f}%)")
    print(f"結果相同數: {same_results}/{total} ({same_results/total*100:.1f}%)")

    print(f"\n詳細結果:")
    for i, r in enumerate(results, 1):
        status_with = "✓" if r['match_with'] else "✗"
        status_without = "✓" if r['match_without'] else "✗"
        same_mark = "=" if r['same'] else "≠"
        print(f"  {i}. '{r['text']}'")
        print(f"     With:    {status_with} '{r['with_fusion']}'")
        print(f"     Without: {status_without} '{r['without_fusion']}' {same_mark}")

    if same_results == total:
        print(f"\n⚠️  注意: 所有結果都相同，Fusion 可能沒有作用")
        print(f"     可能原因:")
        print(f"     1. 圖片太簡單，TrOCR 本身就很準確")
        print(f"     2. LLM 影響權重太低 (fusing_r={config_with.fusing_r})")
        print(f"     3. Beam search 選擇了相同的路徑")
    elif with_correct > without_correct:
        print(f"\n✓ Fusion 有正向效果！準確率提升 {(with_correct-without_correct)/total*100:.1f}%")
    elif with_correct < without_correct:
        print(f"\n⚠️  注意: Fusion 反而降低了準確率")
    else:
        print(f"\n→ Fusion 對準確率沒有影響")

    print("\n" + "=" * 80)
    print("✓ 測試完成")
    print("=" * 80)

def main():
    try:
        test_fusion_comparison()
        return 0
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

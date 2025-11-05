#!/usr/bin/env python3
"""
快速版困難場景測試 - 只測試幾個代表性案例
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import argparse

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "images", "quick_hard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_image(text, size=(900, 150), font_size=28,
                     add_noise=False, noise_level=0.0,
                     blur=False, blur_radius=0,
                     low_contrast=False):
    """創建測試圖片"""
    if low_contrast:
        bg_color = (200, 200, 200)
        text_color = (80, 80, 80)
    else:
        bg_color = "white"
        text_color = "black"

    img = Image.new("RGB", size, color=bg_color)
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
    draw.text((x, y), text, fill=text_color, font=font)

    if blur and blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if add_noise and noise_level > 0:
        img_array = np.array(img)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_img)

    return img

def main():
    print("=" * 80)
    print("快速困難場景測試")
    print("=" * 80)

    # 載入模型
    print("\n載入模型...")
    prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')

    # With Fusion (fusing_r=0.3)
    override_with = argparse.Namespace(fusing_r=0.3)
    model_config_with = process_config('config_files/model/gfd-ocr-en.yaml', args=override_with)
    config_with = combine_config(prompt_config, model_config_with)

    # Without Fusion (fusing_r=0.0)
    override_without = argparse.Namespace(fusing_r=0.0)
    model_config_without = process_config('config_files/model/gfd-ocr-en.yaml', args=override_without)
    config_without = combine_config(prompt_config, model_config_without)

    print(f"✓ With Fusion: fusing_r = {config_with.fusing_r}")
    print(f"✓ Without Fusion: fusing_r = {config_without.fusing_r}")

    model_with = Breezper(config_with)
    model_without = Breezper(config_without)
    print("✓ 模型載入完成\n")

    # 測試案例
    test_cases = [
        {
            'name': '長句子',
            'text': 'The quick brown fox jumps over the lazy dog',
            'params': {}
        },
        {
            'name': '專業術語',
            'text': 'Convolutional Neural Networks for Image Classification',
            'params': {}
        },
        {
            'name': '噪音干擾',
            'text': 'This text has moderate noise interference',
            'params': {'add_noise': True, 'noise_level': 0.15}
        },
        {
            'name': '低對比度',
            'text': 'Low contrast makes text harder to read',
            'params': {'low_contrast': True}
        },
        {
            'name': '模糊',
            'text': 'Blurred images challenge OCR systems',
            'params': {'blur': True, 'blur_radius': 1.5}
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print("=" * 80)
        print(f"測試 {i}/{len(test_cases)}: {test['name']}")
        print("=" * 80)
        print(f"文字: '{test['text']}'")

        # 創建圖片
        img = create_test_image(test['text'], **test['params'])
        img_path = os.path.join(OUTPUT_DIR, f"test_{i}_{test['name'].replace(' ', '_')}.png")
        img.save(img_path)
        print(f"圖片: {img_path}")

        # With Fusion
        print(f"\n[With Fusion (r=0.3)]")
        result_with = model_with.get_transcription(
            img, num_beams=5,
            ocr_prompt=config_with.ocr_prompt,
            llm_prompt=config_with.llm_prompt
        )

        # Without Fusion
        print(f"\n[Without Fusion (r=0.0)]")
        result_without = model_without.get_transcription(
            img, num_beams=5,
            ocr_prompt=config_without.ocr_prompt,
            llm_prompt=config_without.llm_prompt
        )

        # 比較
        match_with = result_with.strip().upper() == test['text'].strip().upper()
        match_without = result_without.strip().upper() == test['text'].strip().upper()
        same = result_with.strip() == result_without.strip()

        print(f"\n{'─' * 80}")
        print(f"結果:")
        print(f"  原文:    '{test['text']}'")
        print(f"  With:    '{result_with}' {'✓' if match_with else '✗'}")
        print(f"  Without: '{result_without}' {'✓' if match_without else '✗'}")

        if not same:
            print(f"  → 結果不同！", end=" ")
            if match_with and not match_without:
                print(f"💡 Fusion 修正了錯誤！")
            elif not match_with and match_without:
                print(f"⚠️  Fusion 反而產生錯誤")
            else:
                print(f"ℹ️  兩者都有錯但錯誤不同")
        else:
            print(f"  → 結果相同")
        print(f"{'─' * 80}\n")

        results.append({
            'name': test['name'],
            'text': test['text'],
            'with': result_with,
            'without': result_without,
            'match_with': match_with,
            'match_without': match_without,
            'same': same
        })

    # 總結
    print("\n" + "=" * 80)
    print("測試總結")
    print("=" * 80)

    total = len(results)
    with_correct = sum(1 for r in results if r['match_with'])
    without_correct = sum(1 for r in results if r['match_without'])
    diff_count = sum(1 for r in results if not r['same'])

    print(f"\n總測試數: {total}")
    print(f"With Fusion 正確: {with_correct}/{total} ({with_correct/total*100:.1f}%)")
    print(f"Without Fusion 正確: {without_correct}/{total} ({without_correct/total*100:.1f}%)")
    print(f"結果不同: {diff_count}/{total} ({diff_count/total*100:.1f}%)")

    print(f"\n詳細:")
    for r in results:
        status_with = "✓" if r['match_with'] else "✗"
        status_without = "✓" if r['match_without'] else "✗"
        diff = "≠" if not r['same'] else "="
        print(f"  {r['name']:15s}: With {status_with} vs Without {status_without} {diff}")

    print("\n" + "=" * 80)
    if with_correct > without_correct:
        print(f"✅ Fusion 提升準確率 {(with_correct-without_correct)/total*100:.1f}%")
    elif with_correct < without_correct:
        print(f"⚠️  Fusion 降低準確率 {(without_correct-with_correct)/total*100:.1f}%")
    else:
        print(f"→ Fusion 對準確率無影響")

    if diff_count > 0:
        print(f"💡 Fusion 在 {diff_count} 個案例中產生不同結果")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ 錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

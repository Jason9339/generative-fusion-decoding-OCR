#!/usr/bin/env python3
"""
困難場景的 Fusion 對比測試

測試場景：
1. 長文本（多行、完整句子）
2. 複雜詞彙（專業術語、罕見詞）
3. 低品質圖片（噪音、模糊、低對比度）
4. 混合數字和符號
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import argparse

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "images", "hard_cases")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_image(text, size=(800, 200), font_size=24,
                     add_noise=False, noise_level=0.0,
                     blur=False, blur_radius=0,
                     low_contrast=False,
                     multiline=False):
    """創建測試圖片（支援多種困難條件）"""

    # 調整大小以容納長文本
    if multiline:
        lines = text.split('\n')
        max_line_length = max(len(line) for line in lines)
        size = (max_line_length * font_size // 2 + 100, len(lines) * font_size + 80)

    # 設定顏色
    if low_contrast:
        bg_color = (200, 200, 200)  # 淺灰背景
        text_color = (80, 80, 80)    # 深灰文字
    else:
        bg_color = "white"
        text_color = "black"

    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 繪製文字（支援多行）
    if multiline:
        y_offset = 40
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (size[0] - text_width) / 2
            draw.text((x, y_offset), line, fill=text_color, font=font)
            y_offset += font_size + 10
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        draw.text((x, y), text, fill=text_color, font=font)

    # 添加模糊
    if blur and blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 添加噪音
    if add_noise and noise_level > 0:
        img_array = np.array(img)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_img)

    return img

def load_models(fusing_r_with=0.2, fusing_r_without=0.0):
    """載入 With/Without Fusion 兩個模型"""
    print("=" * 80)
    print("載入模型配置")
    print("=" * 80)

    prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')

    # With Fusion
    override_with = argparse.Namespace(fusing_r=fusing_r_with)
    model_config_with = process_config('config_files/model/gfd-ocr-en.yaml', args=override_with)
    config_with = combine_config(prompt_config, model_config_with)
    print(f"✓ With Fusion: fusing_r = {config_with.fusing_r}")

    # Without Fusion
    override_without = argparse.Namespace(fusing_r=fusing_r_without)
    model_config_without = process_config('config_files/model/gfd-ocr-en.yaml', args=override_without)
    config_without = combine_config(prompt_config, model_config_without)
    print(f"✓ Without Fusion: fusing_r = {config_without.fusing_r}")

    print("\n載入模型...")
    print("  [1/2] 載入 With Fusion 模型...")
    model_with = Breezper(config_with)
    print("  ✓ With Fusion 模型載入完成")

    print("  [2/2] 載入 Without Fusion 模型...")
    model_without = Breezper(config_without)
    print("  ✓ Without Fusion 模型載入完成")

    return model_with, model_without, config_with, config_without

def run_test(model_with, model_without, config_with, config_without,
             test_name, test_cases, num_beams=5):
    """執行測試並比較結果"""

    print("\n" + "=" * 80)
    print(f"{test_name}")
    print("=" * 80)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        img_params = test_case.get('params', {})

        print(f"\n{'─' * 80}")
        print(f"測試 {i}/{len(test_cases)}")
        print(f"文字: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        if img_params:
            print(f"條件: {img_params}")
        print(f"{'─' * 80}")

        # 創建圖片
        img = create_test_image(text, **img_params)
        img_path = os.path.join(OUTPUT_DIR, f"{test_name.lower().replace(' ', '_')}_{i}.png")
        img.save(img_path)
        print(f"圖片: {img_path}")

        # With Fusion
        print(f"\n[With Fusion]")
        result_with = model_with.get_transcription(
            img, num_beams=num_beams,
            ocr_prompt=config_with.ocr_prompt,
            llm_prompt=config_with.llm_prompt
        )

        # Without Fusion
        print(f"\n[Without Fusion]")
        result_without = model_without.get_transcription(
            img, num_beams=num_beams,
            ocr_prompt=config_without.ocr_prompt,
            llm_prompt=config_without.llm_prompt
        )

        # 比較結果
        match_with = result_with.strip().upper() == text.strip().upper()
        match_without = result_without.strip().upper() == text.strip().upper()
        same = result_with.strip() == result_without.strip()

        print(f"\n{'─' * 80}")
        print(f"結果比較:")
        print(f"  原文: '{text}'")
        print(f"  With:    '{result_with}' {'✓' if match_with else '✗'}")
        print(f"  Without: '{result_without}' {'✓' if match_without else '✗'}")

        if same:
            print(f"  → 結果相同")
        else:
            print(f"  → 結果不同！")
            if match_with and not match_without:
                print(f"     💡 Fusion 修正了錯誤！")
            elif not match_with and match_without:
                print(f"     ⚠️  Fusion 反而產生錯誤")
            else:
                print(f"     ℹ️  兩者都有錯，但錯誤不同")
        print(f"{'─' * 80}")

        results.append({
            'text': text,
            'with_fusion': result_with,
            'without_fusion': result_without,
            'match_with': match_with,
            'match_without': match_without,
            'same': same,
            'params': img_params
        })

    return results

def print_summary(test_name, results):
    """打印測試總結"""
    print(f"\n{'═' * 80}")
    print(f"{test_name} - 總結")
    print(f"{'═' * 80}")

    total = len(results)
    with_correct = sum(1 for r in results if r['match_with'])
    without_correct = sum(1 for r in results if r['match_without'])
    same_results = sum(1 for r in results if r['same'])
    diff_results = total - same_results

    print(f"\n📊 統計:")
    print(f"  總測試數: {total}")
    print(f"  With Fusion 正確: {with_correct}/{total} ({with_correct/total*100:.1f}%)")
    print(f"  Without Fusion 正確: {without_correct}/{total} ({without_correct/total*100:.1f}%)")
    print(f"  結果相同: {same_results}/{total} ({same_results/total*100:.1f}%)")
    print(f"  結果不同: {diff_results}/{total} ({diff_results/total*100:.1f}%)")

    if diff_results > 0:
        fusion_better = sum(1 for r in results if not r['same'] and r['match_with'] and not r['match_without'])
        fusion_worse = sum(1 for r in results if not r['same'] and not r['match_with'] and r['match_without'])
        fusion_both_wrong = sum(1 for r in results if not r['same'] and not r['match_with'] and not r['match_without'])

        print(f"\n📈 結果不同的案例分析:")
        print(f"  Fusion 較好: {fusion_better}")
        print(f"  Fusion 較差: {fusion_worse}")
        print(f"  兩者都錯: {fusion_both_wrong}")

    print(f"\n📝 詳細結果:")
    for i, r in enumerate(results, 1):
        status_with = "✓" if r['match_with'] else "✗"
        status_without = "✓" if r['match_without'] else "✗"
        diff_mark = "=" if r['same'] else "≠"

        text_short = r['text'][:50] + "..." if len(r['text']) > 50 else r['text']
        print(f"\n  {i}. {text_short}")
        print(f"     With:    {status_with} '{r['with_fusion']}'")
        print(f"     Without: {status_without} '{r['without_fusion']}' {diff_mark}")

    # 整體評價
    print(f"\n{'─' * 80}")
    if with_correct > without_correct:
        improvement = (with_correct - without_correct) / total * 100
        print(f"✅ 結論: Fusion 有正向效果！準確率提升 {improvement:.1f}%")
    elif with_correct < without_correct:
        decline = (without_correct - with_correct) / total * 100
        print(f"⚠️  結論: Fusion 降低了準確率 {decline:.1f}%")
    elif diff_results > 0:
        print(f"ℹ️  結論: Fusion 改變了結果但整體準確率相同")
    else:
        print(f"→ 結論: Fusion 對此場景無影響（結果完全相同）")
    print(f"{'─' * 80}")

def main():
    print("困難場景 Fusion 對比測試")
    print("=" * 80)

    # 載入模型
    model_with, model_without, config_with, config_without = load_models(
        fusing_r_with=0.3,  # 提高 fusion 權重
        fusing_r_without=0.0
    )

    all_results = {}

    # ========== 測試 1: 長句子（完整語境） ==========
    print("\n\n")
    test_cases_long = [
        {
            'text': 'The quick brown fox jumps over the lazy dog',
            'params': {'size': (900, 150), 'font_size': 28}
        },
        {
            'text': 'Machine learning enables computers to learn from data',
            'params': {'size': (950, 150), 'font_size': 26}
        },
        {
            'text': 'Natural language processing is a subfield of artificial intelligence',
            'params': {'size': (1100, 150), 'font_size': 24}
        },
        {
            'text': 'Deep neural networks have revolutionized computer vision tasks',
            'params': {'size': (1050, 150), 'font_size': 25}
        },
    ]

    results_long = run_test(model_with, model_without, config_with, config_without,
                           "測試 1: 長句子", test_cases_long, num_beams=5)
    all_results['long_sentences'] = results_long
    print_summary("測試 1: 長句子", results_long)

    # ========== 測試 2: 專業術語和複雜詞彙 ==========
    print("\n\n")
    test_cases_technical = [
        {
            'text': 'Convolutional Neural Networks for Image Classification',
            'params': {'size': (950, 150), 'font_size': 26}
        },
        {
            'text': 'Transformers Architecture with Self-Attention Mechanism',
            'params': {'size': (980, 150), 'font_size': 25}
        },
        {
            'text': 'Recurrent Neural Networks with LSTM and GRU cells',
            'params': {'size': (920, 150), 'font_size': 26}
        },
        {
            'text': 'Generative Adversarial Networks for Image Synthesis',
            'params': {'size': (950, 150), 'font_size': 26}
        },
    ]

    results_technical = run_test(model_with, model_without, config_with, config_without,
                                "測試 2: 專業術語", test_cases_technical, num_beams=5)
    all_results['technical_terms'] = results_technical
    print_summary("測試 2: 專業術語", results_technical)

    # ========== 測試 3: 噪音干擾 ==========
    print("\n\n")
    test_cases_noisy = [
        {
            'text': 'This text has moderate noise interference',
            'params': {'size': (850, 150), 'font_size': 28, 'add_noise': True, 'noise_level': 0.15}
        },
        {
            'text': 'Machine learning models require clean data',
            'params': {'size': (900, 150), 'font_size': 28, 'add_noise': True, 'noise_level': 0.12}
        },
        {
            'text': 'Deep learning advances artificial intelligence',
            'params': {'size': (900, 150), 'font_size': 28, 'add_noise': True, 'noise_level': 0.18}
        },
    ]

    results_noisy = run_test(model_with, model_without, config_with, config_without,
                            "測試 3: 噪音干擾", test_cases_noisy, num_beams=5)
    all_results['noisy'] = results_noisy
    print_summary("測試 3: 噪音干擾", results_noisy)

    # ========== 測試 4: 低對比度 ==========
    print("\n\n")
    test_cases_low_contrast = [
        {
            'text': 'Low contrast makes text harder to read',
            'params': {'size': (850, 150), 'font_size': 28, 'low_contrast': True}
        },
        {
            'text': 'Computer vision algorithms process images',
            'params': {'size': (900, 150), 'font_size': 28, 'low_contrast': True}
        },
        {
            'text': 'Neural networks learn hierarchical features',
            'params': {'size': (900, 150), 'font_size': 28, 'low_contrast': True}
        },
    ]

    results_low_contrast = run_test(model_with, model_without, config_with, config_without,
                                    "測試 4: 低對比度", test_cases_low_contrast, num_beams=5)
    all_results['low_contrast'] = results_low_contrast
    print_summary("測試 4: 低對比度", results_low_contrast)

    # ========== 測試 5: 模糊圖片 ==========
    print("\n\n")
    test_cases_blur = [
        {
            'text': 'Blurred images challenge OCR systems',
            'params': {'size': (850, 150), 'font_size': 28, 'blur': True, 'blur_radius': 1.5}
        },
        {
            'text': 'Image preprocessing improves recognition',
            'params': {'size': (900, 150), 'font_size': 28, 'blur': True, 'blur_radius': 1.8}
        },
    ]

    results_blur = run_test(model_with, model_without, config_with, config_without,
                           "測試 5: 模糊圖片", test_cases_blur, num_beams=5)
    all_results['blur'] = results_blur
    print_summary("測試 5: 模糊圖片", results_blur)

    # ========== 測試 6: 混合條件（最困難） ==========
    print("\n\n")
    test_cases_mixed = [
        {
            'text': 'Combining noise blur and low contrast',
            'params': {
                'size': (850, 150),
                'font_size': 28,
                'add_noise': True,
                'noise_level': 0.1,
                'blur': True,
                'blur_radius': 1.2,
                'low_contrast': True
            }
        },
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
        },
    ]

    results_mixed = run_test(model_with, model_without, config_with, config_without,
                            "測試 6: 混合困難條件", test_cases_mixed, num_beams=5)
    all_results['mixed'] = results_mixed
    print_summary("測試 6: 混合困難條件", results_mixed)

    # ========== 整體總結 ==========
    print("\n\n")
    print("=" * 80)
    print("整體測試總結")
    print("=" * 80)

    total_tests = sum(len(results) for results in all_results.values())
    total_with_correct = sum(sum(1 for r in results if r['match_with'])
                            for results in all_results.values())
    total_without_correct = sum(sum(1 for r in results if r['match_without'])
                               for results in all_results.values())
    total_diff = sum(sum(1 for r in results if not r['same'])
                    for results in all_results.values())

    print(f"\n📊 總體統計:")
    print(f"  總測試數: {total_tests}")
    print(f"  With Fusion 正確率: {total_with_correct}/{total_tests} ({total_with_correct/total_tests*100:.1f}%)")
    print(f"  Without Fusion 正確率: {total_without_correct}/{total_tests} ({total_without_correct/total_tests*100:.1f}%)")
    print(f"  結果不同數: {total_diff}/{total_tests} ({total_diff/total_tests*100:.1f}%)")

    print(f"\n📈 各測試場景正確率對比:")
    for test_name, results in all_results.items():
        with_acc = sum(1 for r in results if r['match_with']) / len(results) * 100
        without_acc = sum(1 for r in results if r['match_without']) / len(results) * 100
        diff_pct = sum(1 for r in results if not r['same']) / len(results) * 100

        diff_symbol = "+" if with_acc > without_acc else "-" if with_acc < without_acc else "="
        print(f"  {test_name:20s}: With {with_acc:5.1f}% vs Without {without_acc:5.1f}% "
              f"[{diff_symbol}{abs(with_acc-without_acc):.1f}%] (不同 {diff_pct:.0f}%)")

    print("\n" + "=" * 80)
    if total_with_correct > total_without_correct:
        improvement = (total_with_correct - total_without_correct) / total_tests * 100
        print(f"✅ 最終結論: Fusion 整體提升準確率 {improvement:.1f}%")
    elif total_with_correct < total_without_correct:
        decline = (total_without_correct - total_with_correct) / total_tests * 100
        print(f"⚠️  最終結論: Fusion 整體降低準確率 {decline:.1f}%")
    else:
        print(f"→ 最終結論: Fusion 對整體準確率無影響")

    if total_diff > 0:
        print(f"💡 Fusion 在 {total_diff/total_tests*100:.1f}% 的案例中產生不同結果")

    print("=" * 80)
    print(f"\n✓ 所有測試完成！")
    print(f"圖片已保存至: {OUTPUT_DIR}")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
OCR 測試效果評估腳本

此腳本提供多種測試場景來評估 TrOCR + LLM Fusion 的效果：
1. 基礎文字識別測試
2. 不同字體大小測試
3. 噪音干擾測試
4. 低對比度測試
5. 與純 TrOCR（無 fusion）的對比測試
"""

import sys
import os
import time
import json
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "images", "evaluation")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_output", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_test_image(text, size=(400, 100), font_size=40,
                     bg_color="white", text_color="black",
                     add_noise=False, noise_level=0.1,
                     low_contrast=False):
    """創建測試圖片，支援多種測試條件"""
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 計算文字位置
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2

    # 繪製文字
    if low_contrast:
        # 低對比度：灰色背景 + 深灰色文字
        img = Image.new("RGB", size, color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        text_color = (80, 80, 80)

    draw.text((x, y), text, fill=text_color, font=font)

    # 添加噪音
    if add_noise:
        img_array = np.array(img)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_img)

    return img

class OCRTester:
    """OCR 測試器"""

    def __init__(self, with_fusion=True):
        """初始化測試器

        Args:
            with_fusion: 是否使用 fusion（False 則設定 fusing_r=0）
        """
        self.with_fusion = with_fusion

        # 載入配置
        # 如果不使用 fusion，透過 args 覆蓋 fusing_r
        import argparse
        if not with_fusion:
            override_args = argparse.Namespace(fusing_r=0.0)
            model_config = process_config('config_files/model/gfd-ocr-en.yaml', args=override_args)
        else:
            model_config = process_config('config_files/model/gfd-ocr-en.yaml')

        prompt_config = process_config('config_files/prompt/ocr-default-prompt.yaml')
        self.config = combine_config(prompt_config, model_config)

        # 載入模型
        print(f"載入模型 (fusion={'ON' if with_fusion else 'OFF'})...")
        self.model = Breezper(self.config)
        print("✓ 模型載入完成")

    def test_image(self, img, description=""):
        """測試單張圖片"""
        start_time = time.time()

        result = self.model.get_transcription(
            img,
            num_beams=3,
            ocr_prompt=self.config.ocr_prompt,
            llm_prompt=self.config.llm_prompt
        )

        elapsed_time = time.time() - start_time

        return {
            'result': result,
            'time': elapsed_time,
            'description': description
        }

def test_basic_recognition():
    """測試 1: 基礎文字識別"""
    print("\n" + "=" * 80)
    print("測試 1: 基礎文字識別")
    print("=" * 80)

    test_cases = [
        "HELLO WORLD",
        "QUICK BROWN FOX",
        "THE AI REVOLUTION",
        "DEEP LEARNING 2024",
        "NATURAL LANGUAGE PROCESSING",
    ]

    tester = OCRTester(with_fusion=True)
    results = []

    for i, text in enumerate(test_cases, 1):
        print(f"\n測試 {i}/{len(test_cases)}: '{text}'")
        img = create_test_image(text, size=(500, 120))
        img.save(os.path.join(OUTPUT_DIR, f"test_basic_{i}.png"))

        result = tester.test_image(img, description=f"基礎識別: {text}")
        match = result['result'].upper().strip() == text.upper().strip()

        print(f"  識別結果: '{result['result']}'")
        print(f"  匹配: {'✓' if match else '✗'}")
        print(f"  耗時: {result['time']:.2f}s")

        results.append({
            'original': text,
            'recognized': result['result'],
            'match': match,
            'time': result['time']
        })

    return results

def test_font_sizes():
    """測試 2: 不同字體大小"""
    print("\n" + "=" * 80)
    print("測試 2: 不同字體大小")
    print("=" * 80)

    test_text = "FONT SIZE TEST"
    font_sizes = [20, 30, 40, 50, 60]

    tester = OCRTester(with_fusion=True)
    results = []

    for i, font_size in enumerate(font_sizes, 1):
        print(f"\n測試 {i}/{len(font_sizes)}: 字體大小 {font_size}")
        img = create_test_image(test_text, size=(450, 100), font_size=font_size)
        img.save(os.path.join(OUTPUT_DIR, f"test_fontsize_{font_size}.png"))

        result = tester.test_image(img, description=f"字體大小 {font_size}")
        match = result['result'].upper().strip() == test_text.upper().strip()

        print(f"  識別結果: '{result['result']}'")
        print(f"  匹配: {'✓' if match else '✗'}")
        print(f"  耗時: {result['time']:.2f}s")

        results.append({
            'font_size': font_size,
            'original': test_text,
            'recognized': result['result'],
            'match': match,
            'time': result['time']
        })

    return results

def test_with_noise():
    """測試 3: 噪音干擾"""
    print("\n" + "=" * 80)
    print("測試 3: 噪音干擾")
    print("=" * 80)

    test_text = "NOISE TEST"
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

    tester = OCRTester(with_fusion=True)
    results = []

    for i, noise_level in enumerate(noise_levels, 1):
        print(f"\n測試 {i}/{len(noise_levels)}: 噪音等級 {noise_level}")
        img = create_test_image(test_text, size=(400, 100),
                               add_noise=True, noise_level=noise_level)
        img.save(os.path.join(OUTPUT_DIR, f"test_noise_{int(noise_level*100)}.png"))

        result = tester.test_image(img, description=f"噪音等級 {noise_level}")
        match = result['result'].upper().strip() == test_text.upper().strip()

        print(f"  識別結果: '{result['result']}'")
        print(f"  匹配: {'✓' if match else '✗'}")
        print(f"  耗時: {result['time']:.2f}s")

        results.append({
            'noise_level': noise_level,
            'original': test_text,
            'recognized': result['result'],
            'match': match,
            'time': result['time']
        })

    return results

def test_fusion_comparison():
    """測試 4: Fusion 開關對比"""
    print("\n" + "=" * 80)
    print("測試 4: Fusion 開關對比測試")
    print("=" * 80)

    test_cases = [
        "FUSION COMPARISON",
        "MACHINE LEARNING",
        "COMPUTER VISION",
    ]

    print("\n載入兩個模型進行對比...")
    tester_with = OCRTester(with_fusion=True)
    tester_without = OCRTester(with_fusion=False)

    results = []

    for i, text in enumerate(test_cases, 1):
        print(f"\n測試 {i}/{len(test_cases)}: '{text}'")
        img = create_test_image(text, size=(500, 120))
        img.save(os.path.join(OUTPUT_DIR, f"test_comparison_{i}.png"))

        # 使用 fusion
        result_with = tester_with.test_image(img, description=f"With Fusion: {text}")
        match_with = result_with['result'].upper().strip() == text.upper().strip()

        # 不使用 fusion
        result_without = tester_without.test_image(img, description=f"Without Fusion: {text}")
        match_without = result_without['result'].upper().strip() == text.upper().strip()

        print(f"  With Fusion:    '{result_with['result']}' {'✓' if match_with else '✗'} ({result_with['time']:.2f}s)")
        print(f"  Without Fusion: '{result_without['result']}' {'✓' if match_without else '✗'} ({result_without['time']:.2f}s)")

        results.append({
            'original': text,
            'with_fusion': {
                'result': result_with['result'],
                'match': match_with,
                'time': result_with['time']
            },
            'without_fusion': {
                'result': result_without['result'],
                'match': match_without,
                'time': result_without['time']
            }
        })

    return results

def print_summary(all_results):
    """打印總結報告"""
    print("\n" + "=" * 80)
    print("測試總結報告")
    print("=" * 80)

    # 基礎識別統計
    if 'basic' in all_results:
        basic = all_results['basic']
        total = len(basic)
        matched = sum(1 for r in basic if r['match'])
        avg_time = sum(r['time'] for r in basic) / total if total > 0 else 0

        print(f"\n基礎識別測試:")
        print(f"  總數: {total}")
        print(f"  成功: {matched}")
        print(f"  準確率: {matched/total*100:.1f}%")
        print(f"  平均耗時: {avg_time:.2f}s")

    # 字體大小統計
    if 'font_sizes' in all_results:
        font_results = all_results['font_sizes']
        print(f"\n字體大小測試:")
        for r in font_results:
            status = "✓" if r['match'] else "✗"
            print(f"  {status} 字體 {r['font_size']}: {r['recognized']} ({r['time']:.2f}s)")

    # 噪音測試統計
    if 'noise' in all_results:
        noise_results = all_results['noise']
        print(f"\n噪音干擾測試:")
        for r in noise_results:
            status = "✓" if r['match'] else "✗"
            print(f"  {status} 噪音 {r['noise_level']:.2f}: {r['recognized']} ({r['time']:.2f}s)")

    # Fusion 對比統計
    if 'comparison' in all_results:
        comp_results = all_results['comparison']
        with_fusion_matched = sum(1 for r in comp_results if r['with_fusion']['match'])
        without_fusion_matched = sum(1 for r in comp_results if r['without_fusion']['match'])
        total = len(comp_results)

        print(f"\nFusion 對比測試:")
        print(f"  總數: {total}")
        print(f"  With Fusion 成功: {with_fusion_matched} ({with_fusion_matched/total*100:.1f}%)")
        print(f"  Without Fusion 成功: {without_fusion_matched} ({without_fusion_matched/total*100:.1f}%)")

        with_time = sum(r['with_fusion']['time'] for r in comp_results) / total
        without_time = sum(r['without_fusion']['time'] for r in comp_results) / total
        print(f"  With Fusion 平均耗時: {with_time:.2f}s")
        print(f"  Without Fusion 平均耗時: {without_time:.2f}s")

def save_results(results, output_path=None):
    """保存結果到 JSON 檔案"""
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "ocr_evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 結果已保存到: {output_path}")

def main():
    try:
        print("OCR 測試效果評估")
        print("=" * 80)

        all_results = {}

        # 執行各項測試
        all_results['basic'] = test_basic_recognition()
        all_results['font_sizes'] = test_font_sizes()
        all_results['noise'] = test_with_noise()
        all_results['comparison'] = test_fusion_comparison()

        # 打印總結
        print_summary(all_results)

        # 保存結果
        save_results(all_results)

        print("\n" + "=" * 80)
        print("✓ 所有評估測試完成！")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
完整的 OCR 評估流程
整合 Pure OCR 和 GFD with LLM 的結果，並生成 CSV 報告
"""
import sys
import os
import json
import csv
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
import re
import string

# 將 repo 根目錄加入 Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gfd.gfd_chinese import ChineseOCRBreezper
from gfd.utils import process_config, combine_config

def remove_punctuation(text):
    """移除標點符號（包含中文和英文標點符號）"""
    # 中文標點符號
    chinese_punctuation = '，。！？；：「」『』（）〔〕【】《》〈〉、·—…～'
    # 英文標點符號
    english_punctuation = string.punctuation
    # 合併所有標點符號
    all_punctuation = chinese_punctuation + english_punctuation

    # 移除所有標點符號
    pattern = f"[{re.escape(all_punctuation)}]"
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

def levenshtein_distance(s1, s2):
    """計算編輯距離"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_cer(ref, pred, remove_punct=True):
    """計算 CER"""
    ref_str = str(ref)
    pred_str = str(pred)

    # 如果需要，移除標點符號
    if remove_punct:
        ref_str = remove_punctuation(ref_str)
        pred_str = remove_punctuation(pred_str)

    # 如果移除標點符號後為空，返回0
    if len(ref_str) == 0:
        return 0.0

    dist = levenshtein_distance(ref_str, pred_str)
    return dist / len(ref_str)

def load_test_data():
    """載入所有測試資料"""
    test_data = []

    # 1. 載入 inference_example (42個區域)
    print("="*80)
    print("載入測試資料...")
    print("="*80)

    image_path = os.path.join(REPO_ROOT, "test_data/inference_example/rec_1_id_2226.jpg")
    json_path = os.path.join(REPO_ROOT, "test_data/inference_example/rec_1_id_2226.json")

    if os.path.exists(image_path) and os.path.exists(json_path):
        image = Image.open(image_path).convert("RGB")
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        shapes = annotation.get("shapes", [])
        print(f"✓ inference_example: {len(shapes)} 個文字區域")

        for idx, shape in enumerate(shapes):
            text = shape.get("text", "")
            points = shape.get("points", [])

            if not text or not points:
                continue

            # 計算 bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            left, top = int(min(x_coords)), int(min(y_coords))
            right, bottom = int(max(x_coords)), int(max(y_coords))

            if right <= left or bottom <= top:
                continue

            # 裁切圖片
            crop = image.crop((left, top, right, bottom))

            if crop.size[0] < 10 or crop.size[1] < 10:
                continue

            test_data.append({
                "source": "inference_example",
                "image_name": "rec_1_id_2226.jpg",
                "region_id": idx,
                "bbox": f"[{left},{top},{right},{bottom}]",
                "ground_truth": text,
                "image": crop
            })

    # 2. 載入 test_samples_40 (40個單字圖片)
    samples_dir = os.path.join(REPO_ROOT, "test_data/test_samples_40")
    labels_path = os.path.join(samples_dir, "labels.json")

    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        print(f"✓ test_samples_40: {len(samples)} 個單字樣本")

        for sample in samples:
            img_path = os.path.join(samples_dir, sample['image'])
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                test_data.append({
                    "source": "test_samples_40",
                    "image_name": sample['image'],
                    "region_id": sample['index'],
                    "bbox": "N/A",
                    "ground_truth": sample['label'],
                    "image": image
                })

    print(f"\n總共載入 {len(test_data)} 個測試樣本\n")
    return test_data

def run_inference(test_data, use_llm=True, num_beams=5):
    """運行推理"""
    mode_name = "GFD with LLM" if use_llm else "Pure OCR"
    print("="*80)
    print(f"階段: {mode_name} 推理")
    print("="*80)

    # 載入模型
    print(f"\n載入 {mode_name} 模型...")
    model_config = process_config(os.path.join(REPO_ROOT, 'config_files/model/gfd-ocr-multilmdb-zhtw.yaml'))
    prompt_config = process_config(os.path.join(REPO_ROOT, 'config_files/prompt/ocr-zhtw-prompt.yaml'))
    combined_config = combine_config(prompt_config, model_config)

    # 如果不使用 LLM，將 fusing_r 設為 0
    if not use_llm:
        combined_config = combined_config._replace(fusing_r=0.0)

    print(f"  - Fusing ratio: {combined_config.fusing_r}")
    print(f"  - Num beams: {num_beams}")

    model = ChineseOCRBreezper(combined_config)
    print("✓ 模型載入完成\n")

    # 推理
    results = []
    for item in tqdm(test_data, desc=f"{mode_name} 推理中"):
        try:
            pred_text = model.get_transcription(
                item['image'],
                num_beams=num_beams,
                ocr_prompt=combined_config.ocr_prompt,
                llm_prompt=combined_config.llm_prompt
            )

            # 移除空格
            pred_text = pred_text.replace(' ', '')

            results.append({
                **item,
                "prediction": pred_text,
                "prediction_no_punct": remove_punctuation(pred_text),
                "cer": compute_cer(item['ground_truth'], pred_text, remove_punct=True)
            })

        except Exception as e:
            print(f"\n❌ 推理錯誤: {e}")
            results.append({
                **item,
                "prediction": "",
                "prediction_no_punct": "",
                "cer": 1.0
            })

    return results

def generate_comparison_csv(pure_ocr_results, gfd_llm_results, output_path):
    """生成比較 CSV"""
    print("\n" + "="*80)
    print("生成比較 CSV...")
    print("="*80)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'source', 'image_name', 'region_id', 'bbox',
            'ground_truth', 'ground_truth_no_punct',
            'pure_ocr_pred', 'pure_ocr_pred_no_punct', 'pure_ocr_cer',
            'gfd_llm_pred', 'gfd_llm_pred_no_punct', 'gfd_llm_cer',
            'cer_improvement', 'match_pure_ocr', 'match_gfd_llm'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pure, gfd in zip(pure_ocr_results, gfd_llm_results):
            gt_no_punct = remove_punctuation(pure['ground_truth'])
            cer_improvement = pure['cer'] - gfd['cer']

            writer.writerow({
                'source': pure['source'],
                'image_name': pure['image_name'],
                'region_id': pure['region_id'],
                'bbox': pure['bbox'],
                'ground_truth': pure['ground_truth'],
                'ground_truth_no_punct': gt_no_punct,
                'pure_ocr_pred': pure['prediction'],
                'pure_ocr_pred_no_punct': pure['prediction_no_punct'],
                'pure_ocr_cer': f"{pure['cer']:.4f}",
                'gfd_llm_pred': gfd['prediction'],
                'gfd_llm_pred_no_punct': gfd['prediction_no_punct'],
                'gfd_llm_cer': f"{gfd['cer']:.4f}",
                'cer_improvement': f"{cer_improvement:.4f}",
                'match_pure_ocr': gt_no_punct == pure['prediction_no_punct'],
                'match_gfd_llm': gt_no_punct == gfd['prediction_no_punct']
            })

    print(f"✓ 比較 CSV 已保存至: {output_path}")

def generate_summary_csv(pure_ocr_results, gfd_llm_results, output_path):
    """生成統計摘要 CSV"""
    print("\n生成統計摘要 CSV...")

    # 計算統計
    pure_cer = sum(r['cer'] for r in pure_ocr_results) / len(pure_ocr_results)
    gfd_cer = sum(r['cer'] for r in gfd_llm_results) / len(gfd_llm_results)

    pure_matches = sum(1 for r in pure_ocr_results if remove_punctuation(r['ground_truth']) == r['prediction_no_punct'])
    gfd_matches = sum(1 for r in gfd_llm_results if remove_punctuation(r['ground_truth']) == r['prediction_no_punct'])

    total_samples = len(pure_ocr_results)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['mode', 'total_samples', 'average_cer', 'perfect_matches', 'match_rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            'mode': 'Pure OCR',
            'total_samples': total_samples,
            'average_cer': f"{pure_cer:.4f}",
            'perfect_matches': pure_matches,
            'match_rate': f"{pure_matches/total_samples:.4f}"
        })

        writer.writerow({
            'mode': 'GFD with LLM',
            'total_samples': total_samples,
            'average_cer': f"{gfd_cer:.4f}",
            'perfect_matches': gfd_matches,
            'match_rate': f"{gfd_matches/total_samples:.4f}"
        })

        writer.writerow({
            'mode': 'Improvement',
            'total_samples': total_samples,
            'average_cer': f"{pure_cer - gfd_cer:.4f}",
            'perfect_matches': gfd_matches - pure_matches,
            'match_rate': f"{(gfd_matches - pure_matches)/total_samples:.4f}"
        })

    print(f"✓ 統計摘要 CSV 已保存至: {output_path}")

def generate_analysis_report(pure_ocr_results, gfd_llm_results, output_path):
    """生成分析報告"""
    print("\n生成分析報告...")

    # 計算統計
    total_samples = len(pure_ocr_results)
    pure_cer = sum(r['cer'] for r in pure_ocr_results) / total_samples
    gfd_cer = sum(r['cer'] for r in gfd_llm_results) / total_samples

    pure_matches = sum(1 for r in pure_ocr_results if remove_punctuation(r['ground_truth']) == r['prediction_no_punct'])
    gfd_matches = sum(1 for r in gfd_llm_results if remove_punctuation(r['ground_truth']) == r['prediction_no_punct'])

    # 找出改善和惡化的案例
    improvements = []
    degradations = []

    for pure, gfd in zip(pure_ocr_results, gfd_llm_results):
        cer_diff = pure['cer'] - gfd['cer']
        if cer_diff > 0.1:  # 改善超過10%
            improvements.append((pure, gfd, cer_diff))
        elif cer_diff < -0.1:  # 惡化超過10%
            degradations.append((pure, gfd, cer_diff))

    improvements.sort(key=lambda x: x[2], reverse=True)
    degradations.sort(key=lambda x: x[2])

    # 寫入報告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# OCR 評估分析報告\n\n")
        f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 總體統計\n\n")
        f.write(f"- **總樣本數**: {total_samples}\n")
        f.write(f"- **Pure OCR 平均 CER**: {pure_cer:.4f} ({pure_cer*100:.2f}%)\n")
        f.write(f"- **GFD with LLM 平均 CER**: {gfd_cer:.4f} ({gfd_cer*100:.2f}%)\n")
        f.write(f"- **CER 改善**: {(pure_cer - gfd_cer):.4f} ({(pure_cer - gfd_cer)*100:.2f}%)\n\n")

        f.write(f"- **Pure OCR 完全匹配**: {pure_matches}/{total_samples} ({pure_matches/total_samples*100:.2f}%)\n")
        f.write(f"- **GFD with LLM 完全匹配**: {gfd_matches}/{total_samples} ({gfd_matches/total_samples*100:.2f}%)\n")
        f.write(f"- **匹配率改善**: {(gfd_matches - pure_matches)/total_samples*100:.2f}%\n\n")

        f.write("## 改善案例 (Top 10)\n\n")
        for i, (pure, gfd, diff) in enumerate(improvements[:10], 1):
            f.write(f"### {i}. {pure['image_name']} (Region {pure['region_id']})\n\n")
            f.write(f"- **Ground Truth**: {pure['ground_truth']}\n")
            f.write(f"- **Pure OCR**: {pure['prediction']} (CER: {pure['cer']:.4f})\n")
            f.write(f"- **GFD with LLM**: {gfd['prediction']} (CER: {gfd['cer']:.4f})\n")
            f.write(f"- **改善**: {diff:.4f}\n\n")

        f.write("## 惡化案例 (Top 10)\n\n")
        for i, (pure, gfd, diff) in enumerate(degradations[:10], 1):
            f.write(f"### {i}. {pure['image_name']} (Region {pure['region_id']})\n\n")
            f.write(f"- **Ground Truth**: {pure['ground_truth']}\n")
            f.write(f"- **Pure OCR**: {pure['prediction']} (CER: {pure['cer']:.4f})\n")
            f.write(f"- **GFD with LLM**: {gfd['prediction']} (CER: {gfd['cer']:.4f})\n")
            f.write(f"- **惡化**: {abs(diff):.4f}\n\n")

    print(f"✓ 分析報告已保存至: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="完整的 OCR 評估流程")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam search 寬度")
    parser.add_argument("--output_dir", type=str, default=None, help="輸出目錄")
    args = parser.parse_args()

    # 創建輸出目錄
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw_results"), exist_ok=True)

    print("\n" + "="*80)
    print("完整 OCR 評估流程")
    print("="*80)
    print(f"輸出目錄: {output_dir}\n")

    try:
        # 階段1: 載入測試資料
        test_data = load_test_data()

        # 階段2: 運行 Pure OCR 推理
        pure_ocr_results = run_inference(test_data, use_llm=False, num_beams=args.num_beams)

        # 保存 Pure OCR 原始結果
        with open(os.path.join(output_dir, "raw_results", "pure_ocr_results.json"), 'w', encoding='utf-8') as f:
            json.dump([{k: v for k, v in r.items() if k != 'image'} for r in pure_ocr_results],
                     f, ensure_ascii=False, indent=2)

        # 階段3: 運行 GFD with LLM 推理
        gfd_llm_results = run_inference(test_data, use_llm=True, num_beams=args.num_beams)

        # 保存 GFD with LLM 原始結果
        with open(os.path.join(output_dir, "raw_results", "gfd_llm_results.json"), 'w', encoding='utf-8') as f:
            json.dump([{k: v for k, v in r.items() if k != 'image'} for r in gfd_llm_results],
                     f, ensure_ascii=False, indent=2)

        # 階段4: 生成比較 CSV
        generate_comparison_csv(
            pure_ocr_results,
            gfd_llm_results,
            os.path.join(output_dir, "comparison.csv")
        )

        # 階段5: 生成統計摘要 CSV
        generate_summary_csv(
            pure_ocr_results,
            gfd_llm_results,
            os.path.join(output_dir, "summary.csv")
        )

        # 階段6: 生成分析報告
        generate_analysis_report(
            pure_ocr_results,
            gfd_llm_results,
            os.path.join(output_dir, "analysis_report.md")
        )

        print("\n" + "="*80)
        print("✅ 評估完成！")
        print("="*80)
        print(f"\n結果保存至: {output_dir}/")
        print(f"  - comparison.csv: 逐筆比較結果")
        print(f"  - summary.csv: 統計摘要")
        print(f"  - analysis_report.md: 詳細分析報告")
        print(f"  - raw_results/: 原始JSON結果\n")

        return 0

    except Exception as e:
        print(f"\n❌ 評估失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

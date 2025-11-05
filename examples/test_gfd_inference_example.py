#!/usr/bin/env python3
"""
測試 GFD (Generative Fusion Decoding) 在 inference_example 上的效果
使用訓練好的 Multi-LMDB TrOCR 模型 + Breeze LLM
"""

import sys
import os
import json
from PIL import Image
from tqdm import tqdm
import torch

# 將 repo 根目錄加入 Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gfd.gfd_chinese import ChineseOCRBreezper
from gfd.utils import process_config, combine_config

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

def remove_punctuation(text):
    """移除標點符號（包含中文和英文標點符號）"""
    import re
    import string

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

def compute_cer(ref, pred, remove_punct=True):
    """計算 CER

    Args:
        ref: 參考文字
        pred: 預測文字
        remove_punct: 是否在計算前移除標點符號（預設為 True）
    """
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

def load_gfd_model(use_llm=True):
    """載入 GFD 模型"""
    print("="*80)
    print("載入 GFD 模型配置...")
    print("="*80)

    # 載入配置
    model_config = process_config(os.path.join(REPO_ROOT, 'config_files/model/gfd-ocr-multilmdb-zhtw.yaml'))
    prompt_config = process_config(os.path.join(REPO_ROOT, 'config_files/prompt/ocr-zhtw-prompt.yaml'))
    combined_config = combine_config(prompt_config, model_config)

    # 如果不使用 LLM，將 fusing_r 設為 0（namedtuple 是 immutable，使用 _replace()）
    if not use_llm:
        combined_config = combined_config._replace(fusing_r=0.0)
        print("⚠️  LLM 已停用 (fusing_r = 0.0)")

    print(f"✓ 配置載入成功")
    print(f"  - OCR model: {combined_config.ocr_model_path}")
    print(f"  - LLM model: {combined_config.llm_model_path}")
    print(f"  - OCR device: {combined_config.ocr_device}")
    print(f"  - LLM device: {combined_config.llm_device}")
    print(f"  - Fusing ratio: {combined_config.fusing_r}")
    print(f"  - OCR prompt: '{combined_config.ocr_prompt}'")
    print(f"  - LLM prompt: '{combined_config.llm_prompt}'")

    # 載入模型
    print("\n載入 ChineseOCRBreezper 模型...")
    try:
        model = ChineseOCRBreezper(combined_config)
        print("✓ ChineseOCRBreezper 模型載入成功\n")
        return model, combined_config
    except Exception as e:
        print(f"✗ 模型載入失敗: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_on_inference_example(model, config, num_beams=5, max_samples=10):
    """在 inference_example 上測試"""
    print("="*80)
    print(f"測試 inference_example (最多 {max_samples} 個區域)")
    print("="*80)

    # 載入圖片和標註
    image_path = os.path.join(REPO_ROOT, "test_data/inference_example/rec_1_id_2226.jpg")
    json_path = os.path.join(REPO_ROOT, "test_data/inference_example/rec_1_id_2226.json")

    if not os.path.exists(image_path):
        print(f"❌ 圖片不存在: {image_path}")
        return None

    if not os.path.exists(json_path):
        print(f"❌ JSON 標註不存在: {json_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)

    shapes = annotation.get("shapes", [])
    print(f"找到 {len(shapes)} 個文字區域")
    print(f"將測試前 {min(max_samples, len(shapes))} 個區域\n")

    results = []
    total_cer = 0
    total_chars = 0

    # 測試每個區域
    for idx, shape in enumerate(tqdm(shapes[:max_samples], desc="推理中")):
        text = shape.get("text", "")
        points = shape.get("points", [])

        if not text or not points:
            continue

        # 計算 bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        left, top = int(min(x_coords)), int(min(y_coords))
        right, bottom = int(max(x_coords)), int(max(y_coords))

        # 檢查有效性
        if right <= left or bottom <= top:
            continue

        # 裁切圖片
        crop = image.crop((left, top, right, bottom))

        if crop.size[0] < 10 or crop.size[1] < 10:
            continue

        # 使用 GFD 進行識別
        try:
            pred_text = model.get_transcription(
                crop,
                num_beams=num_beams,
                ocr_prompt=config.ocr_prompt,
                llm_prompt=config.llm_prompt
            )

            # 移除空格
            pred_text = pred_text.replace(' ', '')

            cer = compute_cer(text, pred_text)
            total_cer += cer * len(text)
            total_chars += len(text)

            result_item = {
                "region_id": idx,
                "shape_id": shape.get("id", ""),
                "bbox": [left, top, right, bottom],
                "ground_truth": text,
                "prediction": pred_text,
                "cer": float(cer),
                "match": text == pred_text
            }
            results.append(result_item)

            # 打印結果
            print(f"\n區域 {idx}:")
            print(f"  GT:   '{text}'")
            print(f"  Pred: '{pred_text}'")
            print(f"  CER:  {cer:.4f} {'✅' if text == pred_text else '❌'}")

        except Exception as e:
            print(f"❌ 區域 {idx} 預測錯誤: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 計算統計
    summary = {
        "total_regions": len(results),
        "total_characters": total_chars,
        "average_cer": total_cer / total_chars if total_chars > 0 else 0.0,
        "perfect_matches": sum(1 for r in results if r["match"]),
        "match_rate": sum(1 for r in results if r["match"]) / len(results) if results else 0.0
    }

    # 打印統計
    print(f"\n{'='*80}")
    print(f"統計摘要")
    print(f"{'='*80}")
    print(f"總區域數: {summary['total_regions']}")
    print(f"總字符數: {summary['total_characters']}")
    print(f"平均 CER: {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)")
    print(f"完全匹配: {summary['perfect_matches']}/{summary['total_regions']} ({summary['match_rate']*100:.2f}%)")
    print(f"{'='*80}\n")

    return {
        "results": results,
        "summary": summary
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="測試 GFD 在 inference_example 上的效果")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam search 寬度")
    parser.add_argument("--max_samples", type=int, default=10, help="最大測試樣本數")
    parser.add_argument("--no_llm", action="store_true", help="不使用 LLM (僅 OCR)")
    parser.add_argument("--output", type=str, default="gfd_results.json", help="輸出結果檔案")
    args = parser.parse_args()

    try:
        # 載入模型
        use_llm = not args.no_llm
        model, config = load_gfd_model(use_llm=use_llm)

        # 在 inference_example 上測試
        results = test_on_inference_example(
            model,
            config,
            num_beams=args.num_beams,
            max_samples=args.max_samples
        )

        if results:
            # 保存結果
            results["config"] = {
                "ocr_model": config.ocr_model_path,
                "llm_model": config.llm_model_path if use_llm else "None",
                "fusing_r": config.fusing_r,
                "num_beams": args.num_beams,
                "use_llm": use_llm
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"✅ 結果已保存至: {args.output}")

        print("\n" + "="*80)
        print("✓ 測試完成！")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

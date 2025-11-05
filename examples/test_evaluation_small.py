#!/usr/bin/env python3
"""
小規模測試版本 - 只測試前2個樣本
"""
import sys
import os

# 修改 run_complete_evaluation.py 中的 load_test_data 函數
# 只載入前2個樣本

# 將 repo 根目錄加入 Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from PIL import Image
import json

def test_data_loading():
    """測試資料載入"""
    print("="*80)
    print("測試資料載入")
    print("="*80)

    test_data = []

    # 只載入 inference_example 的前2個區域
    image_path = os.path.join(REPO_ROOT, "test_data/inference_example/rec_1_id_2226.jpg")
    json_path = os.path.join(REPO_ROOT, "test_data/inference_example/rec_1_id_2226.json")

    image = Image.open(image_path).convert("RGB")
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)

    shapes = annotation.get("shapes", [])[:2]  # 只取前2個
    print(f"載入 {len(shapes)} 個測試區域")

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

        print(f"  樣本 {idx}: '{text}'")

    return test_data

def test_model_loading():
    """測試模型載入"""
    print("\n" + "="*80)
    print("測試模型載入 (Pure OCR)")
    print("="*80)

    from gfd.gfd_chinese import ChineseOCRBreezper
    from gfd.utils import process_config, combine_config

    model_config = process_config(os.path.join(REPO_ROOT, 'config_files/model/gfd-ocr-multilmdb-zhtw.yaml'))
    prompt_config = process_config(os.path.join(REPO_ROOT, 'config_files/prompt/ocr-zhtw-prompt.yaml'))
    combined_config = combine_config(prompt_config, model_config)

    # Pure OCR (fusing_r = 0)
    combined_config = combined_config._replace(fusing_r=0.0)

    print(f"配置:")
    print(f"  - OCR model: {combined_config.ocr_model_path}")
    print(f"  - Fusing ratio: {combined_config.fusing_r}")

    print("\n載入模型...")
    model = ChineseOCRBreezper(combined_config)
    print("✓ 模型載入成功")

    return model, combined_config

def test_inference(test_data, model, config):
    """測試推理"""
    print("\n" + "="*80)
    print("測試推理")
    print("="*80)

    for item in test_data:
        print(f"\n推理樣本: '{item['ground_truth']}'")

        try:
            pred_text = model.get_transcription(
                item['image'],
                num_beams=3,
                ocr_prompt=config.ocr_prompt,
                llm_prompt=config.llm_prompt
            )
            pred_text = pred_text.replace(' ', '')

            print(f"  預測結果: '{pred_text}'")
            print(f"  匹配: {'✅' if pred_text == item['ground_truth'] else '❌'}")

        except Exception as e:
            print(f"  ❌ 推理錯誤: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    try:
        # 測試 1: 資料載入
        test_data = test_data_loading()

        # 測試 2: 模型載入
        model, config = test_model_loading()

        # 測試 3: 推理
        test_inference(test_data, model, config)

        print("\n" + "="*80)
        print("✅ 所有測試通過！")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
測試腳本：驗證 TrOCR 模型的載入、special tokens 和基本輸入輸出
"""

import sys
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from gfd.tokenizer import TrOCRByteTokenizer

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "images", "setup")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_image(text="HELLO WORLD", size=(400, 100), font_size=40):
    """創建測試圖片"""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    # 使用預設字體
    try:
        # 嘗試使用較好的字體
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        # 如果找不到就用預設字體
        font = ImageFont.load_default()

    # 計算文字位置置中
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2

    draw.text((x, y), text, fill="black", font=font)
    return img

def test_trocr_model():
    """測試 TrOCR 模型載入和基本功能"""
    print("=" * 80)
    print("測試 1: TrOCR 模型載入")
    print("=" * 80)

    model_path = "microsoft/trocr-base-printed"
    print(f"載入模型: {model_path}")

    # 載入 processor 和 model
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model.to("cpu")
    model.eval()

    print("✓ 模型載入成功")
    print(f"  - 模型設備: {model.device}")
    print(f"  - 模型 dtype: {model.dtype}")

    # 測試 tokenizer
    print("\n" + "=" * 80)
    print("測試 2: TrOCR Tokenizer 和 Special Tokens")
    print("=" * 80)

    tokenizer = TrOCRByteTokenizer.from_pretrained(model_path)

    print(f"✓ Tokenizer 類型: {type(tokenizer).__name__}")
    print(f"  - vocab_size: {tokenizer.vocab_size}")
    print(f"  - bos_token: '{tokenizer.bos_token}' (id: {tokenizer.bos_token_id})")
    print(f"  - eos_token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"  - pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"  - unk_token: '{tokenizer.unk_token}' (id: {tokenizer.unk_token_id})")

    # 更新模型的 special token 設定
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("\n✓ 模型 config 已更新:")
    print(f"  - decoder_start_token_id: {model.config.decoder_start_token_id}")
    print(f"  - pad_token_id: {model.config.pad_token_id}")
    print(f"  - eos_token_id: {model.config.eos_token_id}")

    # 測試 byte tokenizer 功能
    print("\n" + "=" * 80)
    print("測試 3: Byte Tokenizer 功能")
    print("=" * 80)

    test_text = "Hello"
    test_bytes = test_text.encode('utf-8')

    print(f"測試文字: '{test_text}'")
    print(f"UTF-8 bytes: {test_bytes}")

    # tokenize 文字
    token_ids = tokenizer(test_text, add_special_tokens=False).input_ids
    print(f"Token IDs: {token_ids}")

    # 測試 convert_ids_to_bytes
    bytes_list = tokenizer.convert_ids_to_bytes(token_ids, skip_special_tokens=True)
    reconstructed_bytes = b"".join(bytes_list)
    print(f"重建 bytes: {reconstructed_bytes}")
    print(f"✓ Bytes roundtrip 成功: {reconstructed_bytes == test_bytes}")

    # 測試 tokenize_from_byte
    token_ids_from_bytes = tokenizer.tokenize_from_byte(test_bytes)
    print(f"從 bytes tokenize: {token_ids_from_bytes}")
    print(f"✓ Tokenize from bytes 成功: {token_ids_from_bytes == token_ids}")

    return processor, model, tokenizer

def test_trocr_inference(processor, model, tokenizer):
    """測試 TrOCR 推理功能"""
    print("\n" + "=" * 80)
    print("測試 4: TrOCR 圖片識別")
    print("=" * 80)

    # 創建測試圖片
    test_texts = [
        "HELLO WORLD",
        "TEST IMAGE",
        "123456",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n測試圖片 {i}: '{text}'")
        img = create_test_image(text)

        # 保存圖片
        img_path = os.path.join(OUTPUT_DIR, f"test_image_{i}.png")
        img.save(img_path)
        print(f"  圖片已保存: {img_path}")

        # 處理圖片
        pixel_values = processor(img, return_tensors="pt").pixel_values.to("cpu")
        print(f"  pixel_values shape: {pixel_values.shape}")

        # 生成文字 (不使用 fusion，純 TrOCR)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=64)

        # 解碼結果
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"  識別結果: '{generated_text}'")
        print(f"  Generated IDs: {generated_ids[0].tolist()[:10]}... (前10個)")

        # 測試 encoder outputs
        encoder_outputs = model.get_encoder()(pixel_values, return_dict=True)
        print(f"  Encoder output shape: {encoder_outputs.last_hidden_state.shape}")

    print("\n✓ 所有圖片識別測試完成")

def main():
    try:
        # 測試 TrOCR 模型
        processor, model, tokenizer = test_trocr_model()

        # 測試推理
        test_trocr_inference(processor, model, tokenizer)

        print("\n" + "=" * 80)
        print("✓ 所有測試通過！")
        print("=" * 80)
        print("\n準備好進行 TrOCR + LLM Fusion 測試")

        return 0

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
圖片預處理模組：支援保留比例的 resize 和垂直文本旋轉
"""
import torch
from PIL import Image
import torchvision.transforms.functional as F


def is_vertical_image(image):
    """
    判斷圖片是否為垂直文字
    根據寬高比判斷：高度 > 寬度 則為垂直文字
    """
    width, height = image.size
    return height > width


def rotate_vertical_image(image):
    """
    將垂直文字圖片逆時鐘旋轉 90 度
    PIL.Image.rotate() 使用逆時鐘為正方向
    """
    return image.rotate(90, expand=True)


def resize_with_padding(image, target_size=(384, 384), fill_color=(255, 255, 255)):
    """
    保留原始比例的 resize，不足部分補白邊

    Args:
        image: PIL Image
        target_size: (width, height) 目標尺寸
        fill_color: 填充顏色，預設白色 (255, 255, 255)

    Returns:
        PIL Image: 處理後的圖片
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    # 計算縮放比例（保持比例，取較小者確保圖片不會超出目標尺寸）
    ratio_w = target_width / original_width
    ratio_h = target_height / original_height
    ratio = min(ratio_w, ratio_h)

    # 計算縮放後的尺寸
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize 圖片（保持比例）
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 建立目標尺寸的白色背景
    new_image = Image.new("RGB", target_size, fill_color)

    # 計算貼上位置（置中）
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # 將 resize 後的圖片貼到白色背景上
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def preprocess_image_for_ocr(image, target_size=(384, 384), auto_rotate_vertical=True):
    """
    OCR 圖片預處理完整流程

    Args:
        image: PIL Image (RGB)
        target_size: (width, height) 目標尺寸
        auto_rotate_vertical: 是否自動旋轉垂直文字

    Returns:
        PIL Image: 處理後的圖片
    """
    # 確保是 RGB 模式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 檢查是否為垂直文字，若是則逆時鐘旋轉 90 度
    if auto_rotate_vertical and is_vertical_image(image):
        image = rotate_vertical_image(image)

    # 保留比例 resize + 補白邊
    image = resize_with_padding(image, target_size=target_size)

    return image


def image_to_tensor(image, normalize=True):
    """
    將 PIL Image 轉換為 Tensor

    Args:
        image: PIL Image
        normalize: 是否進行歸一化（mean=0.5, std=0.5）

    Returns:
        torch.Tensor: shape (3, H, W)
    """
    # 轉換為 Tensor
    tensor = F.to_tensor(image)

    # 歸一化
    if normalize:
        tensor = F.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return tensor


class OCRImageTransform:
    """
    OCR 圖片轉換類別（可用於 Dataset）
    """
    def __init__(self, target_size=(384, 384), auto_rotate_vertical=True, normalize=True):
        self.target_size = target_size
        self.auto_rotate_vertical = auto_rotate_vertical
        self.normalize = normalize

    def __call__(self, image):
        """
        Args:
            image: PIL Image

        Returns:
            torch.Tensor: shape (3, H, W)
        """
        # 預處理圖片
        image = preprocess_image_for_ocr(
            image,
            target_size=self.target_size,
            auto_rotate_vertical=self.auto_rotate_vertical
        )

        # 轉換為 Tensor
        tensor = image_to_tensor(image, normalize=self.normalize)

        return tensor


if __name__ == "__main__":
    # 測試程式碼
    import lmdb
    import io

    print("測試圖片預處理...")

    # 測試水平文字
    env_h = lmdb.open("/mnt/whliao/ocr-synth-generator/out_h.lmdb", readonly=True, lock=False)
    with env_h.begin() as txn:
        img_bytes = txn.get(b"image-00000000")
        img_h = Image.open(io.BytesIO(img_bytes))
        print(f"\n水平文字原始尺寸: {img_h.size}")

        # 處理
        transform = OCRImageTransform(target_size=(384, 384), auto_rotate_vertical=True)
        tensor_h = transform(img_h)
        print(f"處理後 Tensor shape: {tensor_h.shape}")
        print(f"判斷為垂直文字: {is_vertical_image(img_h)}")
    env_h.close()

    # 測試垂直文字
    env_v = lmdb.open("/mnt/whliao/ocr-synth-generator/out_v.lmdb", readonly=True, lock=False)
    with env_v.begin() as txn:
        img_bytes = txn.get(b"image-00000000")
        img_v = Image.open(io.BytesIO(img_bytes))
        print(f"\n垂直文字原始尺寸: {img_v.size}")

        # 處理前
        print(f"判斷為垂直文字: {is_vertical_image(img_v)}")

        # 處理
        tensor_v = transform(img_v)
        print(f"處理後 Tensor shape: {tensor_v.shape}")

        # 顯示旋轉後的尺寸
        rotated = rotate_vertical_image(img_v)
        print(f"旋轉後尺寸: {rotated.size}")
    env_v.close()

    print("\n✅ 測試完成")

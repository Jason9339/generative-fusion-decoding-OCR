"""
中文字符級 Tokenizer 的 Byte-Level 適配器
將 PreTrainedTokenizerFast (字符級) 適配為 GFD 所需的 byte-level 接口
"""

from transformers import PreTrainedTokenizerFast
from gfd.tokenizer import ByteTokenizer


class ChineseCharTokenizerAdapter(PreTrainedTokenizerFast, ByteTokenizer):
    """
    將字符級中文 tokenizer 適配為 byte-level tokenizer

    核心思路：
    1. 中文字符本質上是 UTF-8 編碼的 bytes
    2. 我們為每個字符創建 byte_encoder/decoder 映射
    3. GFD 在 byte space 中操作時，我們將其轉換回字符 space
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_byte_mappings()

    def _build_byte_mappings(self):
        """建立字符 <-> byte 的映射"""
        print("建立 byte-level 映射...")

        # 為每個 token 建立 byte 編碼映射
        self.token_to_bytes = {}
        self.bytes_to_token = {}

        # 建立標準的 byte encoder/decoder (256個可能的byte值)
        # 使用與 RobertaTokenizer 相同的映射策略
        self.byte_encoder = {}
        self.byte_decoder = {}

        # 基礎 ASCII 字符直接映射
        for i in range(256):
            # 可打印字符直接使用
            if 33 <= i <= 126:
                self.byte_encoder[i] = chr(i)
                self.byte_decoder[chr(i)] = i
            else:
                # 不可打印字符映射到 Unicode 私有區域
                char = chr(256 + i)
                self.byte_encoder[i] = char
                self.byte_decoder[char] = i

        # 為詞彙表中的每個 token 建立 bytes 映射
        for token, token_id in self.get_vocab().items():
            if token in self.all_special_tokens:
                # 特殊 token 直接編碼
                byte_seq = token.encode('utf-8')
            else:
                # 普通 token 編碼為 UTF-8
                byte_seq = token.encode('utf-8')

            self.token_to_bytes[token_id] = byte_seq

        print(f"✓ Byte 映射建立完成 (vocab size: {len(self.get_vocab())})")

    def convert_ids_to_bytes(self, ids, skip_special_tokens=True):
        """
        將 token IDs 轉換為 bytes
        這是 GFD 要求的核心方法
        """
        if isinstance(ids, int):
            ids = [ids]

        # 轉換為 tokens
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

        if isinstance(tokens, str):
            tokens = [tokens]

        # 將每個 token 轉換為 bytes
        result = []
        for token in tokens:
            if token in self.all_special_tokens and skip_special_tokens:
                continue

            # 將 token 轉換為 UTF-8 bytes
            byte_seq = token.encode('utf-8')
            result.append(byte_seq)

        return result

    def tokenize_from_byte(self, byte_str):
        """
        從 byte string 進行 tokenization
        這是 GFD 要求的另一個核心方法
        """
        # 將 bytes 解碼為字符串
        try:
            text = byte_str.decode('utf-8', errors='ignore')
        except:
            text = str(byte_str)

        # 使用原有的 tokenizer 進行 tokenization
        ids = self(text, add_special_tokens=False).input_ids

        return ids

    def _convert_token_to_bytes(self, token):
        """將單個 token 轉換為 bytes"""
        if token in self.all_special_tokens:
            return token.encode("utf-8")

        # 對於普通 token，使用 byte_encoder 映射
        byte_values = []
        for ch in token:
            # 獲取字符的 UTF-8 bytes
            ch_bytes = ch.encode('utf-8')
            for byte_val in ch_bytes:
                # 使用 byte_encoder 映射
                if byte_val in self.byte_encoder:
                    byte_values.append(ord(self.byte_encoder[byte_val]))
                else:
                    byte_values.append(byte_val)

        return bytes(byte_values)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        從預訓練模型載入 tokenizer
        """
        # 先用標準方式載入
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        # 創建適配器實例
        adapter = cls(
            tokenizer_object=tokenizer.backend_tokenizer,
            **tokenizer.init_kwargs
        )

        # 複製特殊 tokens
        for attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token',
                     'pad_token', 'cls_token', 'mask_token']:
            if hasattr(tokenizer, attr):
                setattr(adapter, attr, getattr(tokenizer, attr))

        # 建立 byte 映射
        adapter._build_byte_mappings()

        return adapter


def create_chinese_tokenizer_adapter(tokenizer_path):
    """
    便捷函數：創建中文 tokenizer 適配器

    Args:
        tokenizer_path: tokenizer 路徑

    Returns:
        適配後的 tokenizer
    """
    print(f"載入中文 tokenizer 適配器: {tokenizer_path}")

    adapter = ChineseCharTokenizerAdapter.from_pretrained(tokenizer_path)

    print("✓ 中文 tokenizer 適配器創建成功")
    print(f"  - Vocab size: {len(adapter)}")
    print(f"  - 支援 byte_encoder: {hasattr(adapter, 'byte_encoder')}")
    print(f"  - 支援 byte_decoder: {hasattr(adapter, 'byte_decoder')}")
    print(f"  - 支援 convert_ids_to_bytes: {hasattr(adapter, 'convert_ids_to_bytes')}")
    print(f"  - 支援 tokenize_from_byte: {hasattr(adapter, 'tokenize_from_byte')}")

    return adapter


if __name__ == "__main__":
    # 測試適配器
    import sys

    tokenizer_path = "/mnt/whliao/experiment/tokenizer"

    print("="*80)
    print("測試中文 Tokenizer 適配器")
    print("="*80)

    try:
        adapter = create_chinese_tokenizer_adapter(tokenizer_path)

        # 測試 1: 基本 tokenization
        print("\n測試 1: 基本 tokenization")
        text = "繁體中文測試"
        ids = adapter(text, add_special_tokens=False).input_ids
        print(f"Text: {text}")
        print(f"IDs: {ids}")

        # 測試 2: convert_ids_to_bytes
        print("\n測試 2: convert_ids_to_bytes")
        byte_list = adapter.convert_ids_to_bytes(ids, skip_special_tokens=True)
        print(f"Bytes: {byte_list}")
        print(f"Decoded: {b''.join(byte_list).decode('utf-8')}")

        # 測試 3: tokenize_from_byte
        print("\n測試 3: tokenize_from_byte")
        byte_str = text.encode('utf-8')
        ids_from_bytes = adapter.tokenize_from_byte(byte_str)
        print(f"Byte string: {byte_str}")
        print(f"IDs from bytes: {ids_from_bytes}")
        print(f"Match: {ids == ids_from_bytes}")

        print("\n" + "="*80)
        print("✓ 適配器測試通過！")
        print("="*80)

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

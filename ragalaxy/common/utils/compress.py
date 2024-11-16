from typing import List, Optional
import re
from transformers import AutoTokenizer

class CompressUtils:
    """压缩工具类"""
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_length: int = 2048
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def compress_text(
        self,
        text: str,
        target_length: Optional[int] = None
    ) -> str:
        """压缩文本到目标长度"""
        if target_length is None:
            target_length = self.max_length
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= target_length:
            return text
            
        # 按句子分割
        sentences = re.split(r'[.!?]+', text)
        compressed = []
        current_length = 0
        
        for sent in sentences:
            sent_tokens = self.tokenizer.encode(sent)
            if current_length + len(sent_tokens) > target_length:
                break
            compressed.append(sent)
            current_length += len(sent_tokens)
            
        return ' '.join(compressed)
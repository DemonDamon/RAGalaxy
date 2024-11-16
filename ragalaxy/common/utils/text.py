import re
from typing import List, Optional
import unicodedata

class TextUtils:
    """文本处理工具类"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """文本标准化"""
        # 统一为 NFKC 格式
        text = unicodedata.normalize('NFKC', text)
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def split_text(
        text: str,
        max_length: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """文本分段"""
        if len(text) <= max_length:
            return [text]
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # 寻找合适的分割点
            split_point = text.rfind(' ', start + max_length - overlap, end)
            if split_point == -1:
                split_point = end
            chunks.append(text[start:split_point])
            start = split_point
            
        return chunks
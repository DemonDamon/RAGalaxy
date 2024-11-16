from typing import List
import nltk
from ..base import BaseChunker

class SemanticChunker(BaseChunker):
    """基于语义的文本分块器"""
    
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        nltk.download('punkt', quiet=True)
        
    def split(self, text: str, **kwargs) -> List[str]:
        # 分句
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size:
                # 当前块已满,保存并创建新块
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
                
            current_chunk.append(sentence)
            current_size += sentence_size
            
        # 处理最后一个块
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
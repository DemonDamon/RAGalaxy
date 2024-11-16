import nltk
from typing import List
from ..base import BaseChunker

class SentenceChunker(BaseChunker):
    """基于句子的分块器"""
    
    def __init__(self, min_length: int = 50):
        self.min_length = min_length
        nltk.download('punkt', quiet=True)
        
    def split(self, text: str, **kwargs) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) >= self.min_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
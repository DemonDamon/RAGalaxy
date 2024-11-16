from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingUtils:
    """向量嵌入工具类"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        self.model = SentenceTransformer(model_name).to(device)
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """文本编码"""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embeddings
        
    def similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]]
    ) -> np.ndarray:
        """计算文本相似度"""
        emb1 = self.encode(texts1)
        emb2 = self.encode(texts2)
        return np.dot(emb1, emb2.T)
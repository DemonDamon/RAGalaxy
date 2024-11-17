from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .base import BaseSemanticRetriever

class SparseRetriever(BaseSemanticRetriever):
    """基于稀疏向量的检索实现"""
    
    def __init__(self, model_name: str, vector_store, **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.vector_store = vector_store

    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. 获取稀疏向量表示
        tokens = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 2. 向量检索
        results = await self.vector_store.search(
            tokens,
            top_k=top_k,
            **self.kwargs
        )
        
        return self._format_results(results)

    async def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        # 批量获取稀疏向量并检索
        batch_tokens = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        batch_results = await self.vector_store.batch_search(
            batch_tokens,
            top_k=top_k,
            **self.kwargs
        )
        
        return [self._format_results(results) for results in batch_results]
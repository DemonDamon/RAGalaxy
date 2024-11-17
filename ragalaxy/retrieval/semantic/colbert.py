from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
from .base import BaseSemanticRetriever

class ColBERTRetriever(BaseSemanticRetriever):
    def __init__(self, model_name: str, vector_store, **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.vector_store = vector_store

    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. 获取查询的 token 级别表示
        query_tokens = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        query_embeddings = self.model(**query_tokens).last_hidden_state
        
        # 2. 检索相似文档
        results = await self.vector_store.search(
            query_embeddings,
            top_k=top_k,
            **self.kwargs
        )
        
        return self._format_results(results)

    async def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        return [await self.retrieve(query, top_k) for query in queries]
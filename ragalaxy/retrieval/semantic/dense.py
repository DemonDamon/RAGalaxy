from typing import List, Dict, Any
import numpy as np
from .base import BaseSemanticRetriever

class DenseRetriever(BaseSemanticRetriever):
    """基于稠密向量的检索实现"""
    
    def __init__(self, embedding_model: str, vector_store, **kwargs):
        super().__init__(embedding_model, **kwargs)
        self.vector_store = vector_store

    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. 获取查询向量
        query_embedding = await self._get_embedding(query)
        
        # 2. 向量检索
        results = await self.vector_store.search(
            query_embedding,
            top_k=top_k,
            **self.kwargs
        )
        
        return self._format_results(results)

    async def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        # 1. 批量获取查询向量
        query_embeddings = await self._batch_get_embeddings(queries)
        
        # 2. 批量检索
        batch_results = await self.vector_store.batch_search(
            query_embeddings,
            top_k=top_k,
            **self.kwargs
        )
        
        return [self._format_results(results) for results in batch_results]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本向量"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.embedding_model)
        return model.encode(text)

    async def _batch_get_embeddings(self, texts: List[str]) -> np.ndarray:
        """批量获取文本向量"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.embedding_model)
        return model.encode(texts)

    def _format_results(self, results: List) -> List[Dict[str, Any]]:
        """格式化检索结果"""
        return [
            {
                'content': result['content'],
                'score': float(result['score'])
            }
            for result in results
        ]
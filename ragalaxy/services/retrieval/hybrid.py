from typing import List, Dict, Any
from .base import BaseRetriever, SearchResult
from common.config.base import BaseConfig

class HybridRetriever:
    """混合检索服务"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._init_retrievers()
        
    def _init_retrievers(self):
        """初始化检索器"""
        # 向量检索
        if self.config.get("retrieval.vector.enabled", True):
            from .vector import VectorRetriever
            self.retrievers["vector"] = VectorRetriever(
                self.config.get("retrieval.vector")
            )
            
        # BM25检索
        if self.config.get("retrieval.bm25.enabled", True):
            from .keyword import BM25Retriever
            self.retrievers["bm25"] = BM25Retriever(
                self.config.get("retrieval.bm25")
            )
            
        # 知识图谱检索
        if self.config.get("retrieval.kg.enabled", False):
            from .knowledge import KGRetriever
            self.retrievers["kg"] = KGRetriever(
                self.config.get("retrieval.kg")
            )
    
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """混合检索"""
        all_results = []
        
        # 并行调用各检索器
        import asyncio
        tasks = [
            retriever.search(query, top_k) 
            for retriever in self.retrievers.values()
        ]
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        for retriever_results in results:
            all_results.extend(retriever_results)
            
        # 重排序
        return self._rerank(all_results)[:top_k]
        
    def _rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """重排序结果"""
        # 1. 基于源权重
        weights = self.config.get("retrieval.weights", {
            "vector": 1.0,
            "bm25": 0.5,
            "kg": 0.8
        })
        
        # 2. 应用权重
        for result in results:
            result.score *= weights.get(result.source, 1.0)
            
        # 3. 按分数排序
        return sorted(results, key=lambda x: x.score, reverse=True) 
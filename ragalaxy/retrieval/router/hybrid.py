from typing import List, Dict, Any
import numpy as np
from .base import BaseRetrievalRouter

class HybridRouter(BaseRetrievalRouter):
    """混合检索路由实现"""
    
    def __init__(self, retrievers: Dict[str, Any], weights: Dict[str, float] = None, **kwargs):
        super().__init__(retrievers, **kwargs)
        self.weights = weights or {name: 1.0 for name in retrievers.keys()}

    def _adjust_weights(self, query: str) -> Dict[str, float]:
        """根据查询特征动态调整检索器权重"""
        weights = self.weights.copy()
        
        # 1. 查询长度特征
        if len(query.split()) > 10:
            # 长查询偏向语义检索
            weights['semantic'] = weights.get('semantic', 1.0) * 1.2
            weights['keyword'] = weights.get('keyword', 1.0) * 0.8
        else:
            # 短查询偏向关键词检索
            weights['semantic'] = weights.get('semantic', 1.0) * 0.8
            weights['keyword'] = weights.get('keyword', 1.0) * 1.2
        
        return weights

    async def route(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        # 动态调整权重
        adjusted_weights = self._adjust_weights(query)
        
        # 使用调整后的权重进行检索
        all_results = {}
        for name, retriever in self.retrievers.items():
            results = await retriever.retrieve(query, top_k=top_k)
            all_results[name] = results
        
        # 使用调整后的权重合并结果
        merged_results = self._merge_results(all_results, top_k, adjusted_weights)
        return merged_results

    def _merge_results(self, all_results: Dict[str, List[Dict]], top_k: int, weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """合并多个检索器的结果"""
        # 1. 标准化分数
        for name, results in all_results.items():
            scores = [r['score'] for r in results]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                for r in results:
                    r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score) if max_score > min_score else 0
                    r['weighted_score'] = r['normalized_score'] * weights[name]

        # 2. 合并所有结果
        merged = []
        for name, results in all_results.items():
            merged.extend(results)

        # 3. 按加权分数排序并返回top-k
        merged.sort(key=lambda x: x['weighted_score'], reverse=True)
        return merged[:top_k]
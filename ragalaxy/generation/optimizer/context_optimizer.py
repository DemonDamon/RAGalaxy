from typing import List, Dict, Any
import numpy as np

class ContextOptimizer:
    """上下文优化器,用于优化检索结果"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_tokens = config.get("max_tokens", 2000)
        self.min_similarity = config.get("min_similarity", 0.5)
        
    def optimize(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化上下文
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文列表,每个上下文包含text和score
            
        Returns:
            优化后的上下文列表
        """
        # 1. 按相关性分数过滤
        filtered_contexts = [
            ctx for ctx in contexts 
            if ctx.get("score", 0) > self.min_similarity
        ]
        
        # 2. 按相关性排序
        sorted_contexts = sorted(
            filtered_contexts,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        # 3. 控制总token数量
        optimized_contexts = []
        total_tokens = 0
        
        for ctx in sorted_contexts:
            tokens = len(ctx["text"].split())
            if total_tokens + tokens <= self.max_tokens:
                optimized_contexts.append(ctx)
                total_tokens += tokens
            else:
                break
                
        return optimized_contexts
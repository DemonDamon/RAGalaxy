from typing import List, Dict, Any
from .base_generator import BaseGenerator

class GenerationChain(BaseGenerator):
    """生成链实现,支持多步生成"""
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        # 1. 优化上下文
        optimized_contexts = self.optimizer.optimize(query, contexts)
        
        # 2. 构建提示词
        prompt = self._build_prompt(query, optimized_contexts)
        
        # 3. 生成回答
        response = self.llm.generate(prompt, **kwargs)
        
        return response
        
    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """构建提示词"""
        context_text = "\n".join([
            f"[{i+1}] {ctx['text']}" 
            for i, ctx in enumerate(contexts)
        ])
        
        return f"""Based on the following context:
{context_text}

Please answer the question: {query}"""
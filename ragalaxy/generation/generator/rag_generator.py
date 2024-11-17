from typing import Any, Dict, List, Optional
from .base_generator import BaseGenerator

class RAGGenerator(BaseGenerator):
    """RAG生成器实现"""
    
    def __init__(self, llm_provider, context_optimizer, config):
        super().__init__(llm_provider, context_optimizer, config)
        self.prompt_template = config.get("prompt_template", 
            "Based on the following context:\n{context}\n\n"
            "Please answer the question: {query}"
        )
        
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        # 1. 优化上下文
        optimized_contexts = self.optimizer.optimize(query, contexts)
        
        # 2. 构建提示词
        context_text = "\n".join([
            f"[{i+1}] {ctx['text']}" 
            for i, ctx in enumerate(optimized_contexts)
        ])
        
        prompt = self.prompt_template.format(
            context=context_text,
            query=query
        )
        
        # 3. 生成回复
        response = self.llm.generate(prompt, **kwargs)
        
        return response
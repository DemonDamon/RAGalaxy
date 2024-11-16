from typing import Dict, Any, Optional
import json
from string import Template

class PromptUtils:
    """提示词工具类"""
    
    @staticmethod
    def load_prompt_template(template_path: str) -> Dict[str, str]:
        """加载提示词模板"""
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def format_prompt(
        template: str,
        **kwargs: Any
    ) -> str:
        """格式化提示词"""
        return Template(template).safe_substitute(**kwargs)
    
    @staticmethod
    def build_rag_prompt(
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """构建RAG提示词"""
        if system_prompt:
            return f"{system_prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        return f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
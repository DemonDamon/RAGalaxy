from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BasePromptBuilder(ABC):
    """提示词构建器基类"""
    
    @abstractmethod
    def build_prompt(self, query: str, context: str) -> str:
        """构建提示词"""
        pass
    
    @abstractmethod
    def build_system_prompt(self) -> str:
        """构建系统提示词"""
        pass

class RAGPromptBuilder(BasePromptBuilder):
    """RAG场景的提示词构建器"""
    
    def __init__(self, template: Optional[str] = None, system_template: Optional[str] = None):
        self.template = template or (
            "Based on the following context:\n"
            "{context}\n\n"
            "Please answer the question: {query}"
        )
        self.system_template = system_template or (
            "You are a helpful AI assistant that answers questions based on the provided context."
        )
    
    def build_prompt(self, query: str, context: str) -> str:
        return self.template.format(
            context=context,
            query=query
        )
    
    def build_system_prompt(self) -> str:
        return self.system_template
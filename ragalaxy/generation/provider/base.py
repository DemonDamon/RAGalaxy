from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, List, Optional

class BaseLLMProvider(ABC):
    """LLM提供者的基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """初始化LLM相关资源"""
        pass
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        pass
        
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成回复"""
        pass
        
    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成回复"""
        pass
        
    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """异步生成回复"""
        pass
        
    @abstractmethod
    async def abatch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """异步批量生成回复"""
        pass
        
    @abstractmethod
    async def astream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """异步流式生成回复"""
        pass
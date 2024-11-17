from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseKeywordRetriever(ABC):
    """关键词检索的基础接口类"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """检索相关文档"""
        pass

    @abstractmethod
    async def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """批量检索"""
        pass
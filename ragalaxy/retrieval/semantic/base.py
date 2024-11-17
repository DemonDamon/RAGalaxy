from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSemanticRetriever(ABC):
    """语义检索的基础接口类"""
    
    def __init__(self, embedding_model: str, **kwargs):
        self.embedding_model = embedding_model
        self.kwargs = kwargs

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 检索结果列表,每个结果包含文档内容和相关性分数
        """
        pass

    @abstractmethod
    async def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """批量检索"""
        pass
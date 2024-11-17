from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetrievalRouter(ABC):
    """检索路由的基础接口类"""
    
    def __init__(self, retrievers: Dict[str, Any], **kwargs):
        self.retrievers = retrievers
        self.kwargs = kwargs

    @abstractmethod
    async def route(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        根据查询选择合适的检索策略并执行检索
        
        Args:
            query: 查询文本
            kwargs: 额外参数
            
        Returns:
            List[Dict]: 检索结果
        """
        pass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    """检索结果模型"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # 召回源标识

class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """检索接口"""
        pass
        
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档"""
        pass 
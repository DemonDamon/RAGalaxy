from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseGraphStore(ABC):
    """图存储基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    async def connect(self) -> bool:
        """连接数据库"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass
        
    @abstractmethod
    async def add_entities(self, entities: List[Dict]) -> bool:
        """添加实体"""
        pass
        
    @abstractmethod
    async def add_relations(self, relations: List[Dict]) -> bool:
        """添加关系"""
        pass
        
    @abstractmethod
    async def query(self, cypher: str) -> List[Dict]:
        """执行查询"""
        pass
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
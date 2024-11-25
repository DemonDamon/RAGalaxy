from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class Entity(BaseModel):
    """实体模型"""
    id: str
    text: str
    type: str
    start: int
    end: int
    properties: Dict[str, Any] = {}

class Relation(BaseModel):
    """关系模型"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = {}

class BaseExtractor(ABC):
    """实体关系抽取器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    async def extract_entities(self, text: str) -> List[Entity]:
        """抽取实体"""
        pass
        
    @abstractmethod
    async def extract_relations(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Relation]:
        """抽取关系"""
        pass 
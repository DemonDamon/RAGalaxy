from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class EntityMention(BaseModel):
    """实体提及"""
    id: str
    text: str
    type: str
    start: int
    end: int

class EntityCandidate(BaseModel):
    """实体候选"""
    id: str
    name: str
    type: str
    score: float
    description: str = ""
    attributes: Dict[str, Any] = {}

class BaseLinker(ABC):
    """实体链接器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    async def get_candidates(
        self,
        mention: EntityMention,
        top_k: int = 5
    ) -> List[EntityCandidate]:
        """获取候选实体"""
        pass
        
    @abstractmethod
    async def link(
        self,
        mention: EntityMention,
        candidates: List[EntityCandidate]
    ) -> EntityCandidate:
        """实体消歧"""
        pass 
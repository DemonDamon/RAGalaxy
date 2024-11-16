from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseExtractor(ABC):
    """实体关系抽取器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表,每个实体包含:
            - text: 实体文本
            - type: 实体类型
            - start: 起始位置
            - end: 结束位置
        """
        pass
    
    @abstractmethod
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """抽取实体间关系
        
        Returns:
            关系列表,每个关系包含:
            - head: 头实体
            - tail: 尾实体
            - type: 关系类型
        """
        pass
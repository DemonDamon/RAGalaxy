from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Rule(BaseModel):
    """推理规则"""
    id: str
    name: str
    pattern: Dict[str, Any]  # 规则模式
    confidence: float        # 规则置信度

class Inference(BaseModel):
    """推理结果"""
    source_id: str
    target_id: str
    relation_type: str
    rule_id: str
    confidence: float
    evidence: List[Dict[str, Any]]

class BaseReasoner(ABC):
    """推理器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[Rule] = []
        
    @abstractmethod
    async def load_rules(self, rules_path: str):
        """加载规则"""
        pass
        
    @abstractmethod
    async def infer(
        self,
        graph: Any,
        entity_id: str,
        max_depth: int = 2
    ) -> List[Inference]:
        """执行推理"""
        pass
        
    @abstractmethod
    async def explain(
        self,
        inference: Inference
    ) -> Dict[str, Any]:
        """解释推理过程"""
        pass 
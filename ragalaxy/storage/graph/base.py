from typing import List, Dict, Any, Optional
from ..base import BaseStorage

class BaseGraphStorage(BaseStorage):
    """图存储基类"""
    
    @abstractmethod
    def add_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """添加节点"""
        pass
        
    @abstractmethod
    def add_edge(self, 
                 from_node: str, 
                 to_node: str, 
                 edge_type: str,
                 properties: Optional[Dict[str, Any]] = None) -> bool:
        """添加边"""
        pass
        
    @abstractmethod
    def get_neighbors(self, 
                     node_id: str, 
                     edge_type: Optional[str] = None,
                     max_depth: int = 1) -> List[Dict[str, Any]]:
        """获取邻居节点"""
        pass
        
    @abstractmethod
    def query(self, cypher: str) -> List[Dict[str, Any]]:
        """执行图查询"""
        pass
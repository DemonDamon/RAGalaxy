from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseChunker(ABC):
    """文本分块器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
    
    @abstractmethod
    def split(self, text: str) -> List[Dict[str, Any]]:
        """将文本分割成块
        
        Args:
            text: 输入文本
            
        Returns:
            分块结果列表,每个块包含:
            - content: 块内容
            - metadata: 块元数据
        """
        pass
    
    @abstractmethod
    def merge(self, chunks: List[Dict[str, Any]]) -> str:
        """将文本块合并成完整文本"""
        pass
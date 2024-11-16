from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

class BaseStorage(ABC):
    """存储基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.namespace = config.get("namespace", "default")
        
    @abstractmethod
    def connect(self) -> bool:
        """建立存储连接"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """断开存储连接"""
        pass
        
    @abstractmethod
    def health_check(self) -> bool:
        """检查存储健康状态"""
        pass

    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from datetime import timedelta

class BaseCache(ABC):
    """缓存基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.namespace = config.get("namespace", "default")
        self.default_ttl = config.get("default_ttl", timedelta(hours=1))
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
        
    @abstractmethod
    def set(self, 
            key: str, 
            value: Any,
            ttl: Optional[timedelta] = None) -> bool:
        """设置缓存值"""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
        
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
        
    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass
        
    def get_key(self, key: str) -> str:
        """生成带命名空间的键名"""
        return f"{self.namespace}:{key}"
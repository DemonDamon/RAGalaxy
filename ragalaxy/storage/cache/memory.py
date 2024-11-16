from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from .base import BaseCache

class MemoryCache(BaseCache):
    """内存缓存实现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._cache: Dict[str, Dict[str, Any]] = {}
        
    def get(self, key: str) -> Optional[Any]:
        cache_key = self.get_key(key)
        if cache_key not in self._cache:
            return None
            
        cache_item = self._cache[cache_key]
        if self._is_expired(cache_item):
            self.delete(key)
            return None
            
        return cache_item["value"]
        
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        cache_key = self.get_key(key)
        expiry = datetime.now() + (ttl or self.default_ttl)
        
        self._cache[cache_key] = {
            "value": value,
            "expiry": expiry
        }
        return True
        
    def delete(self, key: str) -> bool:
        cache_key = self.get_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False
        
    def exists(self, key: str) -> bool:
        cache_key = self.get_key(key)
        return cache_key in self._cache
        
    def clear(self) -> bool:
        self._cache.clear()
        return True
        
    def _is_expired(self, cache_item: Dict[str, Any]) -> bool:
        return datetime.now() > cache_item["expiry"]
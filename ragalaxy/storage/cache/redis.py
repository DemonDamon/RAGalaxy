from typing import Any, Optional, Dict
from datetime import timedelta
import json
import redis
from .base import BaseCache

class RedisCache(BaseCache):
    """Redis缓存实现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self._client = self._create_client()
        
    def _create_client(self) -> redis.Redis:
        """创建Redis客户端"""
        return redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )
        
    def get(self, key: str) -> Optional[Any]:
        cache_key = self.get_key(key)
        value = self._client.get(cache_key)
        if value is None:
            return None
        return json.loads(value)
        
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        cache_key = self.get_key(key)
        try:
            serialized = json.dumps(value)
            if ttl:
                self._client.setex(cache_key, int(ttl.total_seconds()), serialized)
            else:
                self._client.set(cache_key, serialized)
            return True
        except Exception:
            return False
            
    def delete(self, key: str) -> bool:
        cache_key = self.get_key(key)
        return bool(self._client.delete(cache_key))
        
    def exists(self, key: str) -> bool:
        cache_key = self.get_key(key)
        return bool(self._client.exists(cache_key))
        
    def clear(self) -> bool:
        try:
            keys = self._client.keys(f"{self.namespace}:*")
            if keys:
                self._client.delete(*keys)
            return True
        except Exception:
            return False
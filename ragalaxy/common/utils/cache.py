from typing import Any, Optional, Callable
import hashlib
import json
import pickle
from pathlib import Path
import time
from functools import wraps

class CacheUtils:
    """缓存工具类"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        cache_data = {
            'args': args,
            'kwargs': kwargs
        }
        data = json.dumps(cache_data, sort_keys=True).encode()
        return hashlib.md5(data).hexdigest()
    
    def cache(
        self,
        ttl: Optional[int] = None
    ) -> Callable:
        """缓存装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self.get_cache_key(*args, **kwargs)
                cache_file = self.cache_dir / f"{key}.pkl"
                
                # 检查缓存是否存在且未过期
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    if ttl is None or time.time() - cached_data['timestamp'] < ttl:
                        return cached_data['result']
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'result': result,
                        'timestamp': time.time()
                    }, f)
                return result
            return wrapper
        return decorator
import hashlib
from typing import Any
import pickle
import json

class HashUtils:
    """哈希工具类"""
    
    @staticmethod
    def hash_object(obj: Any) -> str:
        """计算对象的哈希值"""
        if isinstance(obj, (str, int, float, bool)):
            data = str(obj).encode()
        elif isinstance(obj, dict):
            data = json.dumps(obj, sort_keys=True).encode()
        else:
            data = pickle.dumps(obj)
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def hash_file(file_path: str, chunk_size: int = 8192) -> str:
        """计算文件的哈希值"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
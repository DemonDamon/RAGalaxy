import os
from typing import List, Optional
from pathlib import Path
import json
import pickle

class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def ensure_dir(dir_path: str) -> None:
        """确保目录存在"""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_files(
        dir_path: str,
        ext: Optional[List[str]] = None
    ) -> List[str]:
        """列出目录下的文件"""
        files = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if ext is None or any(filename.endswith(e) for e in ext):
                    files.append(os.path.join(root, filename))
        return files
    
    @staticmethod
    def save_json(data: dict, file_path: str) -> None:
        """保存 JSON 文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_json(file_path: str) -> dict:
        """加载 JSON 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    @staticmethod
    def save_pickle(data: object, file_path: str) -> None:
        """保存 Pickle 文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(file_path: str) -> object:
        """加载 Pickle 文件"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
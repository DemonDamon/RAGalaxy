from pathlib import Path
from typing import List, Union, Dict
import hashlib
import json
import yaml

class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def load_yaml_config(file_path: str) -> Dict:
        """加载YAML配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def process_batch(items: List[Any], func: callable, batch_size: int = 32) -> List[Any]:
        """批量处理"""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            results.extend(func(batch))
        return results
    
    @staticmethod
    def get_file_type(file_path: Union[str, Path]) -> str:
        """获取文件类型
        Args:
            file_path: 文件路径
        Returns:
            文件类型（不含点号的小写扩展名）
        """
        suffix = Path(file_path).suffix.lower()
        return suffix[1:] if suffix else ""
    
    @staticmethod
    def hash_text(text: str) -> str:
        """计算文本哈希值"""
        return hashlib.md5(text.encode()).hexdigest()
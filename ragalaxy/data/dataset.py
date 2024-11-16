from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

class Dataset:
    """数据集处理类"""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_files(self, pattern: str = "*") -> List[Path]:
        """加载符合模式的所有文件"""
        return list(self.data_dir.glob(pattern))
    
    def process_batch(self, files: List[Path], processor: callable) -> List[Dict[str, Any]]:
        """批量处理文件"""
        results = []
        for file in files:
            try:
                result = processor(file)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
        return results
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

class BaseParser(ABC):
    """文档解析器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    @abstractmethod
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """解析单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析结果字典,包含:
            - content: 文本内容
            - metadata: 元数据
        """
        pass
    
    @abstractmethod
    def batch_parse(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """批量解析文件"""
        pass

    def validate_file(self, file_path: Path) -> bool:
        """验证文件是否可以被该解析器处理"""
        return file_path.suffix.lower() in self.supported_extensions

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表"""
        pass
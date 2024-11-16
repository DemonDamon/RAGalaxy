from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path

class BaseParser(ABC):
    """文档解析器基类"""
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> str:
        """解析文档内容
        Args:
            file_path: 文件路径
        Returns:
            解析后的文本内容
        """
        pass

class BaseChunker(ABC):
    """文本分块器基类"""
    
    @abstractmethod
    def split(self, text: str, **kwargs) -> List[str]:
        """将文本分割成块
        Args:
            text: 输入文本
            **kwargs: 额外参数
        Returns:
            文本块列表
        """
        pass

class BaseExtractor(ABC):
    """实体关系抽取器基类"""
    
    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """抽取实体和关系
        Args:
            text: 输入文本
        Returns:
            包含实体和关系的字典
        """
        pass
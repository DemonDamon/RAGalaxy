from pathlib import Path
from typing import List
import magic
from common.utils.file import FileUtils

class DocumentUtils:
    """文档处理工具类"""
    
    @staticmethod
    def validate_file_type(file_path: Path, allowed_types: List[str]) -> bool:
        """验证文件类型"""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(file_path))
        return any(t in file_type for t in allowed_types)
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """获取文件大小"""
        return file_path.stat().st_size 
from docx import Document
from ..base import BaseParser
from pathlib import Path
from typing import Union

class DocxParser(BaseParser):
    """Word文档解析器"""
    
    def parse(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
import pdfplumber
from pathlib import Path
from ..base import BaseParser

class PDFParser(BaseParser):
    """PDF文档解析器"""
    
    def parse(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
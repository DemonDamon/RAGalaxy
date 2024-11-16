import markdown
from ..base import BaseParser
from pathlib import Path
from typing import Union

class MarkdownParser(BaseParser):
    """Markdown文档解析器"""
    
    def parse(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        html = markdown.markdown(md_text)
        # 简单移除HTML标签
        return html.replace('<p>', '').replace('</p>', '\n')
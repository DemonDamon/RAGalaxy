from typing import Dict, Any, List, Optional
import re
import json

class ParserUtils:
    """解析工具类"""
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """解析JSON格式的响应"""
        try:
            # 提取JSON字符串
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception:
            return {}
    
    @staticmethod
    def parse_list_response(response: str) -> List[str]:
        """解析列表格式的响应"""
        # 移除空行和空格
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        # 移除序号和特殊字符
        cleaned_lines = []
        for line in lines:
            # 移除序号 (1. 2. 等)
            line = re.sub(r'^\d+\.\s*', '', line)
            # 移除项目符号 (- * 等)
            line = re.sub(r'^[-*]\s*', '', line)
            cleaned_lines.append(line)
        return cleaned_lines
    
    @staticmethod
    def extract_code_blocks(text: str) -> List[Dict[str, str]]:
        """提取代码块"""
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.finditer(pattern, text, re.DOTALL)
        code_blocks = []
        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code
            })
        return code_blocks
from typing import Dict, Any, Optional
from string import Template
import re

class TemplateUtils:
    """模板工具类"""
    
    @staticmethod
    def render_template(
        template: str,
        variables: Dict[str, Any],
        default: Optional[str] = None
    ) -> str:
        """渲染模板"""
        try:
            return Template(template).safe_substitute(variables)
        except Exception:
            return default if default is not None else template
            
    @staticmethod
    def extract_variables(template: str) -> List[str]:
        """提取模板变量"""
        pattern = r'\$\{([^}]*)\}'
        matches = re.finditer(pattern, template)
        return [match.group(1) for match in matches]
    
    @staticmethod
    def validate_template(
        template: str,
        required_vars: List[str]
    ) -> bool:
        """验证模板"""
        variables = set(TemplateUtils.extract_variables(template))
        return all(var in variables for var in required_vars)
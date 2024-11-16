from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """执行工具"""
        pass
        
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description
        }

class ToolRegistry:
    """工具注册器"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
        
    def unregister(self, name: str):
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
            
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(name)
        
    def list_tools(self) -> Dict[str, str]:
        """列出所有工具"""
        return {name: tool.description for name, tool in self.tools.items()}
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json

@dataclass
class QueryState:
    """查询状态类"""
    query: str
    context: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self):
        """清空历史"""
        self.history.clear()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "context": self.context,
            "history": self.history,
            "metadata": self.metadata
        }
        
    def save(self, file_path: str):
        """保存状态"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, file_path: str) -> 'QueryState':
        """加载状态"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return cls(**data)
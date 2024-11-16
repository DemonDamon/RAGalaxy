from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

class CallbackManager:
    """回调管理器"""
    
    def __init__(self):
        self.callbacks: Dict[str, List[Callable]] = {}
        
    def register(self, event: str, callback: Callable):
        """注册回调函数"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
        
    def unregister(self, event: str, callback: Callable):
        """注销回调函数"""
        if event in self.callbacks:
            self.callbacks[event].remove(callback)
            
    def trigger(self, event: str, *args, **kwargs):
        """触发回调事件"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(*args, **kwargs)

class BaseHandler(ABC):
    """基础处理器"""
    
    def __init__(self):
        self.callback_manager = CallbackManager()
        
    @abstractmethod
    def handle(self, *args, **kwargs):
        """处理方法"""
        pass

class StreamHandler(BaseHandler):
    """流式处理器"""
    
    def __init__(self):
        super().__init__()
        self.buffer = []
        
    def handle(self, token: str):
        """处理token"""
        self.buffer.append(token)
        self.callback_manager.trigger("on_token", token)
        
    def get_text(self) -> str:
        """获取完整文本"""
        return "".join(self.buffer)
        
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
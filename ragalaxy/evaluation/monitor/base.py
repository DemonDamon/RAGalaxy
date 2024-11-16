from abc import ABC, abstractmethod
from typing import Dict, Any
import time

class BaseMonitor(ABC):
    """性能监控器基类"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        
    def stop(self):
        """停止监控"""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics["duration"] = duration
            self.start_time = None
            
    @abstractmethod
    def record(self, metric_name: str, value: Any):
        """记录指标"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        pass
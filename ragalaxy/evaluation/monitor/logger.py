import logging
from typing import Dict, Any
from .base import BaseMonitor

class LogMonitor(BaseMonitor):
    """日志监控器"""
    
    def __init__(self, logger_name: str = "RAGalaxy"):
        super().__init__()
        self.logger = logging.getLogger(logger_name)
        
    def record(self, metric_name: str, value: Any):
        """记录指标到日志"""
        self.metrics[metric_name] = value
        self.logger.info(f"{metric_name}: {value}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有监控指标"""
        return self.metrics.copy()
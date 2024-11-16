from typing import Dict, Any, List
from .base import BaseMetric
from .retrieval import RecallMetric, PrecisionMetric
from .generation import BleuMetric, RougeMetric, ExactMatchMetric, F1ScoreMetric

class MetricRegistry:
    """评估指标注册表"""
    
    _metrics = {
        'recall': RecallMetric,
        'precision': PrecisionMetric,
        'bleu': BleuMetric,
        'rouge': RougeMetric,
        'exact_match': ExactMatchMetric,
        'f1': F1ScoreMetric
    }
    
    @classmethod
    def get_metric(cls, name: str, **kwargs) -> BaseMetric:
        """获取评估指标实例"""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return cls._metrics[name](**kwargs)
        
    @classmethod
    def get_all_metrics(cls, **kwargs) -> Dict[str, BaseMetric]:
        """获取所有评估指标实例"""
        return {name: metric(**kwargs) 
                for name, metric in cls._metrics.items()}
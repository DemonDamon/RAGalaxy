from typing import List, Dict, Any
from .base import BaseMetric

class RecallMetric(BaseMetric):
    """召回率评估指标"""
    
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

    def __call__(self, prediction: List[str], ground_truth: List[str]) -> float:
        """
        计算 Top-K 召回率
        prediction: 检索到的文档ID列表
        ground_truth: 相关文档ID列表
        """
        if not ground_truth:
            return 0.0
            
        pred_set = set(prediction[:self.k])
        gt_set = set(ground_truth)
        
        if not gt_set:
            return 0.0
            
        return len(pred_set.intersection(gt_set)) / len(gt_set)

class PrecisionMetric(BaseMetric):
    """精确率评估指标"""
    
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

    def __call__(self, prediction: List[str], ground_truth: List[str]) -> float:
        if not prediction[:self.k]:
            return 0.0
            
        pred_set = set(prediction[:self.k])
        gt_set = set(ground_truth)
        
        return len(pred_set.intersection(gt_set)) / len(pred_set)
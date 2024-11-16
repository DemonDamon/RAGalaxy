from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MetricInput:
    """评估输入数据结构"""
    query: str
    retrieval_gt: Optional[List[str]] = None  # 检索的ground truth
    generation_gt: Optional[str] = None       # 生成的ground truth
    retrieved_docs: Optional[List[str]] = None # 实际检索结果
    generated_text: Optional[str] = None      # 实际生成结果

class BaseEvaluator:
    """评估器基类"""
    def __init__(self, metrics: List[str]):
        from .metrics import MetricRegistry
        self.metrics = {name: MetricRegistry.get_metric(name) 
                       for name in metrics}
    
    def evaluate(self, input_data: MetricInput) -> dict:
        """执行评估"""
        raise NotImplementedError
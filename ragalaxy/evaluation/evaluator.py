from typing import Dict, Any
from .base import BaseEvaluator, MetricInput

class RetrievalEvaluator(BaseEvaluator):
    """检索评估器"""
    def evaluate(self, input_data: MetricInput) -> Dict[str, float]:
        if not input_data.retrieval_gt or not input_data.retrieved_docs:
            return {}
        
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(input_data.retrieved_docs, 
                                 input_data.retrieval_gt)
        return results

class GenerationEvaluator(BaseEvaluator):
    """生成评估器"""
    def evaluate(self, input_data: MetricInput) -> Dict[str, float]:
        if not input_data.generation_gt or not input_data.generated_text:
            return {}
            
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(input_data.generated_text, 
                                 input_data.generation_gt)
        return results
from typing import List, Dict, Set
from .base import BaseMetric
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import re

class BleuMetric(BaseMetric):
    """BLEU评分指标"""
    
    def __call__(self, prediction: str, ground_truth: str) -> float:
        if not prediction or not ground_truth:
            return 0.0
            
        prediction_tokens = prediction.split()
        ground_truth_tokens = [ground_truth.split()]
        
        return sentence_bleu(ground_truth_tokens, prediction_tokens)

class RougeMetric(BaseMetric):
    """ROUGE评分指标"""
    
    def __init__(self):
        super().__init__()
        self.rouge = Rouge()
        
    def __call__(self, prediction: str, ground_truth: str) -> float:
        if not prediction or not ground_truth:
            return 0.0
            
        try:
            scores = self.rouge.get_scores(prediction, ground_truth)[0]
            return scores["rouge-l"]["f"]
        except:
            return 0.0

class ExactMatchMetric(BaseMetric):
    """精确匹配评估指标"""
    
    def _normalize_answer(self, text: str) -> str:
        """标准化答案文本"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
        
    def __call__(self, prediction: str, ground_truth: str) -> float:
        if not prediction or not ground_truth:
            return 0.0
        return float(self._normalize_answer(prediction) == 
                    self._normalize_answer(ground_truth))

class F1ScoreMetric(BaseMetric):
    """F1分数评估指标"""
    
    def _get_tokens(self, text: str) -> Set[str]:
        """获取文本的标记集合"""
        text = self._normalize_answer(text)
        tokens = text.split()
        return set(tokens)
        
    def __call__(self, prediction: str, ground_truth: str) -> float:
        if not prediction or not ground_truth:
            return 0.0
            
        pred_tokens = self._get_tokens(prediction)
        truth_tokens = self._get_tokens(ground_truth)
        
        common = pred_tokens & truth_tokens
        if not pred_tokens or not truth_tokens:
            return 0.0
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
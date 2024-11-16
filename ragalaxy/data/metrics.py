from abc import ABC, abstractmethod
from collections import Counter
import re
import string
from typing import List, Dict, Any

class BaseMetric(ABC):
    """评估指标基类"""
    def __init__(self):
        self._total = 0.0
        self._count = 0
        
    @abstractmethod
    def __call__(self, prediction: str, ground_truth: str) -> float:
        pass
        
    def update(self, score: float):
        self._total += score
        self._count += 1
        
    def compute(self) -> float:
        return self._total / self._count if self._count > 0 else 0.0
        
    def reset(self):
        self._total = 0.0
        self._count = 0
        
    @staticmethod
    def normalize_answer(text: str) -> str:
        """标准化答案文本"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return text
        
    @staticmethod
    def get_tokens(text: str) -> List[str]:
        return BaseMetric.normalize_answer(text).split()

class ExactMatch(BaseMetric):
    def __call__(self, prediction: str, ground_truth: str) -> float:
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

class F1Score(BaseMetric):
    def __call__(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.get_tokens(prediction)
        gold_tokens = self.get_tokens(ground_truth)
        
        if not pred_tokens and not gold_tokens:
            return 1.0
            
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)
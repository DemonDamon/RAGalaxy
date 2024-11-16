from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    """评估指标的基类"""
    
    def __init__(self):
        self._total_score = 0.0
        self._count = 0

    @abstractmethod
    def __call__(self, prediction: Any, ground_truth: Any) -> float:
        """计算单个预测的指标分数"""
        pass

    def update(self, prediction: Any, ground_truth: Any):
        """更新评估指标"""
        score = self.__call__(prediction, ground_truth)
        self._total_score += score
        self._count += 1

    def get_metric(self) -> Dict[str, float]:
        """获取评估结果"""
        if self._count == 0:
            return {"score": 0.0}
        return {"score": self._total_score / self._count}

    def reset(self):
        """重置评估指标"""
        self._total_score = 0.0
        self._count = 0
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable

class BaseOptimizer(ABC):
    """参数优化器基类"""
    
    def __init__(self, param_space: Dict[str, List[Any]], 
                 objective_func: Callable[[Dict[str, Any]], float]):
        self.param_space = param_space
        self.objective_func = objective_func
        self.best_params = None
        self.best_score = float('-inf')
        
    @abstractmethod
    def optimize(self, n_trials: int = 10) -> Dict[str, Any]:
        """执行参数优化"""
        pass
    
    def update_best(self, params: Dict[str, Any], score: float):
        """更新最佳参数"""
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
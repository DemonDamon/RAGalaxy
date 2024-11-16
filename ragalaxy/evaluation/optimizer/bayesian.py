from typing import Dict, Any, List
import optuna
from .base import BaseOptimizer

class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""
    
    def __init__(self, param_space: Dict[str, List[Any]], 
                 objective_func: Callable[[Dict[str, Any]], float]):
        super().__init__(param_space, objective_func)
        
    def optimize(self, n_trials: int = 10) -> Dict[str, Any]:
        """执行贝叶斯优化"""
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, 
                                                           min(param_range), 
                                                           max(param_range))
                elif isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, 
                                                         min(param_range), 
                                                         max(param_range))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, 
                                                                 param_range)
            
            score = self.objective_func(params)
            self.update_best(params, score)
            return score
            
        study.optimize(objective, n_trials=n_trials)
        return self.best_params
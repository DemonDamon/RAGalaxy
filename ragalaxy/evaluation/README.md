# evaluation

## 设计思路
本包负责评估层实现,主要参考了:
- AutoRAG 的评估框架(参考 autorag.md:733-741)
- HippoRAG 的评估指标(参考 hipporag.md:296-300)
- FlashRAG 的性能监控(参考 flashrag.md:90-92)

### 主要功能
1. 评估指标收集
2. 参数优化
3. 性能监控

### 目录结构
```
evaluation/
├── metrics/           # 评估指标模块
│   ├── __init__.py
│   ├── base.py       # 基础指标接口
│   ├── retrieval.py  # 检索评估指标
│   └── generation.py # 生成评估指标
├── optimizer/        # 参数优化模块
│   ├── __init__.py
│   ├── base.py      # 基础优化器接口
│   └── bayesian.py  # 贝叶斯优化实现
└── monitor/         # 性能监控模块
    ├── __init__.py
    ├── base.py     # 基础监控接口
    └── logger.py   # 日志监控实现
```

#### 使用示例1
```python
from ragalaxy.evaluation.metrics import BleuMetric, RecallMetric
from ragalaxy.evaluation.optimizer import BayesianOptimizer
from ragalaxy.evaluation.monitor import LogMonitor

# 使用评估指标
bleu = BleuMetric()
score = bleu("generated text", "ground truth")

# 使用优化器
param_space = {
    "temperature": [0.1, 0.2, 0.3, 0.4, 0.5],
    "top_p": [0.1, 0.3, 0.5, 0.7, 0.9]
}
optimizer = BayesianOptimizer(param_space, objective_func)
best_params = optimizer.optimize(n_trials=10)

# 使用监控器
monitor = LogMonitor()
monitor.start()
# ... 执行操作 ...
monitor.record("memory_usage", memory_usage)
monitor.stop()
metrics = monitor.get_metrics()
```

#### 使用示例2
```python
from ragalaxy.evaluation.metrics import MetricRegistry

# 获取单个指标
f1_metric = MetricRegistry.get_metric('f1')
score = f1_metric("predicted answer", "ground truth")

# 获取所有指标
metrics = MetricRegistry.get_all_metrics()
scores = {name: metric("predicted", "truth") 
          for name, metric in metrics.items()}
```

#### 使用示例3
```python
from ragalaxy.evaluation.base import MetricInput
from ragalaxy.evaluation.evaluator import RetrievalEvaluator, GenerationEvaluator

# 评估检索结果
retrieval_evaluator = RetrievalEvaluator(['recall', 'precision'])
retrieval_input = MetricInput(
    query="查询文本",
    retrieval_gt=["doc1", "doc2"],
    retrieved_docs=["doc1", "doc3"]
)
retrieval_scores = retrieval_evaluator.evaluate(retrieval_input)

# 评估生成结果
generation_evaluator = GenerationEvaluator(['bleu', 'rouge', 'f1'])
generation_input = MetricInput(
    query="查询文本",
    generation_gt="参考答案",
    generated_text="生成答案"
)
generation_scores = generation_evaluator.evaluate(generation_input)
```

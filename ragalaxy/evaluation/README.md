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
# 目录结构
```
tests/
├── retrieval/
│   ├── __init__.py
│   ├── conftest.py                # 测试配置和fixtures
│   ├── test_performance/         # 性能测试
│   │   ├── __init__.py
│   │   ├── test_dense_perf.py
│   │   └── test_hybrid_perf.py
│   ├── test_units/              # 单元测试
│   │   ├── __init__.py
│   │   ├── semantic/
│   │   │   ├── test_dense.py
│   │   │   └── test_sparse.py
│   │   ├── keyword/
│   │   │   └── test_bm25.py
│   │   └── router/
│   │       └── test_hybrid.py
│   └── examples/                # 示例代码
│       ├── dense_retrieval.py
│       ├── keyword_search.py
│       └── hybrid_search.py
```
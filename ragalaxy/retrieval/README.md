# retrieval

## 设计思路
本包负责检索层实现,主要参考了:
- FlashRAG 的检索模块(参考 flashrag.md:120-124)
- HippoRAG 的检索策略(参考 hipporag.md:288-294)
- GraphRAG 的查询系统(参考 graphrag.md:54-59)

### 主要功能
1. 语义检索
2. 关键词检索
3. 混合检索策略
4. 检索路由

### 目录结构
```
retrieval/
├── semantic/           # 语义检索模块
│   ├── __init__.py
│   ├── base.py        # 基础语义检索接口
│   ├── dense.py       # 稠密检索实现
│   └── sparse.py      # 稀疏检索实现
├── keyword/           # 关键词检索模块
│   ├── __init__.py
│   ├── base.py       # 基础关键词检索接口
│   └── bm25.py       # BM25实现
└── router/           # 检索路由模块
    ├── __init__.py
    ├── base.py      # 基础路由接口
    └── hybrid.py    # 混合检索路由实现
```
# generation

## 设计思路
本包负责生成层实现,主要参考了:
- LightRAG 的 LLM 集成(参考 lightrag.md:41-43)
- FastRAG 的生成模块(参考 fastrag.md:27-31)
- FlashRAG 的上下文优化(参考 flashrag.md:123-124)

### 主要功能
1. LLM 提供者接口
2. 上下文优化
3. 响应生成

### 目录结构
```
generation/
├── provider/          # LLM提供者模块
│   ├── __init__.py
│   ├── base.py       # 基础LLM接口
│   ├── openai.py     # OpenAI实现
│   └── anthropic.py  # Anthropic实现
├── optimizer/        # 上下文优化模块
│   ├── __init__.py
│   ├── base.py      # 基础优化器接口
│   └── rerank.py    # 重排序优化实现
└── generator/       # 响应生成模块
    ├── __init__.py
    ├── base.py     # 基础生成器接口
    └── chain.py    # 生成链实现
```

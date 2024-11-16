# data

## 设计思路
本包负责数据处理相关的功能,主要参考了:
- LightRAG 的 textract 实现(参考 lightrag.md:10-12)
- AutoRAG 的数据处理模块(参考 autorag.md:146-196)
- FlashRAG 的预处理流程(参考 flashrag.md:110)

### 主要功能
1. 多源文档解析
2. 智能文本分块
3. 实体关系抽取

### 目录结构
```
data/
├── parser/ # 文档解析模块
│   ├── init.py
│   ├── base.py # 基础解析器接口
│   ├── pdf.py # PDF解析实现
│   ├── docx.py # Word文档解析实现
│   └── markdown.py # Markdown解析实现
├── chunker/ # 文本分块模块
│   ├── init.py
│   ├── base.py # 基础分块器接口
│   ├── sentence.py # 基于句子的分块
│   └── semantic.py # 基于语义的分块
└── extractor/ # 实体抽取模块
    ├── init.py
    ├── base.py # 基础抽取器接口
    ├── spacy.py # 基于spacy的实体抽取
    └── bert.py # 基于BERT的实体抽取
```
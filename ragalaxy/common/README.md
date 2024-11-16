# common

## 设计思路
本包提供通用功能,主要参考了:
- GraphRAG 的配置管理(参考 graphrag.md:338-341)
- FlashRAG 的工具函数(参考 flashrag.md:714-719)
- AutoRAG 的工具类(参考 autorag.md:192-196)

### 主要功能
1. 配置管理
2. 日志系统
3. 通用工具函数

### 目录结构
```
common/
├── config/            # 配置管理模块
│   ├── __init__.py
│   ├── base.py       # 基础配置接口
│   └── yaml.py       # YAML配置实现
├── logger/           # 日志模块
│   ├── __init__.py
│   ├── base.py      # 基础日志接口
│   └── file.py      # 文件日志实现
└── utils/           # 工具函数模块
    ├── __init__.py
    ├── io.py       # IO相关工具
    └── text.py     # 文本处理工具
```
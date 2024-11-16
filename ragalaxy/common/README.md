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

#### common/utils 包

已实现的工具类:

1. 基础工具:
- CLIUtils: 命令行交互
- FileUtils: 文件操作
- TextUtils: 文本处理
- ParserUtils: 解析工具

2. 数据处理:
- DataUtils: 数据转换
- PreprocessUtils: 数据预处理
- MetricsUtils: 指标计算

3. 提示词相关:
- PromptUtils: 提示词处理
- TemplateUtils: 模板处理
- CompressUtils: 文本压缩

4. 工具管理:
- BaseTool: 工具基类
- ToolRegistry: 工具注册

5. 状态与缓存:
- QueryState: 查询状态
- CacheUtils: 缓存管理

6. 进度与回调:
- ProgressUtils: 进度显示
- CallbackManager: 回调管理

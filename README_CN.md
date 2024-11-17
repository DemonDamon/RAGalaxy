# RAGalaxy

![RAGalaxy](./assets/RAGalaxy_1.png)
每一份信息就如同浩瀚宇宙中的一颗星辰，而 "RAGalaxy" 则象征着通过 RAG 技术，将这些信息汇聚成结构化且意义深远的智慧体系，宛若星系中璀璨交汇的星河。同时，这一名称也彰显了我们追求知识与创新的雄心，期待在信息化的挑战中开拓出属于人类智慧的璀璨星河。


## 快速开始

### 环境要求
- Anaconda 或 Miniconda
- Python 3.11
- CUDA (可选，用于GPU加速)

### 环境配置

1. 创建并激活虚拟环境
```bash
# 创建环境
conda create -n ragalaxy python=3.11
# 激活环境
conda activate ragalaxy
```

2. 安装依赖
```
# 基础依赖
numpy>=1.24.0
pytest>=7.0.0
asyncio>=3.4.3

# 向量检索相关
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2

# 深度学习框架
torch>=2.0.0
transformers>=4.30.0

# 文本处理
rank-bm25>=0.2.2
tiktoken>=0.5.0

# 性能测试
pytest-asyncio>=0.21.0
pytest-benchmark>=4.0.0
# 开发工具
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0

# 可选依赖
# GPU 支持
# faiss-gpu>=1.7.4  # 如果使用 GPU
# cuda-python>=12.0  # 如果使用 NVIDIA GPU

```

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果需要GPU支持，还需要安装：
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

3. 验证安装
```bash
# 运行测试
pytest tests/retrieval/test_units
```

### 使用示例

```python
from ragalaxy.retrieval.semantic.dense import DenseRetriever

async def main():
    retriever = DenseRetriever(
        model_name="BAAI/bge-small-zh",
        vector_store="faiss"
    )
    
    results = await retriever.retrieve(
        query="人工智能的发展",
        top_k=2
    )
    
    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['content']}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 注意事项

1. 确保已安装 Anaconda 或 Miniconda
2. 建议使用独立的虚拟环境以避免依赖冲突
3. GPU 加速需要 NVIDIA 显卡和相应的 CUDA 支持
4. 首次运行时会自动下载所需的模型文件

## 问题排查

如果遇到问题，请检查：
1. Python 版本是否为 3.11
2. 虚拟环境是否正确激活
3. 所有依赖是否正确安装
4. GPU 版本是否安装了对应的 CUDA 版本
```

这个 README 包含了：
1. 基于 Anaconda 的环境配置说明
2. 清晰的步骤指引
3. 示例代码（参考：```python:ragalaxy/tests/retrieval/examples/dense_retrieval.py startLine:1 endLine:30```）
4. 常见问题排查指南
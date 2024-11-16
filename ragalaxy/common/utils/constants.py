from enum import Enum

class StorageType(Enum):
    """存储类型"""
    LOCAL = "local"
    NEO4J = "neo4j"
    MILVUS = "milvus"
    FAISS = "faiss"

class LLMType(Enum):
    """LLM类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class SearchType(Enum):
    """搜索类型"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

DEFAULT_CONFIG = {
    "storage": {
        "type": StorageType.LOCAL.value,
        "path": "./data"
    },
    "llm": {
        "type": LLMType.OPENAI.value,
        "model": "gpt-3.5-turbo"
    },
    "search": {
        "type": SearchType.HYBRID.value,
        "top_k": 5
    }
}
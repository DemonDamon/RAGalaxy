from typing import Dict, Any, List
from .base import BaseConfig

class RAGConfig(BaseConfig):
    """RAG 系统配置类"""
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.default_config = {
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cuda",
                "batch_size": 32
            },
            "retrieval": {
                "top_k": 5,
                "score_threshold": 0.5
            },
            "llm": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "storage": {
                "vector_store": "faiss",
                "graph_store": "networkx"
            }
        }
        self._init_default_config()
    
    def _init_default_config(self):
        """初始化默认配置"""
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
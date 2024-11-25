from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from .base import BaseRetriever, SearchResult

class VectorRetriever(BaseRetriever):
    """向量检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = self._load_model()
        self.vector_store = self._init_vector_store()
        
    def _load_model(self) -> SentenceTransformer:
        """加载编码模型"""
        model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return SentenceTransformer(model_name).to(device)
        
    def _init_vector_store(self):
        """初始化向量存储"""
        store_type = self.config.get("store_type", "faiss")
        if store_type == "milvus":
            from ragalaxy.storage.vector.milvus import MilvusStore
            return MilvusStore(self.config.get("milvus", {}))
        else:
            from ragalaxy.storage.vector.faiss import FaissStore
            return FaissStore(self.config.get("faiss", {}))
    
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """向量检索"""
        # 1. 文本向量化
        query_embedding = self.model.encode(query)
        
        # 2. 向量检索
        results = await self.vector_store.search(
            query_embedding, 
            top_k=top_k
        )
        
        # 3. 格式化结果
        return [
            SearchResult(
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"],
                source="vector"
            ) 
            for r in results
        ]
        
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档到向量库"""
        try:
            # 1. 批量编码文本
            texts = [doc["content"] for doc in documents]
            embeddings = self.model.encode(texts)
            
            # 2. 添加到向量库
            await self.vector_store.add(
                embeddings=embeddings,
                documents=documents
            )
            return True
        except Exception as e:
            print(f"添加文档失败: {str(e)}")
            return False 
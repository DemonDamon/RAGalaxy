from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
import numpy as np
from ..base import BaseStorage

class BaseVectorStorage(BaseStorage):
    """向量存储基类"""
    
    support_similarity_metrics = ["L2", "IP", "COSINE"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dimension = config.get("dimension", 768)
        self.metric_type = config.get("metric_type", "L2")
        self.index_type = config.get("index_type", "IVF")
        self.embedding_batch = config.get("embedding_batch", 32)
        self._connected = False
        
    def connect(self) -> bool:
        """建立存储连接"""
        if self._connected:
            return True
        try:
            self._do_connect()
            self._connected = True
            return True
        except Exception:
            return False
            
    def disconnect(self) -> bool:
        """断开存储连接"""
        if not self._connected:
            return True
        try:
            self._do_disconnect()
            self._connected = False
            return True
        except Exception:
            return False
            
    def health_check(self) -> bool:
        """检查存储健康状态"""
        return self._connected and self._do_health_check()
        
    @abstractmethod
    def _do_connect(self):
        """实际的连接实现"""
        pass
        
    @abstractmethod  
    def _do_disconnect(self):
        """实际的断开连接实现"""
        pass
        
    @abstractmethod
    def _do_health_check(self) -> bool:
        """实际的健康检查实现"""
        pass
        
    def index_done_callback(self):
        """索引完成回调"""
        pass
        
    def query_done_callback(self):
        """查询完成回调"""
        pass
        
    def truncate_inputs(self, inputs: List[str]) -> List[str]:
        """截断输入文本"""
        max_length = self.config.get("max_length", 512)
        return [text[:max_length] for text in inputs]
        
    @abstractmethod
    def is_exist(self, collection_name: str, ids: List[str]) -> List[bool]:
        """检查向量ID是否存在"""
        pass
        
    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """删除向量"""
        pass
        
    @abstractmethod
    def fetch(self, collection_name: str, ids: List[str]) -> List[np.ndarray]:
        """获取向量"""
        pass
        
    def batch_search(self,
                    collection_name: str,
                    query_vectors: List[np.ndarray],
                    top_k: int = 10,
                    filter: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """批量搜索向量"""
        results = []
        for vectors in self._batch_iterate(query_vectors, self.embedding_batch):
            batch_results = self.search(collection_name, np.array(vectors), top_k, filter)
            results.extend(batch_results)
        return results
        
    def _batch_iterate(self, items: List[Any], batch_size: int):
        """批量迭代器"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
        
    @abstractmethod
    def create_collection(self, 
                         collection_name: str,
                         dimension: Optional[int] = None) -> bool:
        """创建集合"""
        pass
        
    @abstractmethod
    def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass
        
    @abstractmethod
    def insert(self,
              collection_name: str,
              vectors: np.ndarray,
              metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """插入向量,返回向量ID列表"""
        pass
        
    @abstractmethod
    def search(self,
              collection_name: str,
              query_vectors: np.ndarray,
              top_k: int = 10,
              filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """搜索向量,返回结果列表"""
        pass
        
    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        pass
        
    @abstractmethod
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        pass
        
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合信息"""
        pass
        
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        pass
        
    def _match_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, tuple)):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
        
    async def batch_insert(self,
                          collection_name: str,
                          vectors: List[np.ndarray],
                          metadatas: Optional[List[Dict[str, Any]]] = None,
                          batch_size: Optional[int] = None) -> List[str]:
        """批量插入向量"""
        batch_size = batch_size or self.embedding_batch
        all_ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
            ids = await self.insert(collection_name, batch_vectors, batch_metadatas)
            all_ids.extend(ids)
            
        return all_ids
        
    async def batch_search(self,
                          collection_name: str,
                          query_vectors: List[np.ndarray],
                          top_k: int = 10,
                          filter: Optional[Dict[str, Any]] = None,
                          batch_size: Optional[int] = None) -> AsyncIterator[List[Dict[str, Any]]]:
        """批量搜索向量"""
        batch_size = batch_size or self.embedding_batch
        
        for i in range(0, len(query_vectors), batch_size):
            batch_vectors = query_vectors[i:i + batch_size]
            results = await self.search(collection_name, batch_vectors, top_k, filter)
            yield results
            
    def _validate_vectors(self, vectors: np.ndarray) -> bool:
        """验证向量格式"""
        if not isinstance(vectors, np.ndarray):
            return False
        if len(vectors.shape) != 2:
            return False
        if vectors.shape[1] != self.dimension:
            return False
        return True
        
    def _validate_collection(self, collection_name: str) -> bool:
        """验证集合是否存在且可用"""
        if not self._connected:
            raise RuntimeError("Storage not connected")
        if not collection_name:
            raise ValueError("Collection name cannot be empty")
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
        return True
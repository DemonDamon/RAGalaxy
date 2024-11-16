from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from .base import BaseVectorStorage

class FAISSVectorStorage(BaseVectorStorage):
    """FAISS向量存储实现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._collections: Dict[str, faiss.Index] = {}
        self._id_maps: Dict[str, Dict[int, str]] = {}  # 内部ID到外部ID的映射
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}  # 元数据存储
        
    def create_collection(self, 
                         collection_name: str,
                         dimension: Optional[int] = None) -> bool:
        if collection_name in self._collections:
            return False
            
        dim = dimension or self.dimension
        if self.index_type == "IVF":
            # IVF索引需要训练,这里使用简单的L2索引
            index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.IndexFlatL2(dim)
            
        self._collections[collection_name] = index
        self._id_maps[collection_name] = {}
        self._metadata[collection_name] = []
        return True
        
    def drop_collection(self, collection_name: str) -> bool:
        if collection_name not in self._collections:
            return False
            
        del self._collections[collection_name]
        del self._id_maps[collection_name]
        del self._metadata[collection_name]
        return True
        
    def insert(self,
              collection_name: str,
              vectors: np.ndarray,
              metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        index = self._collections[collection_name]
        id_map = self._id_maps[collection_name]
        
        # 获取当前最大ID
        start_id = len(id_map)
        
        # 生成ID列表
        ids = [f"{collection_name}_{i}" for i in range(start_id, start_id + len(vectors))]
        
        # 更新ID映射
        for i, ext_id in enumerate(ids):
            id_map[start_id + i] = ext_id
            
        # 更新元数据
        if metadatas:
            self._metadata[collection_name].extend(metadatas)
            
        # 添加向量
        index.add(vectors.astype(np.float32))
        
        return ids
        
    def search(self,
              collection_name: str,
              query_vectors: np.ndarray,
              top_k: int = 10,
              filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        index = self._collections[collection_name]
        id_map = self._id_maps[collection_name]
        
        # 执行搜索
        distances, indices = index.search(query_vectors.astype(np.float32), top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            matches = []
            for d, j in zip(dist, idx):
                if j == -1:  # FAISS未找到匹配
                    continue
                    
                result = {
                    "id": id_map[j],
                    "distance": float(d)
                }
                
                # 添加元数据
                if self._metadata[collection_name]:
                    result["metadata"] = self._metadata[collection_name][j]
                    
                # 应用过滤
                if filter and not self._match_filter(result.get("metadata", {}), filter):
                    continue
                    
                matches.append(result)
            results.append(matches)
            
        return results
        
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        index = self._collections[collection_name]
        return {
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "index_type": self.index_type
        }
        
    def _match_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        for k, v in filter.items():
            if k not in metadata or metadata[k] != v:
                return False
        return True
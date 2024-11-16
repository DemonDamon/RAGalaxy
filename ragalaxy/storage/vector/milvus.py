from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
from .base import BaseVectorStorage

class MilvusVectorStorage(BaseVectorStorage):
    """Milvus向量存储实现"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.user = config.get("user", "")
        self.password = config.get("password", "")
        self._collections: Dict[str, Collection] = {}
        self._connect()
        
    def _connect(self):
        """连接Milvus服务器"""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password
        )
        
    def create_collection(self, 
                         collection_name: str,
                         dimension: Optional[int] = None) -> bool:
        if collection_name in self._collections:
            return False
            
        dim = dimension or self.dimension
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        
        self._collections[collection_name] = collection
        return True
        
    def drop_collection(self, collection_name: str) -> bool:
        if collection_name not in self._collections:
            return False
            
        collection = self._collections[collection_name]
        collection.drop()
        del self._collections[collection_name]
        return True
        
    def insert(self,
              collection_name: str,
              vectors: np.ndarray,
              metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        collection = self._collections[collection_name]
        
        # 生成ID列表
        ids = [f"{collection_name}_{i}" for i in range(len(vectors))]
        
        # 准备插入数据
        entities = [
            ids,
            vectors.tolist(),
            metadatas or [{} for _ in range(len(vectors))]
        ]
        
        collection.insert(entities)
        return ids
        
    def search(self,
              collection_name: str,
              query_vectors: np.ndarray,
              top_k: int = 10,
              filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        collection = self._collections[collection_name]
        
        # 构建搜索参数
        search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}
        
        # 构建过滤表达式
        expr = None
        if filter:
            conditions = []
            for k, v in filter.items():
                conditions.append(f'metadata["{k}"] == {repr(v)}')
            if conditions:
                expr = " && ".join(conditions)
        
        # 执行搜索
        results = collection.search(
            data=query_vectors.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["metadata"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            matches = []
            for hit in hits:
                matches.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "metadata": hit.entity.get("metadata", {})
                })
            formatted_results.append(matches)
            
        return formatted_results
        
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        collection = self._collections[collection_name]
        stats = collection.get_stats()
        return {
            "total_vectors": stats["row_count"],
            "dimension": self.dimension,
            "index_type": self.index_type
        }
from typing import Dict, Any, Optional, Type, List
import yaml
from pathlib import Path
from .base import BaseVectorStorage
from .faiss import FAISSVectorStorage
from .milvus import MilvusVectorStorage

class VectorStorageFactory:
    """向量存储工厂类"""
    
    _storage_registry: Dict[str, Type[BaseVectorStorage]] = {
        "faiss": FAISSVectorStorage,
        "milvus": MilvusVectorStorage
    }
    
    @classmethod
    def register_storage(cls, name: str, storage_class: Type[BaseVectorStorage]):
        """注册新的存储类型"""
        cls._storage_registry[name] = storage_class
        
    @classmethod
    def create_storage(cls, 
                      storage_type: str,
                      config: Optional[Dict[str, Any]] = None) -> BaseVectorStorage:
        """创建存储实例"""
        if storage_type not in cls._storage_registry:
            raise ValueError(f"Unknown storage type: {storage_type}")
            
        storage_class = cls._storage_registry[storage_type]
        return storage_class(config)
        
    @classmethod
    def load_from_yaml(cls, yaml_path: str, storage_type: str) -> BaseVectorStorage:
        """从YAML配置文件加载存储实例"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        storage_config = config.get("vector_storage", {}).get(storage_type)
        if not storage_config:
            raise ValueError(f"No config found for storage type: {storage_type}")
            
        return cls.create_storage(storage_type, storage_config)
        
    @classmethod
    def load_all_from_yaml(cls, yaml_path: str) -> Dict[str, BaseVectorStorage]:
        """从YAML配置文件加载所有存储实例"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        storage_configs = config.get("vector_storage", {})
        storages = {}
        
        for storage_type, storage_config in storage_configs.items():
            if storage_type in cls._storage_registry:
                storages[storage_type] = cls.create_storage(storage_type, storage_config)
                
        return storages
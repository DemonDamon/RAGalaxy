from typing import Dict, Any, Type, Optional
from .constants import StorageType, LLMType, SearchType

class ComponentFactory:
    """组件工厂类"""
    
    _storage_registry: Dict[str, Type] = {}
    _llm_registry: Dict[str, Type] = {}
    _search_registry: Dict[str, Type] = {}
    
    @classmethod
    def register_storage(cls, storage_type: str):
        """注册存储组件"""
        def wrapper(storage_class: Type):
            cls._storage_registry[storage_type] = storage_class
            return storage_class
        return wrapper
    
    @classmethod
    def register_llm(cls, llm_type: str):
        """注册LLM组件"""
        def wrapper(llm_class: Type):
            cls._llm_registry[llm_type] = llm_class
            return llm_class
        return wrapper
        
    @classmethod
    def register_search(cls, search_type: str):
        """注册搜索组件"""
        def wrapper(search_class: Type):
            cls._search_registry[search_type] = search_class
            return search_class
        return wrapper
    
    @classmethod
    def create_component(cls, component_type: str, config: Dict[str, Any]):
        """创建组件实例"""
        if component_type == "storage":
            storage_type = config.get("type", StorageType.LOCAL.value)
            storage_class = cls._storage_registry.get(storage_type)
            if not storage_class:
                raise ValueError(f"未注册的存储类型: {storage_type}")
            return storage_class(**config)
            
        elif component_type == "llm":
            llm_type = config.get("type", LLMType.OPENAI.value)
            llm_class = cls._llm_registry.get(llm_type)
            if not llm_class:
                raise ValueError(f"未注册的LLM类型: {llm_type}")
            return llm_class(**config)
            
        elif component_type == "search":
            search_type = config.get("type", SearchType.HYBRID.value)
            search_class = cls._search_registry.get(search_type)
            if not search_class:
                raise ValueError(f"未注册的搜索类型: {search_type}")
            return search_class(**config)
            
        else:
            raise ValueError(f"未知的组件类型: {component_type}")
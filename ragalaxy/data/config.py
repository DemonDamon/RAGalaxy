from typing import Dict, Any
import yaml
from pathlib import Path

class DataConfig:
    """数据处理配置管理"""
    
    def __init__(self, config_path: str = None, **kwargs):
        self.config = {}
        if config_path:
            self.load_config(config_path)
        self.config.update(kwargs)
    
    def load_config(self, config_path: str) -> None:
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config.update(yaml.safe_load(f))
            
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
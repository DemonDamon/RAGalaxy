from typing import Dict, Any, Optional
import yaml
import os

class BaseConfig:
    """基础配置类"""
    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        self.config[key] = value
        
    def save(self, config_path: str) -> None:
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
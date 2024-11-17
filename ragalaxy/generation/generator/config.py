from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """生成器配置类"""
    
    # 模型相关
    model_name: str
    model_path: Optional[str] = None
    max_input_len: int = 2048
    batch_size: int = 1
    device: str = "cuda"
    
    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # 系统资源
    gpu_num: int = 1
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GenerationConfig":
        return cls(**config)
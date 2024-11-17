from typing import Any, Dict, List, Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseLLMProvider

class OptimizedLLMProvider(BaseLLMProvider):
    """优化的本地LLM提供者，支持多种硬件加速"""
    
    def _initialize(self) -> None:
        self.model_name = self.config.get("model_name")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config.get("batch_size", 1)
        self.max_tokens = self.config.get("max_tokens", 2000)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 根据设备类型选择优化方案
        if self.device == "cuda":
            self._init_cuda_model()
        else:
            self._init_cpu_model()
            
    def _init_cuda_model(self):
        """CUDA优化初始化"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def _init_cpu_model(self):
        """CPU优化初始化"""
        try:
            import intel_extension_for_pytorch as ipex
            self.model = ipex.optimize(
                AutoModelForCausalLM.from_pretrained(self.model_name)
            )
        except ImportError:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                **kwargs
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
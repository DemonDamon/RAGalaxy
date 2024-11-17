from typing import Any, Dict, List, Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseLLMProvider

class LocalLLMProvider(BaseLLMProvider):
    """本地LLM模型提供者"""
    
    def _initialize(self) -> None:
        self.model_name = self.config.get("model_name")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = self.config.get("max_tokens", 2000)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]
        
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        raise NotImplementedError("Local models don't support streaming yet")
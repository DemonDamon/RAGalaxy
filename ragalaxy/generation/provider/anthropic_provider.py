from typing import Any, Dict, List, Iterator
import anthropic
from .base import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider"""
    
    def _initialize(self) -> None:
        self.client = anthropic.Anthropic(api_key=self.config.get("api_key"))
        self.model = self.config.get("model", "claude-3-sonnet")
        self.max_tokens = self.config.get("max_tokens", 2000)
        
    def generate(self, prompt: str, **kwargs) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return message.content[0].text
        
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]
        
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.content:
                yield chunk.content[0].text
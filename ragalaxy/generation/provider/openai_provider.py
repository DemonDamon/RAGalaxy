import openai
from openai import AsyncOpenAI
from typing import Any, Dict, List, Iterator, AsyncGenerator
from .base import BaseLLMProvider

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider"""
    
    def _initialize(self) -> None:
        self.sync_client = openai.OpenAI(api_key=self.config.get("api_key"))
        self.async_client = AsyncOpenAI(api_key=self.config.get("api_key"))
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.max_tokens = self.config.get("max_tokens", 2000)
        
    async def agenerate(self, prompt: str, **kwargs) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
        
    async def abatch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        responses = []
        for prompt in prompts:
            response = await self.agenerate(prompt, **kwargs)
            responses.append(response)
        return responses
        
    async def astream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
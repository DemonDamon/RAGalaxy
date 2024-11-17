from typing import List, Dict, Any, Optional
from ..provider import BaseLLMProvider
from ..optimizer import ContextOptimizer
from .prompt import BasePromptBuilder, RAGPromptBuilder
from .config import GenerationConfig

class BaseGenerator:
    """生成器基类"""
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        context_optimizer: ContextOptimizer,
        config: Dict[str, Any],
        prompt_builder: Optional[BasePromptBuilder] = None
    ):
        self.llm = llm_provider
        self.optimizer = context_optimizer
        self.config = GenerationConfig.from_dict(config)
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
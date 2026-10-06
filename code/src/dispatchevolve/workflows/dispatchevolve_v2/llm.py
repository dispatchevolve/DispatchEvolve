"""Single provider-layer LLM gateway for all V2 roles."""

from __future__ import annotations

import asyncio
import os

from dispatchevolve.config import LLMProvider, ModelSpec
from dispatchevolve.provider import LiteLLMWrapper

from .config import ModelConfig
from .prompt_store import RenderedPrompt


class V2LLM:
    def __init__(self, config: ModelConfig):
        key = os.environ.get(config.api_key_env or "", "") or None
        self.config = config
        self.client = LiteLLMWrapper(
            ModelSpec(name=config.model), api_base=config.api_base, api_key=key,
            provider=LLMProvider(config.provider),
            transport_max_retries=config.transport_max_retries,
        )

    def complete(self, prompt: RenderedPrompt) -> str:
        return asyncio.run(self.client.generate_with_context(
            prompt.system, [{"role": "user", "content": prompt.user}]
        )).strip()

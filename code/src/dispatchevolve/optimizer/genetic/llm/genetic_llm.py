# Adapted for DispatchEvolve; see THIRD_PARTY_NOTICES.md for origin and changes.
"""
LLM interface for the genetic optimizer, backed by dispatchevolve's provider layer.

Uses litellm for all providers. Configuration is read from LLMModelConfig.
"""

import asyncio
import logging
import os
import time
import json
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dispatchevolve.provider import LLMInterface
from dispatchevolve.prompt_limits import PromptSizeError, require_prompt_within_limit

logger = logging.getLogger(__name__)

_MAX_RETRIES = int(os.environ.get("DISPATCHEVOLVE_LLM_MAX_RETRIES", "8"))
_RETRY_DELAY = int(os.environ.get("DISPATCHEVOLVE_LLM_RETRY_DELAY", "60"))


def _with_retry(func):
    async def wrapper(self, *args, **kwargs):
        retries = 0
        max_retries = _MAX_RETRIES if self.retries is None else int(self.retries)
        while True:
            try:
                return await func(
                    self,
                    *args,
                    **kwargs,
                    _transport_retry_count=retries,
                )
            except Exception as e:
                if isinstance(e, PromptSizeError):
                    raise
                retries += 1
                if retries > max_retries:
                    logger.error(f"LLM call failed after {max_retries} retries: {e}")
                    raise
                logger.warning(
                    f"LLM error ({e}), retrying in {_RETRY_DELAY}s... "
                    f"(Attempt {retries}/{max_retries})"
                )
                await asyncio.sleep(_RETRY_DELAY)
    return wrapper


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


class GeneticLLM(LLMInterface):
    """LLM interface for the genetic optimizer, using litellm via dispatchevolve's provider layer."""

    def __init__(self, model_cfg=None):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            logger.info(f"Initialized genetic optimizer LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    def _build_messages(
        self, system_message: str, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        formatted = []
        if system_message:
            formatted.append({"role": "system", "content": system_message})
        formatted.extend(messages)
        return formatted

    def _build_litellm_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        # The observable wrapper loop is the sole retry owner.  Disabling the
        # LiteLLM layer keeps physical attempts countable and bounded.
        kwargs["num_retries"] = 0
        return kwargs

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    @_with_retry
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        from dispatchevolve.call_logger import track_success, track_failure

        transport_retry_count = int(kwargs.pop("_transport_retry_count", 0))
        formatted_messages = self._build_messages(system_message, messages)
        prompt_size = require_prompt_within_limit(
            system_message,
            messages,
            role="genetic_optimizer",
        )
        prompt_text = _messages_to_prompt(formatted_messages)
        response_root = os.environ.get("DISPATCHEVOLVE_GENETIC_RESPONSE_STORE")
        response_path = None
        if response_root:
            identity = hashlib.sha256(
                json.dumps(
                    {"model": self.model, "messages": formatted_messages},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            response_path = Path(response_root) / f"{identity}.json"
            if response_path.is_file():
                return str(json.loads(response_path.read_text(encoding="utf-8"))["response"])

        loop = asyncio.get_event_loop()
        t0 = time.time()

        def _query():
            import litellm
            litellm.drop_params = True

            call_kwargs = self._build_litellm_kwargs()
            call_kwargs.update(
                {k: v for k, v in kwargs.items() if k not in call_kwargs}
            )
            call_kwargs["num_retries"] = 0

            # Gemini models must use litellm's `gemini/` route (with api_base set
            # to the Vertex express endpoint), not the OpenAI-compatible handler;
            # otherwise a custom api_base sends `openai/gemini-...` and 404s.
            if self.model.startswith("gemini"):
                effective_model = (
                    self.model if self.model.startswith("gemini/") else f"gemini/{self.model}"
                )
            else:
                effective_model = f"openai/{self.model}" if self.api_base else self.model
            try:
                response = litellm.completion(
                    model=effective_model,
                    messages=formatted_messages,
                    **call_kwargs,
                )
                # Persist in the provider worker before returning control to the
                # optimizer/event loop. This matches the critic raw-response
                # transaction and closes the prior post-future checkpoint gap.
                if response_path is not None:
                    response_content = ""
                    if response and response.choices:
                        response_content = response.choices[0].message.content or ""
                    response_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = response_path.with_name(
                        f".{response_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                    )
                    temporary.write_text(
                        json.dumps({"response": response_content}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    os.replace(temporary, response_path)
                return response
            except Exception as e:
                raise RuntimeError(
                    f"LiteLLM call failed for model {self.model}: {e}"
                )

        try:
            response = await loop.run_in_executor(None, _query)
        except Exception as e:
            latency = time.time() - t0
            track_failure(
                model=self.model,
                provider="genetic_optimizer",
                prompt=prompt_text,
                error=e,
                latency_seconds=latency,
                retry_count=transport_retry_count,
                metadata={
                    "api_base": self.api_base,
                    "prompt_utf8_bytes": prompt_size,
                    "conservative_token_upper_bound": prompt_size,
                    "transport_attempt": transport_retry_count + 1,
                    "transport_retry_count": transport_retry_count,
                },
            )
            raise

        latency = time.time() - t0
        content = ""
        if response and response.choices:
            content = response.choices[0].message.content or ""

        track_success(
            response=response,
            model=self.model,
            provider="genetic_optimizer",
            prompt=prompt_text,
            response_text=content,
            latency_seconds=latency,
            metadata={
                "api_base": self.api_base,
                "prompt_utf8_bytes": prompt_size,
                "conservative_token_upper_bound": prompt_size,
                "transport_attempt": transport_retry_count + 1,
                "transport_retry_count": transport_retry_count,
            },
        )

        return content

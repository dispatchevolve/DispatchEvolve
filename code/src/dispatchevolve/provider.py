import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from dispatchevolve.config import DispatchEvolveConfig, LLMProvider, ModelSpec

logger = logging.getLogger(__name__)

_VERTEX_ONLY_ARGUMENTS = (
    "vertex_project",
    "vertex_location",
    "vertex_credentials",
)
_VERTEX_EXPRESS_API_BASE = "https://aiplatform.googleapis.com/v1/publishers/google"


def _is_gemini_location_error(error: Exception) -> bool:
    message = str(error).lower()
    return "user location is not supported" in message and "failed_precondition" in message


def _gemini_generate_via_vertex_express(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_message: str,
    messages: List[Dict[str, str]],
) -> tuple[str, Any]:
    """Retry a geo-blocked AI Studio call through location-free Express Mode."""

    import litellm

    formatted_messages: List[Dict[str, str]] = []
    if system_message:
        formatted_messages.append({"role": "system", "content": system_message})
    formatted_messages.extend(messages)
    effective_model = model if model.startswith("gemini/") else f"gemini/{model}"
    response = litellm.completion(
        model=effective_model,
        messages=formatted_messages,
        api_base=api_base,
        api_key=api_key,
    )
    content = ""
    if response and response.choices:
        content = response.choices[0].message.content or ""
    return content, response


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context."""
        pass


def with_retry(max_retries=8, delay=60):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            instance = args[0] if args else None
            frozen_retries = getattr(instance, "transport_max_retries", None)
            configured_retries = (
                int(frozen_retries)
                if frozen_retries is not None
                else int(os.environ.get("DISPATCHEVOLVE_LLM_MAX_RETRIES", str(max_retries)))
            )
            configured_delay = float(
                os.environ.get("DISPATCHEVOLVE_LLM_RETRY_DELAY", str(delay))
            )
            if configured_retries < 0 or configured_delay < 0:
                raise ValueError("LLM retry count and delay must be non-negative")
            retries = 0
            while True:
                try:
                    return await func(
                        *args,
                        **kwargs,
                        _transport_retry_count=retries,
                    )
                except Exception as e:
                    retries += 1
                    if retries > configured_retries:
                        logger.error(
                            f"API call failed after {configured_retries} retries: {e}"
                        )
                        raise
                    logger.warning(
                        f"API Error ({e}), retrying in {configured_delay:g} seconds... "
                        f"(Attempt {retries}/{configured_retries})"
                    )
                    await asyncio.sleep(configured_delay)

        return wrapper

    return decorator


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


class LiteLLMWrapper(LLMInterface):
    """Unified LLM wrapper using litellm for all providers.

    Supports:
    - openai_compatible / openai: routed through litellm with api_base and api_key
    - gemini: uses gemini/ prefix with GOOGLE_API_KEY
    - vertex_ai: uses vertex_ai/ prefix with GOOGLE_APPLICATION_CREDENTIALS
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        transport_max_retries: Optional[int] = None,
        allow_gemini_fallback: bool = True,
    ):
        self.model = model_spec.name
        self.temperature = model_spec.temperature
        self.top_p = model_spec.top_p
        self.max_tokens = model_spec.max_tokens
        self.api_base = api_base
        self.api_key = api_key
        self.provider = provider
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        if transport_max_retries is not None and transport_max_retries < 0:
            raise ValueError("transport_max_retries must be non-negative")
        self.transport_max_retries = transport_max_retries
        self.allow_gemini_fallback = bool(allow_gemini_fallback)

    @property
    def _provider_name(self) -> str:
        if self.provider:
            return self.provider.value if isinstance(self.provider, LLMProvider) else str(self.provider)
        return "unknown"

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message="",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    @with_retry()
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        import litellm
        from dispatchevolve.call_logger import track_success, track_failure

        formatted_messages = []
        if system_message:
            formatted_messages.append(
                {"role": "system", "content": system_message}
            )
        formatted_messages.extend(messages)

        transport_retry_count = int(kwargs.pop("_transport_retry_count", 0))
        call_kwargs = {}
        if self.temperature is not None:
            call_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            call_kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        call_kwargs.update(kwargs)
        # Exactly one retry layer owns the budget.  LiteLLM must not perform
        # hidden retries beneath the wrapper's observable retry loop.
        call_kwargs["num_retries"] = 0

        if self.provider in (LLMProvider.GEMINI, LLMProvider.VERTEX_AI):
            effective_model = self.model
            if self.provider == LLMProvider.GEMINI and not self.model.startswith("gemini/"):
                effective_model = f"gemini/{self.model}"
            elif self.provider == LLMProvider.VERTEX_AI and not self.model.startswith("vertex_ai/"):
                effective_model = f"vertex_ai/{self.model}"
            if self.provider == LLMProvider.VERTEX_AI:
                if self.vertex_project:
                    call_kwargs["vertex_project"] = self.vertex_project
                if self.vertex_location:
                    call_kwargs["vertex_location"] = self.vertex_location
                if self.api_key:
                    call_kwargs["vertex_credentials"] = self.api_key
            if self.provider == LLMProvider.GEMINI:
                # Gemini API does not use Vertex project, location, or service
                # account credentials. Keep accidental caller kwargs from
                # leaking into LiteLLM's Gemini route.
                for argument in _VERTEX_ONLY_ARGUMENTS:
                    call_kwargs.pop(argument, None)
                # Custom api_base lets us reach Gemini through the Vertex express
                # endpoint (x-goog-api-key auth), which is reachable where the
                # default AI Studio host is geo-restricted.
                if self.api_base:
                    call_kwargs["api_base"] = self.api_base
                if self.api_key:
                    call_kwargs["api_key"] = self.api_key
        else:
            effective_model = f"openai/{self.model}" if self.api_base else self.model
            if self.api_base:
                call_kwargs["api_base"] = self.api_base
            if self.api_key:
                call_kwargs["api_key"] = self.api_key

        litellm.drop_params = True

        prompt_text = _messages_to_prompt(formatted_messages)
        t0 = time.time()

        loop = asyncio.get_event_loop()
        route_metadata = {
            "effective_model": effective_model,
            "route": "primary",
            "transport_attempt": transport_retry_count + 1,
            "transport_retry_count": transport_retry_count,
        }

        def _query():
            try:
                return litellm.completion(
                    model=effective_model,
                    messages=formatted_messages,
                    **call_kwargs,
                )
            except Exception as e:
                if (
                    self.provider == LLMProvider.GEMINI
                    and self.api_base is None
                    and self.api_key
                    and self.allow_gemini_fallback
                    and _is_gemini_location_error(e)
                ):
                    try:
                        _content, fallback_response = _gemini_generate_via_vertex_express(
                            api_base=_VERTEX_EXPRESS_API_BASE,
                            api_key=self.api_key,
                            model=self.model,
                            system_message=system_message,
                            messages=messages,
                        )
                        route_metadata["route"] = "vertex_express"
                        return fallback_response
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"Gemini AI Studio location failure and Vertex Express fallback "
                            f"failed for model {self.model}: {fallback_error}"
                        ) from fallback_error
                raise RuntimeError(
                    f"LiteLLM call failed for model {self.model}: {e}"
                )

        try:
            response = await loop.run_in_executor(None, _query)
        except Exception as e:
            latency = time.time() - t0
            track_failure(
                model=self.model,
                provider=self._provider_name,
                prompt=prompt_text,
                error=e,
                latency_seconds=latency,
                retry_count=transport_retry_count,
                metadata=route_metadata,
            )
            raise

        latency = time.time() - t0
        content = ""
        if response and response.choices:
            content = response.choices[0].message.content or ""

        track_success(
            response=response,
            model=self.model,
            provider=self._provider_name,
            prompt=prompt_text,
            response_text=content,
            latency_seconds=latency,
            metadata=route_metadata,
        )

        return content


def create_llm_from_config(
    config: DispatchEvolveConfig, stage: str = "evolve"
) -> LLMInterface:
    """Create an LLM instance from a DispatchEvolve config for a given stage."""

    model_spec = config.models.get(stage) or config.models.get("evolve")
    if not model_spec:
        raise ValueError(
            f"No model spec for stage '{stage}' and no fallback 'evolve' model."
        )

    if config.provider in (LLMProvider.OPENAI_COMPATIBLE, LLMProvider.OPENAI):
        return LiteLLMWrapper(model_spec, config.api_base, config.api_key, provider=config.provider)
    elif config.provider == LLMProvider.GEMINI:
        return LiteLLMWrapper(
            model_spec, api_base=config.api_base, api_key=config.api_key,
            provider=config.provider,
        )
    elif config.provider == LLMProvider.VERTEX_AI:
        vertex_project = (
            config.scene_evolve.get("vertex_project")
            or config.general_evolve.get("vertex_project")
            or os.environ.get("VERTEXAI_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        vertex_location = (
            config.scene_evolve.get("vertex_location")
            or config.general_evolve.get("vertex_location")
            or os.environ.get("VERTEXAI_LOCATION")
            or "us-central1"
        )
        return LiteLLMWrapper(
            model_spec, provider=config.provider, api_key=config.api_key,
            vertex_project=vertex_project, vertex_location=vertex_location,
        )
    elif config.provider == LLMProvider.CUSTOM:
        if not config.init_client:
            raise ValueError(
                "CUSTOM provider requires 'init_client' callable in config."
            )
        return config.init_client(model_spec, config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

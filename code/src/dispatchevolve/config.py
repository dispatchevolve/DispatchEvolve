import os
import warnings
import yaml
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from dacite import from_dict, Config as DaciteConfig

_PROVIDER_ALIASES = {}

class LLMProvider(str, Enum):
    OPENAI_COMPATIBLE = "openai_compatible"
    OPENAI = "openai"
    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            lower = value.lower()
            for member in cls:
                if member.value.lower() == lower:
                    return member
        return None

@dataclass
class ModelSpec:
    name: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    weight: float = 1.0

@dataclass
class DispatchEvolveConfig:
    provider: LLMProvider = LLMProvider.OPENAI_COMPATIBLE
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    scene_evolve: Dict[str, Any] = field(default_factory=dict)
    general_evolve: Dict[str, Any] = field(default_factory=dict)
    init_client: Optional[Callable] = None
    call_logging: bool = True

    def __post_init__(self):
        # Auto-convert dict values in models to ModelSpec
        if self.models:
            for key, value in list(self.models.items()):
                if isinstance(value, dict):
                    self.models[key] = ModelSpec(**value)
        # Resolve environment variables in api_key
        if self.api_key and isinstance(self.api_key, str):
            if self.api_key.startswith("${") and self.api_key.endswith("}"):
                env_var = self.api_key[2:-1]
                self.api_key = os.environ.get(env_var, self.api_key)

        # Fallback to known env vars if not set
        if not self.api_key:
            if self.provider in ("openai_compatible", LLMProvider.OPENAI_COMPATIBLE):
                self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
            elif self.provider in ("openai", LLMProvider.OPENAI):
                self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif self.provider in ("gemini", LLMProvider.GEMINI):
                self.api_key = os.environ.get("GOOGLE_API_KEY")
            elif self.provider in ("vertex_ai", LLMProvider.VERTEX_AI):
                self.api_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    @classmethod
    def from_yaml(cls, path: str) -> "DispatchEvolveConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        provider_val = data.get("provider", "")
        if isinstance(provider_val, str) and provider_val.lower() in _PROVIDER_ALIASES:
            new_name = _PROVIDER_ALIASES[provider_val.lower()]
            warnings.warn(
                f"Provider '{provider_val}' is deprecated, use '{new_name}' instead. "
                f"Auto-mapping applied.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["provider"] = new_name

        return from_dict(data_class=cls, data=data, config=DaciteConfig(cast=[Enum]))

    def validate(self) -> None:
        """Validate the configuration. Raises ValueError on invalid config."""
        errors = []

        if not self.models:
            errors.append("No models configured. At least 'models.evolve' is required.")
        elif "evolve" not in self.models:
            errors.append("Missing 'models.evolve' model spec. This is the minimum required model.")

        if "evolve" in self.models:
            evolve_model = self.models["evolve"]
            if not evolve_model.name:
                errors.append("models.evolve.name must not be empty.")

        if self.provider not in (LLMProvider.CUSTOM, LLMProvider.GEMINI, LLMProvider.VERTEX_AI):
            if not self.api_key:
                errors.append(
                    f"api_key is not set for provider '{self.provider.value if isinstance(self.provider, LLMProvider) else self.provider}'. "
                    f"Set it in config or via environment variable."
                )

        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

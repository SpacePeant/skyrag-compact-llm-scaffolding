"""Pluggable LLM backends.

`get_backend(config)` is the single place a pipeline acquires an LLM, so
swapping local Ollama for hosted OpenRouter is a config change, not a code
change.
"""

from __future__ import annotations

from ..config import Config, LLMConfig
from .base import AuthError, GenerationParams, LLMBackend, LLMError, LLMResponse
from .echo import EchoBackend
from .ollama import OllamaBackend
from .openrouter import OpenRouterBackend

__all__ = [
    "AuthError",
    "EchoBackend",
    "GenerationParams",
    "LLMBackend",
    "LLMError",
    "LLMResponse",
    "OllamaBackend",
    "OpenRouterBackend",
    "get_backend",
    "params_from_config",
]

_BACKENDS = ("openrouter", "ollama", "echo")


def get_backend(config: Config | LLMConfig, *, backend: str | None = None) -> LLMBackend:
    """Instantiate the configured backend.

    `backend` overrides config (used by the CLI's --backend flag).
    """
    llm = config.llm if isinstance(config, Config) else config
    name = (backend or llm.backend or "openrouter").lower()

    if name == "openrouter":
        return OpenRouterBackend(
            api_key=llm.openrouter_api_key,
            base_url=llm.openrouter_base_url,
            default_model=llm.model,
            referer=llm.openrouter_referer,
            title=llm.openrouter_title,
            timeout=llm.timeout,
            max_retries=llm.max_retries,
            provider=llm.provider,
        )
    if name == "ollama":
        return OllamaBackend(
            base_url=llm.ollama_base_url,
            default_model=llm.model,
            timeout=llm.timeout,
        )
    if name == "echo":
        return EchoBackend(default_model=llm.model)

    raise ValueError(f"Unknown LLM backend {name!r}. Choose one of: {', '.join(_BACKENDS)}")


def params_from_config(llm: LLMConfig, **overrides) -> GenerationParams:
    """Build GenerationParams from config, with per-call overrides."""
    return GenerationParams(
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        seed=llm.seed,
        num_ctx=llm.num_ctx,
    ).merged(**overrides)

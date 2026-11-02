"""Backend-agnostic LLM interface.

The notebooks called the LLM three different ways -- `requests.post` to
Ollama's /api/generate, the `ollama` python package, and a commented-out
local transformers model. All three are replaced by this one protocol so a
pipeline never knows which provider it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GenerationParams:
    """Provider-neutral generation settings.

    Each backend maps these onto its own wire format and ignores what it
    cannot express (e.g. OpenRouter has no `num_ctx` -- context length is a
    property of the served model, not a request parameter).
    """

    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None
    num_ctx: int | None = None
    stop: list[str] = field(default_factory=list)

    def merged(self, **overrides) -> "GenerationParams":
        base = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "stop": list(self.stop),
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return GenerationParams(**base)


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    raw: dict = field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    # Why the model stopped. "length" means the reply was cut off mid-token by
    # max_tokens -- a budget event, not a model decision. Callers that parse
    # structured output MUST check this: a truncated JSON object parses to
    # nothing, which is indistinguishable from a model that answered "nothing"
    # unless the cutoff is detected explicitly.
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def __str__(self) -> str:  # so `print(response)` behaves like the notebooks
        return self.text


class LLMError(RuntimeError):
    """Raised when a backend cannot produce a completion."""


class AuthError(LLMError):
    """Missing or rejected credentials -- retrying will not help."""


@runtime_checkable
class LLMBackend(Protocol):
    """Anything that can turn a prompt into text.

    Implementations must be safe to reuse across calls.
    """

    name: str
    retries: int = 0
    throttled_calls: int = 0

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> LLMResponse: ...

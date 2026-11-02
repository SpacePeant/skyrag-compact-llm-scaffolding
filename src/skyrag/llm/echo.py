"""Offline fake backend.

Lets the full pipeline run without network access or spending credits --
used by the tests and by `--backend echo` for smoke-testing wiring.
"""

from __future__ import annotations

from .base import GenerationParams, LLMBackend, LLMResponse


class EchoBackend(LLMBackend):
    name = "echo"

    def __init__(
        self,
        default_model: str = "echo/stub",
        responses: list[str] | None = None,
        finish_reasons: list[str] | None = None,
    ):
        self.default_model = default_model
        # Optional canned replies, popped in order; falls back to echoing.
        self._responses = list(responses or [])
        # Optional canned finish_reasons, popped in step with `responses`. Lets a
        # test script a reply that was cut off by the token budget ("length"),
        # which parses to nothing and must not be read as model reticence.
        self._finish_reasons = list(finish_reasons or [])
        self.calls: list[dict] = []
        self.retries: int = 0
        self.throttled_calls: int = 0

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> LLMResponse:
        params = params or GenerationParams()
        # `params` is recorded alongside the prompt so tests can assert on the
        # token budget a stage asked for. Arm 3 sizes that budget differently
        # from Arm 2 -- a schema entry is structurally longer than a
        # `"key": score` pair -- and an under-budget there reinstates the
        # 2026-08-17 truncation defect in one arm only, which is invisible in
        # every other observable this double offers.
        self.calls.append(
            {
                "prompt": prompt,
                "model": model or self.default_model,
                "system": system,
                "params": params,
            }
        )

        if self._responses:
            text = self._responses.pop(0)
        else:
            text = f"[echo:{model or self.default_model}] {prompt[:400]}"

        finish = self._finish_reasons.pop(0) if self._finish_reasons else "stop"

        return LLMResponse(
            text=text,
            model=model or self.default_model,
            backend=self.name,
            raw={},
            finish_reason=finish,
        )

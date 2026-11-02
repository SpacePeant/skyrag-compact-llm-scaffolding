"""Ollama backend -- preserves the notebooks' original local behaviour.

Kept so results from the paper's setup can still be reproduced on a machine
with the GPU capacity for llama3.1:70b.
"""

from __future__ import annotations

import random
import time

import requests

from .base import GenerationParams, LLMBackend, LLMError, LLMResponse

# Same set the OpenRouter backend retries. A local Ollama returns 503 while a model
# is still loading into VRAM and 429 under concurrent load, so a long index run hit
# these too -- it just had no retry path, and the first blip ended the stage.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        default_model: str = "llama3.1:70b",
        timeout: int = 600,
        session: requests.Session | None = None,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
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
        model = model or self.default_model

        options: dict = {"temperature": params.temperature}
        if params.num_ctx:
            options["num_ctx"] = params.num_ctx
        if params.seed is not None:
            options["seed"] = params.seed
        if params.max_tokens:
            options["num_predict"] = params.max_tokens
        if params.stop:
            options["stop"] = params.stop

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        data = self._post_with_retry(payload)
        if "response" not in data:
            raise LLMError(f"Unexpected Ollama response shape: {data}")

        return LLMResponse(
            text=data["response"],
            model=model,
            backend=self.name,
            raw=data,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason=_finish_reason(data, options.get("num_predict")),
        )

    def _post_with_retry(self, payload: dict) -> dict:
        """POST /api/generate, retrying transient failures with backoff.

        Brought to parity with the OpenRouter backend deliberately. The two
        backends are meant to be interchangeable arms of the same measurement, so a
        difference in *resilience* between them is a difference in results: a
        transient 503 that one backend rides out and the other converts into an
        LLMError becomes OUTCOME_BACKEND_ERROR for one platform and a normal
        selection for the other, from identical model behaviour.
        """
        url = f"{self.base_url}/api/generate"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                self.retries += 1
                last_error = exc
                self._sleep(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS:
                self.retries += 1
                if response.status_code == 429:
                    self.throttled_calls += 1
                last_error = LLMError(
                    f"HTTP {response.status_code} from Ollama: {response.text[:300]}"
                )
                retry_after = response.headers.get("Retry-After")
                self._sleep(attempt, float(retry_after) if _is_number(retry_after) else None)
                continue

            if not response.ok:
                raise LLMError(
                    f"HTTP {response.status_code} from Ollama: {response.text[:500]}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise LLMError(
                    f"Ollama returned non-JSON: {response.text[:300]}"
                ) from exc

        raise LLMError(
            f"Could not reach Ollama at {self.base_url} after {self.max_retries} "
            f"attempt(s). Is `ollama serve` running? Last error: {last_error}"
        )

    def _sleep(self, attempt: int, explicit: float | None = None) -> None:
        # See the note in openrouter.py: sleeping on the last attempt delays the
        # raise and retries nothing.
        if attempt >= self.max_retries - 1:
            return
        delay = explicit if explicit is not None else min(2**attempt, 30) + random.random()
        time.sleep(delay)


def _is_number(value: str | None) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _finish_reason(data: dict, num_predict: int | None) -> str:
    """Why generation stopped, in the vocabulary `LLMResponse.truncated` expects.

    Ollama reports this as `done_reason` and spells the two cases that matter
    the same way the OpenAI-shaped backends do ("stop", "length"), so the value
    passes through unmapped.

    Older Ollama builds omit the field, and recording `None` there is not a
    neutral act: `.truncated` is then False for a reply that was cut off, and
    `select_keywords` books the outcome as OUTCOME_UNPARSEABLE -- "the model
    answered fully and emitted garbage". That charges our own token budget
    against the model under test, in a repository whose entire purpose is
    comparing 8B against 70B. A silent None here does not lose the signal, it
    inverts it.

    When `done_reason` is absent the answer is usually still decidable, and
    decidable exactly, rather than guessed: a reply that emitted precisely the
    number of tokens it was permitted was stopped by the cap. Where even that
    evidence is missing, raise -- a backend that cannot say why it stopped
    cannot be used for this measurement.
    """
    reason = data.get("done_reason")
    if reason:
        return str(reason)

    emitted = data.get("eval_count")
    if num_predict and emitted is not None:
        return "length" if int(emitted) >= int(num_predict) else "stop"

    raise LLMError(
        "Ollama reported neither done_reason nor eval_count, so a truncated reply "
        "cannot be told from a complete one. Every keyword-selection outcome from "
        "this backend would be uninterpretable; upgrade Ollama to a build that "
        f"returns done_reason. Payload keys: {sorted(data)}"
    )

"""OpenRouter backend -- remote LLM provider interface.

Uses httpx to interact with OpenRouter's OpenAI-compatible /chat/completions endpoint.
Supports both synchronous (`generate`) and asynchronous (`generate_async`) calls
with exponential backoff retries.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any

import httpx

from .base import AuthError, GenerationParams, LLMBackend, LLMError, LLMResponse

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterBackend(LLMBackend):
    name = "openrouter"

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "meta-llama/llama-3.1-70b-instruct",
        referer: str | None = None,
        title: str | None = "SkyRag Research Suite",
        timeout: float = 300.0,
        max_retries: int = 4,
        provider: str | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise AuthError(
                "OPENROUTER_API_KEY is not set. Put it in your environment or in a .env "
                "file at the repo root (see .env.example). Get a key at "
                "https://openrouter.ai/keys"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.referer = referer
        self.title = title
        self.timeout = timeout
        env_retries = os.getenv("OPENROUTER_MAX_RETRIES")
        self.max_retries = int(env_retries) if env_retries and env_retries.isdigit() else (max_retries if max_retries != 4 else 10)
        self.provider = provider
        self.retries: int = 0
        self.throttled_calls: int = 0

        # One client per backend, not one per request. Building and closing an
        # httpx.Client inside every `generate` call opened a fresh TCP connection
        # (and a fresh TLS handshake) per LLM call and left each one in TIME_WAIT;
        # the keyword stage issues one call per course, so a full index run
        # exhausted the local ephemeral port range and surfaced as OSError rather
        # than as anything resembling its cause. A persistent client also gets
        # connection reuse, which is most of the per-call latency on this endpoint.
        #
        # `_owns_client` records whether teardown is ours to do: a caller that
        # injected its own client owns that client's lifetime, and closing it here
        # would break the next user of it.
        self._client = client or httpx.Client(timeout=timeout)
        self._async_client = async_client
        self._owns_client = client is None
        self._owns_async_client = False

    def _ensure_async_client(self) -> httpx.AsyncClient:
        """Create the async client on first use.

        Not built in `__init__`: instantiating an AsyncClient binds it to the
        running event loop, and a backend constructed outside `asyncio.run` would
        bind to a loop that is closed by the time anyone awaits it.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_async_client = True
        return self._async_client

    def close(self) -> None:
        """Release the HTTP connections this backend owns. Idempotent."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
            self._owns_client = False

    async def aclose(self) -> None:
        """Async counterpart to `close`, for the `generate_async` path."""
        if self._owns_async_client and self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._owns_async_client = False
        self.close()

    def __enter__(self) -> "OpenRouterBackend":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    async def __aenter__(self) -> "OpenRouterBackend":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def _build_payload(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> tuple[str, dict[str, Any]]:
        params = params or GenerationParams()
        target_model = model or self.default_model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.stop:
            payload["stop"] = params.stop
        # Pin routing when a provider was named, and refuse the silent fallback:
        # a benchmark that lets the request land wherever is cheapest measures the
        # provider as much as the model. `allow_fallbacks=False` is the half that
        # makes it a pin rather than a preference -- with fallbacks on, an outage
        # at the named provider quietly produces a row served by another one, and
        # nothing in the export would say so.
        if self.provider:
            payload["provider"] = {
                "order": [self.provider],
                "allow_fallbacks": False,
            }

        return target_model, payload

    def _parse_response(
        self, data: dict[str, Any], target_model: str, latency_ms: float
    ) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            text = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Unexpected OpenRouter response shape for model {target_model!r}: {data}"
            ) from exc

        if not text.strip():
            finish = choice.get("finish_reason")
            raise LLMError(
                f"Model {target_model!r} returned empty content (finish_reason={finish!r}). "
                "This often means the prompt exceeded the model's context window."
            )

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", target_model),
            backend=self.name,
            raw=data,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=round(latency_ms, 2),
            finish_reason=choice.get("finish_reason"),
        )

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> LLMResponse:
        target_model, payload = self._build_payload(prompt, model=model, system=system, params=params)
        url = f"{self.base_url}/chat/completions"
        start_time = time.perf_counter()

        if self._client is None:
            raise LLMError(
                "This OpenRouterBackend has been closed. Build a new one rather "
                "than reusing a closed backend."
            )

        data = self._post_sync_with_retry(self._client, url, payload)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return self._parse_response(data, target_model, latency_ms)

    async def generate_async(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> LLMResponse:
        target_model, payload = self._build_payload(prompt, model=model, system=system, params=params)
        url = f"{self.base_url}/chat/completions"
        start_time = time.perf_counter()

        data = await self._post_async_with_retry(self._ensure_async_client(), url, payload)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return self._parse_response(data, target_model, latency_ms)

    def _post_sync_with_retry(self, client: httpx.Client, url: str, payload: dict) -> dict:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = client.post(url, headers=self._headers(), json=payload)
            except httpx.RequestError as exc:
                self.retries += 1
                last_error = exc
                self._sleep(attempt)
                continue

            if response.status_code in (401, 403):
                raise AuthError(
                    f"OpenRouter rejected the API key (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )

            if response.status_code in _RETRYABLE_STATUS:
                self.retries += 1
                if response.status_code == 429:
                    self.throttled_calls += 1
                last_error = LLMError(
                    f"HTTP {response.status_code} from OpenRouter: {response.text[:300]}"
                )
                retry_after = response.headers.get("Retry-After")
                sleep_delay = float(retry_after) if _is_number(retry_after) else None
                if response.status_code == 429 and sleep_delay is None:
                    sleep_delay = min(2 ** (attempt + 1), 60) + random.random()
                self._sleep(attempt, sleep_delay)
                continue

            if not response.is_success:
                raise LLMError(
                    f"HTTP {response.status_code} from OpenRouter: {response.text[:500]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError(f"OpenRouter returned non-JSON: {response.text[:300]}") from exc

            if "error" in data and "choices" not in data:
                message = data["error"]
                message = message.get("message", message) if isinstance(message, dict) else message
                msg_str = str(message).lower()
                if any(fatal in msg_str for fatal in ("credit", "insufficient", "unauthorized", "invalid key", "payment")):
                    raise LLMError(f"OpenRouter error: {message}")
                self.retries += 1
                last_error = LLMError(f"OpenRouter error: {message}")
                self._sleep(attempt, min(2 ** (attempt + 1), 60) + random.random())
                continue

            return data

        raise LLMError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_error}"
        )

    async def _post_async_with_retry(self, client: httpx.AsyncClient, url: str, payload: dict) -> dict:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(url, headers=self._headers(), json=payload)
            except httpx.RequestError as exc:
                self.retries += 1
                last_error = exc
                await self._async_sleep(attempt)
                continue

            if response.status_code in (401, 403):
                raise AuthError(
                    f"OpenRouter rejected the API key (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )

            if response.status_code in _RETRYABLE_STATUS:
                self.retries += 1
                if response.status_code == 429:
                    self.throttled_calls += 1
                last_error = LLMError(
                    f"HTTP {response.status_code} from OpenRouter: {response.text[:300]}"
                )
                retry_after = response.headers.get("Retry-After")
                sleep_delay = float(retry_after) if _is_number(retry_after) else None
                if response.status_code == 429 and sleep_delay is None:
                    sleep_delay = min(2 ** (attempt + 1), 60) + random.random()
                await self._async_sleep(attempt, sleep_delay)
                continue

            if not response.is_success:
                raise LLMError(
                    f"HTTP {response.status_code} from OpenRouter: {response.text[:500]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError(f"OpenRouter returned non-JSON: {response.text[:300]}") from exc

            if "error" in data and "choices" not in data:
                message = data["error"]
                message = message.get("message", message) if isinstance(message, dict) else message
                msg_str = str(message).lower()
                if any(fatal in msg_str for fatal in ("credit", "insufficient", "unauthorized", "invalid key", "payment")):
                    raise LLMError(f"OpenRouter error: {message}")
                self.retries += 1
                last_error = LLMError(f"OpenRouter error: {message}")
                await self._async_sleep(attempt, min(2 ** (attempt + 1), 60) + random.random())
                continue

            return data

        raise LLMError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_error}"
        )

    def _sleep(self, attempt: int, explicit: float | None = None) -> None:
        # No sleep after the final attempt: the loop is about to exit and raise, so
        # waiting first delays the error by up to 30s and retries nothing. At the
        # default max_retries=4 that was a pointless ~8s on every exhausted request.
        if attempt >= self.max_retries - 1:
            return
        delay = explicit if explicit is not None else min(2**attempt, 30) + random.random()
        time.sleep(delay)

    async def _async_sleep(self, attempt: int, explicit: float | None = None) -> None:
        if attempt >= self.max_retries - 1:
            return
        delay = explicit if explicit is not None else min(2**attempt, 30) + random.random()
        await asyncio.sleep(delay)


def _is_number(value: str | None) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

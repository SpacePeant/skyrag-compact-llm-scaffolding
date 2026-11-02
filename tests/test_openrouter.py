"""Unit tests for OpenRouter LLM Backend."""

import pytest
import httpx
from skyrag.llm import AuthError, LLMError, OpenRouterBackend, GenerationParams


def test_openrouter_missing_key():
    """Verify AuthError is raised if api_key is None or empty."""
    with pytest.raises(AuthError):
        OpenRouterBackend(api_key=None)
    with pytest.raises(AuthError):
        OpenRouterBackend(api_key="")


def test_openrouter_headers():
    """Verify request headers include Auth, Referer, and Title."""
    backend = OpenRouterBackend(
        api_key="test-key",
        referer="https://example.com",
        title="Test App",
    )
    headers = backend._headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "Test App"


def test_openrouter_generate_sync():
    """Verify synchronous generate call using httpx MockTransport."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Generated learning path."}}],
                "model": "meta-llama/llama-3.1-70b-instruct",
                "usage": {"prompt_tokens": 50, "completion_tokens": 100},
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)
    backend = OpenRouterBackend(api_key="test-key", client=client)

    res = backend.generate("Recommend a learning path.", params=GenerationParams(temperature=0.5))
    assert res.text == "Generated learning path."
    assert res.model == "meta-llama/llama-3.1-70b-instruct"
    assert res.prompt_tokens == 50
    assert res.completion_tokens == 100
    assert res.latency_ms is not None and res.latency_ms >= 0


def test_openrouter_generate_async():
    """Verify asynchronous generate_async call using httpx MockTransport."""
    import asyncio

    async def _run():
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Async generated content."}}],
                    "model": "anthropic/claude-3.5-sonnet",
                    "usage": {"prompt_tokens": 40, "completion_tokens": 80},
                },
            )

        transport = httpx.MockTransport(mock_handler)
        async_client = httpx.AsyncClient(transport=transport)
        backend = OpenRouterBackend(api_key="test-key", async_client=async_client)

        res = await backend.generate_async("Async query", model="anthropic/claude-3.5-sonnet")
        assert res.text == "Async generated content."
        assert res.model == "anthropic/claude-3.5-sonnet"
        assert res.prompt_tokens == 40
        assert res.completion_tokens == 80

    asyncio.run(_run())


def test_openrouter_auth_error_status():
    """Verify HTTP 401 raises AuthError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized API Key")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)
    backend = OpenRouterBackend(api_key="bad-key", client=client, max_retries=1)

    with pytest.raises(AuthError, match="rejected the API key"):
        backend.generate("Test prompt")

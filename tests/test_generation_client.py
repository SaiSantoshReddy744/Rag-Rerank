from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from app.core.generation_client import GenerationClient, GenerationProviderError

KEY = "test-openai-key"
_REAL_HTTPX_CLIENT = httpx.Client


def _settings(**overrides):
    values = {
        "openai_api_key": KEY,
        "openai_model": "gpt-6-astra",
        "openai_api_base": "https://api.openai.com/v1",
        "generation_timeout_seconds": 10,
        "generation_retry_count": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install(monkeypatch: pytest.MonkeyPatch, handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, len(seen))

    def make_client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(wrapped)
        return _REAL_HTTPX_CLIENT(**kwargs)

    monkeypatch.setattr(httpx, "Client", make_client)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    return seen


def _response(text: str) -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
    }


def test_generate_sends_responses_api_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, lambda request, _count: httpx.Response(200, json=_response("Hello")))
    result = GenerationClient(_settings()).generate("Say hello")

    assert result.text == "Hello"
    assert len(seen) == 1
    assert str(seen[0].url) == "https://api.openai.com/v1/responses"
    assert seen[0].method == "POST"
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"
    payload = __import__("json").loads(seen[0].content)
    assert payload == {"model": "gpt-6-astra", "input": "Say hello", "store": False}


def test_authentication_rate_limit_and_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, lambda _request, _count: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(GenerationProviderError) as auth_error:
        GenerationClient(_settings()).generate("p")
    assert auth_error.value.code == "openai_invalid_key"
    assert KEY not in str(auth_error.value)

    _install(monkeypatch, lambda _request, _count: httpx.Response(429, json={"error": {"message": "quota"}}))
    with pytest.raises(GenerationProviderError) as rate_error:
        GenerationClient(_settings()).generate("p")
    assert rate_error.value.code == "openai_rate_limited"
    assert rate_error.value.http_status == 429

    _install(monkeypatch, lambda _request, _count: httpx.Response(404, json={"error": {"message": "not found"}}))
    with pytest.raises(GenerationProviderError) as model_error:
        GenerationClient(_settings()).generate("p")
    assert model_error.value.code == "openai_model_unavailable"


def test_empty_output_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, lambda _request, _count: httpx.Response(200, json={"status": "completed", "output": []}))
    with pytest.raises(GenerationProviderError) as exc:
        GenerationClient(_settings()).generate("p")
    assert exc.value.code == "openai_empty"


def test_retry_on_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(
        monkeypatch,
        lambda _request, count: (
            httpx.Response(503, json={"error": {"message": "unavailable"}})
            if count == 1
            else httpx.Response(200, json=_response("Recovered"))
        ),
    )
    result = GenerationClient(_settings(generation_retry_count=1)).generate("p")
    assert result.text == "Recovered"
    assert len(seen) == 2

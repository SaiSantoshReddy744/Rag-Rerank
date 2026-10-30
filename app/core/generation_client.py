from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GenerationResult:
    text: str


class GenerationProviderError(RuntimeError):
    """Raised for OpenAI API failures. `code` and `http_status` are safe to show."""

    def __init__(self, message: str, code: str = "generation_failed", http_status: int = 502):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def _api_error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", {})
        return str(error.get("message", ""))[:300] if isinstance(error, dict) else ""
    except Exception:
        return ""


class GenerationClient:
    """Calls OpenAI's Responses API using a server-side bearer token."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.openai_api_key.strip()
        self._model = settings.openai_model.strip()
        self._base = settings.openai_api_base.rstrip("/")
        self._timeout = settings.generation_timeout_seconds
        self._retry_count = settings.generation_retry_count

    @property
    def model(self) -> str:
        return self._model

    def check_configured(self) -> None:
        self.check_key_only()
        if not self._model:
            raise GenerationProviderError(
                "OPENAI_MODEL is not set. Choose a model available to your API project and set it in .env.",
                "openai_model_missing",
                503,
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> GenerationResult:
        self.check_configured()
        url = f"{self._base}/responses"
        payload: dict[str, str | bool] = {"model": self._model, "input": prompt}
        # Groq's Responses API does not accept OpenAI's `store` option.
        if "api.groq.com" not in self._base:
            payload["store"] = False
        last: GenerationProviderError | None = None
        with httpx.Client(timeout=self._timeout) as client:
            for attempt in range(self._retry_count + 1):
                try:
                    response = client.post(url, json=payload, headers=self._headers())
                except httpx.TimeoutException:
                    last = GenerationProviderError(
                        "OpenAI request timed out.", "openai_timeout", 504
                    )
                except httpx.HTTPError as exc:
                    last = GenerationProviderError(
                        f"Could not reach the OpenAI API ({type(exc).__name__}). Check your internet connection.",
                        "openai_unreachable",
                        502,
                    )
                else:
                    if response.status_code == 200:
                        return GenerationResult(text=self._parse(response))
                    last = self._map_error(response)
                    if response.status_code not in _TRANSIENT_STATUS:
                        raise last
                if attempt < self._retry_count:
                    time.sleep(min(1.0 * (2**attempt), 8.0))
        assert last is not None
        raise last

    def _parse(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError as exc:
            raise GenerationProviderError("OpenAI returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise GenerationProviderError("OpenAI returned an invalid response object.")
        blocks: list[str] = []
        output_items = data.get("output", [])
        if not isinstance(output_items, list):
            output_items = []
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        blocks.append(text)
        text = "\n".join(blocks).strip()
        if not text:
            status = data.get("status", "unknown")
            raise GenerationProviderError(
                f"OpenAI returned no text (response status={status}).", "openai_empty", 502
            )
        return text

    def _map_error(self, response: httpx.Response) -> GenerationProviderError:
        status = response.status_code
        detail = _api_error_message(response)
        suffix = f" OpenAI says: {detail}" if detail else ""
        if status in (401, 403):
            return GenerationProviderError(
                "OpenAI rejected the API key or project access." + suffix,
                "openai_invalid_key",
                status,
            )
        if status == 404:
            return GenerationProviderError(
                f"Model '{self._model}' was not found or is unavailable to this API project." + suffix,
                "openai_model_unavailable",
                502,
            )
        if status == 429:
            return GenerationProviderError(
                "OpenAI rate limit or quota was reached. Check the API project limits and billing." + suffix,
                "openai_rate_limited",
                429,
            )
        if status in (500, 502, 503, 504):
            return GenerationProviderError(
                "OpenAI is temporarily unavailable. Try again shortly." + suffix,
                "openai_unavailable",
                503,
            )
        return GenerationProviderError(
            f"OpenAI request failed with HTTP {status}." + suffix, "openai_error", 502
        )

    def list_models(self) -> list[dict[str, object]]:
        """List model IDs visible to this API key."""
        self.check_key_only()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(f"{self._base}/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise GenerationProviderError(
                f"Could not reach the OpenAI API ({type(exc).__name__}).",
                "openai_unreachable",
            ) from exc
        if response.status_code != 200:
            raise self._map_error(response)
        try:
            data: Any = response.json()
        except ValueError as exc:
            raise GenerationProviderError("OpenAI returned invalid model-list JSON.") from exc
        return [
            {"name": item["id"], "display_name": item["id"]}
            for item in data.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

    def check_key_only(self) -> None:
        if not self._api_key:
            raise GenerationProviderError(
                "OPENAI_API_KEY is not set. Add your API key to .env; do not paste it into chat.",
                "openai_key_missing",
                503,
            )

    def is_configured(self) -> bool:
        return bool(self._api_key and self._model)

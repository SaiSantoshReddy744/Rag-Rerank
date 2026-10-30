from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[Any]: ...

    def decode(self, tokens: list[Any]) -> str: ...


class _WhitespaceEncoder:
    """Offline approximation used only when tiktoken's vocabulary is unavailable."""

    def encode(self, text: str) -> list[Any]:
        return re.findall(r"\s+|[^\s]+", text)

    def decode(self, tokens: list[Any]) -> str:
        return "".join(str(token) for token in tokens)


@lru_cache(maxsize=1)
def get_token_encoder() -> TokenEncoder:
    try:
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get(
            "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")
        )
        cache_key = hashlib.sha1(
            b"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        ).hexdigest()
        if not Path(cache_dir, cache_key).is_file():
            raise FileNotFoundError("cl100k_base vocabulary is not cached")
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("tiktoken vocabulary unavailable; using approximate offline token counts")
        return _WhitespaceEncoder()

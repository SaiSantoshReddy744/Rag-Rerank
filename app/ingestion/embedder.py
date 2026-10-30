from __future__ import annotations

from functools import cached_property
from typing import Any, cast

from app.config import Settings


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @cached_property
    def _model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self._settings.embedding_model_name, device=self._settings.embedding_device
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self._settings.embedding_batch_size,
        )
        return cast(list[list[float]], embeddings.astype(float).tolist())

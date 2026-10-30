from __future__ import annotations

import sys
import types

import pytest

from app.config import Settings
from app.ingestion.embedder import Embedder


class _FakeArray:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def astype(self, _: type[float]) -> _FakeArray:
        return self

    def tolist(self) -> list[list[float]]:
        return self._values


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
        batch_size: int,
    ) -> _FakeArray:
        _ = normalize_embeddings, convert_to_numpy, show_progress_bar, batch_size
        values = [[float(len(text)), 1.0] for text in texts]
        return _FakeArray(values)


def test_embedder_uses_configured_device(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.__dict__["SentenceTransformer"] = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    settings = Settings(embedding_model_name="fake-model", embedding_device="cuda")
    embedder = Embedder(settings)

    assert embedder.embed_texts(["hello"]) == [[5.0, 1.0]]
    assert embedder._model.device == "cuda"
    assert embedder._model.model_name == "fake-model"

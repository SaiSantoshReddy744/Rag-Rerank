from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.core.reranker import RankedChunk
from app.core.retrieval import Retriever
from app.core.vectorstore import RetrievalFilters, VectorStore, normalize_similarity_score


@dataclass
class _FakeResult:
    score: float
    payload: dict[str, object]


class _FakeClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []

    def get_collections(self) -> object:
        return type("Collections", (), {"collections": []})()

    def create_collection(self, **_: object) -> None:
        return None

    def search(self, **kwargs: object) -> list[_FakeResult]:
        self.search_calls.append(kwargs)
        return [
            _FakeResult(
                score=1.4,
                payload={
                    "doc_id": "doc-b",
                    "document_fingerprint": "fingerprint-b",
                    "filename": "b.pdf",
                    "chunk_id": "chunk-b",
                    "chunk_index": 2,
                    "text": "B chunk",
                    "page_number": 2,
                    "section_title": "Beta",
                    "source_path": "/tmp/b.pdf",
                },
            ),
            _FakeResult(
                score=1.0,
                payload={
                    "doc_id": "doc-a",
                    "document_fingerprint": "fingerprint-a",
                    "filename": "a.pdf",
                    "chunk_id": "chunk-a",
                    "chunk_index": 1,
                    "text": "A chunk",
                    "page_number": 1,
                    "section_title": "Alpha",
                    "source_path": "/tmp/a.pdf",
                },
            ),
            _FakeResult(
                score=0.2,
                payload={
                    "doc_id": "doc-c",
                    "document_fingerprint": "fingerprint-c",
                    "filename": "c.pdf",
                    "chunk_id": "chunk-c",
                    "chunk_index": 3,
                    "text": "C chunk",
                    "page_number": 3,
                    "section_title": "Gamma",
                    "source_path": "/tmp/c.pdf",
                },
            ),
        ]

    def scroll(self, **_: object) -> tuple[list[object], None]:
        return ([], None)

    def upsert(self, **_: object) -> None:
        return None

    def delete(self, **_: object) -> None:
        return None


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


class _FakeReranker:
    def rank(self, query: str, passages: list[str]) -> list[RankedChunk]:
        del query
        if len(passages) < 2:
            return [RankedChunk(index=index, score=0.0) for index in range(len(passages))]
        return [
            RankedChunk(index=1, score=2.0),
            RankedChunk(index=0, score=1.0),
        ]


def test_similarity_score_is_normalized() -> None:
    assert normalize_similarity_score(-0.3) == 0.0
    assert normalize_similarity_score(0.4) == 0.4
    assert normalize_similarity_score(1.7) == 1.0


def test_retriever_applies_threshold_and_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(qdrant_url="http://example", retrieval_similarity_threshold=0.3)
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore)

    result = retriever.retrieve(
        "query",
        top_k=2,
        similarity_threshold=0.3,
        filters=RetrievalFilters(doc_id="doc-a"),
    )

    assert [chunk.doc_id for chunk in result.chunks] == ["doc-a", "doc-b"]
    assert result.chunks[0].score == 1.0
    assert result.chunks[1].score == 1.0
    assert result.candidate_reranker_scores is None
    assert fake_client.search_calls
    assert fake_client.search_calls[0]["query_filter"] is not None


def test_retriever_preserves_order_when_reranking_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        retrieval_similarity_threshold=0.3,
        re_rank_enabled=False,
        rerank_candidate_k=5,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(
        settings,
        _FakeEmbedder(),
        vectorstore,
        reranker=_FakeReranker(),
    )

    result = retriever.retrieve(
        "query",
        top_k=2,
        candidate_k=3,
        similarity_threshold=0.0,
    )

    assert [chunk.doc_id for chunk in result.chunks] == ["doc-a", "doc-b"]
    assert len(result.candidate_chunks) == 3
    assert len(result.chunks) == 2
    assert result.candidate_reranker_scores is None
    assert result.retrieval_config.candidate_k == 3
    assert result.retrieval_config.top_k == 2
    assert result.retrieval_config.reranker_enabled is False


def test_retriever_supports_candidate_pool_larger_than_top_k_when_reranking_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        re_rank_enabled=False,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(
        settings,
        _FakeEmbedder(),
        vectorstore,
        reranker=_FakeReranker(),
    )

    result = retriever.retrieve(
        "query",
        top_k=2,
        candidate_k=3,
    )

    assert len(result.candidate_chunks) == 3
    assert len(result.chunks) == 2
    assert result.candidate_reranker_scores is None
    assert result.retrieval_config.candidate_k == 3
    assert result.retrieval_config.top_k == 2
    assert result.retrieval_config.reranker_enabled is False


def test_retriever_never_uses_candidate_pool_smaller_than_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        re_rank_enabled=False,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore)

    result = retriever.retrieve(
        "query",
        top_k=3,
        candidate_k=1,
    )

    assert result.retrieval_config.candidate_k == 3
    assert result.retrieval_config.top_k == 3


def test_retriever_uses_reranker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        retrieval_similarity_threshold=0.3,
        re_rank_enabled=True,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore, reranker=_FakeReranker())

    result = retriever.retrieve("query", top_k=2, similarity_threshold=0.3)

    assert [chunk.doc_id for chunk in result.chunks] == ["doc-b", "doc-a"]
    assert result.candidate_reranker_scores == [1.0, 2.0]
    assert result.retrieval_config.candidate_k == 20
    assert result.retrieval_config.top_k == 2
    assert result.retrieval_config.reranker_enabled is True


def test_retriever_leaves_candidate_pool_unchanged_when_filter_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        re_rank_enabled=True,
        retrieval_structural_filter_enabled=False,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore, reranker=_FakeReranker())

    result = retriever.retrieve("query", top_k=2, candidate_k=3)

    assert result.original_candidate_count == 3
    assert result.filtered_candidate_count == 3
    assert result.candidate_chunks == result.filtered_candidate_chunks
    assert result.candidate_structure_categories == ["SUBSTANTIVE", "SUBSTANTIVE", "SUBSTANTIVE"]
    assert result.filtered_candidate_structure_categories == [
        "SUBSTANTIVE",
        "SUBSTANTIVE",
        "SUBSTANTIVE",
    ]
    assert result.candidate_reranker_scores == [1.0, 2.0, None]
    assert result.retrieval_config.structural_filter_enabled is False


def test_retriever_structural_filter_excludes_only_structural_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_results = [
        _FakeResult(
            score=0.99,
            payload={
                "doc_id": "doc-1",
                "document_fingerprint": "fingerprint-1",
                "filename": "sample.pdf",
                "chunk_id": "chunk-contents",
                "chunk_index": 0,
                "text": "Table of contents",
                "page_number": 1,
                "section_title": "Contents",
                "source_path": "/tmp/sample.pdf",
            },
        ),
        _FakeResult(
            score=0.98,
            payload={
                "doc_id": "doc-1",
                "document_fingerprint": "fingerprint-1",
                "filename": "sample.pdf",
                "chunk_id": "chunk-index",
                "chunk_index": 1,
                "text": "Index",
                "page_number": 2,
                "section_title": "Index",
                "source_path": "/tmp/sample.pdf",
            },
        ),
        _FakeResult(
            score=0.97,
            payload={
                "doc_id": "doc-1",
                "document_fingerprint": "fingerprint-1",
                "filename": "sample.pdf",
                "chunk_id": "chunk-copyright",
                "chunk_index": 2,
                "text": "Copyright 2024 Example",
                "page_number": 3,
                "section_title": "Copyright",
                "source_path": "/tmp/sample.pdf",
            },
        ),
        _FakeResult(
            score=0.96,
            payload={
                "doc_id": "doc-1",
                "document_fingerprint": "fingerprint-1",
                "filename": "sample.pdf",
                "chunk_id": "chunk-glossary",
                "chunk_index": 3,
                "text": "Glossary",
                "page_number": 4,
                "section_title": "Glossary",
                "source_path": "/tmp/sample.pdf",
            },
        ),
        _FakeResult(
            score=0.50,
            payload={
                "doc_id": "doc-1",
                "document_fingerprint": "fingerprint-1",
                "filename": "sample.pdf",
                "chunk_id": "chunk-substantive-1",
                "chunk_index": 4,
                "text": "Substantive evidence one",
                "page_number": 5,
                "section_title": "Chapter 1",
                "source_path": "/tmp/sample.pdf",
            },
        ),
        _FakeResult(
            score=0.40,
            payload={
                "doc_id": "doc-1",
                "document_fingerprint": "fingerprint-1",
                "filename": "sample.pdf",
                "chunk_id": "chunk-substantive-2",
                "chunk_index": 5,
                "text": "Substantive evidence two",
                "page_number": 6,
                "section_title": "Chapter 2",
                "source_path": "/tmp/sample.pdf",
            },
        ),
    ]

    class StructuralFakeClient(_FakeClient):
        def search(self, **kwargs: object) -> list[_FakeResult]:
            self.search_calls.append(kwargs)
            return fake_results

    fake_client = StructuralFakeClient()

    def fake_qdrant_client(url: str) -> StructuralFakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        re_rank_enabled=True,
        retrieval_structural_filter_enabled=True,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore, reranker=_FakeReranker())

    result = retriever.retrieve("query", top_k=2, candidate_k=20, similarity_threshold=0.0)

    assert fake_client.search_calls[0]["limit"] == 20
    assert result.original_candidate_count == 6
    assert result.filtered_candidate_count == 2
    assert result.candidate_structure_categories == [
        "CONTENTS",
        "INDEX",
        "COPYRIGHT",
        "GLOSSARY",
        "SUBSTANTIVE",
        "SUBSTANTIVE",
    ]
    assert result.filtered_candidate_structure_categories == ["SUBSTANTIVE", "SUBSTANTIVE"]
    assert [chunk.chunk_id for chunk in result.candidate_chunks] == [
        "chunk-contents",
        "chunk-index",
        "chunk-copyright",
        "chunk-glossary",
        "chunk-substantive-1",
        "chunk-substantive-2",
    ]
    assert [chunk.score for chunk in result.candidate_chunks] == [0.99, 0.98, 0.97, 0.96, 0.5, 0.4]
    assert [chunk.chunk_id for chunk in result.filtered_candidate_chunks] == [
        "chunk-substantive-1",
        "chunk-substantive-2",
    ]
    assert result.candidate_reranker_scores == [None, None, None, None, 1.0, 2.0]
    assert [chunk.chunk_id for chunk in result.chunks] == [
        "chunk-substantive-2",
        "chunk-substantive-1",
    ]
    assert [chunk.score for chunk in result.chunks] == [0.4, 0.5]
    assert result.retrieval_config.structural_filter_enabled is True

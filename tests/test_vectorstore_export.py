from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from app.config import Settings
from app.core.vectorstore import VectorStore


@dataclass(frozen=True)
class _FakeRecord:
    payload: dict[str, object]


class _FakeClient:
    def __init__(self) -> None:
        self.scroll_calls: list[dict[str, object]] = []

    def get_collections(self) -> object:
        return type("Collections", (), {"collections": []})()

    def create_collection(self, **_: object) -> None:
        return None

    def scroll(self, **kwargs: object) -> tuple[list[_FakeRecord], str | None]:
        self.scroll_calls.append(kwargs)
        offset = kwargs.get("offset")
        if offset is None:
            return (
                [
                    _FakeRecord(
                        payload={
                            "doc_id": "doc-123",
                            "document_fingerprint": "fingerprint-1",
                            "filename": "sample.pdf",
                            "chunk_id": "chunk-2",
                            "chunk_index": 2,
                            "page_number": 3,
                            "text": "third chunk",
                            "section_title": "Section C",
                            "source_path": "/tmp/sample.pdf",
                        }
                    ),
                    _FakeRecord(
                        payload={
                            "doc_id": "doc-123",
                            "document_fingerprint": "fingerprint-1",
                            "filename": "sample.pdf",
                            "chunk_id": "chunk-0",
                            "chunk_index": 0,
                            "page_number": 1,
                            "text": "first chunk",
                            "section_title": "Section A",
                            "source_path": "/tmp/sample.pdf",
                        }
                    ),
                ],
                "page-2",
            )
        if offset == "page-2":
            return (
                [
                    _FakeRecord(
                        payload={
                            "doc_id": "doc-123",
                            "document_fingerprint": "fingerprint-1",
                            "filename": "sample.pdf",
                            "chunk_id": "chunk-1",
                            "chunk_index": 1,
                            "page_number": 2,
                            "text": "second chunk",
                            "section_title": None,
                            "source_path": "/tmp/sample.pdf",
                        }
                    )
                ],
                None,
            )
        return ([], None)


def test_list_chunks_for_doc_id_scrolls_all_pages_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(qdrant_url="http://example")
    vectorstore = VectorStore(settings)

    chunks = vectorstore.list_chunks_for_doc_id("doc-123")

    assert [chunk["chunk_id"] for chunk in chunks] == ["chunk-0", "chunk-1", "chunk-2"]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert [chunk["page_number"] for chunk in chunks] == [1, 2, 3]
    assert [chunk["text"] for chunk in chunks] == ["first chunk", "second chunk", "third chunk"]
    assert [chunk["section_title"] for chunk in chunks] == ["Section A", None, "Section C"]
    assert fake_client.scroll_calls

    scroll_filter = cast(Any, fake_client.scroll_calls[0]["scroll_filter"])
    assert scroll_filter.must[0].key == "doc_id"
    assert scroll_filter.must[0].match.value == "doc-123"
    assert fake_client.scroll_calls[0]["with_payload"] is True
    assert fake_client.scroll_calls[0]["with_vectors"] is False

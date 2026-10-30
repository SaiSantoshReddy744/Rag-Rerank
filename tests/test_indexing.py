from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client.http import models as qmodels

from app.config import Settings
from app.ingestion.chunker import Chunk
from app.ingestion.indexer import Indexer, build_chunk_id


@dataclass
class FakeVectorStore:
    points: list[qmodels.PointStruct]

    def __init__(self) -> None:
        self.points = []

    def upsert(self, points: list[qmodels.PointStruct]) -> None:
        self.points.extend(points)


def test_indexer_uses_document_fingerprint_for_chunk_ids() -> None:
    settings = Settings(
        embedding_model_name="model-a", embedding_dimension=3, embedding_version="v1"
    )
    vectorstore = FakeVectorStore()
    indexer = Indexer(settings, vectorstore)
    chunks = [
        Chunk(
            chunk_index=0,
            page_number=1,
            section_title="Intro",
            source_path="/tmp/sample.pdf",
            text="hello world",
            token_count=2,
        )
    ]
    embeddings = [[0.1, 0.2, 0.3]]

    result = indexer.index_chunks(
        doc_id="doc-123",
        document_fingerprint="fingerprint-abc",
        filename="sample.pdf",
        chunks=chunks,
        embeddings=embeddings,
    )

    expected_id = build_chunk_id("fingerprint-abc", 0, "hello world")
    assert result[0].chunk_id == expected_id
    assert len(vectorstore.points) == 1
    point = vectorstore.points[0]
    assert point.payload is not None
    assert point.id == str(uuid.uuid5(uuid.NAMESPACE_URL, expected_id))
    assert point.payload["document_fingerprint"] == "fingerprint-abc"
    assert point.payload["filename"] == "sample.pdf"
    assert point.payload["doc_id"] == "doc-123"

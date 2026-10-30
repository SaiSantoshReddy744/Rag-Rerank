from __future__ import annotations

from app.ingestion.chunker import RecursiveChunker
from app.ingestion.extractor import ExtractedPage
from app.ingestion.indexer import build_chunk_id


def test_chunker_generates_chunks_with_metadata() -> None:
    chunker = RecursiveChunker(max_tokens=40, overlap_tokens=5)
    page = ExtractedPage(
        page_number=2,
        text="Sentence one. Sentence two. " * 20,
        section_title="Overview",
        source_path="/tmp/sample.pdf",
    )

    chunks = chunker.chunk_pages([page])

    assert len(chunks) > 1
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_chunk_id_is_deterministic() -> None:
    fingerprint = "abc123"
    first = build_chunk_id(fingerprint, 0, "hello world")
    second = build_chunk_id(fingerprint, 0, "hello world")
    third = build_chunk_id(fingerprint, 1, "hello world")

    assert first == second
    assert first != third

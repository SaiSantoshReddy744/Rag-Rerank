from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.chunker import Chunk
from app.ingestion.pipeline import IngestionResult
from app.main import app


class _FakeVectorStore:
    def __init__(self, duplicate_exists: bool) -> None:
        self.duplicate_exists = duplicate_exists
        self.deleted_fingerprints: list[str] = []

    def has_document_fingerprint(self, fingerprint: str) -> bool:
        _ = fingerprint
        return self.duplicate_exists

    def delete_document_fingerprint(self, fingerprint: str) -> None:
        self.deleted_fingerprints.append(fingerprint)

    def ping(self) -> bool:
        return True

    def ensure_collection(self) -> None:
        return None


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, Path]] = []
        self.embedder = _FakeEmbedder()

    def ingest_pdf(
        self, doc_id: str, document_fingerprint: str, filename: str, pdf_path: Path
    ) -> IngestionResult:
        self.calls.append((doc_id, document_fingerprint, filename, pdf_path))
        return IngestionResult(
            doc_id=doc_id,
            document_fingerprint=document_fingerprint,
            filename=filename,
            source_path=str(pdf_path),
            chunks=[
                Chunk(
                    chunk_index=0,
                    page_number=1,
                    section_title="Intro",
                    source_path=str(pdf_path),
                    text="hello",
                    token_count=1,
                )
            ],
        )


class _FakeServices:
    def __init__(
        self,
        data_dir: Path,
        duplicate_exists: bool,
        duplicate_policy: Literal["reject", "replace", "allow"],
    ) -> None:
        settings = Settings(data_dir=data_dir, duplicate_upload_policy=duplicate_policy)
        settings.ensure_data_dirs()
        self.settings = settings
        self.vectorstore = _FakeVectorStore(duplicate_exists=duplicate_exists)
        self.pipeline = _FakePipeline()


def test_upload_rejects_duplicate_pdf(tmp_path: Path) -> None:
    original_services = app.state.services
    app.state.services = _FakeServices(tmp_path, duplicate_exists=True, duplicate_policy="reject")
    client = TestClient(app)

    try:
        response = client.post(
            "/v1/upload",
            files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    finally:
        app.state.services = original_services

    assert response.status_code == 409


def test_upload_replaces_duplicate_pdf(tmp_path: Path) -> None:
    original_services = app.state.services
    services = _FakeServices(tmp_path, duplicate_exists=True, duplicate_policy="replace")
    app.state.services = services
    client = TestClient(app)

    try:
        response = client.post(
            "/v1/upload",
            files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    finally:
        app.state.services = original_services

    assert response.status_code == 200
    # Identical-content points have deterministic IDs, so successful upsert replaces them in place.
    # Deleting the fingerprint before ingestion would lose the old document if ingestion failed.
    assert not services.vectorstore.deleted_fingerprints
    assert services.pipeline.calls
    assert services.pipeline.calls[0][2] == "sample.pdf"

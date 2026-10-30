from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.core.vectorstore import VectorStore
from app.ingestion.chunker import Chunk, RecursiveChunker
from app.ingestion.embedder import Embedder
from app.ingestion.extractor import PdfExtractionError, extract_pdf_text
from app.ingestion.indexer import Indexer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    document_fingerprint: str
    filename: str
    source_path: str
    chunks: list[Chunk]


class IngestionPipeline:
    def __init__(self, settings: Settings, vectorstore: VectorStore, embedder: Embedder) -> None:
        self._settings = settings
        self._chunker = RecursiveChunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        self.embedder = embedder
        self._indexer = Indexer(settings, vectorstore)

    def ingest_pdf(
        self, doc_id: str, document_fingerprint: str, filename: str, pdf_path: Path
    ) -> IngestionResult:
        logger.info(
            "ingestion_started",
            extra={
                "doc_id": doc_id,
                "uploaded_filename": filename,
                "source_path": str(pdf_path),
            },
        )
        logger.info(
            "extraction_started",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
        )
        pages = extract_pdf_text(pdf_path)
        logger.info(
            "extraction_completed",
            extra={
                "doc_id": doc_id,
                "uploaded_filename": filename,
                "page_count": len(pages),
            },
        )
        logger.info(
            "chunking_started",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
        )
        chunks = self._chunker.chunk_pages(pages)
        if not chunks:
            raise PdfExtractionError("No extractable text was found in the PDF")
        logger.info(
            "chunking_completed",
            extra={
                "doc_id": doc_id,
                "uploaded_filename": filename,
                "chunk_count": len(chunks),
            },
        )

        logger.info(
            "embedding_started",
            extra={
                "doc_id": doc_id,
                "uploaded_filename": filename,
                "chunk_count": len(chunks),
            },
        )
        logger.info(
            "indexing_started",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
        )
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunk_batch])
            self._indexer.index_chunks(
                doc_id, document_fingerprint, filename, chunk_batch, embeddings
            )
        logger.info(
            "embedding_completed",
            extra={"doc_id": doc_id, "uploaded_filename": filename, "chunk_count": len(chunks)},
        )
        logger.info(
            "indexing_completed",
            extra={
                "doc_id": doc_id,
                "uploaded_filename": filename,
                "chunk_count": len(chunks),
            },
        )
        return IngestionResult(
            doc_id=doc_id,
            document_fingerprint=document_fingerprint,
            filename=filename,
            source_path=str(pdf_path),
            chunks=chunks,
        )

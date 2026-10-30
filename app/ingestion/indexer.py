from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client.http import models as qmodels

from app.config import Settings
from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    chunk_index: int


def build_chunk_id(document_fingerprint: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha256(f"{document_fingerprint}{chunk_index}{chunk_text}".encode()).hexdigest()
    return digest


def build_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class Indexer:
    def __init__(self, settings: Settings, vectorstore: VectorStoreLike) -> None:
        self._settings = settings
        self._vectorstore = vectorstore

    def index_chunks(
        self,
        doc_id: str,
        document_fingerprint: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[IndexedChunk]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points: list[qmodels.PointStruct] = []
        indexed: list[IndexedChunk] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = build_chunk_id(document_fingerprint, chunk.chunk_index, chunk.text)
            payload = {
                "doc_id": doc_id,
                "document_fingerprint": document_fingerprint,
                "filename": filename,
                "chunk_id": chunk_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "token_count": chunk.token_count,
                "source_path": chunk.source_path,
                "embedding_model": self._settings.embedding_model_name,
                "embedding_dimension": self._settings.embedding_dimension,
                "embedding_version": self._settings.embedding_version,
                "text": chunk.text,
            }
            points.append(
                qmodels.PointStruct(
                    id=build_point_id(chunk_id),
                    vector=embedding,
                    payload=payload,
                )
            )
            indexed.append(IndexedChunk(chunk_id=chunk_id, chunk_index=chunk.chunk_index))

        self._vectorstore.upsert(points)
        return indexed


class VectorStoreLike(Protocol):
    def upsert(self, points: list[qmodels.PointStruct]) -> None: ...

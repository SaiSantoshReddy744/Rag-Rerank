from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings

logger = logging.getLogger(__name__)

FilterCondition = (
    qmodels.FieldCondition
    | qmodels.IsEmptyCondition
    | qmodels.IsNullCondition
    | qmodels.HasIdCondition
    | qmodels.HasVectorCondition
    | qmodels.NestedCondition
    | qmodels.Filter
)


@dataclass(frozen=True)
class RetrievalFilters:
    doc_id: str | None = None
    document_fingerprint: str | None = None
    filename: str | None = None
    source_path: str | None = None
    page_number: int | None = None
    page_numbers: tuple[int, ...] | None = None
    chunk_index: int | None = None
    chunk_indices: tuple[int, ...] | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    document_fingerprint: str
    filename: str
    chunk_id: str
    chunk_index: int
    text: str
    raw_score: float
    score: float
    page_number: int
    section_title: str | None
    source_path: str


def normalize_similarity_score(raw_score: float) -> float:
    if raw_score <= 0.0:
        return 0.0
    if raw_score >= 1.0:
        return 1.0
    return raw_score


def _build_filter(filters: RetrievalFilters | None) -> qmodels.Filter | None:
    if filters is None:
        return None

    conditions: list[FilterCondition] = []
    if filters.doc_id is not None:
        conditions.append(
            qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=filters.doc_id))
        )
    if filters.document_fingerprint is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="document_fingerprint",
                match=qmodels.MatchValue(value=filters.document_fingerprint),
            )
        )
    if filters.filename is not None:
        conditions.append(
            qmodels.FieldCondition(key="filename", match=qmodels.MatchValue(value=filters.filename))
        )
    if filters.source_path is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="source_path", match=qmodels.MatchValue(value=filters.source_path)
            )
        )
    if filters.page_number is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="page_number", match=qmodels.MatchValue(value=filters.page_number)
            )
        )
    if filters.page_numbers is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="page_number", match=qmodels.MatchAny(any=list(filters.page_numbers))
            )
        )
    if filters.chunk_index is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="chunk_index", match=qmodels.MatchValue(value=filters.chunk_index)
            )
        )
    if filters.chunk_indices is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="chunk_index", match=qmodels.MatchAny(any=list(filters.chunk_indices))
            )
        )
    if filters.section_title is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="section_title", match=qmodels.MatchValue(value=filters.section_title)
            )
        )

    if not conditions:
        return None
    return qmodels.Filter(must=cast(Any, conditions))


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.qdrant_url.strip():
            self._client = QdrantClient(url=settings.qdrant_url.strip())
        else:
            # Embedded on-disk mode: no Qdrant server or Docker required.
            settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(settings.qdrant_local_path))

    def ensure_collection(self) -> None:
        existing = {collection.name for collection in self._client.get_collections().collections}
        if self._settings.qdrant_collection in existing:
            return

        self._client.create_collection(
            collection_name=self._settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=self._settings.embedding_dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            logger.exception("qdrant ping failed")
            return False

    def upsert(self, points: list[qmodels.PointStruct]) -> None:
        self.ensure_collection()
        self._client.upsert(collection_name=self._settings.qdrant_collection, points=points)

    def has_document_fingerprint(self, document_fingerprint: str) -> bool:
        self.ensure_collection()
        query_filter = qmodels.Filter(
            must=cast(
                Any,
                [
                    qmodels.FieldCondition(
                        key="document_fingerprint",
                        match=qmodels.MatchValue(value=document_fingerprint),
                    )
                ],
            )
        )
        points, _ = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=query_filter,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0

    def delete_document_fingerprint(self, document_fingerprint: str) -> None:
        self.ensure_collection()
        query_filter = qmodels.Filter(
            must=cast(
                Any,
                [
                    qmodels.FieldCondition(
                        key="document_fingerprint",
                        match=qmodels.MatchValue(value=document_fingerprint),
                    )
                ],
            )
        )
        self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=qmodels.FilterSelector(filter=query_filter),
        )

    def list_documents(self) -> list[dict[str, object]]:
        """One entry per indexed document, derived from stored chunk payloads."""
        self.ensure_collection()
        docs: dict[str, dict[str, object]] = {}
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "filename", "page_number"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = str(payload.get("doc_id", ""))
                entry = docs.setdefault(
                    doc_id,
                    {"doc_id": doc_id, "filename": str(payload.get("filename", "")),
                     "chunk_count": 0, "page_count": 0},
                )
                entry["chunk_count"] = cast(int, entry["chunk_count"]) + 1
                entry["page_count"] = max(
                    cast(int, entry["page_count"]), int(payload.get("page_number", 0))
                )
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: str(d["filename"]))

    def has_documents(self) -> bool:
        """Check whether the collection has any points without scanning every chunk."""
        self.ensure_collection()
        points, _ = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    def delete_document(self, doc_id: str) -> None:
        self.ensure_collection()
        flt = qmodels.Filter(
            must=cast(Any, [qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=doc_id))])
        )
        self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=qmodels.FilterSelector(filter=flt),
        )

    def list_chunks_for_doc_id(self, doc_id: str) -> list[dict[str, object]]:
        self.ensure_collection()
        query_filter = qmodels.Filter(
            must=cast(
                Any,
                [qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))],
            )
        )

        offset: Any = None
        exported: list[dict[str, object]] = []

        while True:
            points, offset = self._client.scroll(
                collection_name=self._settings.qdrant_collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                section_title = payload.get("section_title")
                chunk_index = cast(int, payload.get("chunk_index", 0))
                page_number = cast(int, payload.get("page_number", 0))
                exported.append(
                    {
                        "doc_id": str(payload.get("doc_id", "")),
                        "document_fingerprint": str(payload.get("document_fingerprint", "")),
                        "filename": str(payload.get("filename", "")),
                        "chunk_id": str(payload.get("chunk_id", "")),
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                        "text": str(payload.get("text", "")),
                        "section_title": section_title if isinstance(section_title, str) else None,
                        "source_path": str(payload.get("source_path", "")),
                    }
                )
            if offset is None:
                break

        exported.sort(key=lambda item: cast(int, item["chunk_index"]))
        return exported

    def search(
        self,
        vector: list[float],
        top_k: int,
        similarity_threshold: float = 0.0,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        self.ensure_collection()
        query_filter = _build_filter(filters)
        client = cast(Any, self._client)
        if hasattr(client, "query_points"):
            search_result = client.query_points(
                collection_name=self._settings.qdrant_collection,
                query=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            results = search_result.points
        else:
            results = client.search(
                collection_name=self._settings.qdrant_collection,
                query_vector=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        retrieved: list[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            raw_score = float(result.score or 0.0)
            score = normalize_similarity_score(raw_score)
            if score < similarity_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    doc_id=str(payload.get("doc_id", "")),
                    document_fingerprint=str(payload.get("document_fingerprint", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_id=str(payload.get("chunk_id", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    raw_score=raw_score,
                    score=score,
                    page_number=int(payload.get("page_number", 0)),
                    section_title=payload.get("section_title")
                    if isinstance(payload.get("section_title"), str)
                    else None,
                    source_path=str(payload.get("source_path", "")),
                )
            )
        retrieved.sort(
            key=lambda chunk: (
                -chunk.score,
                chunk.doc_id,
                chunk.page_number,
                chunk.chunk_index,
                chunk.chunk_id,
            )
        )
        return retrieved[:top_k]

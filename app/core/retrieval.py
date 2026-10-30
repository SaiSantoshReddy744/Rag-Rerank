from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.core.reranker import RankedChunk
from app.core.structural import (
    STRUCTURAL_FILTER_EXCLUDED_CATEGORIES,
    classify_structure,
)
from app.core.vectorstore import RetrievalFilters, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_name: str
    embedding_version: str
    candidate_k: int
    top_k: int
    similarity_threshold: float
    reranker_enabled: bool
    reranker_model_name: str | None
    structural_filter_enabled: bool


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    candidate_chunks: list[RetrievedChunk]
    filtered_candidate_chunks: list[RetrievedChunk]
    candidate_reranker_scores: list[float | None] | None
    original_candidate_count: int
    filtered_candidate_count: int
    embedding_latency_ms: int
    retrieval_latency_ms: int
    rerank_latency_ms: int
    retrieval_config: RetrievalConfig
    candidate_structure_categories: list[str] | None = None
    filtered_candidate_structure_categories: list[str] | None = None


class EmbedderLike(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class RerankerLike(Protocol):
    def rank(self, query: str, passages: list[str]) -> list[RankedChunk]: ...


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderLike,
        vectorstore: VectorStore,
        reranker: RerankerLike | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
        structural_filter_enabled: bool | None = None,
        rerank_enabled: bool | None = None,
    ) -> RetrievalResult:
        effective_top_k = top_k if top_k is not None else self._settings.retrieval_top_k_default

        if similarity_threshold is None:
            effective_threshold = self._settings.retrieval_similarity_threshold
        else:
            effective_threshold = similarity_threshold

        configured_reranking = self._settings.re_rank_enabled
        reranking_enabled = (
            self._reranker is not None
            and (configured_reranking if rerank_enabled is None else rerank_enabled)
        )
        effective_structural_filter_enabled = (
            structural_filter_enabled
            if structural_filter_enabled is not None
            else self._settings.retrieval_structural_filter_enabled
        )

        effective_candidate_k = (
            candidate_k
            if candidate_k is not None
            else max(effective_top_k, self._settings.rerank_candidate_k)
        )

        effective_candidate_k = max(effective_top_k, effective_candidate_k)

        embedding_start = time.perf_counter()
        query_embedding = self._embedder.embed_texts([query])[0]
        embedding_latency_ms = int((time.perf_counter() - embedding_start) * 1000)

        retrieval_start = time.perf_counter()
        chunks = self._vectorstore.search(
            query_embedding,
            top_k=effective_candidate_k,
            similarity_threshold=effective_threshold,
            filters=filters,
        )
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

        candidate_chunks = list(chunks)
        candidate_structural_categories = [classify_structure(chunk) for chunk in candidate_chunks]
        filtered_candidate_chunks = [
            chunk
            for chunk, category in zip(
                candidate_chunks,
                candidate_structural_categories,
                strict=True,
            )
            if not effective_structural_filter_enabled
            or category not in STRUCTURAL_FILTER_EXCLUDED_CATEGORIES
        ]
        filtered_candidate_structure_categories = [
            category
            for category in candidate_structural_categories
            if not effective_structural_filter_enabled
            or category not in STRUCTURAL_FILTER_EXCLUDED_CATEGORIES
        ]
        filtered_candidate_indices = [
            index
            for index, category in enumerate(candidate_structural_categories)
            if not effective_structural_filter_enabled
            or category not in STRUCTURAL_FILTER_EXCLUDED_CATEGORIES
        ]
        rerank_latency_ms = 0
        candidate_reranker_scores: list[float | None] | None = None

        if reranking_enabled and self._reranker is not None:
            rerank_start = time.perf_counter()

            valid_ranked_passages = []
            ranked_passages = self._reranker.rank(
                query,
                [chunk.text for chunk in filtered_candidate_chunks],
            )
            for ranked_chunk in ranked_passages:
                if 0 <= ranked_chunk.index < len(filtered_candidate_chunks):
                    valid_ranked_passages.append(ranked_chunk)
            candidate_reranker_scores = [None for _ in candidate_chunks]
            for ranked_chunk in valid_ranked_passages:
                if 0 <= ranked_chunk.index < len(filtered_candidate_indices):
                    original_index = filtered_candidate_indices[ranked_chunk.index]
                    candidate_reranker_scores[original_index] = ranked_chunk.score

            rerank_latency_ms = int((time.perf_counter() - rerank_start) * 1000)

            chunks = [
                filtered_candidate_chunks[ranked_chunk.index]
                for ranked_chunk in sorted(
                    valid_ranked_passages,
                    key=lambda item: item.score,
                    reverse=True,
                )[:effective_top_k]
            ]
        else:
            chunks = candidate_chunks[:effective_top_k]

        logger.info(
            "retrieval_completed",
            extra={
                "embedding_latency_ms": embedding_latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
                "candidate_k": effective_candidate_k,
                "final_top_k": effective_top_k,
                "reranker_enabled": reranking_enabled,
                "structural_filter_enabled": effective_structural_filter_enabled,
                "original_candidate_count": len(candidate_chunks),
                "filtered_candidate_count": len(filtered_candidate_chunks),
                "total_ms": (embedding_latency_ms + retrieval_latency_ms + rerank_latency_ms),
            },
        )

        return RetrievalResult(
            chunks=chunks,
            candidate_chunks=candidate_chunks,
            filtered_candidate_chunks=filtered_candidate_chunks,
            candidate_reranker_scores=candidate_reranker_scores,
            original_candidate_count=len(candidate_chunks),
            filtered_candidate_count=len(filtered_candidate_chunks),
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            retrieval_config=RetrievalConfig(
                embedding_model_name=self._settings.embedding_model_name,
                embedding_version=self._settings.embedding_version,
                candidate_k=effective_candidate_k,
                top_k=effective_top_k,
                similarity_threshold=effective_threshold,
                reranker_enabled=reranking_enabled,
                reranker_model_name=(
                    self._settings.re_rank_model_name if reranking_enabled else None
                ),
                structural_filter_enabled=effective_structural_filter_enabled,
            ),
            candidate_structure_categories=candidate_structural_categories,
            filtered_candidate_structure_categories=filtered_candidate_structure_categories,
        )

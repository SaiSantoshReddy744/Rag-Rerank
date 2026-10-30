"""Versioned API routes."""
from __future__ import annotations

import logging
from time import perf_counter
from typing import cast

from fastapi import APIRouter, HTTPException, Request

from app.core.generation_client import GenerationProviderError
from app.core.reranker import RerankerError
from app.core.vectorstore import RetrievalFilters
from app.logging_config import request_id_var
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    LatencyMetrics,
    RankedPassage,
    SourceChunk,
)
from app.services import Services

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    if payload.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented")

    services = cast(Services, request.app.state.services)
    request_id = request_id_var.get()
    total_start = perf_counter()

    # Fail fast on missing OpenAI configuration, before spending time on retrieval.
    try:
        services.generation_client.check_configured()
    except GenerationProviderError as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc

    if not services.vectorstore.has_documents():
        raise HTTPException(status_code=404, detail="No documents indexed yet. Upload a PDF first.")

    try:
        retrieval_result = services.retriever.retrieve(
            payload.message,
            top_k=payload.top_k,
            similarity_threshold=services.settings.retrieval_similarity_threshold,
            filters=RetrievalFilters(doc_id=payload.doc_id),
        )
    except RerankerError as exc:
        logger.exception("rerank_failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Neural reranking failed: {exc}") from exc

    if not retrieval_result.candidate_chunks:
        raise HTTPException(status_code=404, detail="No relevant passages found for this question.")

    prompt_bundle = services.prompt_builder.build(payload.message, retrieval_result.chunks)

    llm_start = perf_counter()
    try:
        answer = services.generation_client.generate(prompt_bundle.prompt).text
    except GenerationProviderError as exc:
        logger.warning("generation_failed code=%s", exc.code, extra={"request_id": request_id})
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
    llm_latency_ms = int((perf_counter() - llm_start) * 1000)
    total_latency_ms = int((perf_counter() - total_start) * 1000)

    latency = LatencyMetrics(
        embedding=retrieval_result.embedding_latency_ms,
        retrieval=retrieval_result.retrieval_latency_ms,
        rerank=retrieval_result.rerank_latency_ms,
        llm=llm_latency_ms,
        total=total_latency_ms,
    )
    included = prompt_bundle.included_sources
    source_models = [
        SourceChunk(
            doc_id=c.doc_id, filename=c.filename, chunk_id=c.chunk_id, chunk_index=c.chunk_index,
            text=c.text, score=c.score, page_number=c.page_number, section_title=c.section_title,
        )
        for c in included
    ]

    reranked = retrieval_result.retrieval_config.reranker_enabled
    cand = retrieval_result.candidate_chunks
    scores = retrieval_result.candidate_reranker_scores
    used_ids = {c.chunk_id for c in included}
    vector_rank = {c.chunk_id: i + 1 for i, c in enumerate(cand)}
    rerank_rank: dict[str, int] = {}
    if reranked:
        for i, c in enumerate(retrieval_result.chunks):
            rerank_rank[c.chunk_id] = i + 1
    score_by_id = (
        {c.chunk_id: scores[i] for i, c in enumerate(cand)} if reranked and scores else {}
    )

    def passage(c, rank: int) -> RankedPassage:  # type: ignore[no-untyped-def]
        return RankedPassage(
            rank=rank, chunk_id=c.chunk_id, filename=c.filename, page_number=c.page_number,
            chunk_index=c.chunk_index, text=c.text, vector_score=c.score,
            rerank_score=score_by_id.get(c.chunk_id),
            vector_rank=vector_rank[c.chunk_id], rerank_rank=rerank_rank.get(c.chunk_id),
            used_in_prompt=c.chunk_id in used_ids,
        )

    before = [passage(c, i + 1) for i, c in enumerate(cand)]
    after = [passage(c, i + 1) for i, c in enumerate(retrieval_result.chunks)] if reranked else []

    return ChatResponse(
        answer=answer,
        sources=source_models,
        latency_ms=latency,
        reranker_enabled=reranked,
        reranker_model=retrieval_result.retrieval_config.reranker_model_name,
        generation_model=services.generation_client.model,
        retrieved_count=len(cand),
        reranked_count=len(retrieval_result.chunks) if reranked else 0,
        before_rerank=before,
        after_rerank=after,
    )

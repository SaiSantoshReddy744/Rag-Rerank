from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ErrorPayload(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorPayload


class HealthResponse(BaseModel):
    status: Literal["ok"]
    qdrant_connected: bool
    openai_configured: bool
    openai_model: str | None = None
    reranker_enabled: bool = False
    reranker_model: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: Literal["ready"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class PdfEvaluationQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    relevant_pages: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=20)


class PdfEvaluationRequest(BaseModel):
    questions: list[PdfEvaluationQuestion] = Field(min_length=1, max_length=30)
    top_k: int = Field(default=5, ge=1, le=20)


class LatencyMetrics(BaseModel):
    embedding: int
    retrieval: int
    rerank: int
    llm: int
    total: int


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float
    page_number: int
    section_title: str | None = None


class RankedPassage(BaseModel):
    rank: int
    chunk_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    vector_score: float
    rerank_score: float | None = None
    vector_rank: int
    rerank_rank: int | None = None
    used_in_prompt: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: LatencyMetrics
    reranker_enabled: bool = False
    reranker_model: str | None = None
    generation_model: str | None = None
    retrieved_count: int = 0
    reranked_count: int = 0
    before_rerank: list[RankedPassage] = Field(default_factory=list)
    after_rerank: list[RankedPassage] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int

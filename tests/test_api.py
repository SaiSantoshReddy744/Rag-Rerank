from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.core.prompt_builder import BuiltPrompt
from app.core.retrieval import RetrievalConfig, RetrievalResult
from app.core.vectorstore import RetrievedChunk
from app.config import get_settings
from app.main import app
from app.models.schemas import ChatRequest


@dataclass(frozen=True)
class FakeRetrievedChunk:
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


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: object | None = None,
    ) -> RetrievalResult:
        _ = query, top_k, similarity_threshold, filters
        chunk = RetrievedChunk(
            doc_id="doc-1",
            document_fingerprint="fingerprint-1",
            filename="sample.pdf",
            chunk_id="chunk-1",
            chunk_index=0,
            text="The document explains the setup flow.",
            raw_score=0.92,
            score=0.92,
            page_number=1,
            section_title="Intro",
            source_path="/tmp/sample.pdf",
        )
        return RetrievalResult(
            chunks=[chunk],
            candidate_chunks=[chunk],
            filtered_candidate_chunks=[chunk],
            original_candidate_count=1,
            filtered_candidate_count=1,
            candidate_reranker_scores=None,
            embedding_latency_ms=3,
            retrieval_latency_ms=4,
            rerank_latency_ms=0,
            retrieval_config=RetrievalConfig(
                embedding_model_name="model",
                embedding_version="v1",
                candidate_k=1,
                top_k=1,
                similarity_threshold=0.0,
                reranker_enabled=False,
                reranker_model_name=None,
                structural_filter_enabled=False,
            ),
        )


class FakePromptBuilder:
    def build(self, message: str, sources: list[RetrievedChunk]) -> BuiltPrompt:
        _ = sources
        return BuiltPrompt(prompt=f"Question: {message}\nAnswer:", included_sources=sources)


class FakeVectorStore:
    def ping(self) -> bool:
        return True

    def ensure_collection(self) -> None:
        return None

    def list_documents(self) -> list[dict[str, object]]:
        return [{"doc_id": "doc-1", "filename": "sample.pdf", "chunk_count": 1, "page_count": 1}]

    def has_documents(self) -> bool:
        return True


class FakeGenerationClient:
    @dataclass(frozen=True)
    class Result:
        text: str

    def generate(self, prompt: str) -> FakeGenerationClient.Result:
        return FakeGenerationClient.Result(text=f"mock answer for: {prompt[:20]}")

    model = "test-model"

    def check_configured(self) -> None:
        return None

    def is_configured(self) -> bool:
        return True


class FakeServices:
    def __init__(self) -> None:
        from app.config import get_settings

        self.settings = get_settings()
        self.vectorstore = FakeVectorStore()
        self.retriever = FakeRetriever()
        self.prompt_builder = FakePromptBuilder()
        self.generation_client = FakeGenerationClient()


def test_health_endpoint_returns_structured_status() -> None:
    original_services = app.state.services
    app.state.services = FakeServices()
    client = TestClient(app)
    try:
        response = client.get("/v1/health")
    finally:
        app.state.services = original_services

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "qdrant_connected": True,
        "openai_configured": True,
        "openai_model": "test-model",
        "reranker_enabled": get_settings().re_rank_enabled,
        "reranker_model": (
            get_settings().re_rank_model_name if get_settings().re_rank_enabled else None
        ),
    }


def test_chat_endpoint_returns_answer_and_sources() -> None:
    original_services = app.state.services
    app.state.services = FakeServices()
    client = TestClient(app)
    try:
        response = client.post(
            "/v1/chat",
            json=ChatRequest(
                message="What is the setup?", doc_id=None, top_k=1, stream=False
            ).model_dump(),
        )
    finally:
        app.state.services = original_services

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert len(body["sources"]) == 1
    assert body["sources"][0]["filename"] == "sample.pdf"
    assert body["latency_ms"]["total"] >= 0


def test_chat_reports_missing_openai_key_clearly() -> None:
    from app.core.generation_client import GenerationProviderError

    class Unconfigured(FakeGenerationClient):
        def check_configured(self) -> None:
            raise GenerationProviderError("OPENAI_API_KEY is not set.", "openai_key_missing", 503)

    services = FakeServices()
    services.generation_client = Unconfigured()
    original = app.state.services
    app.state.services = services
    try:
        response = TestClient(app).post("/v1/chat", json={"message": "hi"})
    finally:
        app.state.services = original
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["error"]["message"]


def test_ui_is_served() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "RagEval" in response.text

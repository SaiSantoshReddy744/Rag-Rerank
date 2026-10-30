from __future__ import annotations

import uuid
from typing import cast

from fastapi import APIRouter, HTTPException, Request

from app.core.generation_client import GenerationProviderError
from app.models.schemas import DocumentInfo, HealthResponse
from app.services import Services

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    services = cast(Services, request.app.state.services)
    s = services.settings
    return HealthResponse(
        status="ok",
        qdrant_connected=services.vectorstore.ping(),
        openai_configured=services.generation_client.is_configured(),
        openai_model=services.generation_client.model or None,
        reranker_enabled=s.re_rank_enabled,
        reranker_model=s.re_rank_model_name if s.re_rank_enabled else None,
    )


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[DocumentInfo]:
    services = cast(Services, request.app.state.services)
    return [DocumentInfo(**d) for d in services.vectorstore.list_documents()]  # type: ignore[arg-type]


@router.delete("/documents/{doc_id}")
def delete_document(request: Request, doc_id: str) -> dict[str, str]:
    services = cast(Services, request.app.state.services)
    try:
        safe_doc_id = str(uuid.UUID(doc_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    services.vectorstore.delete_document(safe_doc_id)
    upload_path = services.settings.data_dir / "uploads" / f"{safe_doc_id}.pdf"
    upload_path.unlink(missing_ok=True)
    return {"status": "deleted", "doc_id": doc_id}


@router.get("/models")
def openai_models(request: Request) -> dict[str, object]:
    """Models visible to your OpenAI API key."""
    services = cast(Services, request.app.state.services)
    try:
        models = services.generation_client.list_models()
    except GenerationProviderError as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
    configured = services.generation_client.model
    return {
        "configured": configured,
        "configured_available": configured in {m["name"] for m in models},
        "models": models,
    }

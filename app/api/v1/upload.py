from __future__ import annotations

import hashlib
import logging
import uuid
from threading import Lock
from typing import cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.extractor import PdfExtractionError
from app.models.schemas import UploadResponse
from app.services import Services

router = APIRouter()
logger = logging.getLogger(__name__)
_upload_lock = Lock()


def _is_pdf(file_name: str, content_type: str | None) -> bool:
    return file_name.lower().endswith(".pdf") or content_type == "application/pdf"


@router.post("/upload", response_model=UploadResponse)
def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    # Avoid duplicate-check races between simultaneous uploads in this worker.
    with _upload_lock:
        return _ingest_upload(request, file)


def _ingest_upload(request: Request, file: UploadFile) -> UploadResponse:
    if not _is_pdf(file.filename or "", file.content_type):
        raise HTTPException(status_code=422, detail="Only PDF uploads are supported")

    services = cast(Services, request.app.state.services)
    settings = services.settings
    doc_id = str(uuid.uuid4())
    filename = file.filename or f"{doc_id}.pdf"

    upload_path = settings.data_dir / "uploads" / f"{doc_id}.pdf"
    digest = hashlib.sha256()
    size = 0
    header = b""
    try:
        file.file.seek(0)
        with upload_path.open("wb") as destination:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > settings.max_upload_size_bytes:
                    limit_mb = settings.max_upload_size_bytes // (1024 * 1024)
                    raise HTTPException(
                        status_code=413, detail=f"PDF exceeds the {limit_mb} MB upload limit"
                    )
                if len(header) < 5:
                    header += block[: 5 - len(header)]
                digest.update(block)
                destination.write(block)
        if header != b"%PDF-":
            raise HTTPException(status_code=422, detail="The uploaded file is not a valid PDF")
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise
    document_fingerprint = digest.hexdigest()

    duplicate_exists = services.vectorstore.has_document_fingerprint(document_fingerprint)
    logger.info(
        "upload_received",
        extra={
            "request_id": getattr(request.state, "request_id", "-"),
            "doc_id": doc_id,
            "uploaded_filename": filename,
            "duplicate_exists": duplicate_exists,
            "duplicate_policy": settings.duplicate_upload_policy,
        },
    )

    if duplicate_exists and settings.duplicate_upload_policy == "reject":
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="Duplicate PDF upload detected")

    try:
        result = services.pipeline.ingest_pdf(
            doc_id=doc_id,
            document_fingerprint=document_fingerprint,
            filename=filename,
            pdf_path=upload_path,
        )
    except PdfExtractionError as exc:
        upload_path.unlink(missing_ok=True)
        logger.warning(
            "pdf_ingestion_failed",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        try:
            services.vectorstore.delete_document(doc_id)
        except Exception:
            logger.warning("failed_ingest_cleanup_failed", extra={"doc_id": doc_id}, exc_info=True)
        upload_path.unlink(missing_ok=True)
        logger.exception(
            "pdf_ingestion_failed",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
        )
        raise HTTPException(status_code=500, detail="Failed to ingest uploaded PDF") from exc

    return UploadResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        chunk_count=len(result.chunks),
        status="ready",
    )

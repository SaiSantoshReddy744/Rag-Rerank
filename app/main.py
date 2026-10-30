from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.config import PROJECT_ROOT, get_settings
from app.logging_config import configure_logging, request_id_var
from app.models.schemas import ErrorPayload, ErrorResponse
from app.services import Services


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=ErrorPayload(code=code, message=message)).model_dump(),
    )


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    logger = logging.getLogger(__name__)

    app = FastAPI(title=settings.app_name, debug=settings.debug)
    app.state.services = Services.build(settings)

    @app.middleware("http")
    async def add_request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        start = perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            total_ms = int((perf_counter() - start) * 1000)
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "total_ms": total_ms,
                    "status_code": response.status_code if response is not None else None,
                },
            )
            request_id_var.reset(token)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(422, "validation_error", str(exc.errors()))

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        status_code = exc.status_code
        code_map = {
            400: "bad_request",
            401: "openai_invalid_key",
            429: "rate_limited",
            503: "service_unavailable",
            404: "not_found",
            409: "conflict",
            422: "unprocessable_entity",
            500: "internal_server_error",
        }
        return _error_response(
            status_code, code_map.get(status_code, "http_error"), str(exc.detail)
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception("unhandled error", exc_info=exc)
        return _error_response(500, "internal_server_error", "An unexpected error occurred")

    @app.on_event("startup")
    def startup() -> None:
        services = app.state.services
        try:
            services.vectorstore.ensure_collection()
        except Exception:
            logger.warning("Qdrant collection initialization deferred", exc_info=True)
        if not services.generation_client.is_configured():
            logger.warning(
                "OPENAI_API_KEY and/or OPENAI_MODEL missing in %s - questions will fail until set",
                PROJECT_ROOT / ".env",
            )

    app.include_router(v1_router)

    static_dir = PROJECT_ROOT / "app" / "static"

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()

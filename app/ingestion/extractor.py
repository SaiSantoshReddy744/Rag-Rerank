from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    section_title: str | None
    source_path: str


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


def _guess_section_title(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) <= 80:
            return line
        return line[:80]
    return None


def extract_pdf_text(pdf_path: Path) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    try:
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                try:
                    text = page.get_text("text").strip()
                except Exception:  # pragma: no cover - defensive against fitz parser errors
                    logger.warning(
                        "page_extraction_failed",
                        extra={"page_number": index, "source_path": str(pdf_path)},
                        exc_info=True,
                    )
                    continue
                if not text:
                    continue
                pages.append(
                    ExtractedPage(
                        page_number=index,
                        text=text,
                        section_title=_guess_section_title(text),
                        source_path=str(pdf_path),
                    )
                )
    except Exception as exc:  # pragma: no cover - depends on corrupt PDF fixtures
        logger.warning("pdf_open_failed", extra={"source_path": str(pdf_path)}, exc_info=True)
        raise PdfExtractionError(f"Failed to extract text from {pdf_path.name}") from exc
    return pages

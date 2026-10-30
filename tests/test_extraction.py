from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.ingestion.extractor import PdfExtractionError, extract_pdf_text


def test_extract_pdf_text_reads_page_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello PDF\nSetup")
    document.save(pdf_path)
    document.close()

    pages = extract_pdf_text(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Hello PDF" in pages[0].text


def test_extract_pdf_text_skips_empty_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    document.new_page()
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Visible text")
    document.save(pdf_path)
    document.close()

    pages = extract_pdf_text(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 2


def test_extract_pdf_text_raises_for_invalid_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_text("not a real pdf", encoding="utf-8")

    with pytest.raises(PdfExtractionError):
        extract_pdf_text(pdf_path)

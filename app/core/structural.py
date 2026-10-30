from __future__ import annotations

import re
from collections.abc import Mapping

from app.core.vectorstore import RetrievedChunk

STRUCTURAL_FILTER_EXCLUDED_CATEGORIES = frozenset(
    {
        "CONTENTS",
        "INDEX",
        "COPYRIGHT",
        "GLOSSARY",
    }
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _chunk_value(chunk: RetrievedChunk | Mapping[str, object], key: str) -> object:
    if isinstance(chunk, Mapping):
        return chunk.get(key)
    return getattr(chunk, key, None)


def classify_structure(chunk: RetrievedChunk | Mapping[str, object]) -> str:
    section_title = _normalize(str(_chunk_value(chunk, "section_title") or ""))
    text = _normalize(str(_chunk_value(chunk, "text") or ""))
    haystack = f"{section_title} {text}"

    if (
        "table of contents" in haystack
        or section_title == "contents"
        or text.startswith("contents")
    ):
        return "CONTENTS"
    if section_title == "index" or re.search(r"\bindex\b", haystack) is not None:
        return "INDEX"
    if (
        "introduction" in haystack
        or section_title.startswith("intro")
        or text.startswith("introduction")
        or text.startswith("intro ")
    ):
        return "INTRODUCTION"
    if "copyright" in haystack:
        return "COPYRIGHT"
    if "glossary" in haystack:
        return "GLOSSARY"
    if "bibliography" in haystack or section_title == "references" or text.startswith("references"):
        return "BIBLIOGRAPHY"

    structural_markers = (
        "cover",
        "title page",
        "dedication",
        "preface",
        "foreword",
        "acknowledg",
        "appendix",
        "about the author",
        "author note",
        "front matter",
    )
    if any(marker in haystack for marker in structural_markers):
        return "STRUCTURAL_OTHER"

    return "SUBSTANTIVE"


def is_rerank_eligible(chunk: RetrievedChunk | Mapping[str, object]) -> bool:
    return classify_structure(chunk) not in STRUCTURAL_FILTER_EXCLUDED_CATEGORIES

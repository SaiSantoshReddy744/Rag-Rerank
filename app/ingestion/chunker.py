from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from app.core.tokenization import TokenEncoder, get_token_encoder

from app.ingestion.extractor import ExtractedPage


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    page_number: int
    section_title: str | None
    source_path: str
    text: str
    token_count: int


class RecursiveChunker:
    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    @cached_property
    def _encoding(self) -> TokenEncoder:
        return get_token_encoder()

    def token_count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        for page in pages:
            page_chunks = self._chunk_text(page.text)
            for page_chunk in page_chunks:
                chunks.append(
                    Chunk(
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        section_title=page.section_title,
                        source_path=page.source_path,
                        text=page_chunk,
                        token_count=self.token_count(page_chunk),
                    )
                )
                chunk_index += 1
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        atomic_segments = self._recursive_split(normalized, ["\n\n", "\n", ". ", " ", ""])
        return self._pack_segments(atomic_segments)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if self.token_count(text) <= self._max_tokens or not separators:
            return [text.strip()]

        separator = separators[0]
        if separator and separator not in text:
            return self._recursive_split(text, separators[1:])

        if not separator:
            return self._split_by_tokens(text)

        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._recursive_split(text, separators[1:])

        results: list[str] = []
        current = pieces[0].strip()
        for piece in pieces[1:]:
            candidate = f"{current}{separator}{piece}".strip()
            if self.token_count(candidate) <= self._max_tokens:
                current = candidate
                continue
            if current.strip():
                results.extend(self._recursive_split(current.strip(), separators[1:]))
            current = piece.strip()
        if current.strip():
            results.extend(self._recursive_split(current.strip(), separators[1:]))
        return [segment for segment in results if segment.strip()]

    def _split_by_tokens(self, text: str) -> list[str]:
        tokens = self._encoding.encode(text)
        if len(tokens) <= self._max_tokens:
            return [text.strip()]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self._max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._encoding.decode(chunk_tokens).strip())
            if end == len(tokens):
                break
            start = max(end - self._overlap_tokens, start + 1)
        return [chunk for chunk in chunks if chunk]

    def _pack_segments(self, segments: list[str]) -> list[str]:
        if not segments:
            return []

        packed: list[str] = []
        current_tokens: list[int] = []

        for segment in segments:
            segment_tokens = self._encoding.encode(segment)
            if not current_tokens:
                current_tokens = segment_tokens
                continue

            proposed = current_tokens + segment_tokens
            if len(proposed) <= self._max_tokens:
                current_tokens = proposed
                continue

            packed.append(self._encoding.decode(current_tokens).strip())
            overlap = current_tokens[-self._overlap_tokens :] if self._overlap_tokens > 0 else []
            current_tokens = overlap + segment_tokens
            while len(current_tokens) > self._max_tokens:
                packed.append(self._encoding.decode(current_tokens[: self._max_tokens]).strip())
                current_tokens = current_tokens[self._max_tokens - self._overlap_tokens :]

        if current_tokens:
            packed.append(self._encoding.decode(current_tokens).strip())

        return [chunk for chunk in packed if chunk]

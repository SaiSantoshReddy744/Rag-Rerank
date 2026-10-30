from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from app.config import Settings
from app.core.tokenization import TokenEncoder, get_token_encoder
from app.core.vectorstore import RetrievedChunk


@dataclass(frozen=True)
class BuiltPrompt:
    prompt: str
    included_sources: list[RetrievedChunk]


class PromptBuilder:
    def __init__(self, settings: Settings) -> None:
        self._token_budget = settings.prompt_token_budget

    @cached_property
    def _encoding(self) -> TokenEncoder:
        return get_token_encoder()

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens]).strip()

    def build(self, message: str, sources: list[RetrievedChunk]) -> BuiltPrompt:
        instructions = [
            "You are a helpful assistant answering only from the provided document context.",
            "If the context does not contain the answer, say so plainly.",
            "Preserve source attribution by citing facts inline with [source N].",
            "Do not invent facts or rely on outside knowledge.",
        ]
        prompt_parts: list[str] = ["Instructions:", *instructions, "", "Retrieved context:"]
        used_tokens = self._count_tokens("\n".join(prompt_parts))
        question_block = f"User question:\n{message}\n\nAnswer:"
        question_tokens = self._count_tokens(question_block)
        remaining_tokens = max(self._token_budget - used_tokens - question_tokens, 0)

        included_sources: list[RetrievedChunk] = []
        rendered_sources: list[str] = []
        remaining_budget = remaining_tokens
        for index, source in enumerate(sources, start=1):
            source_header = (
                f"[source {index}] doc_id={source.doc_id} "
                f"page={source.page_number} chunk_index={source.chunk_index} "
                f"section={source.section_title or 'unknown'} "
                f"score={source.score:.3f}"
            )
            source_body = source.text.strip()
            source_block = f"{source_header}\n{source_body}".strip()
            source_tokens = self._count_tokens(source_block)
            if source_tokens > remaining_budget:
                truncated_budget = remaining_budget - self._count_tokens(f"{source_header}\n")
                if truncated_budget < 0:
                    truncated_budget = 0
                truncated_body = self._truncate_to_tokens(source_body, truncated_budget)
                if not truncated_body:
                    break
                source_block = f"{source_header}\n{truncated_body}".strip()
                source_tokens = self._count_tokens(source_block)
            rendered_sources.append(source_block)
            included_sources.append(source)
            remaining_budget -= source_tokens
            if remaining_budget <= 0:
                break

        prompt_parts.extend(rendered_sources)
        prompt_parts.extend(["", question_block])
        return BuiltPrompt(
            prompt="\n\n".join(prompt_parts).strip(), included_sources=included_sources
        )

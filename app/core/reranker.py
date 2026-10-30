from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from app.config import Settings


@dataclass(frozen=True)
class RankedChunk:
    index: int
    score: float


class RerankerError(RuntimeError):
    """Raised when the cross-encoder cannot be loaded or fails to score passages."""


class _RerankerModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> object: ...


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.re_rank_enabled
        self._model_name = settings.re_rank_model_name
        self._model: _RerankerModel | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def rank(self, query: str, passages: list[str]) -> list[RankedChunk]:
        """Score every (query, passage) pair with the cross-encoder. Never returns fake scores."""
        if not self._enabled:
            raise RerankerError("Reranking is disabled (RE_RANK_ENABLED=false).")
        if not passages:
            return []
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                # Avoid Transformers' meta-device/low-memory initialization path, which can
                # fail on some Windows + PyTorch combinations when CrossEncoder moves to CPU.
                # CPU is the predictable fallback for this local prototype; the model is
                # cached after the first load, so subsequent questions skip initialization.
                self._model = CrossEncoder(
                    self._model_name,
                    device="cpu",
                    automodel_args={"low_cpu_mem_usage": False},
                    max_length=384,
                )
            scores = cast(
                Sequence[float],
                self._model.predict(
                    [(query, p) for p in passages],
                    batch_size=16,
                    show_progress_bar=False,
                ),
            )
        except Exception as exc:
            raise RerankerError(
                f"Reranker '{self._model_name}' failed: {type(exc).__name__}: {exc}"
            ) from exc
        if len(scores) != len(passages):
            raise RerankerError("Reranker returned a wrong number of scores.")
        return [RankedChunk(index=i, score=float(s)) for i, s in enumerate(scores)]

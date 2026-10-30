from __future__ import annotations

import pytest

from app.config import Settings
from app.core.reranker import Reranker, RerankerError


class _Model:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(p[1])) for p in pairs]


def test_reranker_scores_every_pair_and_never_fakes() -> None:
    r = Reranker(Settings(_env_file=None, re_rank_enabled=True))
    r._model = _Model()  # type: ignore[assignment]
    ranked = r.rank("q", ["a", "bbb", "cc"])
    assert [(x.index, x.score) for x in ranked] == [(0, 1.0), (1, 3.0), (2, 2.0)]


def test_disabled_reranker_raises_instead_of_returning_zero_scores() -> None:
    with pytest.raises(RerankerError):
        Reranker(Settings(_env_file=None, re_rank_enabled=False)).rank("q", ["a"])

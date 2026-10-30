from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import cast

from app.config import Settings
from app.core.reranker import Reranker

DEFAULT_INPUT = Path("eval/results/m1-rerank-k20-v2.json")
DEFAULT_OUTPUT = Path("eval/results/rq3-reranker-analysis.json")

STRUCTURAL_CATEGORY_ORDER = [
    "CONTENTS",
    "INDEX",
    "INTRODUCTION",
    "COPYRIGHT",
    "GLOSSARY",
    "BIBLIOGRAPHY",
    "STRUCTURAL_OTHER",
    "SUBSTANTIVE",
]


@dataclass(frozen=True)
class GoldRecord:
    question_id: str
    question: str
    gold_chunk_id: str
    gold_relevance: str
    gold_grade: int | None
    evidence_roles: list[str]
    gold_text_span: str
    candidate_present: bool
    candidate_rank: int | None
    final_present: bool
    final_rank: int | None
    dense_raw_score: float | None
    dense_score: float | None
    reranker_score: float | None
    reranker_score_source: str
    rank_delta: int | None
    page_number: int | None
    section_title: str | None
    structure_category: str
    outcome: str


@dataclass(frozen=True)
class FailureCase:
    question_id: str
    question: str
    gold_chunk_id: str
    gold_candidate_rank: int | None
    gold_final_rank: int | None
    replacement_chunk_id: str | None
    replacement_candidate_rank: int | None
    replacement_final_rank: int | None
    replacement_structure_category: str | None
    dense_raw_score: float | None
    dense_score: float | None
    reranker_score: float | None
    reranker_score_source: str
    page_number: int | None
    section_title: str | None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_structure(chunk: Mapping[str, object]) -> str:
    section_title = _normalize(str(chunk.get("section_title") or ""))
    text = _normalize(str(chunk.get("text") or ""))
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


def rank_displacement(candidate_rank: int | None, final_rank: int | None) -> int | None:
    if candidate_rank is None or final_rank is None:
        return None
    return candidate_rank - final_rank


def classify_outcome(candidate_present: bool, final_present: bool) -> str:
    if not candidate_present:
        return "CANDIDATE_GENERATION_FAILURE"
    if not final_present:
        return "RERANKER_SELECTION_FAILURE"
    return "SUCCESSFUL_SELECTION"


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _question_records(result: Mapping[str, object]) -> tuple[list[GoldRecord], str]:
    question_id = str(result.get("question_id", ""))
    question = str(result.get("question", ""))
    answerable = bool(result.get("answerable", True))
    if not answerable:
        return [], "missing"

    candidate_chunks = cast(list[dict[str, object]], result.get("candidate_chunks", []))
    retrieved_chunks = cast(list[dict[str, object]], result.get("retrieved_chunks", []))

    candidate_by_id = {
        str(chunk["chunk_id"]): chunk
        for chunk in candidate_chunks
        if isinstance(chunk, Mapping) and "chunk_id" in chunk
    }
    final_by_id = {
        str(chunk["chunk_id"]): chunk
        for chunk in retrieved_chunks
        if isinstance(chunk, Mapping) and "chunk_id" in chunk
    }

    persisted_scores = [
        _as_float(chunk.get("reranker_score")) if isinstance(chunk, Mapping) else None
        for chunk in candidate_chunks
    ]
    reranker_scores, reranker_score_source = _resolve_reranker_scores(
        question=question,
        candidate_chunks=candidate_chunks,
        persisted_scores=persisted_scores,
        retrieval_config=cast(dict[str, object], result.get("retrieval_config", {})),
    )
    reranker_score_by_id = (
        {
            str(chunk["chunk_id"]): score
            for chunk, score in zip(candidate_chunks, reranker_scores, strict=True)
        }
        if reranker_scores is not None
        else {}
    )

    evidence_rows: list[GoldRecord] = []
    merged_evidence: OrderedDict[str, dict[str, object]] = OrderedDict()

    for role in ("gold_evidence", "acceptable_evidence"):
        evidences = cast(list[dict[str, object]], result.get(role, []))
        for evidence in evidences:
            chunk_id = str(evidence.get("chunk_id", ""))
            if not chunk_id:
                continue
            merged = merged_evidence.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "roles": [],
                    "grade": evidence.get("grade"),
                    "relevance": evidence.get("relevance", role.removesuffix("_evidence")),
                    "page_number": evidence.get("page_number"),
                    "text_span": evidence.get("text_span", ""),
                },
            )
            roles = cast(list[str], merged["roles"])
            roles.append(role.removesuffix("_evidence"))
            grade = evidence.get("grade")
            if isinstance(grade, int):
                current_grade = merged.get("grade")
                if not isinstance(current_grade, int) or grade > current_grade:
                    merged["grade"] = grade
            if (
                not str(merged.get("text_span", "")).strip()
                and str(evidence.get("text_span", "")).strip()
            ):
                merged["text_span"] = evidence.get("text_span", "")

    for evidence in merged_evidence.values():
        chunk_id = str(evidence["chunk_id"])
        candidate_chunk = candidate_by_id.get(chunk_id)
        final_chunk = final_by_id.get(chunk_id)
        candidate_rank = _as_int(candidate_chunk.get("candidate_rank")) if candidate_chunk else None
        final_rank = _as_int(final_chunk.get("rank")) if final_chunk else None
        gold_source_chunk = candidate_chunk or final_chunk or {}
        structure_category = classify_structure(gold_source_chunk)
        outcome = classify_outcome(candidate_chunk is not None, final_chunk is not None)
        evidence_rows.append(
            GoldRecord(
                question_id=question_id,
                question=question,
                gold_chunk_id=chunk_id,
                gold_relevance=str(evidence.get("relevance", "acceptable")),
                gold_grade=_as_int(evidence.get("grade")),
                evidence_roles=cast(list[str], evidence["roles"]),
                gold_text_span=str(evidence.get("text_span", "")),
                candidate_present=candidate_chunk is not None,
                candidate_rank=candidate_rank,
                final_present=final_chunk is not None,
                final_rank=final_rank,
                dense_raw_score=_as_float(gold_source_chunk.get("raw_score")),
                dense_score=_as_float(gold_source_chunk.get("score")),
                reranker_score=reranker_score_by_id.get(chunk_id),
                reranker_score_source=reranker_score_source,
                rank_delta=rank_displacement(candidate_rank, final_rank),
                page_number=_as_int(
                    gold_source_chunk.get("page_number") or evidence.get("page_number")
                ),
                section_title=_as_str_or_none(gold_source_chunk.get("section_title")),
                structure_category=structure_category,
                outcome=outcome,
            )
        )

    return evidence_rows, reranker_score_source


def _resolve_reranker_scores(
    *,
    question: str,
    candidate_chunks: Sequence[Mapping[str, object]],
    persisted_scores: Sequence[float | None],
    retrieval_config: Mapping[str, object],
) -> tuple[list[float] | None, str]:
    if persisted_scores and all(score is not None for score in persisted_scores):
        resolved_scores: list[float] = []
        for score in persisted_scores:
            if score is None:
                raise AssertionError("persisted_scores unexpectedly contained None")
            resolved_scores.append(score)
        return resolved_scores, "persisted"

    model_name = _as_str_or_none(retrieval_config.get("reranker_model_name"))
    if model_name is None:
        return None, "missing"

    try:
        reranker = _cached_reranker(model_name)
        passages = [str(chunk.get("text", "")) for chunk in candidate_chunks]
        ranked = reranker.rank(question, passages)
        scores = [0.0 for _ in passages]
        for ranked_chunk in ranked:
            if 0 <= ranked_chunk.index < len(scores):
                scores[ranked_chunk.index] = ranked_chunk.score
        return scores, "computed"
    except Exception:
        return None, "missing"


@lru_cache(maxsize=4)
def _cached_reranker(model_name: str) -> Reranker:
    settings = Settings(re_rank_enabled=True, re_rank_model_name=model_name)
    return Reranker(settings)


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _build_failure_cases(
    rows: Sequence[GoldRecord], results: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    question_result_by_id = {
        str(result.get("question_id", "")): result
        for result in results
        if isinstance(result, Mapping)
    }
    relevant_ids_by_question: dict[str, set[str]] = {}
    for row in rows:
        relevant_ids_by_question.setdefault(row.question_id, set()).add(row.gold_chunk_id)

    cases: list[dict[str, object]] = []
    for row in rows:
        if row.outcome != "RERANKER_SELECTION_FAILURE":
            continue
        result = question_result_by_id.get(row.question_id, {})
        candidate_chunks = cast(list[dict[str, object]], result.get("candidate_chunks", []))
        final_chunks = cast(list[dict[str, object]], result.get("retrieved_chunks", []))
        candidate_by_id = {
            _as_str_or_none(chunk.get("chunk_id")): chunk
            for chunk in candidate_chunks
            if _as_str_or_none(chunk.get("chunk_id")) is not None
        }
        replacement = _select_replacement_chunk(
            final_chunks, relevant_ids_by_question[row.question_id]
        )
        replacement_id = _as_str_or_none(replacement.get("chunk_id")) if replacement else None
        replacement_candidate = candidate_by_id.get(replacement_id) if replacement_id else None
        cases.append(
            {
                "question_id": row.question_id,
                "question": row.question,
                "gold_chunk_id": row.gold_chunk_id,
                "gold_candidate_rank": row.candidate_rank,
                "gold_final_rank": row.final_rank,
                "replacement_chunk_id": replacement_id,
                "replacement_candidate_rank": _as_int(replacement_candidate.get("candidate_rank"))
                if replacement_candidate
                else None,
                "replacement_final_rank": _as_int(replacement.get("rank")) if replacement else None,
                "replacement_structure_category": classify_structure(replacement)
                if replacement
                else None,
                "dense_raw_score": row.dense_raw_score,
                "dense_score": row.dense_score,
                "reranker_score": row.reranker_score,
                "reranker_score_source": row.reranker_score_source,
                "page_number": row.page_number,
                "section_title": row.section_title,
            }
        )

    def sort_key(item: Mapping[str, object]) -> tuple[int, int, int]:
        category = _as_str_or_none(item.get("replacement_structure_category")) or "SUBSTANTIVE"
        priority = 0 if category != "SUBSTANTIVE" else 1
        final_rank = _as_int(item.get("replacement_final_rank")) or 99
        candidate_rank = _as_int(item.get("replacement_candidate_rank")) or 99
        return (priority, final_rank, candidate_rank)

    return sorted(cases, key=sort_key)[:10]


def _select_replacement_chunk(
    final_chunks: Sequence[Mapping[str, object]],
    relevant_ids: set[str],
) -> Mapping[str, object] | None:
    for chunk in final_chunks:
        chunk_id = _as_str_or_none(chunk.get("chunk_id"))
        if chunk_id is None:
            continue
        if chunk_id not in relevant_ids:
            return chunk
    return final_chunks[0] if final_chunks else None


def analyze_report(results_path: Path) -> dict[str, object]:
    report = _load_json(results_path)
    results = cast(list[dict[str, object]], report.get("results", []))

    all_rows: list[GoldRecord] = []
    for result in results:
        rows, _score_source = _question_records(result)
        all_rows.extend(rows)

    total_answerable_questions = sum(
        1 for result in results if bool(result.get("answerable", True))
    )
    total_relevant_chunks = len(all_rows)
    candidate_absent = sum(1 for row in all_rows if row.outcome == "CANDIDATE_GENERATION_FAILURE")
    candidate_present = total_relevant_chunks - candidate_absent
    retained = sum(1 for row in all_rows if row.outcome == "SUCCESSFUL_SELECTION")
    lost_after_rerank = sum(1 for row in all_rows if row.outcome == "RERANKER_SELECTION_FAILURE")

    present_and_final_rows = [
        row for row in all_rows if row.candidate_present and row.final_present
    ]
    rank_deltas = [row.rank_delta for row in present_and_final_rows if row.rank_delta is not None]
    reranker_score_sources = [row.reranker_score_source for row in all_rows]

    structural_summary = {
        category: {"promoted": 0, "retained": 0, "demoted": 0, "total": 0}
        for category in STRUCTURAL_CATEGORY_ORDER
    }
    for result in results:
        candidate_chunks = cast(list[dict[str, object]], result.get("candidate_chunks", []))
        for chunk in candidate_chunks:
            category = classify_structure(chunk)
            if category not in structural_summary:
                continue
            structural_summary[category]["total"] += 1
        final_chunks = cast(list[dict[str, object]], result.get("retrieved_chunks", []))
        candidate_by_id = {
            _as_str_or_none(chunk.get("chunk_id")): chunk
            for chunk in candidate_chunks
            if _as_str_or_none(chunk.get("chunk_id")) is not None
        }
        for final_chunk in final_chunks:
            chunk_id = _as_str_or_none(final_chunk.get("chunk_id"))
            if chunk_id is None:
                continue
            category = classify_structure(final_chunk)
            if category not in structural_summary:
                continue
            candidate_chunk = candidate_by_id.get(chunk_id)
            candidate_rank = (
                _as_int(candidate_chunk.get("candidate_rank")) if candidate_chunk else None
            )
            final_rank = _as_int(final_chunk.get("rank"))
            if candidate_rank is None or final_rank is None:
                continue
            delta = candidate_rank - final_rank
            if delta > 0:
                structural_summary[category]["promoted"] += 1
            elif delta < 0:
                structural_summary[category]["demoted"] += 1
            else:
                structural_summary[category]["retained"] += 1

    analysis: dict[str, object] = {
        "input_file": str(results_path),
        "reranker_score_sources": {
            "persisted": sum(1 for source in reranker_score_sources if source == "persisted"),
            "computed": sum(1 for source in reranker_score_sources if source == "computed"),
            "missing": sum(1 for source in reranker_score_sources if source == "missing"),
        },
        "summary": {
            "total_answerable_questions": total_answerable_questions,
            "total_relevant_evidence_chunks": total_relevant_chunks,
            "gold_absent_from_candidate_20": candidate_absent,
            "gold_present_in_candidate_20": candidate_present,
            "gold_retained_in_final_top_5": retained,
            "gold_lost_after_reranking": lost_after_rerank,
            "gold_absent_from_candidate_20_pct": _pct(candidate_absent, total_relevant_chunks),
            "gold_present_in_candidate_20_pct": _pct(candidate_present, total_relevant_chunks),
            "gold_retained_in_final_top_5_pct": _pct(retained, total_relevant_chunks),
            "gold_lost_after_reranking_pct": _pct(lost_after_rerank, total_relevant_chunks),
            "rank_delta_mean": mean(rank_deltas) if rank_deltas else None,
            "rank_delta_median": median(rank_deltas) if rank_deltas else None,
            "rank_delta_positive": sum(1 for delta in rank_deltas if delta > 0),
            "rank_delta_zero": sum(1 for delta in rank_deltas if delta == 0),
            "rank_delta_negative": sum(1 for delta in rank_deltas if delta < 0),
        },
        "structural_promotion": structural_summary,
        "failure_cases": _build_failure_cases(all_rows, results),
    }
    return analysis


def _pct(value: int, total: int) -> float | None:
    if total == 0:
        return None
    return value / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RQ3 reranker failure modes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    analysis = analyze_report(args.input)
    output = json.dumps(analysis, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

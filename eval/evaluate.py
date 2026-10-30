from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import log2
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.config import get_settings
from app.core.structural import classify_structure
from app.core.vectorstore import RetrievalFilters, RetrievedChunk
from app.services import Services


@dataclass(frozen=True)
class EvidenceSpec:
    page_number: int
    chunk_id: str
    text_span: str
    relevance: str = "acceptable"
    grade: int | None = None


@dataclass(frozen=True)
class QuestionSpec:
    id: str | None
    question: str
    expected_answer_contains: str
    expected_doc_id: str | None
    expected_chunk_keywords: list[str]
    gold_answer: str | None = None
    answerable: bool = True
    gold_evidence: list[EvidenceSpec] = field(default_factory=list)
    acceptable_evidence: list[EvidenceSpec] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    question: str
    answerable: bool
    gold_answer: str | None
    gold_evidence: list[dict[str, object]]
    acceptable_evidence: list[dict[str, object]]
    validation_errors: list[str]
    candidate_recall: dict[str, float | None]
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    recall_at_20: float | None
    mrr: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    candidate_chunks: list[dict[str, object]]
    retrieved_chunks: list[dict[str, object]]
    original_candidate_count: int
    filtered_candidate_count: int
    candidate_structure_categories: list[str]
    filtered_candidate_structure_categories: list[str]
    precision: float
    retrieval_hit_at_k: float
    generation_latency_ms: int
    latency_ms: int
    faithfulness: float
    context_utilisation: float
    answer: str
    retrieval_config: dict[str, object]


def _parse_evidence(raw: object) -> EvidenceSpec:
    if not isinstance(raw, dict):
        raise ValueError("gold evidence entries must be JSON objects")

    chunk_id = raw.get("chunk_id")
    page_number = raw.get("page_number")
    text_span = raw.get("text_span")
    relevance = raw.get("relevance", "acceptable")
    grade = raw.get("grade")

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("gold evidence chunk_id must be a non-empty string")
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError("gold evidence page_number must be a positive integer")
    if not isinstance(text_span, str) or not text_span.strip():
        raise ValueError("gold evidence text_span must be a non-empty string")
    if not isinstance(relevance, str) or not relevance.strip():
        raise ValueError("gold evidence relevance must be a non-empty string")
    if grade is not None and (not isinstance(grade, int) or grade < 0):
        raise ValueError("gold evidence grade must be a non-negative integer when provided")

    return EvidenceSpec(
        page_number=page_number,
        chunk_id=chunk_id,
        text_span=text_span,
        relevance=relevance,
        grade=grade,
    )


def _parse_question(raw: object, index: int) -> QuestionSpec:
    if not isinstance(raw, dict):
        raise ValueError("each question entry must be a JSON object")

    question = raw.get("question")
    expected_answer_contains = raw.get("expected_answer_contains")
    expected_doc_id = raw.get("expected_doc_id")
    expected_chunk_keywords = raw.get("expected_chunk_keywords", [])
    question_id = raw.get("id") or raw.get("question_id") or f"q{index + 1:02d}"
    gold_answer = raw.get("gold_answer")
    answerable = raw.get("answerable", True)
    gold_evidence = raw.get("gold_evidence", [])
    acceptable_evidence = raw.get("acceptable_evidence", [])

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question text must be a non-empty string")
    if not isinstance(expected_answer_contains, str):
        raise ValueError("expected_answer_contains must be a string")
    if expected_doc_id is not None and not isinstance(expected_doc_id, str):
        raise ValueError("expected_doc_id must be a string or null")
    if not isinstance(expected_chunk_keywords, list) or any(
        not isinstance(keyword, str) for keyword in expected_chunk_keywords
    ):
        raise ValueError("expected_chunk_keywords must be a list of strings")
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question id must be a non-empty string when provided")
    if gold_answer is not None and not isinstance(gold_answer, str):
        raise ValueError("gold_answer must be a string or null")
    if not isinstance(answerable, bool):
        raise ValueError("answerable must be a boolean")
    if not isinstance(gold_evidence, list):
        raise ValueError("gold_evidence must be a list")
    if not isinstance(acceptable_evidence, list):
        raise ValueError("acceptable_evidence must be a list")

    return QuestionSpec(
        id=question_id,
        question=question,
        expected_answer_contains=expected_answer_contains,
        expected_doc_id=expected_doc_id,
        expected_chunk_keywords=expected_chunk_keywords,
        gold_answer=gold_answer,
        answerable=answerable,
        gold_evidence=[_parse_evidence(item) for item in gold_evidence],
        acceptable_evidence=[_parse_evidence(item) for item in acceptable_evidence],
    )


def _load_questions(path: Path) -> list[QuestionSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("questions file must contain a JSON array")
    return [_parse_question(item, index) for index, item in enumerate(raw)]


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords}


def _question_relevant_chunks(question: QuestionSpec) -> dict[str, int]:
    relevant: dict[str, int] = {}
    for evidence in question.gold_evidence + question.acceptable_evidence:
        relevance = evidence.grade if evidence.grade is not None else 1
        existing = relevant.get(evidence.chunk_id)
        if existing is None or relevance > existing:
            relevant[evidence.chunk_id] = relevance
    return relevant


def _validate_question(
    question: QuestionSpec,
    *,
    known_chunks: dict[str, int] | None = None,
) -> list[str]:
    errors: list[str] = []
    relevant_chunks = _question_relevant_chunks(question)

    if question.answerable and not relevant_chunks:
        errors.append("answerable questions must define gold_evidence or acceptable_evidence")
    if not question.answerable and relevant_chunks:
        errors.append("unanswerable questions must not define gold evidence")

    for evidence in question.gold_evidence + question.acceptable_evidence:
        if not evidence.chunk_id.strip():
            errors.append("gold evidence chunk_id must not be empty")
        if evidence.page_number < 1:
            errors.append(
                f"gold evidence page_number must be positive for chunk_id={evidence.chunk_id}"
            )
        if not evidence.text_span.strip():
            errors.append(
                f"gold evidence text_span must not be empty for chunk_id={evidence.chunk_id}"
            )
        if known_chunks is not None:
            if evidence.chunk_id not in known_chunks:
                errors.append(f"unknown gold evidence chunk_id={evidence.chunk_id}")
            else:
                expected_page_number = known_chunks[evidence.chunk_id]
                if evidence.page_number != expected_page_number:
                    errors.append(
                        "gold evidence page_number does not match indexed chunk "
                        f"for chunk_id={evidence.chunk_id} "
                        f"(expected {expected_page_number}, got {evidence.page_number})"
                    )

    return errors


def _score_heuristic_retrieval(
    question: QuestionSpec,
    chunks: Sequence[RetrievedChunk],
) -> tuple[float, float]:
    keywords = [keyword.lower() for keyword in question.expected_chunk_keywords if keyword]
    if question.expected_doc_id is None and not keywords:
        return 0.0, 0.0
    relevant_count = 0
    for chunk in chunks:
        text = chunk.text.lower()
        doc_id = chunk.doc_id
        if question.expected_doc_id is not None and doc_id != question.expected_doc_id:
            continue
        if keywords and not all(keyword in text for keyword in keywords):
            continue
        relevant_count += 1
    precision = relevant_count / max(len(chunks), 1)
    retrieval_hit_at_k = 1.0 if relevant_count > 0 else 0.0
    return precision, retrieval_hit_at_k


def _score_answer(answer: str, context_text: str) -> tuple[float, float]:
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context_text)
    if not answer_tokens or not context_tokens:
        return 0.0, 0.0
    overlap = answer_tokens.intersection(context_tokens)
    faithfulness = len(overlap) / len(answer_tokens)
    context_utilisation = len(overlap) / len(context_tokens)
    return faithfulness, context_utilisation


def _recall_at_k(
    chunks: Sequence[RetrievedChunk],
    relevant_chunks: dict[str, int],
    k: int,
) -> float | None:
    if not relevant_chunks:
        return None
    if k > len(chunks):
        return None

    window = chunks[:k]
    return 1.0 if any(chunk.chunk_id in relevant_chunks for chunk in window) else 0.0


def _mrr(chunks: Sequence[RetrievedChunk], relevant_chunks: dict[str, int]) -> float | None:
    if not relevant_chunks:
        return None
    for index, chunk in enumerate(chunks, start=1):
        if chunk.chunk_id in relevant_chunks:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(
    chunks: Sequence[RetrievedChunk],
    relevant_chunks: dict[str, int],
    k: int,
) -> float | None:
    if not relevant_chunks:
        return None
    if k > len(chunks):
        return None

    def dcg(scores: Sequence[int]) -> float:
        value = 0.0
        for rank, score in enumerate(scores):
            if score <= 0:
                continue
            value += (2**score - 1) / log2(rank + 2)
        return value

    gains = [relevant_chunks.get(chunk.chunk_id, 0) for chunk in chunks[:k]]
    ideal_gains = sorted(relevant_chunks.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_gains)

    if ideal_dcg == 0.0:
        return None

    return dcg(gains) / ideal_dcg


def _serialize_chunk(
    chunk: RetrievedChunk,
    *,
    rank: int | None = None,
    rank_label: str = "rank",
    reranker_score: float | None = None,
) -> dict[str, object]:
    payload = asdict(chunk)
    if rank is not None:
        payload[rank_label] = rank
    payload["reranker_score"] = reranker_score
    payload["structure_category"] = classify_structure(payload)
    return payload


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    non_null = [value for value in values if value is not None]
    return mean(non_null) if non_null else None


def run_evaluation(
    questions_path: Path,
    *,
    candidate_k: int | None = None,
    structural_filter_enabled: bool | None = None,
) -> dict[str, object]:
    settings = get_settings()
    services = Services.build(settings)
    questions = _load_questions(questions_path)
    effective_structural_filter_enabled = (
        settings.retrieval_structural_filter_enabled
        if structural_filter_enabled is None
        else structural_filter_enabled
    )

    results: list[QuestionResult] = []
    validation_errors: list[dict[str, object]] = []
    for index, item in enumerate(questions):
        question_id = item.id or f"q{index + 1:02d}"
        question_errors = _validate_question(item)
        if question_errors:
            validation_errors.append({"question_id": question_id, "errors": question_errors})

        filters = (
            None if item.expected_doc_id is None else RetrievalFilters(doc_id=item.expected_doc_id)
        )
        retrieval = services.retriever.retrieve(
            item.question,
            top_k=settings.retrieval_top_k_default,
            candidate_k=candidate_k,
            similarity_threshold=settings.retrieval_similarity_threshold,
            filters=filters,
            structural_filter_enabled=effective_structural_filter_enabled,
        )
        prompt_bundle = services.prompt_builder.build(item.question, retrieval.chunks)
        generation_start = perf_counter()
        print(f"Generating {question_id}: prompt_chars={len(prompt_bundle.prompt)}")
        answer = services.generation_client.generate(prompt_bundle.prompt).text
        generation_latency_ms = int((perf_counter() - generation_start) * 1000)

        precision, retrieval_hit_at_k = _score_heuristic_retrieval(item, retrieval.chunks)
        context_text = "\n".join(chunk.text for chunk in retrieval.chunks)
        faithfulness, context_utilisation = _score_answer(answer, context_text)

        relevant_chunks = _question_relevant_chunks(item)
        reranker_score_by_chunk_id: dict[str, float | None] = {}
        if retrieval.candidate_reranker_scores is not None:
            for candidate_chunk, reranker_score in zip(
                retrieval.candidate_chunks,
                retrieval.candidate_reranker_scores,
                strict=True,
            ):
                reranker_score_by_chunk_id[candidate_chunk.chunk_id] = reranker_score

        candidate_recall = {
            "5": _recall_at_k(retrieval.candidate_chunks, relevant_chunks, 5),
            "20": _recall_at_k(retrieval.candidate_chunks, relevant_chunks, 20),
        }

        result = QuestionResult(
            question_id=question_id,
            question=item.question,
            answerable=item.answerable,
            gold_answer=item.gold_answer,
            gold_evidence=[asdict(evidence) for evidence in item.gold_evidence],
            acceptable_evidence=[asdict(evidence) for evidence in item.acceptable_evidence],
            validation_errors=question_errors,
            candidate_recall=candidate_recall,
            recall_at_1=_recall_at_k(retrieval.chunks, relevant_chunks, 1),
            recall_at_5=_recall_at_k(retrieval.chunks, relevant_chunks, 5),
            recall_at_10=_recall_at_k(retrieval.chunks, relevant_chunks, 10),
            recall_at_20=_recall_at_k(retrieval.chunks, relevant_chunks, 20),
            mrr=_mrr(retrieval.chunks, relevant_chunks),
            ndcg_at_5=_ndcg_at_k(retrieval.chunks, relevant_chunks, 5),
            ndcg_at_10=_ndcg_at_k(retrieval.chunks, relevant_chunks, 10),
            candidate_chunks=[
                _serialize_chunk(
                    chunk,
                    rank=index + 1,
                    rank_label="candidate_rank",
                    reranker_score=reranker_score_by_chunk_id.get(chunk.chunk_id),
                )
                for index, chunk in enumerate(retrieval.candidate_chunks)
            ],
            retrieved_chunks=[
                _serialize_chunk(
                    chunk,
                    rank=index + 1,
                    reranker_score=reranker_score_by_chunk_id.get(chunk.chunk_id),
                )
                for index, chunk in enumerate(retrieval.chunks)
            ],
            original_candidate_count=retrieval.original_candidate_count,
            filtered_candidate_count=retrieval.filtered_candidate_count,
            candidate_structure_categories=(retrieval.candidate_structure_categories or []),
            filtered_candidate_structure_categories=(
                retrieval.filtered_candidate_structure_categories or []
            ),
            precision=precision,
            retrieval_hit_at_k=retrieval_hit_at_k,
            generation_latency_ms=generation_latency_ms,
            latency_ms=(
                retrieval.embedding_latency_ms
                + retrieval.retrieval_latency_ms
                + retrieval.rerank_latency_ms
                + generation_latency_ms
            ),
            faithfulness=faithfulness,
            context_utilisation=context_utilisation,
            answer=answer,
            retrieval_config=asdict(retrieval.retrieval_config),
        )
        results.append(result)

    summary = {
        "questions": len(results),
        "strict_metrics_questions": sum(1 for result in results if result.recall_at_1 is not None),
        "retrieval_precision": mean(result.precision for result in results) if results else 0.0,
        "retrieval_hit_at_k": mean(result.retrieval_hit_at_k for result in results)
        if results
        else 0.0,
        "generation_latency_ms": mean(result.generation_latency_ms for result in results)
        if results
        else 0.0,
        "latency_ms": mean(result.latency_ms for result in results) if results else 0.0,
        "answer_faithfulness": mean(result.faithfulness for result in results) if results else 0.0,
        "context_utilisation": mean(result.context_utilisation for result in results)
        if results
        else 0.0,
        "candidate_recall_at_5": _mean_or_none(
            [result.candidate_recall["5"] for result in results]
        ),
        "candidate_recall_at_20": _mean_or_none(
            [result.candidate_recall["20"] for result in results]
        ),
        "recall_at_1": _mean_or_none([result.recall_at_1 for result in results]),
        "recall_at_5": _mean_or_none([result.recall_at_5 for result in results]),
        "recall_at_10": _mean_or_none([result.recall_at_10 for result in results]),
        "recall_at_20": _mean_or_none([result.recall_at_20 for result in results]),
        "mrr": _mean_or_none([result.mrr for result in results]),
        "ndcg_at_5": _mean_or_none([result.ndcg_at_5 for result in results]),
        "ndcg_at_10": _mean_or_none([result.ndcg_at_10 for result in results]),
    }
    return {
        "summary": summary,
        "validation_errors": validation_errors,
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PDF RAG evaluation suite.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of dense retrieval candidates to retain before optional reranking.",
    )
    parser.add_argument(
        "--structural-filter-enabled",
        action="store_true",
        default=None,
        help="Exclude structural decoy chunks before reranking.",
    )
    args = parser.parse_args()

    report = run_evaluation(
        args.questions,
        candidate_k=args.candidate_k,
        structural_filter_enabled=args.structural_filter_enabled,
    )
    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

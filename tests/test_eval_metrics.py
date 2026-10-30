from __future__ import annotations

import json
from pathlib import Path

from app.core.vectorstore import RetrievedChunk
from eval.evaluate import (
    EvidenceSpec,
    QuestionSpec,
    _load_questions,
    _mrr,
    _ndcg_at_k,
    _question_relevant_chunks,
    _recall_at_k,
    _validate_question,
)


def _chunk(chunk_id: str, text: str = "chunk text", page_number: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id="doc-1",
        document_fingerprint="fingerprint-1",
        filename="sample.pdf",
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        raw_score=0.9,
        score=0.9,
        page_number=page_number,
        section_title="Section",
        source_path="/tmp/sample.pdf",
    )


def test_load_questions_parses_gold_evidence_schema(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-1",
                    "question": "What is the topic?",
                    "expected_answer_contains": "topic",
                    "expected_doc_id": "doc-1",
                    "expected_chunk_keywords": ["topic"],
                    "gold_answer": "It is about the topic.",
                    "answerable": False,
                    "gold_evidence": [],
                    "acceptable_evidence": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = _load_questions(questions_path)

    assert questions[0].id == "q-1"
    assert questions[0].gold_answer == "It is about the topic."
    assert questions[0].answerable is False


def test_validation_accepts_answerable_question_with_known_gold_chunk() -> None:
    question = QuestionSpec(
        id="q-1",
        question="What is the chapter about?",
        expected_answer_contains="chapter",
        expected_doc_id="doc-1",
        expected_chunk_keywords=["chapter"],
        gold_answer="It is about a chapter.",
        answerable=True,
        gold_evidence=[
            EvidenceSpec(
                page_number=1,
                chunk_id="chunk-a",
                text_span="This chapter explains the chapter topic.",
                relevance="primary",
            )
        ],
    )

    errors = _validate_question(question, known_chunks={"chunk-a": 1})

    assert errors == []


def test_validation_flags_missing_and_invalid_gold_chunk_ids() -> None:
    question = QuestionSpec(
        id="q-2",
        question="What is the chapter about?",
        expected_answer_contains="chapter",
        expected_doc_id="doc-1",
        expected_chunk_keywords=["chapter"],
        gold_answer="It is about a chapter.",
        answerable=True,
        gold_evidence=[
            EvidenceSpec(
                page_number=2,
                chunk_id="chunk-missing",
                text_span="This chapter explains the chapter topic.",
            )
        ],
    )

    errors = _validate_question(question, known_chunks={"chunk-a": 1})

    assert any("unknown gold evidence chunk_id=chunk-missing" in error for error in errors)


def test_unanswerable_question_allows_empty_evidence_and_no_strict_metrics() -> None:
    question = QuestionSpec(
        id="q-3",
        question="Which option is correct?",
        expected_answer_contains="unknown",
        expected_doc_id=None,
        expected_chunk_keywords=[],
        answerable=False,
    )

    errors = _validate_question(question)

    assert errors == []
    assert _question_relevant_chunks(question) == {}
    assert _recall_at_k([_chunk("chunk-a")], {}, 1) is None
    assert _mrr([_chunk("chunk-a")], {}) is None
    assert _ndcg_at_k([_chunk("chunk-a")], {}, 5) is None


def test_multiple_acceptable_evidence_chunks_count_as_relevant() -> None:
    question = QuestionSpec(
        id="q-4",
        question="What does the section discuss?",
        expected_answer_contains="section",
        expected_doc_id="doc-1",
        expected_chunk_keywords=["section"],
        gold_answer="It discusses the section.",
        answerable=True,
        gold_evidence=[
            EvidenceSpec(
                page_number=1,
                chunk_id="chunk-primary",
                text_span="Primary supporting passage.",
            )
        ],
        acceptable_evidence=[
            EvidenceSpec(
                page_number=1,
                chunk_id="chunk-alt",
                text_span="Alternative supporting passage.",
            )
        ],
    )

    relevant_chunks = _question_relevant_chunks(question)
    retrieved_chunks = [_chunk("chunk-alt")]

    assert relevant_chunks == {"chunk-primary": 1, "chunk-alt": 1}
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 1) == 1.0


def test_strict_metrics_use_exact_chunk_id_matches() -> None:
    relevant_chunks = {"gold-chunk": 1}
    retrieved_chunks = [_chunk("distractor"), _chunk("gold-chunk")]

    assert _recall_at_k(retrieved_chunks, relevant_chunks, 1) == 0.0
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 2) == 1.0
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 5) is None
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 10) is None
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 20) is None

    assert _mrr(retrieved_chunks, relevant_chunks) == 0.5

    assert (
        _ndcg_at_k(
            retrieved_chunks,
            relevant_chunks,
            2,
        )
        == 0.6309297535714575
    )
    assert _ndcg_at_k(retrieved_chunks, relevant_chunks, 5) is None
    assert _ndcg_at_k(retrieved_chunks, relevant_chunks, 10) is None


def test_metrics_return_none_when_k_exceeds_available_chunks() -> None:
    relevant_chunks = {"gold-chunk": 1}
    retrieved_chunks = [
        _chunk("distractor"),
        _chunk("gold-chunk"),
    ]

    assert _recall_at_k(retrieved_chunks, relevant_chunks, 2) == 1.0
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 5) is None
    assert _recall_at_k(retrieved_chunks, relevant_chunks, 20) is None

    assert _ndcg_at_k(retrieved_chunks, relevant_chunks, 2) is not None
    assert _ndcg_at_k(retrieved_chunks, relevant_chunks, 5) is None
    assert _ndcg_at_k(retrieved_chunks, relevant_chunks, 20) is None


def test_candidate_recall_is_distinct_from_final_recall() -> None:
    relevant_chunks = {"gold-chunk": 1}
    candidate_chunks = [
        _chunk("distractor-1"),
        _chunk("distractor-2"),
        _chunk("gold-chunk"),
        _chunk("distractor-3"),
        _chunk("distractor-4"),
    ]
    final_chunks = [
        _chunk("distractor-1"),
        _chunk("distractor-2"),
        _chunk("distractor-3"),
        _chunk("distractor-4"),
        _chunk("distractor-5"),
    ]

    candidate_recall_at_5 = _recall_at_k(
        candidate_chunks,
        relevant_chunks,
        5,
    )
    final_recall_at_5 = _recall_at_k(
        final_chunks,
        relevant_chunks,
        5,
    )

    assert candidate_recall_at_5 == 1.0
    assert final_recall_at_5 == 0.0

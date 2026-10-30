from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from app.config import Settings
from app.core.reranker import RankedChunk
from app.core.retrieval import RetrievalConfig, RetrievalResult
from app.core.vectorstore import RetrievedChunk
from eval.analyze_reranker import (
    analyze_report,
    classify_outcome,
    classify_structure,
    rank_displacement,
)
from eval.evaluate import run_evaluation


def test_classify_structure_and_rank_displacement() -> None:
    assert (
        classify_structure(
            {
                "section_title": "Index",
                "text": "Index",
            }
        )
        == "INDEX"
    )
    assert (
        classify_structure(
            {
                "section_title": "Chapter 1",
                "text": "This is substantive content.",
            }
        )
        == "SUBSTANTIVE"
    )
    assert rank_displacement(7, 2) == 5
    assert rank_displacement(None, 2) is None
    assert classify_outcome(False, False) == "CANDIDATE_GENERATION_FAILURE"
    assert classify_outcome(True, False) == "RERANKER_SELECTION_FAILURE"
    assert classify_outcome(True, True) == "SUCCESSFUL_SELECTION"


def test_run_evaluation_persists_reranker_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Where is the answer?",
                    "expected_answer_contains": "answer",
                    "expected_doc_id": "doc-1",
                    "expected_chunk_keywords": ["answer"],
                    "gold_answer": "The answer is on page 2.",
                    "answerable": True,
                    "gold_evidence": [
                        {
                            "page_number": 2,
                            "chunk_id": "chunk-2",
                            "text_span": "The answer is here.",
                            "relevance": "primary",
                            "grade": 3,
                        }
                    ],
                    "acceptable_evidence": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    chunk_one = RetrievedChunk(
        doc_id="doc-1",
        document_fingerprint="fingerprint-1",
        filename="sample.pdf",
        chunk_id="chunk-1",
        chunk_index=0,
        text="distractor",
        raw_score=0.8,
        score=0.8,
        page_number=1,
        section_title="Contents",
        source_path="/tmp/sample.pdf",
    )
    chunk_two = RetrievedChunk(
        doc_id="doc-1",
        document_fingerprint="fingerprint-1",
        filename="sample.pdf",
        chunk_id="chunk-2",
        chunk_index=1,
        text="The answer is here.",
        raw_score=0.9,
        score=0.9,
        page_number=2,
        section_title="Chapter 1",
        source_path="/tmp/sample.pdf",
    )

    class FakeRetriever:
        def retrieve(self, *args: object, **kwargs: object) -> RetrievalResult:
            _ = args, kwargs
            return RetrievalResult(
                chunks=[chunk_two],
                candidate_chunks=[chunk_one, chunk_two],
                filtered_candidate_chunks=[chunk_one, chunk_two],
                original_candidate_count=2,
                filtered_candidate_count=2,
                candidate_reranker_scores=[0.2, 0.9],
                embedding_latency_ms=1,
                retrieval_latency_ms=2,
                rerank_latency_ms=3,
                retrieval_config=RetrievalConfig(
                    embedding_model_name="model",
                    embedding_version="v1",
                    candidate_k=2,
                    top_k=1,
                    similarity_threshold=0.0,
                    reranker_enabled=True,
                    reranker_model_name="BAAI/bge-reranker-base",
                    structural_filter_enabled=False,
                ),
            )

    class FakePromptBuilder:
        def build(self, message: str, sources: list[RetrievedChunk]) -> object:
            _ = sources
            return type("Prompt", (), {"prompt": f"{message}\n"})()

    class FakeGenerationClient:
        def generate(self, prompt: str) -> object:
            _ = prompt
            return type("Generation", (), {"text": "generated answer"})()

    class FakeVectorStore:
        def ping(self) -> bool:
            return True

    @dataclass
    class FakeServices:
        settings: Settings
        retriever: FakeRetriever
        prompt_builder: FakePromptBuilder
        generation_client: FakeGenerationClient
        vectorstore: FakeVectorStore

    fake_services = FakeServices(
        settings=Settings(),
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        generation_client=FakeGenerationClient(),
        vectorstore=FakeVectorStore(),
    )
    monkeypatch.setattr("eval.evaluate.get_settings", lambda: Settings())
    monkeypatch.setattr("eval.evaluate.Services.build", lambda settings: fake_services)

    report = run_evaluation(questions_path)

    results = cast(list[dict[str, object]], report["results"])
    result = results[0]
    candidate_chunks = cast(list[dict[str, object]], result["candidate_chunks"])
    retrieved_chunks = cast(list[dict[str, object]], result["retrieved_chunks"])
    assert candidate_chunks[0]["reranker_score"] == 0.2
    assert candidate_chunks[1]["reranker_score"] == 0.9
    assert retrieved_chunks[0]["reranker_score"] == 0.9


def test_analyze_report_handles_missing_reranker_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "summary": {"questions": 1},
                "results": [
                    {
                        "question_id": "Q001",
                        "question": "What is the answer?",
                        "answerable": True,
                        "candidate_chunks": [
                            {
                                "chunk_id": "chunk-contents",
                                "candidate_rank": 1,
                                "raw_score": 0.91,
                                "score": 0.91,
                                "page_number": 1,
                                "section_title": "Contents",
                                "text": "Table of contents",
                            },
                            {
                                "chunk_id": "chunk-gold",
                                "candidate_rank": 2,
                                "raw_score": 0.88,
                                "score": 0.88,
                                "page_number": 2,
                                "section_title": "Chapter 1",
                                "text": "The answer is here.",
                            },
                        ],
                        "retrieved_chunks": [
                            {
                                "chunk_id": "chunk-contents",
                                "rank": 1,
                                "raw_score": 0.91,
                                "score": 0.91,
                                "page_number": 1,
                                "section_title": "Contents",
                                "text": "Table of contents",
                            }
                        ],
                        "retrieval_config": {
                            "reranker_enabled": True,
                            "reranker_model_name": "BAAI/bge-reranker-base",
                        },
                        "gold_evidence": [
                            {
                                "chunk_id": "chunk-gold",
                                "page_number": 2,
                                "text_span": "The answer is here.",
                                "relevance": "primary",
                                "grade": 3,
                            }
                        ],
                        "acceptable_evidence": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeReranker:
        def __init__(self, settings: Settings) -> None:
            _ = settings

        def rank(self, query: str, passages: list[str]) -> list[RankedChunk]:
            _ = query, passages
            return [RankedChunk(index=0, score=0.1), RankedChunk(index=1, score=0.9)]

    monkeypatch.setattr("eval.analyze_reranker.Reranker", FakeReranker)

    analysis = analyze_report(benchmark_path)
    summary = cast(dict[str, object], analysis["summary"])
    failure_cases = cast(list[dict[str, object]], analysis["failure_cases"])
    reranker_sources = cast(dict[str, int], analysis["reranker_score_sources"])

    assert summary["gold_absent_from_candidate_20"] == 0
    assert summary["gold_lost_after_reranking"] == 1
    assert failure_cases[0]["replacement_structure_category"] == "CONTENTS"
    assert failure_cases[0]["reranker_score"] == 0.9
    assert failure_cases[0]["reranker_score_source"] == "computed"
    assert reranker_sources["computed"] == 1

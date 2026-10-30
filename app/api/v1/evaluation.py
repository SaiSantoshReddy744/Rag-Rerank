"""Evaluation endpoints for saved research runs and uploaded-document comparisons."""
from __future__ import annotations

import json
from math import log2
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request

from app.config import PROJECT_ROOT
from app.core.generation_client import GenerationProviderError
from app.core.reranker import RerankerError
from app.core.vectorstore import RetrievedChunk, RetrievalFilters
from app.models.schemas import PdfEvaluationRequest

router = APIRouter()
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
_METRICS = (
    "questions", "recall_at_1", "recall_at_5", "mrr", "ndcg_at_5",
    "candidate_recall_at_5", "candidate_recall_at_20", "retrieval_precision",
    "retrieval_hit_at_k", "answer_faithfulness", "context_utilisation", "latency_ms",
)


def _load(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, dict) or "mrr" not in summary:
        return None  # not a retrieval-evaluation file (e.g. analysis output)
    results = data.get("results") or []
    cfg = results[0].get("retrieval_config", {}) if results and isinstance(results[0], dict) else {}
    return {
        "run": path.stem,
        "reranker_enabled": cfg.get("reranker_enabled"),
        "reranker_model": cfg.get("reranker_model_name"),
        "candidate_k": cfg.get("candidate_k"),
        "top_k": cfg.get("top_k"),
        "structural_filter_enabled": cfg.get("structural_filter_enabled"),
        "metrics": {k: summary.get(k) for k in _METRICS},
    }


@router.get("/evaluation/results")
def evaluation_results() -> dict[str, object]:
    runs = []
    if RESULTS_DIR.is_dir():
        for path in sorted(RESULTS_DIR.glob("*.json")):
            run = _load(path)
            if run is not None:
                runs.append(run)
    return {
        "source": "eval/results/*.json (stored measurements from earlier runs of eval/evaluate.py)",
        "note": "Computed on the repository's evaluation PDF, not on documents you upload here.",
        "runs": runs,
    }


def _page_metrics(
    chunks: list[RetrievedChunk],
    relevant_pages: set[int],
    total_relevant_chunks: int,
    top_k: int,
) -> dict[str, float]:
    ranked = chunks[:top_k]
    hits = [int(chunk.page_number in relevant_pages) for chunk in ranked]
    relevant_hits = sum(hits)
    precision = relevant_hits / top_k
    recall = relevant_hits / total_relevant_chunks if total_relevant_chunks else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
    mrr = 1 / first if first else 0.0
    dcg = sum(hit / log2(i + 2) for i, hit in enumerate(hits))
    ideal_count = min(total_relevant_chunks, top_k)
    idcg = sum(1 / log2(i + 2) for i in range(ideal_count))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "mrr": mrr,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def _roc_auc(
    chunks: list[RetrievedChunk],
    scores: list[float | None],
    relevant_pages: set[int],
) -> float | None:
    paired = [
        (chunk.page_number in relevant_pages, score)
        for chunk, score in zip(chunks, scores, strict=False)
        if score is not None
    ]
    positives = [score for relevant, score in paired if relevant]
    negatives = [score for relevant, score in paired if not relevant]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0
               for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


@router.post("/evaluation/documents/{doc_id}")
def evaluate_document(
    doc_id: str,
    payload: PdfEvaluationRequest,
    request: Request,
) -> dict[str, object]:
    services = request.app.state.services
    documents = services.vectorstore.list_documents()
    document = next((item for item in documents if item["doc_id"] == doc_id), None)
    if document is None:
        raise HTTPException(status_code=404, detail="Selected PDF is not indexed.")
    if not services.settings.re_rank_enabled:
        raise HTTPException(
            status_code=400,
            detail="Enable RE_RANK_ENABLED=true in .env to compare reranking.",
        )
    try:
        services.generation_client.check_configured()
    except GenerationProviderError as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc

    candidate_k = max(services.settings.rerank_candidate_k, payload.top_k)
    doc_chunks = services.vectorstore.list_chunks_for_doc_id(doc_id)
    aggregates: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg_at_k", "roc_auc")}
        for name in ("dense", "reranked")
    }
    latency_values = {"dense": [], "reranked": []}
    answer_latency_values: list[int] = []
    question_rows: list[dict[str, object]] = []

    for index, item in enumerate(payload.questions, start=1):
        filters = RetrievalFilters(doc_id=doc_id)
        try:
            dense = services.retriever.retrieve(
                item.question,
                top_k=payload.top_k,
                candidate_k=candidate_k,
                filters=filters,
                rerank_enabled=False,
            )
            reranked = services.retriever.retrieve(
                item.question,
                top_k=payload.top_k,
                candidate_k=candidate_k,
                filters=filters,
                rerank_enabled=True,
            )
        except RerankerError as exc:
            raise HTTPException(status_code=500, detail=f"Reranker failed: {exc}") from exc

        relevant_pages = set(item.relevant_pages)
        if any(page > int(document["page_count"]) for page in relevant_pages):
            raise HTTPException(
                status_code=422,
                detail=f"Question {index} has a relevant page beyond this PDF's indexed page count.",
            )
        total_relevant_chunks = sum(
            1 for chunk in doc_chunks if int(chunk["page_number"]) in relevant_pages
        )
        if total_relevant_chunks == 0:
            raise HTTPException(
                status_code=422,
                detail=f"Question {index} labels page(s) with no indexed text as relevant.",
            )
        dense_metrics = _page_metrics(
            dense.chunks, relevant_pages, total_relevant_chunks, payload.top_k
        )
        reranked_metrics = _page_metrics(
            reranked.chunks, relevant_pages, total_relevant_chunks, payload.top_k
        )
        dense_auc = _roc_auc(
            dense.candidate_chunks,
            [chunk.score for chunk in dense.candidate_chunks],
            relevant_pages,
        )
        rerank_scores = reranked.candidate_reranker_scores or []
        reranked_auc = _roc_auc(reranked.candidate_chunks, rerank_scores, relevant_pages)
        if dense_auc is not None:
            dense_metrics["roc_auc"] = dense_auc
        if reranked_auc is not None:
            reranked_metrics["roc_auc"] = reranked_auc
        dense_ms = dense.embedding_latency_ms + dense.retrieval_latency_ms
        reranked_ms = (
            reranked.embedding_latency_ms
            + reranked.retrieval_latency_ms
            + reranked.rerank_latency_ms
        )
        prompt_bundle = services.prompt_builder.build(item.question, reranked.chunks)
        answer_start = perf_counter()
        try:
            generated_answer = services.generation_client.generate(prompt_bundle.prompt).text
        except GenerationProviderError as exc:
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
        answer_latency_ms = int((perf_counter() - answer_start) * 1000)
        answer_latency_values.append(answer_latency_ms)
        for name, metrics, latency in (
            ("dense", dense_metrics, dense_ms),
            ("reranked", reranked_metrics, reranked_ms),
        ):
            for metric, value in metrics.items():
                aggregates[name][metric].append(value)
            latency_values[name].append(latency)
        question_rows.append(
            {
                "question": item.question,
                "relevant_pages": sorted(relevant_pages),
                "dense": dense_metrics,
                "reranked": reranked_metrics,
                "dense_latency_ms": dense_ms,
                "reranked_latency_ms": reranked_ms,
                "reranker_ms": reranked.rerank_latency_ms,
                "generated_answer": generated_answer,
                "answer_generation_ms": answer_latency_ms,
            }
        )

    summary = {
        name: {
            metric: (sum(values) / len(values) if values else None)
            for metric, values in metrics.items()
        }
        for name, metrics in aggregates.items()
    }
    summary["dense"]["mean_latency_ms"] = sum(latency_values["dense"]) / len(latency_values["dense"])
    summary["reranked"]["mean_latency_ms"] = sum(latency_values["reranked"]) / len(latency_values["reranked"])
    summary["mean_answer_generation_ms"] = (
        sum(answer_latency_values) / len(answer_latency_values) if answer_latency_values else 0.0
    )
    return {
        "filename": document["filename"],
        "question_count": len(question_rows),
        "top_k": payload.top_k,
        "candidate_k": candidate_k,
        "metrics_are_page_based": True,
        "summary": summary,
        "questions": question_rows,
    }


@router.get("/evaluation/results/{run}")
def evaluation_run(run: str) -> dict[str, object]:
    if "/" in run or "\\" in run or ".." in run:
        raise HTTPException(status_code=400, detail="Invalid run name")
    loaded = _load(RESULTS_DIR / f"{run}.json")
    if loaded is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return loaded

import csv
import json
import time
from pathlib import Path
from typing import TypedDict

import httpx

BASE_URL = "http://localhost:8000"
DOC_ID = "5b038ed7-a74c-4b00-9f61-059a11d55b23"


class SourceData(TypedDict, total=False):
    page_number: int
    score: float
    section_title: str | None


class LatencyData(TypedDict, total=False):
    embedding: int
    retrieval: int
    rerank: int
    llm: int
    total: int


class QuestionResult(TypedDict):
    answer: str
    sources: list[SourceData]
    latency_ms: LatencyData
    wall_clock_ms: float


OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


QUESTIONS = [
    ("Q001", "factual", "What is the 10X Rule?", "Chapter 1"),
    ("Q002", "factual", "What are the two parts of the 10X Rule?", "Chapter 1"),
    ("Q003", "factual", "What are the four degrees of action?", "Chapter 7"),
    ("Q004", "factual", "What does the fourth degree of action mean?", "Chapter 7"),
    (
        "Q005",
        "factual",
        "What are the four major mistakes people make when setting goals?",
        "Chapter 1",
    ),
    (
        "Q006",
        "conceptual",
        "Why does the author believe people commonly underestimate the amount of action required to achieve a goal?",
        "Chapter 1-2",
    ),
    (
        "Q007",
        "conceptual",
        "Why does the author consider average or normal levels of action dangerous?",
        "Chapter 8",
    ),
    (
        "Q008",
        "conceptual",
        "Why does the author argue that massive action can create new problems?",
        "Chapter 7",
    ),
    (
        "Q009",
        "conceptual",
        "Why does the author believe success must be maintained rather than achieved only once?",
        "Chapter 1",
    ),
    (
        "Q010",
        "conceptual",
        "Why does the author argue that setting goals too low can limit the actions required to achieve them?",
        "Chapter 1-9",
    ),
    (
        "Q011",
        "paraphrase",
        "According to the author, what happens when someone dramatically underestimates the effort required for a goal?",
        "Chapter 2",
    ),
    (
        "Q012",
        "paraphrase",
        "If someone keeps operating at ordinary levels of effort, why might they remain stuck at ordinary results?",
        "Chapter 8",
    ),
    (
        "Q013",
        "paraphrase",
        "What does the book suggest about continuing to act after encountering resistance?",
        "Chapter 2-7",
    ),
    (
        "Q014",
        "paraphrase",
        "How does the author connect unusually ambitious thinking with unusually large amounts of action?",
        "Chapter 1",
    ),
    (
        "Q015",
        "paraphrase",
        "Why does the author believe fear should lead to action rather than avoidance?",
        "Chapter 16",
    ),
    (
        "Q016",
        "chunk-boundary",
        "How does the author connect setting ambitious targets with estimating the effort required to achieve them?",
        "Chapter 1",
    ),
    (
        "Q017",
        "chunk-boundary",
        "How does the author connect massive thinking with massive action?",
        "Chapter 1",
    ),
    (
        "Q018",
        "chunk-boundary",
        "Why does the author connect normal levels of action with vulnerability to unexpected events?",
        "Chapter 2-8",
    ),
    (
        "Q019",
        "chunk-boundary",
        "How does the author connect taking massive action with encountering new problems and criticism?",
        "Chapter 7",
    ),
    (
        "Q020",
        "chunk-boundary",
        "How does the author connect achieving success with maintaining and creating additional success?",
        "Chapter 1-3",
    ),
    (
        "Q021",
        "multi-part",
        "What are the four degrees of action, and which one does the author associate with exceptional success?",
        "Chapter 7",
    ),
    (
        "Q022",
        "multi-part",
        "What is the relationship between 10X goals and 10X actions according to the author?",
        "Chapter 1-9",
    ),
    (
        "Q023",
        "multi-part",
        'What does the author mean by "average" and why does he consider it a failing formula?',
        "Chapter 8",
    ),
    (
        "Q024",
        "multi-part",
        "What does the author say about fear, and how should fear influence action?",
        "Chapter 16",
    ),
    (
        "Q025",
        "multi-part",
        "What does the author identify as characteristics of successful people, and why does he believe these characteristics are attainable?",
        "Chapter 22",
    ),
    (
        "Q026",
        "difficult",
        "How does the author distinguish between taking normal action and taking massive action?",
        "Chapter 7-8",
    ),
    (
        "Q027",
        "difficult",
        "Why does the author believe people who retreat may actually be reacting to their interpretation of failure rather than failure itself?",
        "Chapter 7",
    ),
    (
        "Q028",
        "difficult",
        "How does the author's experience of building his first business illustrate his argument about estimating effort?",
        "Chapter 2",
    ),
    (
        "Q029",
        "difficult",
        "What does the author mean when he says that successful people continue creating new levels of success instead of relying on previous achievements?",
        "Chapter 1-3",
    ),
    (
        "Q030",
        "difficult",
        "How does the book connect responsibility, action, and success?",
        "Chapter 4-23",
    ),
    (
        "Q031",
        "unanswerable",
        "What does the author recommend as the best programming language for building AI systems?",
        "NOT FOUND",
    ),
    (
        "Q032",
        "unanswerable",
        "What embedding model does the author recommend for semantic search?",
        "NOT FOUND",
    ),
    (
        "Q033",
        "unanswerable",
        "What is the author's opinion on Qdrant?",
        "NOT FOUND",
    ),
    (
        "Q034",
        "unanswerable",
        "What does the author say about retrieval-augmented generation?",
        "NOT FOUND",
    ),
    (
        "Q035",
        "unanswerable",
        "What GPU does the author recommend for running Qwen3 8B?",
        "NOT FOUND",
    ),
]


def run_question(
    client: httpx.Client,
    question: str,
) -> QuestionResult:
    start = time.perf_counter()

    response = client.post(
        f"{BASE_URL}/v1/chat",
        json={
            "message": question,
            "doc_id": DOC_ID,
            "top_k": 5,
            "stream": False,
        },
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    payload = response.json()

    return {
        "answer": payload.get("answer", ""),
        "sources": payload.get("sources", []),
        "latency_ms": payload.get("latency_ms", {}),
        "wall_clock_ms": round(elapsed_ms, 2),
    }


def main() -> None:
    results = []

    print(f"Running {len(QUESTIONS)} questions against {DOC_ID}")
    print(f"Endpoint: {BASE_URL}/v1/chat")
    print()

    with httpx.Client(timeout=120.0) as client:
        for index, (qid, category, question, expected_section) in enumerate(
            QUESTIONS,
            start=1,
        ):
            print(f"[{index}/{len(QUESTIONS)}] {qid}: {question}")

            try:
                result = run_question(client, question)

                sources = result["sources"]

                top_source = sources[0] if sources else {}

                results.append(
                    {
                        "id": qid,
                        "category": category,
                        "question": question,
                        "expected_section": expected_section,
                        "answer": result["answer"],
                        "top_source_page": top_source.get("page_number"),
                        "top_source_score": top_source.get("score"),
                        "top_source_section": top_source.get("section_title"),
                        "source_count": len(sources),
                        "sources_json": json.dumps(
                            sources,
                            ensure_ascii=False,
                        ),
                        "embedding_ms": result["latency_ms"].get("embedding"),
                        "retrieval_ms": result["latency_ms"].get("retrieval"),
                        "rerank_ms": result["latency_ms"].get("rerank"),
                        "llm_ms": result["latency_ms"].get("llm"),
                        "total_ms": result["latency_ms"].get("total"),
                        "wall_clock_ms": result["wall_clock_ms"],
                        "error": "",
                    }
                )

                print(
                    f"    top page={top_source.get('page_number')} "
                    f"score={top_source.get('score')} "
                    f"total={result['latency_ms'].get('total')}ms"
                )

            except Exception as exc:
                print(f"    ERROR: {exc}")

                results.append(
                    {
                        "id": qid,
                        "category": category,
                        "question": question,
                        "expected_section": expected_section,
                        "answer": "",
                        "top_source_page": "",
                        "top_source_score": "",
                        "top_source_section": "",
                        "source_count": 0,
                        "sources_json": "[]",
                        "embedding_ms": "",
                        "retrieval_ms": "",
                        "rerank_ms": "",
                        "llm_ms": "",
                        "total_ms": "",
                        "wall_clock_ms": "",
                        "error": str(exc),
                    }
                )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"m1_eval_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"m1_eval_{timestamp}.csv"

    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = list(results[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("DONE")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()

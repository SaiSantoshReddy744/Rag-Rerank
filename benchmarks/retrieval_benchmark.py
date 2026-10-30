from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import fitz

from app.config import get_settings
from app.ingestion.extractor import extract_pdf_text
from app.services import Services


@dataclass(frozen=True)
class BenchmarkSeries:
    mean_ms: float
    p95_ms: float
    samples: list[float]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(0.95 * (len(ordered) - 1)))
    return ordered[index]


def _summarize(samples: list[float]) -> BenchmarkSeries:
    return BenchmarkSeries(
        mean_ms=round(mean(samples), 2) if samples else 0.0,
        p95_ms=round(_p95(samples), 2) if samples else 0.0,
        samples=[round(value, 2) for value in samples],
    )


def _create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Architecture overview and setup instructions.")
    page_two = document.new_page()
    page_two.insert_text((72, 72), "Retrieval and generation are driven by Inference Lab.")
    document.save(path)
    document.close()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_benchmark(sample_pdf: Path, queries: list[str], iterations: int) -> dict[str, object]:
    settings = get_settings()
    services = Services.build(settings)
    services.vectorstore.ensure_collection()

    pdf_bytes = sample_pdf.read_bytes()
    services.pipeline.ingest_pdf(
        doc_id=str(uuid.uuid4()),
        document_fingerprint=_hash_bytes(pdf_bytes),
        filename=sample_pdf.name,
        pdf_path=sample_pdf,
    )

    upload_latencies: list[float] = []
    extraction_latencies: list[float] = []
    embedding_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    total_latencies: list[float] = []

    for iteration in range(iterations):
        doc_id = str(uuid.uuid4())
        ingest_path = sample_pdf.parent / f"benchmark-{iteration}-{doc_id}.pdf"

        upload_start = perf_counter()
        fingerprint = _hash_bytes(pdf_bytes)
        ingest_path.write_bytes(pdf_bytes)
        upload_latencies.append((perf_counter() - upload_start) * 1000)

        extraction_start = perf_counter()
        pages = extract_pdf_text(ingest_path)
        extraction_latencies.append((perf_counter() - extraction_start) * 1000)

        chunk_texts = [chunk.text for chunk in services.pipeline._chunker.chunk_pages(pages)]  # noqa: SLF001
        embedding_start = perf_counter()
        embeddings = services.embedder.embed_texts(chunk_texts) if chunk_texts else []
        embedding_latencies.append((perf_counter() - embedding_start) * 1000)

        if chunk_texts and embeddings:
            services.pipeline._indexer.index_chunks(  # noqa: SLF001
                doc_id=doc_id,
                document_fingerprint=fingerprint,
                filename=sample_pdf.name,
                chunks=services.pipeline._chunker.chunk_pages(pages),  # noqa: SLF001
                embeddings=embeddings,
            )

        for query in queries:
            total_start = perf_counter()
            total_doc_id = str(uuid.uuid4())
            total_upload_path = sample_pdf.parent / f"total-{iteration}-{total_doc_id}.pdf"
            total_upload_path.write_bytes(pdf_bytes)

            total_fingerprint = _hash_bytes(pdf_bytes)
            services.pipeline.ingest_pdf(
                doc_id=total_doc_id,
                document_fingerprint=total_fingerprint,
                filename=sample_pdf.name,
                pdf_path=total_upload_path,
            )
            retrieval = services.retriever.retrieve(
                query,
                top_k=settings.retrieval_top_k_default,
                similarity_threshold=settings.retrieval_similarity_threshold,
            )
            retrieval_latencies.append(
                retrieval.retrieval_latency_ms + retrieval.embedding_latency_ms
            )

            prompt_bundle = services.prompt_builder.build(query, retrieval.chunks)
            generation_start = perf_counter()
            services.generation_client.generate(prompt_bundle.prompt)
            generation_latencies.append((perf_counter() - generation_start) * 1000)
            total_latencies.append((perf_counter() - total_start) * 1000)

    report: dict[str, object] = {
        "local_file_write_latency_ms": asdict(_summarize(upload_latencies)),
        "extraction_latency_ms": asdict(_summarize(extraction_latencies)),
        "embedding_latency_ms": asdict(_summarize(embedding_latencies)),
        "retrieval_latency_ms": asdict(_summarize(retrieval_latencies)),
        "generation_latency_ms": asdict(_summarize(generation_latencies)),
        "total_end_to_end_latency_ms": asdict(_summarize(total_latencies)),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PDF RAG latency stages and emit JSON reports."
    )
    parser.add_argument("--sample-pdf", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.sample_pdf is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_pdf = Path(tmpdir) / "benchmark-sample.pdf"
            _create_sample_pdf(sample_pdf)
            report = run_benchmark(sample_pdf, ["What does the document cover?"], args.iterations)
    else:
        report = run_benchmark(args.sample_pdf, ["What does the document cover?"], args.iterations)

    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

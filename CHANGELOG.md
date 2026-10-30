# Changelog

All notable changes to this repository will be documented in this file.

## [1.2.0] - 2026-08-17

### Added

- Added passage-level gold-evidence evaluation for retrieval quality analysis.
- Added candidate Recall@k, Recall@1, Recall@5, Recall@20, MRR, and nDCG@5 retrieval metrics.
- Added exact candidate-pool instrumentation for dense retrieval and reranking analysis.
- Added per-candidate dense scores, reranker scores, candidate ranks, final ranks, and structural categories to evaluation artifacts.
- Added reranker failure analysis to identify cases where gold evidence is retrieved but displaced before final `top_k`.
- Added score-margin analysis between displaced gold evidence and replacement candidates.
- Added document-structure classification for `CONTENTS`, `INDEX`, `COPYRIGHT`, `GLOSSARY`, and substantive content.
- Added structural eligibility filtering between dense retrieval and reranking.
- Added controlled before/after evaluation for structural filtering.
- Added benchmark artifacts for dense retrieval, reranking, reranker failure analysis, and structural filtering.
- Added validation and provenance checks around gold-evidence evaluation artifacts.

### Changed

- Extended the M1 retrieval evaluation flow from answer-level inspection to passage-level evidence evaluation.
- Separated candidate retrieval failures from reranking failures in evaluation and diagnostics.
- Extended retrieval telemetry to preserve candidate-level ranking information required for forensic analysis.
- Updated the evaluation workflow to keep controlled retrieval interventions isolated from unrelated pipeline changes.
- Updated retrieval analysis to distinguish semantic relevance from evidentiary usefulness.
- Updated the retrieval pipeline to optionally exclude structural document chunks from the evidence-oriented reranking path without removing them from the corpus.
- Updated documentation with the results and limitations of the M1 retrieval experiments.

### Research Findings

- Dense retrieval achieved `83.3%` candidate Recall@20 on the strict 30-question evaluation set used for the reranking experiment.
- With `candidate_k=20`, final Recall@5 was `66.7%` before structural filtering.
- Structural filtering improved final Recall@5 from `66.7%` to `70.0%`.
- MRR improved from `0.3911` to `0.4022`.
- nDCG@5 improved from `0.3561` to `0.3722`.
- Recall@1 remained `20.0%`, showing that structural filtering addressed only part of the reranking failure surface.
- Structural document content such as tables of contents, indexes, and glossaries can be semantically relevant while providing weak evidence for answer-oriented retrieval.
- Increasing the reranking candidate pool does not guarantee improved final retrieval quality.
- The evaluation infrastructure now distinguishes candidate coverage from final evidence ranking, enabling more precise retrieval failure diagnosis.

### Documentation

- Updated the README with the M1 retrieval evaluation architecture, benchmark methodology, experimental results, evaluation artifacts, limitations, and next research direction.
- Added v1.2.0 release documentation for retrieval evaluation and reranking failure analysis.

## [1.1.0] - 2026-08-10

### Changed

- Updated the README to match the implemented M1 retrieval, reranking, evaluation, and benchmarking flow.
- Clarified the distinction between rerank candidate pools and final `top_k` context.
- Corrected the offline evaluation section to describe the metrics currently computed by the code.
- Added v1.1.0 release notes.

## [1.0.0] - 2026-07-28

### Added

- Production-oriented PDF ingestion pipeline with PyMuPDF extraction, recursive chunking, embedding generation, and Qdrant indexing.
- Deterministic document fingerprints, document IDs, and duplicate handling policies.
- Semantic retrieval with metadata filtering, score normalization, and deterministic ordering.
- Prompt construction and generation orchestration through Inference Lab (M0) over REST.
- FastAPI endpoints for upload, chat, and health.
- Evaluation and benchmarking commands that emit JSON reports.
- Structured logging, request IDs, and typed error responses.

### Documentation

- Public README with architecture, setup, API examples, troubleshooting, and release guidance.
- Release notes for v1.0.0.
## Unreleased
- Replaced generic `/v1/generate` proxy client with a real Gemini `generateContent` client (header auth, retries, typed errors).
- `.env` now resolved from the project root; added `GEMINI_API_KEY`, `GEMINI_MODEL`.
- Embedded Qdrant fallback (no server); reranker enabled by default and fails loudly instead of returning fake scores.
- `/v1/chat` returns pre- and post-rerank passages; added `/v1/documents`, `/v1/models`, `/v1/evaluation/results`.
- Added single-page UI at `/`, `run.py`, `requirements.txt`, `python -m app.tools.list_models`.

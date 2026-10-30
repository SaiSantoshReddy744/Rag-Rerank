# Knowledge Vault v1.0.0 Release Notes

Knowledge Vault v1.0.0 is the first public release of the Milestone 1 retrieval layer for the local AI stack.

## Scope

This repository provides:

- PDF ingestion
- text extraction
- recursive chunking
- embedding generation
- vector indexing in Qdrant
- semantic retrieval
- prompt construction
- orchestration of generation through Inference Lab (M0)

## What this release is not

- It is not a standalone LLM runtime.
- It does not host model serving locally.
- It does not replace Inference Lab.

## Key implementation notes

- Generation is delegated to M0 via `POST /v1/generate`.
- Duplicate PDF uploads are handled through fingerprint-based detection.
- Retrieval supports top-k, similarity thresholds, and metadata filtering.
- The API exposes `/v1/upload`, `/v1/chat`, and `/v1/health`.

## Compatibility

- Python 3.11+
- Qdrant
- Inference Lab (M0) reachable at the configured generation endpoint

## Notes for operators

- If chat requests return `502`, verify the M0 service and its configured model.
- If uploads return `409`, the uploaded document fingerprint already exists and duplicate detection is active.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.core.vectorstore import VectorStore

DEFAULT_DOC_ID = "5b038ed7-a74c-4b00-9f61-059a11d55b23"
DEFAULT_OUTPUT = Path("eval/data/the-10x-rule-chunks.json")


def export_chunks(doc_id: str, output: Path) -> dict[str, object]:
    settings = get_settings()
    vectorstore = VectorStore(settings)
    chunks = vectorstore.list_chunks_for_doc_id(doc_id)

    filename = chunks[0]["filename"] if chunks else None
    report = {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all indexed chunks for a doc_id.")
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = export_chunks(args.doc_id, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

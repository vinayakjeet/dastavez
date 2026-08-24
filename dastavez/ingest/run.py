"""Ingest documents into chunks under one named configuration.

    uv run python -m dastavez.ingest.run --document pmkisan-og-revised-en
    uv run python -m dastavez.ingest.run --converter docling-routed --splitter section
    uv run python -m dastavez.ingest.run --list-matrix

Every run records the hash of its own configuration, so two runs are comparable
without anyone remembering what was set when.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app import spans
from app.spans import stage_span
from dastavez.ingest.chunking import SPLITTERS
from dastavez.ingest.cleaning import CLEANERS
from dastavez.ingest.converters import (
    DoclingConverter,
    MarkerConverter,
    MinerUConverter,
    PyPdfConverter,
    RoutedDoclingConverter,
    convert_cached,
)
from dastavez.pipeline import CLEANINGS, CONVERTERS, METADATA, PipelineConfig, matrix
from dastavez.store import SqliteChunkStore

DOCUMENTS = Path("corpus/documents")
MANIFEST = Path("corpus/manifest.jsonl")

BUILDERS = {
    "pypdf": PyPdfConverter,
    "docling": lambda: DoclingConverter(ocr=True),
    "docling-noocr": lambda: DoclingConverter(ocr=False),
    "docling-routed": RoutedDoclingConverter,
    "marker": MarkerConverter,
    "mineru": MinerUConverter,
}


def manifest_rows() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    rows = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("sha256"):
            rows[row["id"]] = row
    return rows


def ingest_document(
    document_id: str, row: dict, config: PipelineConfig, store: SqliteChunkStore, *, refresh: bool
) -> tuple[int, str]:
    path = DOCUMENTS / f"{document_id}.pdf"
    if not path.exists():
        return 0, "missing file"

    with stage_span(
        spans.INGEST,
        **{
            "dastavez.document": document_id,
            "dastavez.scheme": row["scheme"],
            "dastavez.config_hash": config.digest,
        },
    ):
        converter = BUILDERS[config.converter]()
        with stage_span(spans.INGEST_CONVERT, **{"dastavez.converter": config.converter}) as sp:
            blocks = convert_cached(path, converter, refresh=refresh)
            sp.record(**{"dastavez.blocks": len(blocks)})

        blocks = CLEANERS[config.cleaning](blocks)

        with stage_span(spans.INGEST_CHUNK, **{"dastavez.splitter": config.splitter}) as sp:
            chunks = list(SPLITTERS[config.splitter](document_id, blocks))
            if config.metadata == "enriched":
                chunks = [
                    replace(c, scheme=row["scheme"], doc_kind=row["kind"]) for c in chunks
                ]
            sp.record(
                **{
                    "dastavez.chunks": len(chunks),
                    "dastavez.chunk_chars_mean": (
                        sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
                    ),
                }
            )

        if not chunks:
            store.replace_document(config.run_id, document_id)
            # An empty result is reported rather than skipped. For the floor converter
            # against a scanned document this is the correct and expected outcome, and
            # it is the population the OCR-cascade study is measured over. Silently
            # producing nothing is how that population becomes invisible.
            return 0, "no extractable text"

        store.replace_document(config.run_id, document_id)
        store.write(config.run_id, chunks)
        pages = f"p{min(c.page for c in chunks)}-{max(c.end_page for c in chunks)}"
        return len(chunks), pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", default="pypdf", choices=sorted(CONVERTERS))
    parser.add_argument("--splitter", default="fixed", choices=sorted(SPLITTERS))
    parser.add_argument("--cleaning", default="none", choices=sorted(CLEANINGS))
    parser.add_argument("--metadata", default="none", choices=sorted(METADATA))
    parser.add_argument("--document", help="one document id, default is all of them")
    parser.add_argument("--refresh", action="store_true", help="ignore the conversion cache")
    parser.add_argument("--list-matrix", action="store_true", help="print every configuration")
    args = parser.parse_args()

    if args.list_matrix:
        for config in matrix():
            print(f"{config.digest}  {config.run_id}")
        print(f"\n{len(matrix())} configurations, {len(CONVERTERS)} conversion passes")
        return 0

    rows = manifest_rows()
    if not rows:
        print("no corpus manifest. Run corpus/fetch.py first.")
        return 1

    wanted = [args.document] if args.document else sorted(rows)
    if missing := [d for d in wanted if d not in rows]:
        print(f"not in the manifest: {', '.join(missing)}")
        return 1

    config = PipelineConfig(
        converter=args.converter,
        cleaning=args.cleaning,
        splitter=args.splitter,
        metadata=args.metadata,
    )

    store = SqliteChunkStore()
    store.start_run(
        config.run_id, started=datetime.now(UTC).isoformat(), **config.as_row()
    )

    total = 0
    empty: list[str] = []
    for document_id in wanted:
        written, note = ingest_document(
            document_id, rows[document_id], config, store, refresh=args.refresh
        )
        total += written
        if written:
            print(f"  {written:>5} chunks  {document_id}  {note}")
        else:
            empty.append(document_id)
            print(f"  {'0 chunks':>12}  {document_id}  ({note})")

    print()
    print(f"run {config.run_id}")
    print(f"{total} chunks from {len(wanted) - len(empty)} of {len(wanted)} documents")
    if empty:
        print(f"{len(empty)} produced nothing: {', '.join(empty)}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

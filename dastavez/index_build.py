"""Embed the chunks of one ingest run.

    uv run python -m dastavez.index_build --run <run_id>
"""

from __future__ import annotations

import argparse
import sys
import time

from dastavez.retrieval import Embedder, VectorIndex
from dastavez.store import SqliteChunkStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    store = SqliteChunkStore()
    chunks = list(store.read(args.run))
    store.close()

    if not chunks:
        print(f"no chunks for run {args.run}")
        return 1

    embedder = Embedder(args.model) if args.model else Embedder()
    index = VectorIndex()
    started = time.monotonic()
    written = index.build(args.run, chunks, embedder)
    index.close()

    elapsed = time.monotonic() - started
    print(f"embedded {written} chunks in {elapsed:.1f}s with {embedder.model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

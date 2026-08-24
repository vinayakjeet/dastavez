"""Convert one document under a heavy converter and cache the blocks.

Marker and MinerU cannot share an environment with Docling (see
`dastavez.ingest.converters`), so their conversions run from whichever environment
holds that converter's dependency and land in the shared block cache everything
else reads. From the repo root, with that converter's environment on the path:

    <env>/bin/python -m dastavez.offline_convert --converter marker \\
        --document pmkisan-og-revised-en

The cache key is content hash plus converter identity, so a re-run is free and a
changed document or converter version is a new entry rather than a stale one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dastavez.chunks import Block
from dastavez.ingest.converters import (
    Converter,
    MarkerConverter,
    MinerUConverter,
    convert_cached,
)

DOCUMENTS = Path("corpus/documents")
HEAVY: dict[str, type[Converter]] = {
    "marker": MarkerConverter,
    "mineru": MinerUConverter,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, choices=sorted(HEAVY))
    parser.add_argument("--document", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    path = DOCUMENTS / f"{args.document}.pdf"
    if not path.exists():
        print(f"no such document: {path}")
        return 1

    converter = HEAVY[args.converter]()
    blocks: list[Block] = convert_cached(path, converter, refresh=args.refresh)
    pages = sorted({b.page for b in blocks})
    print(f"{len(blocks)} blocks over pages {pages[0]}-{pages[-1]} ({converter.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

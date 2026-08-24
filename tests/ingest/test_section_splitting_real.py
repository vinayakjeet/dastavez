"""Section splitting verified against a real converted document, not just fakes.

`tests/ingest/test_chunk_provenance.py` proves the splitter protects a table given
a labelled block stream. This module proves the stream itself: that Docling really
emits TableItem blocks for a corpus document, so there is something for the
splitter to protect. It runs against the local conversion cache and is skipped
where the corpus has not been fetched, which CI is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dastavez.chunks import Block
from dastavez.ingest.chunking import section_respecting

CACHE = Path("corpus/.converted")
DOCUMENT = "pmayu-ahp-sop"


def cached_blocks(document: str, converter: str) -> list[Block] | None:
    matches = sorted(CACHE.glob(f"{document}.{converter}-*.json"))
    if not matches:
        return None
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    return [Block(**row) for row in payload["blocks"]]


@pytest.fixture(scope="module")
def docling_blocks() -> list[Block]:
    converted = cached_blocks(DOCUMENT, "docling")
    reason = (
        f"no {DOCUMENT} conversion cached; run "
        f"`uv run python -m dastavez.ingest.run --converter docling-noocr --document {DOCUMENT}`"
    )
    if converted is None:
        pytest.skip(reason)
    return converted


def test_the_converter_actually_labels_tables(docling_blocks):
    """Without this, the no-partial-table claim below would be vacuous."""
    assert any(b.kind == "TableItem" for b in docling_blocks)


def test_no_chunk_carries_a_partial_table(docling_blocks):
    """Every chunk holding table text holds all of it, or none.

    A table emitted whole produces chunks whose text is exactly one table's text;
    anything else means a fragment slipped through, with the header row in one
    chunk and the numbers in another.
    """
    whole = {b.text for b in docling_blocks if b.kind == "TableItem"}
    chunks = list(section_respecting(DOCUMENT, docling_blocks))

    holding = [c for c in chunks if "TableItem" in c.kinds]
    assert holding
    assert all(c.text in whole for c in holding), "a chunk carries a table fragment"


def test_tables_do_not_merge_with_surrounding_prose(docling_blocks):
    chunks = list(section_respecting(DOCUMENT, docling_blocks))

    for chunk in chunks:
        if "TableItem" in chunk.kinds:
            assert set(chunk.kinds) == {"TableItem"}

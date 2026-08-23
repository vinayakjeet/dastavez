from __future__ import annotations

import pytest

from dastavez.chunks import Block, Chunk
from dastavez.ingest.chunking import fixed_size, section_respecting
from dastavez.store import SqliteChunkStore


def blocks(*spec: tuple[str, int, str | None]) -> list[Block]:
    return [Block(text=t, page=p, kind=k) for t, p, k in spec]


def test_a_chunk_cannot_exist_without_a_page():
    """The invariant this project rests on.

    Every claim has to resolve to a page of a named document. A chunk that lost its
    page is useful for retrieval and useless for citation, and the failure surfaces
    as a citation pointing at the wrong place rather than as an error.
    """
    with pytest.raises(TypeError):
        Chunk(document_id="d", ordinal=0, text="hello")  # type: ignore[call-arg]

    with pytest.raises(ValueError):
        Chunk(document_id="d", ordinal=0, text="hello", page=0, end_page=0)


def test_end_page_cannot_precede_page():
    with pytest.raises(ValueError):
        Chunk(document_id="d", ordinal=0, text="hi", page=5, end_page=4)


def test_empty_chunks_are_refused():
    """An empty chunk indexes nothing and cites a page it did not read."""
    with pytest.raises(ValueError):
        Chunk(document_id="d", ordinal=0, text="   ", page=1, end_page=1)


@pytest.mark.parametrize("split", [fixed_size, section_respecting], ids=["fixed", "section"])
def test_no_block_loses_its_page_through_either_splitter(split):
    source = blocks(
        ("alpha", 1, "TextItem"),
        ("beta", 2, "TextItem"),
        ("gamma", 3, "TextItem"),
    )
    chunks = list(split("doc", source))

    assert chunks
    assert all(c.page >= 1 for c in chunks)
    assert min(c.page for c in chunks) == 1
    assert max(c.end_page for c in chunks) == 3


def test_a_chunk_spanning_pages_reports_a_range_not_a_page():
    """Citing "page 7" for text that began on page 6 is wrong in a way a reader sees."""
    source = blocks(("a", 6, "TextItem"), ("b", 7, "TextItem"))
    chunk = next(iter(fixed_size("doc", source)))

    assert chunk.page == 6
    assert chunk.end_page == 7
    assert chunk.spans_pages


def test_section_respecting_never_merges_across_a_heading():
    source = blocks(
        ("Eligibility", 1, "SectionHeaderItem"),
        ("Farmers holding land.", 1, "TextItem"),
        ("Exclusions", 2, "SectionHeaderItem"),
        ("Institutional landholders are excluded.", 2, "TextItem"),
    )
    chunks = list(section_respecting("doc", source))

    assert len(chunks) == 2
    assert "Exclusions" not in chunks[0].text
    assert "Eligibility" not in chunks[1].text


def test_section_respecting_never_splits_a_table():
    """A table cut in half puts the header row in one chunk and the numbers in
    another, and neither answers a question about either."""
    wide = "col a | col b\n" + "\n".join(f"row {n} | {n}" for n in range(400))
    source = [
        Block(text="Rates", page=1, kind="SectionHeaderItem"),
        Block(text=wide, page=1, kind="TableItem"),
    ]
    chunks = list(section_respecting("doc", source))

    holding = [c for c in chunks if "col a" in c.text]
    assert len(holding) == 1
    assert holding[0].text.count("row 399") == 1


def test_a_converter_that_labels_nothing_makes_both_splitters_identical():
    """Not a bug, and it is the interaction the ablation exists to show.

    The floor converter emits no block kinds, so the section-respecting splitter has
    nothing to respect. Converter and splitter are therefore not independent axes, and
    a results table reporting them as independent would mislead.
    """
    source = blocks(("one", 1, None), ("two", 1, None), ("three", 2, None))

    assert [c.text for c in fixed_size("doc", source)] == [
        c.text for c in section_respecting("doc", source)
    ]


def test_the_store_refuses_a_chunk_without_a_page(tmp_path):
    """The schema enforces it too, because a database that will accept a chunk
    without a page is a database that will eventually contain one."""
    store = SqliteChunkStore(tmp_path / "chunks.sqlite")
    store.start_run(
        "run",
        config_hash="abc",
        converter="pypdf",
        splitter="fixed",
        cleaning="none",
        metadata="none",
        started="2026-08-22T00:00:00Z",
    )

    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run", "doc", 0, "text", None, None, "[]", None, None, None),
        )
    store.close()


def test_chunks_survive_a_round_trip_through_the_store(tmp_path):
    store = SqliteChunkStore(tmp_path / "chunks.sqlite")
    store.start_run(
        "run",
        config_hash="abc",
        converter="pypdf",
        splitter="fixed",
        cleaning="none",
        metadata="none",
        started="2026-08-22T00:00:00Z",
    )
    original = list(fixed_size("doc", blocks(("alpha", 3, "TextItem"), ("beta", 4, "TextItem"))))
    assert store.write("run", original) == len(original)

    recovered = list(store.read("run", "doc"))
    assert [(c.page, c.end_page, c.text) for c in recovered] == [
        (c.page, c.end_page, c.text) for c in original
    ]
    store.close()

"""Where chunks live.

Two backends behind one interface, and the split is architectural rather than a
workaround for a missing credential.

Ingestion and the M6 ablation are offline batch work: every configuration over the
whole corpus, several times. Running that against a free-tier Postgres would be slow,
would burn a quota that the served API needs, and would make a laptop-local
experiment depend on a network. So the ablation writes to SQLite on disk.

The served API is a different job with different needs: one deployment, concurrent
readers, and vector search. That is Postgres with pgvector.

The interface exists so the ablation and the service cannot drift apart. A chunk
written by one has to be readable by the other, and the schema is defined once.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from dastavez.chunks import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    converter   TEXT NOT NULL,
    splitter    TEXT NOT NULL,
    cleaning    TEXT NOT NULL,
    metadata    TEXT NOT NULL,
    started     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    run_id       TEXT NOT NULL,
    document_id  TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page         INTEGER NOT NULL,
    end_page     INTEGER NOT NULL,
    kinds        TEXT NOT NULL,
    section_path TEXT,
    scheme       TEXT,
    doc_kind     TEXT,
    PRIMARY KEY (run_id, document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks (run_id, document_id, page);
"""


class ChunkStore(Protocol):
    def start_run(self, run_id: str, **config: str) -> None: ...
    def replace_document(self, run_id: str, document_id: str) -> int:
        """Drop this document's chunks for this run before writing new ones.

        Needed because writing is INSERT OR REPLACE keyed on ordinal, so a re-ingest
        that produces fewer chunks than the previous one leaves the tail behind. That
        happened: stripping unmapped glyphs took one configuration from 586 chunks to
        584, and 21 null bytes survived in two orphaned rows that nothing had
        overwritten.

        A configuration whose chunk set silently mixes two versions of itself is the
        precise failure an ablation cannot survive, because every number downstream is
        attributed to a configuration that never existed.
        """
        cursor = self._connection.execute(
            "DELETE FROM chunks WHERE run_id = ? AND document_id = ?", (run_id, document_id)
        )
        self._connection.commit()
        return cursor.rowcount

    def write(self, run_id: str, chunks: Iterable[Chunk]) -> int: ...
    def read(self, run_id: str, document_id: str | None = None) -> Iterator[Chunk]: ...


class SqliteChunkStore:
    """The ablation's store. One file, no server, survives a killed run.

    `page` and `end_page` are NOT NULL in the schema on purpose. Page provenance is
    the invariant this whole project rests on, and a database that will accept a
    chunk without a page is a database that will eventually contain one.
    """

    def __init__(self, path: Path = Path("corpus/chunks.sqlite")) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def start_run(self, run_id: str, **config: str) -> None:
        columns = ("config_hash", "converter", "splitter", "cleaning", "metadata", "started")
        missing = [c for c in columns if c not in config]
        if missing:
            # Refused rather than defaulted. A run row with a blank axis is a results
            # row nobody can attribute, and the ablation is entirely attribution.
            raise ValueError(f"run {run_id} is missing {missing}")
        self._connection.execute(
            f"INSERT OR REPLACE INTO runs (run_id, {', '.join(columns)}) "
            f"VALUES (?, {', '.join('?' * len(columns))})",
            (run_id, *(config[c] for c in columns)),
        )
        self._connection.commit()

    def replace_document(self, run_id: str, document_id: str) -> int:
        """Drop this document's chunks for this run before writing new ones.

        Needed because writing is INSERT OR REPLACE keyed on ordinal, so a re-ingest
        that produces fewer chunks than the previous one leaves the tail behind. That
        happened: stripping unmapped glyphs took one configuration from 586 chunks to
        584, and 21 null bytes survived in two orphaned rows that nothing had
        overwritten.

        A configuration whose chunk set silently mixes two versions of itself is the
        precise failure an ablation cannot survive, because every number downstream is
        attributed to a configuration that never existed.
        """
        cursor = self._connection.execute(
            "DELETE FROM chunks WHERE run_id = ? AND document_id = ?", (run_id, document_id)
        )
        self._connection.commit()
        return cursor.rowcount

    def write(self, run_id: str, chunks: Iterable[Chunk]) -> int:
        rows = [
            (
                run_id,
                c.document_id,
                c.ordinal,
                c.text,
                c.page,
                c.end_page,
                json.dumps(list(c.kinds)),
                c.section_path,
                c.scheme,
                c.doc_kind,
            )
            for c in chunks
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()
        return len(rows)

    def read(self, run_id: str, document_id: str | None = None) -> Iterator[Chunk]:
        sql = (
            "SELECT document_id, ordinal, text, page, end_page, kinds, "
            "section_path, scheme, doc_kind FROM chunks WHERE run_id = ?"
        )
        params: list[object] = [run_id]
        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)
        sql += " ORDER BY document_id, ordinal"

        for row in self._connection.execute(sql, params):
            yield Chunk(
                document_id=row[0],
                ordinal=row[1],
                text=row[2],
                page=row[3],
                end_page=row[4],
                kinds=tuple(json.loads(row[5])),
                section_path=row[6],
                scheme=row[7],
                doc_kind=row[8],
            )

    def close(self) -> None:
        self._connection.close()

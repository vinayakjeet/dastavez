"""Dense retrieval over chunks, with the page provenance carried through.

Two decisions here are load-bearing and both would be easy to get wrong quietly.

**The embedding model is multilingual, and that is not a preference.** This corpus
holds Hindi and English, and the eval set tags questions `hi`, `en` and `hinglish`
with results reported per language. An English-only model such as `all-MiniLM-L6-v2`
would embed Devanagari into noise, and the failure would not look like a failure: the
English slice would score well, the aggregate would look acceptable, and the Hindi
slice would be quietly worthless. Reporting per language is what makes that visible,
and picking a multilingual model is what makes it survivable.

**Search is exact, not approximate.** The corpus is 353 pages, a few hundred chunks.
An ANN index is machinery for millions of vectors and it trades recall for speed that
is not needed here. At this size a brute-force cosine scan is both faster than
building an index and lossless, so an approximate-search recall number would be a
number about a decision nobody had to make. pgvector's IVFFlat or HNSW belong in the
served deployment if the corpus grows, not in the ablation.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app import spans
from app.spans import stage_span
from dastavez.chunks import Chunk

# e5 models are trained with asymmetric prefixes and lose real accuracy without them:
# a passage embedded as a query sits in a different part of the space. This is the
# single easiest way to silently degrade an e5 retriever, so the prefixes live here
# rather than at call sites where one of them will eventually be forgotten.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

EMBEDDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    run_id      TEXT NOT NULL,
    document_id TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    model       TEXT NOT NULL,
    vector      BLOB NOT NULL,
    PRIMARY KEY (run_id, document_id, ordinal, model)
);
"""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def citation(self) -> str:
        """How a page anchor is rendered anywhere a reader sees it."""
        pages = (
            f"pages {self.chunk.page} to {self.chunk.end_page}"
            if self.chunk.spans_pages
            else f"page {self.chunk.page}"
        )
        return f"{self.chunk.document_id}, {pages}"


class Embedder:
    """Local sentence-transformers, loaded once.

    Local rather than an embedding API for the same reason the ingestion stack is
    local: the ablation embeds the whole corpus once per configuration, and putting a
    network round trip and a second quota inside that loop would make the study cost
    money and time it does not need to.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model_name = model
        self._model = None

    def warm(self) -> float:
        """Load the model now, and return how long it took.

        Call this at startup. Without it the first query pays the load inside the
        retrieval span, which measured 51 seconds against a warm retrieval median of
        127 milliseconds: a four-hundred-fold difference reported as retrieval
        latency. A sibling project published a session latency of 1184ms against a
        provider's 559ms for the same class of reason, a span wrapping something that
        was not the work.

        The load is a startup cost and belongs in its own number, the way a
        connection setup does, not folded into the thing a caller waits for on every
        question.
        """
        import time

        started = time.monotonic()
        self._load()
        return time.monotonic() - started

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def passages(self, texts: Sequence[str]):
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def query(self, text: str):
        return self._encode([QUERY_PREFIX + text])[0]

    def _encode(self, texts: Sequence[str]):
        return self._load().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )


class VectorIndex:
    """Embeddings beside the chunks they came from, in the same SQLite file.

    Same file on purpose: a chunk and its vector that can drift apart will, and a
    vector pointing at a chunk that no longer exists produces a citation to a page
    nobody can open.
    """

    def __init__(self, path: Path = Path("corpus/chunks.sqlite")) -> None:
        self.path = path
        self._connection = sqlite3.connect(path)
        self._connection.executescript(EMBEDDING_SCHEMA)
        self._connection.commit()

    def build(self, run_id: str, chunks: Iterable[Chunk], embedder: Embedder) -> int:
        import numpy as np

        rows = list(chunks)
        if not rows:
            return 0

        vectors = embedder.passages([c.text for c in rows])
        self._connection.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    c.document_id,
                    c.ordinal,
                    embedder.model_name,
                    np.asarray(v, dtype="float32").tobytes(),
                )
                for c, v in zip(rows, vectors, strict=True)
            ],
        )
        self._connection.commit()
        return len(rows)

    def search(self, run_id: str, question: str, embedder: Embedder, k: int = 5) -> list[Hit]:
        with stage_span(spans.RETRIEVE, **{"dastavez.k": k}) as retrieve:
            hits = self._dense(run_id, question, embedder, k)
            retrieve.record(
                **{
                    "dastavez.candidates": len(hits),
                    "dastavez.top_score": hits[0].score if hits else 0.0,
                }
            )
            return hits

    def _dense(self, run_id: str, question: str, embedder: Embedder, k: int) -> list[Hit]:
        import numpy as np

        rows = list(
            self._connection.execute(
                "SELECT e.document_id, e.ordinal, e.vector, c.text, c.page, c.end_page, "
                "c.kinds, c.section_path, c.scheme, c.doc_kind "
                "FROM embeddings e JOIN chunks c "
                "ON e.run_id = c.run_id AND e.document_id = c.document_id "
                "AND e.ordinal = c.ordinal "
                "WHERE e.run_id = ? AND e.model = ?",
                (run_id, embedder.model_name),
            )
        )
        if not rows:
            return []

        with stage_span(spans.RETRIEVE_DENSE) as dense:
            matrix = np.vstack([np.frombuffer(r[2], dtype="float32") for r in rows])
            # The query encode is inside this span on purpose. It is unavoidable cost
            # of dense retrieval and a caller waits for it, so excluding it would
            # report a latency nobody experiences. See bench/stages.md.
            scores = matrix @ np.asarray(embedder.query(question), dtype="float32")
            best = np.argsort(-scores)[:k]
            dense.record(
                **{
                    "dastavez.candidates": len(rows),
                    "dastavez.top_score": float(scores[best[0]]) if len(best) else 0.0,
                }
            )

        return [
            Hit(
                chunk=Chunk(
                    document_id=rows[i][0],
                    ordinal=rows[i][1],
                    text=rows[i][3],
                    page=rows[i][4],
                    end_page=rows[i][5],
                    kinds=tuple(json.loads(rows[i][6])),
                    section_path=rows[i][7],
                    scheme=rows[i][8],
                    doc_kind=rows[i][9],
                ),
                score=float(scores[i]),
            )
            for i in best
        ]

    def close(self) -> None:
        self._connection.close()

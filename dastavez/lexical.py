"""BM25 over the same chunks the dense index holds.

This exists because of a measured failure, not because hybrid retrieval is
fashionable. Dense retrieval on this corpus returned scores clustered between 0.82
and 0.86 for three eligibility questions, and "who is excluded from receiving
PM-Kisan benefits" did not return the exclusion list on any attempt, before or after
the chunking fix.

The reason is visible once stated: the exclusion criteria are a numbered list of
categories (institutional landholders, serving and retired officials, income tax
payers) and the question shares almost no vocabulary with them beyond the word
"excluded" itself. An embedding matches meaning, and the meaning of a list of job
titles is not close to the meaning of the question that asks which job titles are
listed.

A query naming a scheme code, a circular number, a section, or a specific exclusion
is a lexical problem. That is the half BM25 answers.

Tokenisation is deliberately simple and language-aware only where it must be.
Devanagari has no case to fold and its word boundaries are spaces, so the same
tokeniser serves both scripts; what it must not do is split on the characters that
carry meaning in this corpus, since PM-KISAN and 6000/- are exactly the tokens a
lexical retriever exists to match.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from app import spans
from app.spans import stage_span
from dastavez.chunks import Chunk
from dastavez.retrieval import Hit

# Keep digits, letters from any script, and the internal hyphens and slashes that
# scheme codes and rupee amounts are made of. Splitting PM-KISAN into two tokens is
# how a lexical retriever stops being able to find the thing it is best at finding.
TOKEN = re.compile(r"[0-9]+(?:[/\-][0-9]+)*|[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)


def tokenise(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


class LexicalIndex:
    """BM25 built on demand from the chunk store.

    Built in memory rather than persisted. The corpus is a few hundred chunks, the
    build takes milliseconds, and a persisted lexical index is one more thing that can
    silently disagree with the chunks it claims to describe. The dense index is
    persisted because embedding is expensive; this is not.
    """

    def __init__(self, path: Path = Path("corpus/chunks.sqlite")) -> None:
        self._connection = sqlite3.connect(path)
        self._built_for: str | None = None
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def _build(self, run_id: str) -> None:
        if self._built_for == run_id:
            return

        import json as _json

        from rank_bm25 import BM25Okapi

        rows = self._connection.execute(
            "SELECT document_id, ordinal, text, page, end_page, kinds, section_path, "
            "scheme, doc_kind FROM chunks WHERE run_id = ? ORDER BY document_id, ordinal",
            (run_id,),
        ).fetchall()

        self._chunks = [
            Chunk(
                document_id=r[0],
                ordinal=r[1],
                text=r[2],
                page=r[3],
                end_page=r[4],
                kinds=tuple(_json.loads(r[5])),
                section_path=r[6],
                scheme=r[7],
                doc_kind=r[8],
            )
            for r in rows
        ]
        self._bm25 = BM25Okapi([tokenise(c.text) for c in self._chunks]) if self._chunks else None
        self._built_for = run_id

    def search(self, run_id: str, question: str, k: int = 5) -> list[Hit]:
        with stage_span(spans.RETRIEVE_LEXICAL) as span:
            self._build(run_id)
            if not self._bm25:
                span.record(**{"dastavez.candidates": 0, "dastavez.top_score": 0.0})
                return []

            scores = self._bm25.get_scores(tokenise(question))
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            span.record(
                **{
                    "dastavez.candidates": len(self._chunks),
                    "dastavez.top_score": float(scores[ranked[0]]) if ranked else 0.0,
                }
            )
            return [Hit(chunk=self._chunks[i], score=float(scores[i])) for i in ranked]

    def close(self) -> None:
        self._connection.close()


def reciprocal_rank_fusion(
    runs: Iterable[list[Hit]], k: int = 60, limit: int = 5
) -> list[Hit]:
    """Combine ranked lists by rank rather than by score.

    Score fusion is the obvious approach and it is wrong here. BM25 returns unbounded
    positive scores whose scale depends on corpus statistics; cosine similarity from
    an e5 model returns a compressed band, measured on this corpus between 0.80 and
    0.87. Adding or weighting those directly means whichever retriever happens to
    produce larger numbers wins every tie, and normalising them requires knowing each
    distribution, which changes with the corpus.

    Rank fusion needs neither. A chunk ranked first by either retriever scores the
    same regardless of what the underlying numbers looked like.

    `k` is 60, which is the value the method was published with, and it is **not**
    tuned. Tuning it requires questions with known correct answers, which is the eval
    set, which is not frozen yet. This is recorded rather than quietly defaulted: the
    number in the code today is a citation, not a measurement, and M2.2 replaces it
    with a swept value.
    """
    scores: dict[tuple[str, int], float] = {}
    seen: dict[tuple[str, int], Chunk] = {}

    for run in runs:
        for rank, hit in enumerate(run, start=1):
            key = (hit.chunk.document_id, hit.chunk.ordinal)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            seen[key] = hit.chunk

    best = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [Hit(chunk=seen[key], score=scores[key]) for key in best]

"""Dense and lexical retrieval, fused.

Not built because hybrid retrieval is the accepted answer. Built because on this
corpus the two halves disagree on the top result for five of six probe queries, and
on at least one of them the lexical half is simply right and the dense half is simply
wrong.

The worked case, which `bench/retriever_comparison.py` regenerates:

    "who is excluded from receiving PM-Kisan benefits?"

    dense    pmkisan-og-revised-en page 5, Nagaland land transfer procedure
    lexical  pmkisan-og-revised-en page 3, the exclusion list itself

Page 3 lists the excluded categories: employees of local bodies, retired pensioners
above a pension threshold, anyone who paid income tax last year, named professions.
Page 5 is about reassessing eligibility after a land transfer in Nagaland, and it
happens to contain the sentence "All the exclusions under the Operational Guidelines
will be applicable".

That sentence is why dense loses. It is a passage *about* exclusions, and an
embedding cannot tell the difference between a document that mentions a concept and
a document that enumerates it. BM25 matched the word "Excluding" where it actually
appears in the list, twice.

The general shape: a question naming a scheme code, a circular number, a rupee
amount or a specific excluded category is lexical. A paraphrase, or a question asked
in a different language from the document, is dense. Neither is a general-purpose
retriever on this corpus, which is the argument for running both.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import spans
from app.spans import stage_span
from dastavez.lexical import LexicalIndex, reciprocal_rank_fusion
from dastavez.retrieval import Embedder, Hit, VectorIndex


@dataclass
class Comparison:
    """One query, both halves, and whether they agreed.

    Kept as a first-class result rather than logged, because "how often do the two
    disagree" is a number this project reports, and reconstructing it from logs later
    is how a measurement becomes an anecdote.
    """

    question: str
    dense: list[Hit]
    lexical: list[Hit]
    fused: list[Hit]

    @property
    def agreed(self) -> bool:
        if not self.dense or not self.lexical:
            return False
        return self._key(self.dense[0]) == self._key(self.lexical[0])

    @staticmethod
    def _key(hit: Hit) -> tuple[str, int]:
        return hit.chunk.document_id, hit.chunk.ordinal


class HybridRetriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.dense = VectorIndex()
        self.lexical = LexicalIndex()

    def warm(self) -> float:
        return self.embedder.warm()

    def search(self, run_id: str, question: str, k: int = 5) -> list[Hit]:
        return self.compare(run_id, question, k).fused

    def compare(self, run_id: str, question: str, k: int = 5) -> Comparison:
        """Both halves and the fusion, so a caller can see where they disagreed."""
        with stage_span(spans.RETRIEVE, **{"dastavez.k": k}) as retrieve:
            dense = self.dense.search(run_id, question, self.embedder, k=k)
            lexical = self.lexical.search(run_id, question, k=k)

            with stage_span(spans.RETRIEVE_FUSE) as fuse:
                fused = reciprocal_rank_fusion([dense, lexical], limit=k)
                overlap = len(
                    {(h.chunk.document_id, h.chunk.ordinal) for h in dense}
                    & {(h.chunk.document_id, h.chunk.ordinal) for h in lexical}
                )
                fuse.record(
                    **{"dastavez.candidates": len(fused), "dastavez.overlap": overlap}
                )

            retrieve.record(
                **{
                    "dastavez.candidates": len(fused),
                    "dastavez.top_score": fused[0].score if fused else 0.0,
                }
            )

        return Comparison(question=question, dense=dense, lexical=lexical, fused=fused)

    def close(self) -> None:
        self.dense.close()
        self.lexical.close()

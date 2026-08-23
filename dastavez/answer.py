"""Turn retrieved chunks into a cited answer, or a refusal.

Every model call goes through Tollgate. Dastavez holds no provider credential at all,
which is a property worth keeping rather than an accident of build order: this service
ingests documents from the public internet and puts their text into a prompt, so it is
the last place in the portfolio that should also hold a key.

The refusal path is not an error path. A question the corpus cannot answer has a
correct answer, and that answer is "the corpus does not say", with what was searched.
The eval set carries at least twenty unanswerable questions precisely so refusal
correctness is measurable, and a system that never refuses scores perfectly on a set
that contains none.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app import spans
from app.spans import stage_span
from dastavez.retrieval import Hit

# Deliberately plain. A prompt that coaxes the model into confidence is a prompt that
# produces confident wrong answers, and this project measures exactly that gap.
SYSTEM = (
    "You answer questions about Indian government scheme documents using only the "
    "extracts provided. Every factual claim must cite the document and page it came "
    "from, written as (document-id, page N). If the extracts do not contain the "
    "answer, say so plainly and name what you did look at. Do not use knowledge from "
    "outside the extracts, and do not guess at a number that is not written down."
)


@dataclass(frozen=True)
class Answer:
    text: str
    hits: tuple[Hit, ...]
    refused: bool
    provider: str | None = None

    @property
    def citations(self) -> list[str]:
        return [h.citation() for h in self.hits]


def _extracts(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.document_id}, page {h.chunk.page}]\n{h.chunk.text}" for h in hits
    )


def answer(question: str, hits: list[Hit], *, floor: float = 0.80) -> Answer:
    """Answer from these extracts, or refuse.

    `floor` is a placeholder and is marked as one. A real threshold comes from M3.3,
    derived from the measured score distribution over answerable and unanswerable
    questions, with the false-refusal and false-answer rates published at the chosen
    point and the two either side of it. Picking a number here and calling it tuned is
    the mistake a sibling project already paid for, gating at 2 points against a judge
    whose noise floor was 20.
    """
    with stage_span(spans.ANSWER) as span:
        return _answer(question, hits, floor, span)


def _answer(question: str, hits: list[Hit], floor: float, span) -> Answer:
    if not hits or hits[0].score < floor:
        # A refusal opens the same span as an answer. It is an outcome, not an error
        # path, and closing the tree early would make refusals invisible in a latency
        # distribution while flattering the mean, since refusals are the fast case.
        span.record(**{"dastavez.refused": True, "dastavez.citations": len(hits)})
        searched = ", ".join(sorted({h.chunk.document_id for h in hits})) or "nothing"
        return Answer(
            text=(
                "The documents searched do not contain enough to answer this. "
                f"Searched: {searched}."
            ),
            hits=tuple(hits),
            refused=True,
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get("TOLLGATE_URL", "http://127.0.0.1:8077/v1"),
        api_key="dastavez-holds-no-provider-key",
    )
    completion = client.chat.completions.create(
        model=os.environ.get("DASTAVEZ_MODEL", "groq/llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Extracts:\n\n{_extracts(hits)}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )
    text = completion.choices[0].message.content or ""
    span.record(
        **{
            "dastavez.refused": False,
            "dastavez.citations": len(hits),
            "dastavez.answer_chars": len(text),
            "dastavez.model": getattr(completion, "model", "") or "",
        }
    )
    return Answer(
        text=text,
        hits=tuple(hits),
        refused=False,
        provider=getattr(completion, "model", None),
    )

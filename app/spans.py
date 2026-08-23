"""Every stage span name and `dastavez.*` attribute, in one place.

`bench/stages.md` says what each of these spans starts and ends at, and it is hashed.
This module is the attribute half of that contract as code, and `stage_span` refuses
an attribute the table does not declare, so the contract is enforced where attributes
are written rather than only asserted afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import spanlight
import structlog
from opentelemetry.trace import Span, Status, StatusCode

logger = structlog.get_logger(__name__)

INGEST = "dastavez.ingest"
INGEST_CONVERT = "dastavez.ingest.convert"
INGEST_CHUNK = "dastavez.ingest.chunk"
INGEST_EMBED = "dastavez.ingest.embed"

QUESTION = "dastavez.question"
DECOMPOSE = "dastavez.decompose"
RETRIEVE = "dastavez.retrieve"
RETRIEVE_DENSE = "dastavez.retrieve.dense"
RETRIEVE_LEXICAL = "dastavez.retrieve.lexical"
RETRIEVE_FUSE = "dastavez.retrieve.fuse"
RERANK = "dastavez.rerank"
ANSWER = "dastavez.answer"

CONTRACT: dict[str, frozenset[str]] = {
    INGEST: frozenset({"dastavez.document", "dastavez.scheme", "dastavez.config_hash"}),
    INGEST_CONVERT: frozenset(
        {"dastavez.converter", "dastavez.blocks", "dastavez.cache_hit", "dastavez.pages"}
    ),
    INGEST_CHUNK: frozenset({"dastavez.splitter", "dastavez.chunks", "dastavez.chunk_chars_mean"}),
    INGEST_EMBED: frozenset({"dastavez.embedding_model", "dastavez.embedded"}),
    QUESTION: frozenset(
        {
            "dastavez.language",
            "dastavez.question_type",
            "dastavez.refused",
            "dastavez.run_id",
            "dastavez.citations",
        }
    ),
    DECOMPOSE: frozenset({"dastavez.sub_questions"}),
    RETRIEVE: frozenset({"dastavez.k", "dastavez.candidates", "dastavez.top_score"}),
    RETRIEVE_DENSE: frozenset({"dastavez.candidates", "dastavez.top_score"}),
    RETRIEVE_LEXICAL: frozenset({"dastavez.candidates", "dastavez.top_score"}),
    RETRIEVE_FUSE: frozenset({"dastavez.candidates", "dastavez.overlap"}),
    RERANK: frozenset({"dastavez.reranker", "dastavez.top_changed", "dastavez.top_score"}),
    ANSWER: frozenset(
        {"dastavez.refused", "dastavez.citations", "dastavez.answer_chars", "dastavez.model"}
    ),
}

# This service ingests documents from the public internet and puts their contents into
# a prompt, which makes it the largest untrusted-input surface in the portfolio. Span
# attributes are indexed by the backend and are the documented anti-pattern for
# content: size limits, plus text queryable by anyone with dashboard access.
#
# `dastavez.document` is an identifier and `dastavez.citations` is a count, which is
# why the rule is a substring check rather than a ban on strings.
# The markers name things that hold text, not things that count text. "chunk" was in
# this list first and failed against `dastavez.chunks`, which is a count of them: a
# rule that flags the safe attribute teaches everyone to delete the rule.
FORBIDDEN_SUBSTRINGS = (
    "text",
    "content",
    "prompt",
    "extract",
    "passage",
    "body",
    "snippet",
    "api_key",
    "secret",
)

SHARED_ATTRIBUTES = frozenset(
    {
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "spanlight.cost_usd",
        "error.type",
    }
)


class UndeclaredAttribute(KeyError):
    """An attribute this span is not allowed to carry, per the contract above."""


# Lenient in production, strict in the suite, and the asymmetry is inherited as a
# correction rather than a preference. In a sibling project, raising took a live turn
# down: an attribute was recorded where it was not declared, and the only call site no
# test could reach was the one that was wrong. `tests/conftest.py` turns strict on.
strict = False


class StageSpan:
    def __init__(self, span: Span, stage: str) -> None:
        self.span = span
        self._stage = stage

    def record(self, **attributes: object) -> None:
        allowed = _permitted(self._stage, set(attributes))
        for key, value in attributes.items():
            if key in allowed:
                self.span.set_attribute(key, value)

    def mark(self, event: str) -> None:
        self.span.add_event(event)


def _permitted(stage: str, offered: set[str]) -> set[str]:
    allowed = CONTRACT[stage] | SHARED_ATTRIBUTES
    unknown = offered - allowed
    if unknown:
        message = f"{stage} may not carry {sorted(unknown)}"
        if strict:
            raise UndeclaredAttribute(message)
        logger.error("spans.undeclared_attribute", stage=stage, attributes=sorted(unknown))
    return allowed


@contextmanager
def stage_span(stage: str, **attributes: object) -> Iterator[StageSpan]:
    """Open one of the spans named in `bench/stages.md`.

    Nesting comes from the tracer's current context, so a stage opened inside another
    is its child without either naming the other. That is what makes the two trees in
    `bench/stages.md` real rather than a diagram.
    """
    if stage not in CONTRACT:
        raise UndeclaredAttribute(f"unknown stage {stage!r}, not in bench/stages.md")

    tracer = spanlight.get_tracer()
    with tracer.start_as_current_span(stage) as span:
        handle = StageSpan(span, stage)
        if attributes:
            handle.record(**attributes)
        try:
            yield handle
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise

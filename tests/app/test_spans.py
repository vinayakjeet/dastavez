from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import spans
from app.spans import CONTRACT, FORBIDDEN_SUBSTRINGS, UndeclaredAttribute, stage_span


@pytest.fixture
def exporter(monkeypatch):
    """A real tracer, because the shape being asserted is the tracer's own.

    `spanlight.init` is a no-op without an endpoint and hands back spans that record
    nothing, so a no-op span has no parent to check.
    """
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(spans.spanlight, "get_tracer", lambda: provider.get_tracer("test"))
    return memory


def test_stages_nest_rather_than_arriving_as_unrelated_roots(exporter):
    with stage_span(spans.QUESTION):
        with stage_span(spans.RETRIEVE), stage_span(spans.RETRIEVE_DENSE):
            pass
        with stage_span(spans.ANSWER):
            pass

    finished = {s.name: s for s in exporter.get_finished_spans()}
    roots = [s for s in exporter.get_finished_spans() if s.parent is None]

    assert [s.name for s in roots] == [spans.QUESTION]
    assert finished[spans.RETRIEVE].parent.span_id == finished[spans.QUESTION].context.span_id
    assert finished[spans.RETRIEVE_DENSE].parent.span_id == finished[spans.RETRIEVE].context.span_id
    assert finished[spans.ANSWER].parent.span_id == finished[spans.QUESTION].context.span_id


def test_every_stage_has_a_boundary_written_down():
    """`bench/stages.md` is hashed, so a stage added here without a boundary there is
    a measurement whose extent nobody defined."""
    boundaries = Path("bench/stages.md").read_text(encoding="utf-8")
    assert [stage for stage in CONTRACT if f"`{stage}`" not in boundaries] == []


def test_stages_md_has_not_drifted_from_its_hash():
    """The hash is the mechanism that stops a boundary moving after a result is seen."""
    import hashlib

    recorded = Path("bench/stages.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(Path("bench/stages.md").read_bytes()).hexdigest()
    assert actual == recorded, "bench/stages.md changed; re-run every published number"


def test_the_contract_declares_no_attribute_that_could_carry_document_text():
    """This service puts text from the public internet into a prompt, which makes it
    the largest untrusted-input surface in the portfolio. The check is on the contract
    rather than on a call site, because the contract is what a hurried change edits."""
    offenders = [
        attribute
        for allowed in CONTRACT.values()
        for attribute in allowed
        if any(word in attribute for word in FORBIDDEN_SUBSTRINGS)
    ]
    assert offenders == []


def test_an_undeclared_attribute_fails_in_the_suite(exporter):
    with pytest.raises(UndeclaredAttribute), stage_span(spans.RETRIEVE) as span:
        span.record(**{"dastavez.invented": 1})


def test_an_unknown_stage_is_refused():
    with pytest.raises(UndeclaredAttribute), stage_span("dastavez.not.a.stage"):
        pass


def test_an_error_inside_a_stage_is_recorded_on_it(exporter):
    with pytest.raises(ValueError), stage_span(spans.INGEST_CONVERT):
        raise ValueError("converter exploded")

    span = {s.name: s for s in exporter.get_finished_spans()}[spans.INGEST_CONVERT]
    assert span.attributes["error.type"] == "ValueError"
    assert span.status.status_code is trace.StatusCode.ERROR

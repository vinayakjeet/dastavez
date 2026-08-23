from __future__ import annotations

import os

import pytest

# Pinned before anything imports the app. The sibling gateway learned this the hard
# way: loading a developer `.env` silently made a real provider the test default, and
# a suite whose result depends on an untracked file passes on one machine and fails in
# CI. Dastavez holds no provider key at all, so what matters here is that no test can
# reach a live Tollgate by accident.
os.environ["TOLLGATE_URL"] = "http://127.0.0.1:9/v1"

from app import spans  # noqa: E402  imported after the environment is pinned


@pytest.fixture(autouse=True)
def strict_span_contract(monkeypatch):
    """Strict in CI, lenient in production, and the asymmetry is deliberate.

    `app/spans.py` drops an undeclared attribute and logs it rather than raising,
    because instrumentation that throws turns a mistyped attribute into an outage.
    That leniency would also let a contract breach ship silently, so the suite turns
    strict on: a stage carrying an attribute the table does not declare fails here,
    where it is cheap, instead of producing an empty dashboard panel later, where an
    empty panel and a wrong metric name and an idle service all look identical.
    """
    monkeypatch.setattr(spans, "strict", True)

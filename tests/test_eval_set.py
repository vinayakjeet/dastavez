"""The eval set, and the guards that decide whether it counts as one.

Two kinds of test here. The first checks the real committed set, so a hand edit
that breaks a rule fails CI rather than surfacing as a strange judging result. The
second checks that each guard actually fires, because a validator nobody has seen
reject anything is indistinguishable from one that returns an empty list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dastavez.evalset import (
    MIN_PER_TYPE,
    MIN_UNANSWERABLE,
    QUESTION_TYPES,
    TARGET_SIZE,
    EvalSetError,
    Question,
    content_hash,
    coverage_errors,
    dump,
    load,
    structural_errors,
    verify,
)

REAL = Path("eval/questions.jsonl")
REAL_HASH = Path("eval/questions.sha256")


def q(**over) -> Question:
    base = {
        "id": "dz-001",
        "question": "How much?",
        "gold_answer": "Rs. 6000.",
        "gold_pages": (("pmkisan-faq", 1),),
        "difficulty_tag": "easy",
        "language_tag": "en",
        "question_type": "factual",
    }
    return Question(**{**base, **over})


def filler(n: int, kind: str, start: int) -> list[Question]:
    """Enough questions of one type to clear its floor, so a test about one rule
    is not drowned in complaints about the other five."""
    return [q(id=f"{kind}-{start + i}", question_type=kind) for i in range(n)]


def a_valid_set() -> list[Question]:
    out: list[Question] = []
    for i, kind in enumerate(QUESTION_TYPES):
        if kind == "unanswerable":
            out += [
                q(
                    id=f"u-{i}-{j}",
                    question_type=kind,
                    gold_pages=(),
                    note="searched the FAQs, not stated",
                )
                for j in range(MIN_UNANSWERABLE)
            ]
        else:
            out += filler(MIN_PER_TYPE, kind, i * 100)
    return out


class TestTheCommittedSet:
    def test_it_loads_and_its_hash_still_matches(self) -> None:
        """The artifact promise. A published number is measured against this hash,
        so a silent edit has to be loud somewhere, and this is where."""
        questions, digest = verify(REAL, REAL_HASH)
        assert len(questions) == TARGET_SIZE
        assert digest == REAL_HASH.read_text(encoding="utf-8").strip()

    def test_every_type_clears_the_floor_spec_names(self) -> None:
        counts: dict[str, int] = {}
        for question in load(REAL):
            counts[question.question_type] = counts.get(question.question_type, 0) + 1
        assert set(counts) == set(QUESTION_TYPES)
        for kind, n in counts.items():
            floor = MIN_UNANSWERABLE if kind == "unanswerable" else MIN_PER_TYPE
            assert n >= floor, f"{kind} has {n}, floor is {floor}"

    def test_every_language_slice_is_reportable(self) -> None:
        """SPEC reports language separately, which a slice of two cannot support.

        Ten is not a statistical threshold, it is a floor below which the slice
        should not be quoted as a number at all.
        """
        counts: dict[str, int] = {}
        for question in load(REAL):
            counts[question.language_tag] = counts.get(question.language_tag, 0) + 1
        assert set(counts) == {"en", "hi", "hinglish"}
        for lang, n in counts.items():
            assert n >= 10, f"{lang} slice has {n} questions, too thin to report"

    def test_every_answerable_question_cites_a_real_page(self) -> None:
        """A citation that resolves to nothing cannot be scored for citation
        accuracy, which is one of this project's own metrics."""
        known = {
            (row["id"], page)
            for row in (
                json.loads(line)
                for line in Path("corpus/manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            for page in range(1, row["pages"] + 1)
        }
        assert coverage_errors(load(REAL), known) == []

    def test_unanswerable_questions_say_why(self) -> None:
        """The note is the evidence the absence was checked. Without it the slice
        is an assertion that the corpus is silent, which is what a reviewer needs
        to be able to dispute."""
        for question in load(REAL):
            if question.question_type == "unanswerable":
                assert question.gold_pages == ()
                assert question.note.strip(), f"{question.id} has no note"

    def test_the_draft_provenance_is_recorded_on_every_row(self) -> None:
        """The set is model-drafted and unadjudicated, and the file has to say so
        per row rather than only in the datasheet, because rows get copied and
        datasheets get left behind."""
        for question in load(REAL):
            assert question.annotator_ids, f"{question.id} names no annotator"


class TestTheGuardsFire:
    def test_a_missing_citation_is_refused(self) -> None:
        broken = a_valid_set() + [q(id="no-cite", gold_pages=())]
        assert any("no gold page" in e for e in structural_errors(broken))

    def test_an_unanswerable_question_may_not_cite_a_page(self) -> None:
        """Citing a page means the corpus answers it, which contradicts the type."""
        broken = a_valid_set() + [
            q(id="bad-u", question_type="unanswerable", note="x", gold_pages=(("d", 1),))
        ]
        assert any("unanswerable but cites" in e for e in structural_errors(broken))

    def test_an_unanswerable_question_without_a_note_is_refused(self) -> None:
        broken = a_valid_set() + [q(id="no-note", question_type="unanswerable", gold_pages=())]
        assert any("needs a note" in e for e in structural_errors(broken))

    def test_a_thin_type_is_refused(self) -> None:
        thin = [x for x in a_valid_set() if x.question_type != "table_lookup"]
        thin += filler(3, "table_lookup", 900)
        assert any("type table_lookup: 3" in e for e in structural_errors(thin))

    def test_duplicate_ids_are_refused(self) -> None:
        dupes = a_valid_set() + [q(id="factual-0")]
        assert any("duplicate id" in e for e in structural_errors(dupes))

    def test_an_unknown_tag_is_refused(self) -> None:
        errors = structural_errors(a_valid_set() + [q(id="bad-lang", language_tag="ta")])
        assert any("unknown language_tag" in e for e in errors)

    def test_a_citation_outside_the_corpus_is_refused(self) -> None:
        errors = coverage_errors([q(gold_pages=(("not-a-doc", 3),))], {("pmkisan-faq", 1)})
        assert errors and "not in the corpus" in errors[0]


class TestTheHashMeansSomething:
    def test_reordering_the_file_does_not_change_the_hash(self) -> None:
        """The hash identifies the data, not the byte order of the file. Two people
        who wrote the same set in a different order have the same set."""
        questions = load(REAL)
        assert content_hash(questions) == content_hash(list(reversed(questions)))

    def test_changing_an_answer_changes_the_hash(self) -> None:
        questions = load(REAL)
        edited = [*questions[1:], Question(**{**questions[0].__dict__, "gold_answer": "different"})]
        assert content_hash(edited) != content_hash(questions)

    def test_adding_an_annotator_does_not_change_the_hash(self) -> None:
        """Deliberate. The M4.3 audit adds a second annotator to a subset, and that
        must not invalidate results measured before the audit ran."""
        questions = load(REAL)
        audited = [
            Question(**{**questions[0].__dict__, "annotator_ids": ("a", "b")}),
            *questions[1:],
        ]
        assert content_hash(audited) == content_hash(questions)

    def test_a_tampered_file_fails_verification(self, tmp_path: Path) -> None:
        questions = load(REAL)
        path = tmp_path / "q.jsonl"
        tampered = Question(**{**questions[0].__dict__, "gold_answer": "tampered"})
        dump([tampered, *questions[1:]], path)
        with pytest.raises(EvalSetError, match="hash mismatch"):
            verify(path, REAL_HASH)

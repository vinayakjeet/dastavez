"""Does the page a question cites actually contain the answer it claims?

The second half of M4.2's acceptance. The first half, that every answerable
question cites a page in the manifest, is enforced at build time. This is the
other half: opening that page shows the answer.

A human reading 150 pages is the gold standard and is not what this does. This
checks whether the distinctive tokens of the gold answer appear in the cited
page's text: the numbers, dates and amounts that carry the claim. A gold answer
saying "Rs. 6000 per annum" whose cited page contains neither 6000 nor any of its
words is either citing the wrong page or paraphrasing something that is not there,
and both need a person to look.

    uv run python scripts/check_citations.py
    uv run python scripts/check_citations.py --verbose

What this cannot catch: a citation that contains the right numbers in the wrong
context, and an answer correctly grounded in a page whose extracted text is too
damaged to match. The second is not hypothetical here. pypdf recovers almost
nothing from this corpus's Devanagari pages, which is the OCR-cascade problem
this project exists to study, so a Hindi question is likelier to fail this check
for reasons that have nothing to do with its label.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dastavez.evalset import Question, load  # noqa: E402

RUN_ID = "pypdf.strip.fixed.none.d5c4ff3b99cd"
CHUNKS = Path("corpus/chunks.sqlite")

# Numbers carry the claim in this corpus: amounts, dates, hectares, ages, counts.
# Matching on them rather than on words keeps the check strict without tripping
# over paraphrase, which is the whole point of a gold answer being a paraphrase.
NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def page_text(connection: sqlite3.Connection, document_id: str, page: int) -> str:
    rows = connection.execute(
        "SELECT text FROM chunks WHERE run_id=? AND document_id=? AND page=? ORDER BY ordinal",
        (RUN_ID, document_id, page),
    ).fetchall()
    return " ".join(row[0] for row in rows)


def claim_tokens(answer: str) -> list[str]:
    """The numbers a gold answer asserts, normalised so 6,000 and 6000 match."""
    found = []
    for raw in NUMBER.findall(answer):
        cleaned = raw.replace(",", "").rstrip(".")
        # One and two digit numbers appear on every page by accident. Anything
        # shorter than three digits is not distinctive enough to check on.
        if len(cleaned) >= 3:
            found.append(cleaned)
    return sorted(set(found))


def normalise(text: str) -> str:
    """Squeeze the spaces pypdf inserts inside numbers.

    The corpus is full of them: `pmkisan-faq-revised` page 1 carries the date
    "1 .6.2019", and five gold answers citing it were flagged as unsupported by
    the first version of this check for that reason alone. Extraction damage is
    what this project studies, so a checker that mistakes it for a label error is
    measuring the wrong thing.
    """
    return re.sub(r"(?<=\d)[\s,]+(?=[\d.])", "", text)


def supported(answer: str, text: str, question: str = "") -> tuple[bool, list[str]]:
    tokens = claim_tokens(answer)
    if not tokens:
        # Nothing numeric to check. Not evidence of a problem, just outside what
        # this check can see, and counted separately rather than as a pass.
        return True, []
    # A number the asker supplied is not a claim the page has to support. "A
    # pension of Rs. 8,000 does not trigger the exclusion" is grounded by the page
    # stating the 10,000 threshold, and 8,000 came from the question.
    asked = set(claim_tokens(question))
    haystack = normalise(text)
    missing = [t for t in tokens if t not in asked and t not in haystack]
    return not missing, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check gold citations support gold answers.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    questions = load()
    connection = sqlite3.connect(CHUNKS)

    checkable: list[Question] = []
    unchecked: list[Question] = []
    failures: list[tuple[Question, list[str]]] = []

    for question in questions:
        if question.question_type == "unanswerable":
            continue
        text = " ".join(page_text(connection, d, p) for d, p in question.gold_pages)
        ok, missing = supported(question.gold_answer, text, question.question)
        if not claim_tokens(question.gold_answer):
            unchecked.append(question)
            continue
        checkable.append(question)
        if not ok:
            failures.append((question, missing))

    answerable = [q for q in questions if q.question_type != "unanswerable"]
    passed = len(checkable) - len(failures)
    print(f"{len(answerable)} answerable questions")
    print(f"  {len(checkable)} carry a checkable numeric claim")
    print(f"  {passed} of those have every number present on the cited page")
    print(f"  {len(failures)} do not")
    print(f"  {len(unchecked)} have no number to check, so this says nothing about them")

    if failures:
        print()
        by_language: dict[str, int] = {}
        for question, _ in failures:
            by_language[question.language_tag] = by_language.get(question.language_tag, 0) + 1
        print("failures by question language: " + ", ".join(
            f"{lang} {n}" for lang, n in sorted(by_language.items())
        ))
        print()
        for question, missing in failures if args.verbose else failures[:8]:
            pages = ", ".join(f"{d} p{p}" for d, p in question.gold_pages)
            print(f"  {question.id} [{question.language_tag}/{question.question_type}] {pages}")
            print(f"    missing from the page: {missing}")
            print(f"    answer: {question.gold_answer[:88]}")
        if not args.verbose and len(failures) > 8:
            print(f"  ... {len(failures) - 8} more, use --verbose")

    print()
    if failures:
        print(
            f"{len(failures)} of {len(checkable)} checkable citations do not contain a number "
            "their answer asserts. Each is a question to look at, not a proven error: a "
            "damaged page extraction fails this check for reasons that have nothing to do "
            "with the label."
        )
    else:
        print(
            "Every checkable citation contains the numbers its answer asserts. That is the "
            "mechanical half of M4.2; the reading half stays with the audited subset."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

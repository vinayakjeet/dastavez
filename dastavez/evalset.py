"""The eval set, and the rules it has to satisfy before it counts as one.

SPEC calls this the project's contribution, and says the first thing measured is
not accuracy but how wrong the eval set itself is. That framing only works if the
set is a versioned artifact rather than a file people edit: a question whose gold
answer moved after a number was published invalidates the number silently, and
nobody notices because the file still looks like the file.

So the set is content-hashed, and the hash is committed. `verify` recomputes it
from the questions alone, over a canonical form, so reordering the file or
reformatting the JSON does not change the identity of the data inside it. What
changes the hash is a changed question, a changed answer, or a changed citation,
which is exactly the set of edits that should invalidate a published result.

The structural rules here are SPEC's own acceptance criteria for M4.1 and M4.2,
made executable. The one worth naming is the floor on `unanswerable`: refusal
correctness cannot be measured on a set that contains no unanswerable questions,
and a system that never refuses scores perfectly on such a set. Twenty of them is
what stops that from being an accident.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = (
    "factual",
    "eligibility_multihop",
    "table_lookup",
    "exclusion",
    "procedural",
    "unanswerable",
)

LANGUAGE_TAGS = ("hi", "en", "hinglish")
DIFFICULTY_TAGS = ("easy", "medium", "hard")

# SPEC's floors. Fifteen of each type so no slice is too thin to report on its
# own, and twenty unanswerable because that slice carries refusal correctness by
# itself.
MIN_PER_TYPE = 15
MIN_UNANSWERABLE = 20
TARGET_SIZE = 150

DEFAULT_PATH = Path("eval/questions.jsonl")
DEFAULT_HASH_PATH = Path("eval/questions.sha256")


class EvalSetError(ValueError):
    """The set violates a rule that makes it unusable as an eval set."""


@dataclass(frozen=True)
class Question:
    """One question. The eight fields SPEC names, and nothing else.

    `gold_pages` is a list of `(document_id, page)` pairs rather than free text,
    because citation-page accuracy is a metric here and a citation that cannot be
    resolved to a specific page of a specific named document cannot be scored.
    """

    id: str
    question: str
    gold_answer: str
    gold_pages: tuple[tuple[str, int], ...]
    difficulty_tag: str
    language_tag: str
    question_type: str
    annotator_ids: tuple[str, ...] = field(default=())
    note: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> Question:
        try:
            pages = tuple((str(d), int(p)) for d, p in row.get("gold_pages", []))
        except (TypeError, ValueError) as exc:
            raise EvalSetError(
                f"{row.get('id')}: gold_pages must be [document_id, page] pairs"
            ) from exc
        return cls(
            id=str(row["id"]),
            question=str(row["question"]),
            gold_answer=str(row["gold_answer"]),
            gold_pages=pages,
            difficulty_tag=str(row["difficulty_tag"]),
            language_tag=str(row["language_tag"]),
            question_type=str(row["question_type"]),
            annotator_ids=tuple(str(a) for a in row.get("annotator_ids", [])),
            note=str(row.get("note", "")),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "gold_pages": [[d, p] for d, p in self.gold_pages],
            "difficulty_tag": self.difficulty_tag,
            "language_tag": self.language_tag,
            "question_type": self.question_type,
            "annotator_ids": list(self.annotator_ids),
            "note": self.note,
        }

    def canonical(self) -> str:
        """What the hash is taken over.

        Deliberately not the whole record. `annotator_ids` and `note` describe who
        touched a question and why, and adding a second annotator to the audit
        subset must not invalidate results measured before that audit ran. The
        data being asked about, and the answer being asked for, are what identify
        the set.
        """
        return json.dumps(
            {
                "id": self.id,
                "question": self.question,
                "gold_answer": self.gold_answer,
                "gold_pages": [[d, p] for d, p in self.gold_pages],
                "question_type": self.question_type,
                "language_tag": self.language_tag,
                "difficulty_tag": self.difficulty_tag,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def load(path: Path | str = DEFAULT_PATH) -> list[Question]:
    path = Path(path)
    if not path.exists():
        raise EvalSetError(f"no eval set at {path}")
    out: list[Question] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Question.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise EvalSetError(f"{path}:{n}: {exc}") from exc
    return out


def dump(questions: Iterable[Question], path: Path | str = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(q.to_dict(), ensure_ascii=False, sort_keys=True) for q in questions
    )
    path.write_text(body + "\n", encoding="utf-8", newline="\n")


def content_hash(questions: Sequence[Question]) -> str:
    """A hash of the data, not of the file.

    Sorted by id before hashing, so the order lines happen to sit in does not
    change the identity of the set. Two people who wrote the same questions in a
    different order have the same eval set and should get the same hash.
    """
    joined = "\n".join(q.canonical() for q in sorted(questions, key=lambda q: q.id))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def structural_errors(questions: Sequence[Question]) -> list[str]:
    """Every rule SPEC's acceptance criteria state, checked at once.

    Returns all of them rather than raising on the first. A set being assembled
    fails many of these simultaneously and fixing them one exception at a time is
    the slowest possible way to find that out.
    """
    errors: list[str] = []
    seen: set[str] = set()
    counts = dict.fromkeys(QUESTION_TYPES, 0)

    for q in questions:
        where = q.id or "<no id>"
        if q.id in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(q.id)

        if q.question_type not in QUESTION_TYPES:
            errors.append(f"{where}: unknown question_type {q.question_type!r}")
        else:
            counts[q.question_type] += 1

        if q.language_tag not in LANGUAGE_TAGS:
            errors.append(f"{where}: unknown language_tag {q.language_tag!r}")
        if q.difficulty_tag not in DIFFICULTY_TAGS:
            errors.append(f"{where}: unknown difficulty_tag {q.difficulty_tag!r}")
        if not q.question.strip():
            errors.append(f"{where}: empty question")
        if not q.gold_answer.strip():
            errors.append(f"{where}: empty gold_answer")

        # The rule that carries the refusal slice. An unanswerable question with a
        # citation is answerable, and an answerable one without a citation cannot
        # be scored for citation accuracy.
        if q.question_type == "unanswerable":
            if q.gold_pages:
                errors.append(f"{where}: unanswerable but cites {len(q.gold_pages)} page(s)")
            if not q.note.strip():
                errors.append(
                    f"{where}: unanswerable needs a note on why the corpus cannot answer it"
                )
        elif not q.gold_pages:
            errors.append(f"{where}: no gold page, so the citation cannot be checked")

    for kind, n in counts.items():
        floor = MIN_UNANSWERABLE if kind == "unanswerable" else MIN_PER_TYPE
        if n < floor:
            errors.append(f"type {kind}: {n} questions, floor is {floor}")

    return errors


def coverage_errors(questions: Sequence[Question], known_pages: set[tuple[str, int]]) -> list[str]:
    """Every citation resolves to a page that exists in the corpus.

    Separate from `structural_errors` because it needs the corpus, and a question
    file should be checkable on its own before anyone builds an index.
    """
    errors = []
    for q in questions:
        for doc, page in q.gold_pages:
            if (doc, page) not in known_pages:
                errors.append(f"{q.id}: cites {doc} p{page}, which is not in the corpus")
    return errors


def verify(
    path: Path | str = DEFAULT_PATH, hash_path: Path | str = DEFAULT_HASH_PATH
) -> tuple[list[Question], str]:
    """Load, check structure, and confirm the committed hash still matches."""
    questions = load(path)
    if errors := structural_errors(questions):
        raise EvalSetError("\n".join(errors))
    digest = content_hash(questions)
    recorded = Path(hash_path)
    if recorded.exists():
        expected = recorded.read_text(encoding="utf-8").split()[0].strip()
        if expected != digest:
            raise EvalSetError(
                f"hash mismatch: {hash_path} records {expected[:12]}, "
                f"the questions hash to {digest[:12]}"
            )
    return questions, digest

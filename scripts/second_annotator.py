"""A second annotator over a subset of the eval set, from a different model family.

SPEC's M4.3 and M4.4. The eval set was drafted by one model, and the brief's own
warning is that a model-authored gold answer is exactly the label that looks right
and is not: the two text-to-SQL benchmarks it cites measured annotation error
rates of 52.8% and 62.8%, and correcting them moved leaderboard agents by up to
nine ranks.

    uv run python scripts/second_annotator.py --n 30
    uv run python scripts/second_annotator.py --report      # cached only

## What is being labelled, and why it is not the gold answer

The obvious design is to show the second annotator the gold answer and ask
whether the cited page supports it. That gives no usable kappa: my side of the
table has no variance, because I wrote every answer believing it, so chance
agreement is total and the statistic collapses. ShipGate hit the same wall on a
slice where two judges agreed 92% of the time and both scored zero.

So the label is **answerability**, which both annotators assign with real
variance: 123 of the 150 questions are answerable and 27 are deliberately not.
The second annotator sees the question and a retrieved context, never the gold
answer or the question type, and says whether the context answers it.

That is also the label most worth checking. An unanswerable question that is
actually answerable poisons the refusal slice, and an answerable one whose
citation does not hold poisons everything else.

## Context is retrieved, not handed over

Both kinds of question get the same treatment: the top chunks by BM25 over the
corpus, using the project's own `LexicalIndex`. Handing the cited page to
answerable questions and nothing to unanswerable ones would leak the label
through the shape of the prompt, and the annotator would be scoring the prompt
rather than the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dastavez.evalset import Question, load  # noqa: E402
from dastavez.lexical import LexicalIndex  # noqa: E402

RUN_ID = "pypdf.strip.fixed.none.d5c4ff3b99cd"
MODEL = "gemini-3.6-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
CACHE = Path("eval/second-annotation.jsonl")
SEED = 20260902

# Landis and Koch, the bands kappa is conventionally read against.
BANDS = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "poor"),
]

PROMPT = """You are annotating a question-answering dataset built over Indian
government scheme documents.

Below is a question and the passages a retrieval system found for it. Decide one
thing only: do these passages contain the information needed to answer the
question?

Answer "yes" if the passages state the answer, even partially but sufficiently.
Answer "no" if the passages do not contain it, or contain only related material
that does not actually answer what was asked.

Do not use knowledge you have about these schemes from anywhere else. Judge only
what the passages say.

QUESTION:
{question}

PASSAGES:
{context}

Reply with a JSON object and nothing else:
{{"answerable": "yes" or "no", "why": "one short sentence"}}"""


@dataclass
class Annotation:
    question_id: str
    mine: str
    theirs: str
    why: str
    model: str
    dated: str


def band(kappa: float) -> str:
    for floor, name in BANDS:
        if kappa >= floor:
            return name
    return "poor"


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Returns (kappa, observed agreement, expected agreement).

    Written here rather than imported because Dastavez has no calibration module
    and this is twelve lines. It is the same statistic ShipGate computes.
    """
    n = len(pairs)
    if not n:
        return 0.0, 0.0, 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    mine = Counter(a for a, _ in pairs)
    theirs = Counter(b for _, b in pairs)
    expected = sum((mine[k] / n) * (theirs[k] / n) for k in set(mine) | set(theirs))
    if expected >= 1.0:
        return 0.0, observed, expected
    return (observed - expected) / (1 - expected), observed, expected


def sample(questions: list[Question], n: int) -> list[Question]:
    """A stratified sample, so the unanswerable slice is represented in proportion.

    Seeded, because a subset that changes between runs makes the reported error
    rate unreproducible.
    """
    rng = random.Random(SEED)
    answerable = [q for q in questions if q.question_type != "unanswerable"]
    unanswerable = [q for q in questions if q.question_type == "unanswerable"]
    share = round(n * len(unanswerable) / len(questions))
    picked = rng.sample(unanswerable, min(share, len(unanswerable)))
    picked += rng.sample(answerable, n - len(picked))
    return sorted(picked, key=lambda q: q.id)


def context_for(index: LexicalIndex, question: str, k: int = 5) -> str:
    hits = index.search(RUN_ID, question, k=k)
    return "\n\n".join(
        f"[{h.citation()}]\n{h.chunk.text[:1200]}" for h in hits
    ) or "(no passages retrieved)"


def keys() -> list[str]:
    """Every Gemini key available, in rotation order.

    The free tier allows roughly 20 requests a rolling window with a retry delay
    of 20 to 49 seconds, which QUOTAS.md records. One key annotates about twenty
    questions before it stalls, so a thirty-question subset needs either patience
    or a second key. Rotating is the cheaper of the two.
    """
    found = [os.environ.get("GEMINI_API_KEY", "")]
    for n in range(2, 6):
        found.append(os.environ.get(f"GEMINI_API_KEY_{n}", ""))
    return [k for k in found if k]


def ask(pool: list[str], question: str, context: str, retries: int = 6) -> tuple[str, str]:
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": PROMPT.format(question=question, context=context)}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
    ).encode()
    for attempt in range(retries):
        # Round-robin, so a key that just hit its window is not the one retried.
        key = pool[attempt % len(pool)]
        request = urllib.request.Request(
            f"{ENDPOINT}/{MODEL}:generateContent?key={key}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            verdict = str(parsed.get("answerable", "")).strip().lower()
            return ("yes" if verdict.startswith("y") else "no"), str(parsed.get("why", ""))[:200]
        except urllib.error.HTTPError as exc:
            # 429 is the rolling window and 503 is the model under load. Both are
            # worth waiting out; the documented retry delay runs to 49 seconds, so
            # a backoff shorter than that just burns another attempt.
            if exc.code in (429, 503) and attempt < retries - 1:
                if len(pool) > 1 and attempt % len(pool) != len(pool) - 1:
                    time.sleep(3)  # another key is free, try it before waiting
                else:
                    time.sleep(min(20 + 15 * attempt, 60))
                continue
            raise
        except (TimeoutError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise
    raise RuntimeError("unreachable")


def load_cache() -> dict[str, Annotation]:
    out: dict[str, Annotation] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["question_id"]] = Annotation(**row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Second annotation over the eval set.")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--report", action="store_true", help="use cached annotations only")
    args = parser.parse_args()

    questions = load()
    subset = sample(questions, args.n)
    cache = load_cache()

    todo = [q for q in subset if q.id not in cache]
    print(f"{len(subset)} sampled, {len(cache)} already annotated, {len(todo)} to do")

    if todo and not args.report:
        pool = keys()
        if not pool:
            print("GEMINI_API_KEY is not set; nothing annotated.")
            return 1
        print(f"{len(pool)} key(s) in rotation")
        index = LexicalIndex()
        dated = time.strftime("%Y-%m-%d")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        for n, question in enumerate(todo, start=1):
            mine = "no" if question.question_type == "unanswerable" else "yes"
            theirs, why = ask(pool, question.question, context_for(index, question.question))
            row = Annotation(question.id, mine, theirs, why, MODEL, dated)
            with CACHE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
            cache[question.id] = row
            flag = "" if mine == theirs else "   <- disagreement"
            print(f"  [{n}/{len(todo)}] {question.id}  mine={mine} theirs={theirs}{flag}")
            time.sleep(4.0)

    annotated = [cache[q.id] for q in subset if q.id in cache]
    if len(annotated) < 2:
        print("not enough annotations to compute agreement")
        return 1

    pairs = [(a.mine, a.theirs) for a in annotated]
    kappa, observed, expected = cohens_kappa(pairs)
    disagreements = [a for a in annotated if a.mine != a.theirs]

    print()
    print("=" * 74)
    print(f"second annotator : {MODEL}, {annotated[0].dated}")
    print(f"subset           : {len(annotated)} of {len(questions)}, seeded {SEED}")
    print(f"raw agreement    : {observed:.1%}")
    print(f"chance agreement : {expected:.1%}")
    print(f"Cohen's kappa    : {kappa:.3f}  ({band(kappa)})")
    print("=" * 74)

    if disagreements:
        by_id = {q.id: q for q in questions}
        print(f"\n{len(disagreements)} disagreements, each needing adjudication:\n")
        for a in disagreements:
            q = by_id[a.question_id]
            print(f"  {a.question_id}  [{q.question_type}]  mine={a.mine} theirs={a.theirs}")
            print(f"    Q: {q.question[:96]}")
            print(f"    their reason: {a.why[:110]}")
            print(f"    my citation : {[f'{d} p{p}' for d, p in q.gold_pages] or 'none'}")
            print()

    rate = len(disagreements) / len(annotated)
    print(
        f"Disagreement rate {rate:.1%} on the audited subset. Extrapolated to the full "
        f"{len(questions)}, that is an estimated {rate * len(questions):.0f} questions "
        "carrying a label one of the two annotators would dispute."
    )
    print(
        "An estimate, not a count. Only the subset above was annotated twice, and a "
        "disagreement marks a question worth checking rather than a question known to be "
        "wrong: adjudication decides which annotator was right, and sometimes neither is."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

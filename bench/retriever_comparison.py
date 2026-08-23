"""Where dense and lexical retrieval disagree, and which one is right.

    uv run python bench/retriever_comparison.py
    uv run python bench/retriever_comparison.py --run <run_id>

M2.3 asks for a query where BM25 beats the dense retriever, with the retrieved
chunks shown. This regenerates that rather than quoting it, because a number in a
README that cannot be regenerated does not belong in the README.

The probe set is small and hand-written, and that is stated rather than hidden. It
is not the eval set: it has no gold answers, it was chosen to span the shapes of
question this corpus attracts, and its purpose is to show that the two halves
disagree often enough to be worth running both. The eval set replaces it for any
claim about accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# bench/ scripts run as files rather than as modules, so the repo root is not on the
# path. Adding it here keeps the documented command a single copyable line, which is
# the point of a bench script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dastavez.hybrid import HybridRetriever  # noqa: E402

DEFAULT_RUN = "pypdf.none.fixed.none.a5037433bfc4"

# Chosen to span the shapes, not sampled. Lexical shapes name a code, an amount or a
# category verbatim; semantic shapes paraphrase.
PROBES = [
    "PM-KISAN",
    "Rs.6000 per year",
    "institutional land holders",
    "shall not be eligible for benefit",
    # Both phrasings, deliberately. The short form put BM25 on the exclusion list and
    # dense on an unrelated procedure; the long form moved both somewhere else. A
    # comparison that changes under a rewording is partly about the wording, and
    # keeping only the flattering phrasing would be choosing the result first.
    "who is excluded from PM-Kisan",
    "who is excluded from receiving PM-Kisan benefits",
    "grievance redressal officer",
    "how much money does a farmer get",
    "what documents do I need to apply",
]


def show(hit, prefix: str) -> None:
    body = " ".join(hit.chunk.text.split())[:150]
    print(f"    {prefix:<9} {hit.citation()}")
    print(f"              {body}...")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="show the chunks")
    args = parser.parse_args()

    retriever = HybridRetriever()
    load = retriever.warm()
    print(f"embedder warmed in {load:.1f}s, excluded from every number below\n")

    disagreements = []
    for probe in PROBES:
        try:
            comparison = retriever.compare(args.run, probe, k=args.k)
        except LookupError as exc:
            print(exc)
            return 1

        mark = "agree" if comparison.agreed else "DISAGREE"
        print(f"{mark:<9} {probe}")
        if not comparison.agreed:
            disagreements.append(comparison)
            if args.verbose:
                show(comparison.dense[0], "dense")
                show(comparison.lexical[0], "lexical")
                print()

    retriever.close()

    print()
    print(f"{len(disagreements)} of {len(PROBES)} probes disagree on the top result.")
    print(
        "Disagreement is not evidence either half is better. It is evidence that "
        "one retriever\nis not enough, which is what M2.3 asks this to show. Which "
        "half is right on a given\nquery is a question for the eval set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

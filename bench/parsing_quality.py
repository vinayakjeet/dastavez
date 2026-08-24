"""Parsing quality per converter per gold page, OmniDocBench definitions.

    uv run python bench/parsing_quality.py --converter docling-noocr
    uv run python bench/parsing_quality.py --all-converters --document pmkisan-og-revised-en

Three metrics per page against corpus/gold: text similarity, table TEDS and
reading order, defined in `dastavez/parsing.py` with their sources. A page the
converter produced nothing for scores zero and says so, because an empty page
is a measurement, not a gap in the table.

These scores do not predict downstream answer quality; they locate where each
converter loses content and structure. The end-to-end numbers are M6's job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# bench/ scripts run as files rather than as modules, so the repo root is not on the
# path. Adding it here keeps the documented command a single copyable line, which is
# the point of a bench script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dastavez.chunks import Block  # noqa: E402
from dastavez.ingest.converters import Converter, convert_cached  # noqa: E402
from dastavez.ingest.run import BUILDERS, DOCUMENTS, manifest_rows  # noqa: E402
from dastavez.parsing import (  # noqa: E402
    parse_markdown_table,
    reading_order_score,
    split_gold_blocks,
    teds,
    text_similarity,
)

GOLD_INDEX = Path("corpus/gold/index.jsonl")


def gold_rows() -> dict[str, dict]:
    if not GOLD_INDEX.exists():
        print("no gold index. Transcribe gold pages first (corpus/gold/README.md).")
        raise SystemExit(1)
    rows = {}
    for line in GOLD_INDEX.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[f"{row['document_id']}-p{row['page']}"] = row
    return rows


def score_page(blocks: list[Block], gold_text: str) -> dict[str, float | str | None]:
    output_text = "\n\n".join(block.text for block in blocks)
    gold_blocks = split_gold_blocks(gold_text)
    gold_tables = [t for t in (parse_markdown_table(b) for b in gold_blocks) if t]

    table_scores: list[float] = []
    for gold_table in gold_tables:
        candidates = [b.text for b in blocks if b.kind == "TableItem"] or [
            b.text for b in blocks if parse_markdown_table(b.text)
        ]
        table_scores.append(max((teds(c, _render(gold_table)) for c in candidates), default=0.0))

    order = reading_order_score([b.text for b in blocks], gold_blocks)

    return {
        "text": text_similarity(output_text, gold_text),
        "teds": sum(table_scores) / len(table_scores) if table_scores else None,
        "order": order,
        "blocks": len(blocks),
    }


def _render(table) -> str:
    """A GoldTable back to markdown so TEDS compares like with like."""
    lines = ["| " + " | ".join(table.header) + " |", "|" + "---|" * len(table.header)]
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", choices=sorted(BUILDERS))
    parser.add_argument("--all-converters", action="store_true")
    parser.add_argument("--document", help="one document id, default is every gold document")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if not args.converter and not args.all_converters:
        parser.error("name a converter or pass --all-converters")
    converters: list[str] = (
        sorted(BUILDERS) if args.all_converters else [args.converter]  # type: ignore[list-item]
    )

    rows = gold_rows()
    wanted = sorted({row["document_id"] for row in rows.values()})
    if args.document:
        wanted = [args.document] if args.document in wanted else []
        if not wanted:
            print(f"no gold pages for {args.document}")
            return 1

    manifest = manifest_rows()
    print(
        "text similarity, table TEDS and reading order per page; definitions and"
        " their sources in dastavez/parsing.py. These scores locate content and"
        " structure loss; they do not predict retrieval or answer quality."
    )
    print()

    for name in converters:
        builder: type[Converter] = BUILDERS[name]
        converter = builder()
        per_page: list[tuple[str, dict]] = []
        empty = 0
        for document_id in wanted:
            path = DOCUMENTS / f"{document_id}.pdf"
            if not path.exists() or document_id not in manifest:
                continue
            try:
                all_blocks = convert_cached(path, converter, refresh=args.refresh)
            except RuntimeError as exc:
                print(f"{name}: {exc}")
                return 1
            pages = {
                int(key.rsplit("-p", 1)[1]): value
                for key, value in rows.items()
                if value["document_id"] == document_id
            }
            for page, row in sorted(pages.items()):
                page_blocks = [b for b in all_blocks if b.page == page]
                if not page_blocks:
                    empty += 1
                per_page.append(
                    (f"{document_id}-p{page}", score_page(page_blocks, _gold_text(row)))
                )

        print(f"== {name} ==")
        for key, scores in per_page:
            order = scores["order"]
            order_text = "n/a" if order is None else f"{order:+.2f}"
            teds_text = "n/a" if scores["teds"] is None else f"{scores['teds']:.3f}"
            print(
                f"  {key:<38} text {scores['text']:.3f}  teds {teds_text:>5}"
                f"  order {order_text:>5}  blocks {scores['blocks']}"
            )
        if per_page:
            text_mean = sum(s["text"] for _, s in per_page) / len(per_page)
            teds_values = [s["teds"] for _, s in per_page if s["teds"] is not None]
            orders = [s["order"] for _, s in per_page if s["order"] is not None]
            teds_mean = sum(teds_values) / len(teds_values) if teds_values else float("nan")
            order_mean = sum(orders) / len(orders) if orders else float("nan")
            print(
                f"  {'MEAN':<38} text {text_mean:.3f}  teds {teds_mean:.3f}"
                f"  order {order_mean:+.3f}  pages {len(per_page)} ({empty} empty)"
            )
        print()
    return 0


def _gold_text(row: dict) -> str:
    return Path(row["path"]).read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

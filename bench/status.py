"""Dastavez's status numbers, regenerated from the artifacts on disk.

    uv run python bench/status.py

One JSON object on stdout, in the shape the portfolio site's project frontmatter
consumes: label, value, source. Every number is derived here from the corpus
manifest, the gold index and the conversion cache, so a stale claim requires a
stale artifact rather than a stale sentence.

Parsing means cover whichever converters have cached conversions for gold
documents; an arm with no cache is absent from the list rather than reported as
zero, because absent and measured are different things.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsing_quality import gold_rows, score_page  # noqa: E402

from dastavez.ingest.converters import cached_blocks  # noqa: E402
from dastavez.ingest.run import BUILDERS, DOCUMENTS, manifest_rows  # noqa: E402

SOURCE = "bench/status.py"


def corpus_numbers(manifest: dict[str, dict]) -> dict[str, int]:
    return {
        "documents": len(manifest),
        "pages": sum(int(row.get("pages", 0)) for row in manifest.values()),
    }


def gold_numbers(rows: dict[str, dict]) -> dict[str, int]:
    return {
        "pages": len(rows),
        "schemes": len({row["scheme"] for row in rows.values()}),
        "hindi_pages": sum(1 for row in rows.values() if row["language"] == "hi"),
        "table_pages": sum(1 for row in rows.values() if row["has_table"]),
    }


def parsing_means(rows: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Means per converter, from the cache only, over the pages it covers."""
    manifest = manifest_rows()
    documents = sorted({row["document_id"] for row in rows.values()})
    means: dict[str, dict[str, float]] = {}
    for name, builder in sorted(BUILDERS.items()):
        converter = builder()
        pages: list[dict] = []
        for document_id in documents:
            path = DOCUMENTS / f"{document_id}.pdf"
            if not path.exists() or document_id not in manifest:
                continue
            all_blocks = cached_blocks(path, converter)
            if all_blocks is None:
                continue
            for key, row in rows.items():
                if row["document_id"] != document_id:
                    continue
                page = int(key.rsplit("-p", 1)[1])
                page_blocks = [b for b in all_blocks if b.page == page]
                pages.append(score_page(page_blocks, _gold_text(row)))
        if pages:
            teds_pages = [p["teds"] for p in pages if p["teds"] is not None]
            order_pages = [p["order"] for p in pages if p["order"] is not None]
            entry = {
                "text": round(sum(p["text"] for p in pages) / len(pages), 3),
                "pages": len(pages),
            }
            # An arm whose covered pages carry no table has no TEDS to report.
            # Zero would read as a measurement; absent is the honest value.
            if teds_pages:
                entry["teds"] = round(sum(teds_pages) / len(teds_pages), 3)
            if order_pages:
                entry["order"] = round(sum(order_pages) / len(order_pages), 3)
            means[name] = entry
    return means


def _gold_text(row: dict) -> str:
    return Path(row["path"]).read_text(encoding="utf-8")


def main() -> int:
    manifest = manifest_rows()
    rows = gold_rows()
    corpus = corpus_numbers(manifest)
    gold = gold_numbers(rows)
    parsing = parsing_means(rows)

    metrics = [
        {"label": "corpus documents, four schemes", "value": corpus["documents"]},
        {"label": "corpus pages, scans included", "value": corpus["pages"]},
        {"label": "hand-curated gold pages", "value": gold["pages"]},
        {"label": "gold pages carrying tables", "value": gold["table_pages"]},
        {"label": "gold pages carrying Hindi", "value": gold["hindi_pages"]},
    ]
    for name, scores in parsing.items():
        metrics.append(
            {
                "label": f"parsing text similarity, {name}, {scores['pages']} gold pages",
                "value": scores["text"],
            }
        )
        if "teds" in scores:
            metrics.append(
                {
                    "label": f"parsing table TEDS, {name}, {scores['pages']} gold pages",
                    "value": scores["teds"],
                }
            )
    metrics.append({"label": "gold label audit (M4.4)", "value": "pending"})

    print(
        json.dumps(
            {
                "project": "dastavez",
                "date": date.today().isoformat(),
                "source": SOURCE,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

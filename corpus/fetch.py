"""Build and verify the document corpus.

Two modes, and the second is the one that makes the corpus an artifact rather than a
folder on one laptop:

    uv run python corpus/fetch.py            download anything missing, write manifest
    uv run python corpus/fetch.py --verify   re-hash what is on disk, report drift

Every row of `manifest.jsonl` carries the seven fields BACKLOG M0.1 requires: source
url, publisher, publication date, retrieval date, page count, language, and a content
hash. A document that cannot be fetched is recorded with its failure rather than
omitted, because a silent gap in a corpus is indistinguishable from a document that
was never meant to be there.

Government hosts are slow and sometimes hostile to scripted access. This fetches
politely: one at a time, with a real timeout, and it does not retry aggressively. A
corpus build that hammers a public service is not a corpus build worth having.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import httpx
import yaml
from pypdf import PdfReader

CORPUS = Path(__file__).parent
SOURCES = CORPUS / "sources.yaml"
DOCUMENTS = CORPUS / "documents"
MANIFEST = CORPUS / "manifest.jsonl"

TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0)

# Some government hosts reject the default python-httpx agent outright. Identifying
# honestly as a research crawler is the polite version of setting a browser string.
HEADERS = {"User-Agent": "dastavez-corpus/0.1 (research; contact via repository)"}


@dataclass
class Entry:
    id: str
    scheme: str
    kind: str
    language: str
    url: str
    publisher: str
    published: str | None
    retrieved: str | None
    pages: int | None
    has_text_layer: bool | None
    text_page_ratio: float | None
    sha256: str | None
    bytes: int | None
    status: str


def load_sources() -> list[Entry]:
    raw = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))
    entries: list[Entry] = []
    for scheme, block in (raw.get("schemes") or {}).items():
        for doc in block.get("documents") or []:
            entries.append(
                Entry(
                    id=doc["id"],
                    scheme=scheme,
                    kind=doc["kind"],
                    language=doc["language"],
                    url=doc["url"],
                    publisher=block.get("name_en", scheme),
                    published=doc.get("published"),
                    retrieved=None,
                    pages=None,
                    has_text_layer=None,
                    text_page_ratio=None,
                    sha256=None,
                    bytes=None,
                    status="pending",
                )
            )
    return entries


def prior_manifest() -> dict[str, dict]:
    """What the last run recorded, keyed by document id.

    Needed because a cached document skips the network and would otherwise lose its
    retrieval date, which is one of the seven fields the manifest promises. A corpus
    that forgets when it was collected is a corpus nobody can date a result against.
    """
    if not MANIFEST.exists():
        return {}
    rows = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row["id"]] = row
    return rows


def published_date(path: Path) -> str | None:
    """The document's own creation date, where it states one.

    PDF metadata is frequently absent or wrong, so this is a best effort and null
    means "the file does not state one" rather than "unknown and probably recent".
    Government documents are often re-exported years after publication, so this
    never overrides a date written by hand in sources.yaml.
    """
    try:
        raw = (PdfReader(str(path)).metadata or {}).get("/CreationDate")
    except Exception:
        return None
    if not raw:
        return None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if len(digits) < 8:
        return None
    return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"


def inspect(path: Path) -> tuple[int | None, bool | None, float | None]:
    """Real page count, and whether page one carries extractable text.

    This replaced a regex over the raw bytes that counted `/Type /Page`. That
    heuristic was wrong twice. It over-counted by one per document, because
    `/Type /Pages` is the page-tree node and contains `/Type /Page` as a substring.
    Then, with that fixed, it silently returned nothing for eight of twenty-three
    documents, because PDF 1.5 and later can put those objects inside compressed
    object streams where no regex will find them. The corpus totalled 197 pages by
    heuristic and 284 by parse, a 31 percent undercount, and the corpus size target
    is stated in pages.

    The text-layer signal is not a nicety. It decides which documents need OCR, which
    is the most expensive step in the pipeline, and it defines the population the
    OCR-cascade error category is measured over.

    It is measured over every page, not sampled from page one. The first version
    sampled page one and mislabelled four documents, because a cover page is an image
    even when the body is text: `pmayu-clap-manual` was recorded as image-only while
    46 of its 52 pages carry text. That is roughly an hour of needless OCR at 31
    seconds a page, and a wrong denominator for the OCR-cascade rate.

    So the ratio is recorded as well as the flag. A document that is 90 percent text
    with an image cover and one that is 10 percent text are different problems, and a
    boolean cannot tell them apart.
    """
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        if not pages:
            return 0, None, None
        with_text = sum(
            1 for page in reader.pages if len((page.extract_text() or "").strip()) > 40
        )
        ratio = with_text / pages
        # Majority rather than any. A document with two readable pages out of fifty
        # needs OCR, and calling it "has a text layer" would skip it.
        return pages, ratio >= 0.5, round(ratio, 3)
    except Exception:
        return None, None, None


def fetch_one(client: httpx.Client, entry: Entry, previous: dict[str, dict]) -> Entry:
    target = DOCUMENTS / f"{entry.id}.pdf"

    if target.exists():
        blob = target.read_bytes()
        entry.sha256 = hashlib.sha256(blob).hexdigest()
        entry.bytes = len(blob)
        entry.pages, entry.has_text_layer, entry.text_page_ratio = inspect(target)
        entry.retrieved = (previous.get(entry.id) or {}).get("retrieved") or date.fromtimestamp(
            target.stat().st_mtime
        ).isoformat()
        entry.published = entry.published or published_date(target)
        entry.status = "cached"
        return entry

    try:
        response = client.get(entry.url, headers=HEADERS, follow_redirects=True)
    except httpx.RequestError as exc:
        entry.status = f"unreachable: {type(exc).__name__}"
        return entry

    if response.status_code != 200:
        entry.status = f"http {response.status_code}"
        return entry

    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        # A 200 that is a login page or an error page is the failure mode that
        # quietly poisons a corpus, because it looks like a successful download.
        entry.status = f"not a pdf: {content_type or 'unknown content-type'}"
        return entry

    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    entry.sha256 = hashlib.sha256(response.content).hexdigest()
    entry.bytes = len(response.content)
    entry.pages, entry.has_text_layer, entry.text_page_ratio = inspect(target)
    entry.published = entry.published or published_date(target)
    entry.retrieved = date.today().isoformat()
    entry.status = "fetched"
    return entry


def write_manifest(entries: list[Entry]) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as fh:
        for entry in entries:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest, run without --verify first")
        return 1

    drifted, missing, ok = [], [], 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("sha256"):
            continue
        path = DOCUMENTS / f"{row['id']}.pdf"
        if not path.exists():
            missing.append(row["id"])
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            drifted.append(row["id"])
        else:
            ok += 1

    for name in missing:
        print(f"MISSING  {name}")
    for name in drifted:
        print(f"CHANGED  {name}")
    print(f"{ok} verified, {len(missing)} missing, {len(drifted)} changed")
    return 1 if (missing or drifted) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        return verify()

    entries = load_sources()
    previous = prior_manifest()
    with httpx.Client(timeout=TIMEOUT) as client:
        for entry in entries:
            fetch_one(client, entry, previous)
            print(f"{entry.status:<28} {entry.id}")

    write_manifest(entries)

    usable = [e for e in entries if e.sha256]
    pages = sum(e.pages or 0 for e in usable)
    scanned = [e for e in usable if e.has_text_layer is False]
    print()
    print(f"{len(usable)}/{len(entries)} documents, {pages} pages")
    print(f"{len(scanned)} image-only, needing OCR: {', '.join(e.id for e in scanned) or 'none'}")
    if len(usable) < len(entries):
        print("Failures are recorded in the manifest rather than dropped. Replace a")
        print("dead url in sources.yaml, or record in LEARNING.md why it stays dead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

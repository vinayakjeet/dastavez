"""Find candidate PDF documents on scheme websites.

Written because the first hand-assembled URL list resolved one time in five, and
because that will happen again: government scheme sites reorganise, and a corpus
built from links copied once rots quietly.

    uv run python corpus/discover.py                 crawl seeds, probe, report
    uv run python corpus/discover.py --out cands.tsv write candidates to a file

The output is candidates, not corpus. A human still decides which documents belong,
because "is this the operational guideline or a press release" is not something a
link extractor can answer. What this removes is the part that was never judgement:
finding which links exist and which of them actually return a PDF.

Politeness is not optional here. These are public services on public money, and a
crawl that hammers them is both rude and likely to get blocked. One request at a
time, a real delay between them, one level deep from the seeds, and a hard cap.

TLS verification stays on. Some of these hosts serve incomplete certificate chains,
and the tempting fix is to pass `verify=False`. That is refused here for two reasons:
it would ship in a public repo as an example other people copy, and a certificate
that does not validate is a fact about the source worth recording rather than a
nuisance worth suppressing. A TLS failure is reported as its own status, so the
manifest can say "this document exists but its host cannot be verified" instead of
quietly pretending the problem is not there.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

HEADERS = {"User-Agent": "dastavez-corpus/0.1 (research; contact via repository)"}
TIMEOUT = httpx.Timeout(connect=12.0, read=45.0, write=12.0, pool=12.0)
DELAY_S = 1.0
MAX_PROBES = 120

HREF = re.compile(rb"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

# Seed pages per scheme. Landing pages plus wherever each site parks documents.
# Hosts that did not resolve at all on 2026-08-22 are left in with a note rather
# than deleted, because "this host was unreachable" is itself a corpus fact.
SEEDS: dict[str, list[str]] = {
    "pm-kisan": [
        "https://pmkisan.gov.in/",
        "https://pmkisan.gov.in/Documents.aspx",
        "https://pmkisan.gov.in/HomeNew.aspx",
    ],
    "pmay-u": [
        "https://pmay-urban.gov.in/",
        "https://pmay-urban.gov.in/guidelines",
        "https://pmay-urban.gov.in/documents",
    ],
    "ujjwala": [
        "https://www.pmuy.gov.in/",
        "https://www.pmuy.gov.in/documents.html",
        "https://dfpd.gov.in/",
    ],
    "ayushman-bharat": [
        "https://nha.gov.in/",
        "https://nha.gov.in/PM-JAY",
    ],
    # pmayg.nic.in and pmjay.gov.in did not resolve from here on 2026-08-22.
    # Recorded in LEARNING.md rather than silently dropped.
}


@dataclass
class Candidate:
    scheme: str
    url: str
    status: str
    bytes: int | None


def links_from(client: httpx.Client, page: str) -> list[str]:
    try:
        response = client.get(page, headers=HEADERS, follow_redirects=True)
    except httpx.RequestError:
        return []
    if response.status_code != 200:
        return []

    found = []
    for raw in HREF.findall(response.content):
        try:
            href = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not href or href.startswith(("mailto:", "javascript:", "#")):
            continue
        absolute = urljoin(str(response.url), href)
        if ".pdf" in absolute.lower():
            found.append(absolute.split("#")[0])
    return found


def probe(client: httpx.Client, url: str) -> tuple[str, int | None]:
    """A URL counts only if the body actually starts with %PDF.

    A 200 that returns HTML is the failure this whole module exists to catch: it
    writes a login page to disk under a .pdf name, and the damage only surfaces much
    later as a converter producing nothing.
    """
    try:
        with client.stream("GET", url, headers=HEADERS, follow_redirects=True) as response:
            if response.status_code != 200:
                return f"http {response.status_code}", None
            head = next(response.iter_bytes(chunk_size=1024), b"")
            if not head.startswith(b"%PDF"):
                kind = response.headers.get("content-type", "unknown")
                return f"not a pdf: {kind}", None
            length = response.headers.get("content-length")
            return "pdf", int(length) if length and length.isdigit() else None
    except httpx.ConnectError as exc:
        # A certificate failure arrives as a ConnectError, and it is worth
        # distinguishing: "the host is down" and "the host cannot be verified" call
        # for different decisions about whether the document can be used at all.
        reason = str(exc).lower()
        detail = "tls" if "certificate" in reason or "ssl" in reason else "connect"
        return f"unreachable: {detail}", None
    except httpx.RequestError as exc:
        return f"unreachable: {type(exc).__name__}", None


def discover() -> list[Candidate]:
    seen: set[str] = set()
    candidates: list[Candidate] = []
    probes = 0

    with httpx.Client(timeout=TIMEOUT) as client:
        for scheme, pages in SEEDS.items():
            urls: list[str] = []
            for page in pages:
                urls.extend(links_from(client, page))
                time.sleep(DELAY_S)

            for url in urls:
                if url in seen or probes >= MAX_PROBES:
                    continue
                seen.add(url)
                status, size = probe(client, url)
                probes += 1
                time.sleep(DELAY_S)
                candidates.append(Candidate(scheme, url, status, size))
                if status == "pdf":
                    kb = f"{size // 1024}kb" if size else "size unknown"
                    print(f"  PDF   {kb:>10}  {url}")

    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    candidates = discover()
    good = [c for c in candidates if c.status == "pdf"]

    print()
    print(f"{len(good)} confirmed PDFs from {len(candidates)} candidate links")
    if candidates:
        rejected = len(candidates) - len(good)
        print(f"{rejected} rejected. That rejection rate is corpus data, not noise.")

    if args.out:
        with args.out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("scheme\tstatus\tbytes\turl\n")
            for c in candidates:
                fh.write(f"{c.scheme}\t{c.status}\t{c.bytes or ''}\t{c.url}\n")
        print(f"written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

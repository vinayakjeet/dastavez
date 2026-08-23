"""Converters, behind one interface, with their output cached.

Four converters are one axis of the ablation. The other three axes (cleaning,
splitting, metadata enrichment) operate on the blocks a converter produced, and never
re-open the PDF. That asymmetry is the whole reason this module caches.

Measured on this machine, 2026-08-22: Docling's first conversion of a single page
took 6,371 seconds including model downloads, and a warm conversion of the same page
still exceeded two minutes. The corpus is 353 pages. Converting once per
configuration would put the M6 matrix at thousands of hours and the ablation would
simply not happen.

Converting once per converter and caching the blocks puts it at four passes over the
corpus, after which every combination of the remaining three axes is a cheap
transformation of cached output, and the >=3 repeat runs the credibility rules
require cost nothing extra in conversion.

The cache key is the document's content hash plus the converter's identity and
version. It is deliberately not the file path: two runs against a document that
changed underneath them must not silently share a cache entry, and the corpus
manifest already records exactly this hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from dastavez.chunks import Block

logger = logging.getLogger(__name__)

CACHE = Path("corpus/.converted")


class Converter(Protocol):
    name: str
    version: str

    def blocks(self, path: Path) -> list[Block]: ...


def cache_key(path: Path, converter: Converter) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{path.stem}.{converter.name}-{converter.version}.{digest}.json"


def convert_cached(path: Path, converter: Converter, *, refresh: bool = False) -> list[Block]:
    """Blocks for this document under this converter, converting only if needed."""
    target = CACHE / cache_key(path, converter)

    if target.exists() and not refresh:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return [Block(**block) for block in payload["blocks"]]

    started = time.monotonic()
    blocks = converter.blocks(path)
    elapsed = time.monotonic() - started

    CACHE.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "document": path.stem,
                "converter": converter.name,
                "version": converter.version,
                "seconds": round(elapsed, 2),
                "blocks": [asdict(b) for b in blocks],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        "convert.done", extra={"document": path.stem, "converter": converter.name,
                               "blocks": len(blocks), "seconds": round(elapsed, 1)}
    )
    return blocks


class PyPdfConverter:
    """The floor baseline, and it is not a strawman.

    `PyPDFLoader` is what most tutorials use, so the honest question this project
    answers is how much a better pipeline buys over what someone would build in an
    afternoon. It extracts text per page and labels nothing, which means the
    section-respecting splitter has nothing to work with. That is the point: the
    converter axis and the splitting axis interact, and this converter is where that
    interaction is visible.

    It also cannot read a page with no text layer, and twelve of the thirty documents
    in this corpus are exactly that. It returns nothing for those rather than
    pretending, which is what makes the OCR-cascade category measurable.
    """

    name = "pypdf"
    version = "1"

    def blocks(self, path: Path) -> list[Block]:
        from pypdf import PdfReader

        found: list[Block] = []
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                found.append(Block(text=text, page=number, kind=None))
        return found


class DoclingConverter:
    """Layout-aware conversion, with block types and page numbers preserved.

    OCR is a constructor argument rather than always-on. Docling runs OCR by default,
    and eighteen of this corpus's thirty documents already carry a text layer, so the
    default spends the most expensive part of the pipeline re-reading text that was
    already extractable. Whether that is a quality tradeoff or pure waste is an
    ablation question, so it is a knob rather than a decision made here.
    """

    name = "docling"

    def __init__(self, *, ocr: bool = True) -> None:
        self.ocr = ocr
        self.version = f"2-ocr{int(ocr)}"
        self._converter = None

    def _build(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = self.ocr
        options.do_table_structure = True
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def blocks(self, path: Path) -> list[Block]:
        if self._converter is None:
            self._converter = self._build()

        document = self._converter.convert(str(path)).document
        found: list[Block] = []
        for item, level in document.iterate_items():
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            page = _page_of(item)
            if page is None:
                # A block whose page cannot be recovered is dropped, loudly. Keeping
                # it would put text in the index that no citation can ever point at,
                # which is worse than a retrieval miss because it looks like success.
                logger.warning(
                    "convert.block_without_page",
                    extra={"document": path.stem, "text": text[:60]},
                )
                continue
            found.append(
                Block(text=text, page=page, kind=type(item).__name__, level=level)
            )
        return found


def _page_of(item: object) -> int | None:
    """Docling carries page numbers on provenance entries rather than on the item."""
    for entry in getattr(item, "prov", None) or []:
        number = getattr(entry, "page_no", None)
        if isinstance(number, int) and number >= 1:
            return number
    return None


class RoutedDoclingConverter:
    """Docling with OCR turned on only for documents that have no text layer.

    Measured on this machine, 2026-08-22, on one page carrying a text layer: Docling
    with OCR took 31.0 seconds and produced 4 items; with `do_ocr=False` it took 2.4
    seconds and produced the same 4 items. Thirteen times the cost for output that
    matched.

    Across this corpus, 18 of 30 documents (198 of 353 pages) carry a text layer, so
    routing takes one converter pass from 3.04 hours to 1.47 hours, and the four-arm
    M6 matrix from 12.16 to 5.87 hours.

    **This is an ablation arm, not an optimisation applied to the others.** The
    equivalence above is a single page, n=1, on one document. Whether routed output
    matches always-on output across the whole corpus is exactly the kind of claim this
    project exists to measure rather than assume, so it runs as its own arm and its
    parsing-quality scores get reported beside the other two in M1.7. If it loses
    quality anywhere, the table will say so.

    The routing signal is `has_text_layer` from the corpus manifest, which was
    recorded at fetch time precisely so this decision would not need to re-open every
    file.
    """

    name = "docling-routed"
    version = "2"

    def __init__(self, manifest: Path = Path("corpus/manifest.jsonl")) -> None:
        self._text_layer: dict[str, bool] = {}
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("has_text_layer") is not None:
                    self._text_layer[row["id"]] = bool(row["has_text_layer"])
        self._with_ocr = DoclingConverter(ocr=True)
        self._without = DoclingConverter(ocr=False)

    def blocks(self, path: Path) -> list[Block]:
        # Unknown documents get OCR. Being slow on a document the manifest has not
        # seen is recoverable; silently skipping OCR on a scanned page produces an
        # empty document that looks like a converter failure much later.
        has_text = self._text_layer.get(path.stem, False)
        chosen = self._without if has_text else self._with_ocr
        return chosen.blocks(path)


def converters() -> Iterator[Converter]:
    """Everything the ablation can vary along the converter axis, today.

    MinerU and Marker join this list at M1.2. They are absent rather than stubbed,
    because a stub that returns nothing is indistinguishable in a results table from
    a converter that failed.
    """
    yield PyPdfConverter()
    yield DoclingConverter(ocr=True)
    yield DoclingConverter(ocr=False)
    yield RoutedDoclingConverter()

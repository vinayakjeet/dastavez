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

Marker and MinerU do not install into the same environment as Docling: their
dependency trees pin torch and transformers against each other, and resolving all
four together produces an environment where none of them behaves as it does alone.
So each heavy converter converts from its own environment and writes into this
shared cache, which is what `dastavez.offline_convert` drives. The served service
and the tests never import either library; everything downstream of a conversion
reads blocks that are indistinguishable by origin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, replace
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


def strip_unmapped_glyphs(text: str) -> tuple[str, int]:
    """Remove NULL bytes a converter emits where it could not map a glyph.

    Measured on this corpus: pypdf produced 785 of them across two documents, and
    `pmayu-ahp-sop` carried nulls in 61 of its 63 chunks. The cause is a ligature the
    font maps to no Unicode codepoint, so "operation" arrives with a NULL byte where
    the "ti" should be.

    It is invisible in every normal view. A terminal renders NULL as nothing, so the
    word reads as "operaon" and looks like a typo in the source document rather than a
    defect in the extraction. It also made grep report the output as a binary file,
    which is the only reason it got noticed at all.

    It has to be removed before indexing, because a NULL inside a word produces a
    token no query will ever match and an embedding of text that does not exist. It
    cannot be repaired: which characters were dropped is not recoverable from what is
    left, and guessing "ti" because it is usually "ti" would be inventing source text.

    So the count is returned rather than swallowed. A high rate is a converter failing
    on a document, which is exactly the signal the converter axis exists to expose,
    and it belongs in the parsing-quality table rather than in a silent cleanup.
    """
    count = text.count(chr(0))
    return (text.replace(chr(0), ""), count) if count else (text, 0)


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
        unmapped = 0
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text, dropped = strip_unmapped_glyphs((page.extract_text() or "").strip())
            unmapped += dropped
            if text.strip():
                found.append(Block(text=text, page=number, kind=None))
        if unmapped:
            logger.warning(
                "convert.unmapped_glyphs",
                extra={
                    "document": path.stem,
                    "converter": self.name,
                    "count": unmapped,
                },
            )
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
        return docling_items_to_blocks(document.iterate_items(), document, document_id=path.stem)


def _page_of(item: object) -> int | None:
    """Docling carries page numbers on provenance entries rather than on the item."""
    for entry in getattr(item, "prov", None) or []:
        number = getattr(entry, "page_no", None)
        if isinstance(number, int) and number >= 1:
            return number
    return None


def docling_items_to_blocks(
    items: Iterable[tuple[object, int | None]], document: object, *, document_id: str
) -> list[Block]:
    """Blocks from `(item, level)` pairs, whatever Docling yielded them from.

    Split out from `DoclingConverter` so the table rule is testable without loading
    the models. Tables carry no `.text` attribute at all; their content lives in the
    grid and reaches text only through export. A converter loop that reads `.text`
    alone therefore drops every table in the corpus silently, which is what this
    code did between M0.2 and M1.2, and why the kind vocabulary the splitters key
    on never contained a real TableItem until it was fixed.
    """
    found: list[Block] = []
    for item, level in items:
        page = _page_of(item)
        if page is None:
            # A block whose page cannot be recovered is dropped, loudly. Keeping
            # it would put text in the index that no citation can ever point at,
            # which is worse than a retrieval miss because it looks like success.
            logger.warning(
                "convert.block_without_page",
                extra={"document": document_id, "text": _preview(item)},
            )
            continue
        text = _docling_text(item, document)
        if not text:
            continue
        text, _ = strip_unmapped_glyphs(text)
        if not text:
            continue
        found.append(Block(text=text, page=page, kind=type(item).__name__, level=level))
    return found


def _preview(item: object) -> str:
    return ((getattr(item, "text", "") or "")[:60]).strip()


def _docling_text(item: object, document: object) -> str:
    """Text of one Docling item, exporting tables through their markdown form.

    The export needs the document root for its hyperlink resolution, which is why
    it is carried alongside the items rather than read off each item alone.
    """
    text = (getattr(item, "text", "") or "").strip()
    if text:
        return text
    export = getattr(item, "export_to_markdown", None)
    if export is None or not hasattr(item, "data"):
        return ""
    try:
        return (export(document) or "").strip()
    except Exception:
        logger.exception("convert.table_export_failed")
        return ""


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


class MarkerConverter:
    """Marker: deep-learning layout and OCR, tuned for PDFs to markdown.

    It runs its own model stack on every page whether the page needs it or not,
    which is what makes it the expensive arm rather than a variant of Docling.
    Its native block tree nests lines and spans inside every text block, so taking
    `children` whole would index every sentence several times over; only leaf
    content types are taken and containers are left to their children.

    Marker labels headings and tables with its own type names. The blocks carry
    Docling's names instead, because the splitting axis keys on kinds, and a
    splitter keyed to one converter's vocabulary quietly disables structure-aware
    chunking for every other converter. The vocabulary is shared infrastructure,
    like the chunk type itself.
    """

    name = "marker"
    version = "2.0.0"

    # Marker type name to the shared vocabulary. Types absent here are containers
    # (lines, spans, groups) whose content arrives through their parents.
    KINDS = {
        "SectionHeader": "SectionHeaderItem",
        "Text": "TextItem",
        "ListItem": "ListItem",
        "Table": "TableItem",
        "Caption": "CaptionItem",
        "Footnote": "FootnoteItem",
    }

    def __init__(self) -> None:
        self._converter = None

    def _build(self):
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        return PdfConverter(artifact_dict=create_model_dict())

    def blocks(self, path: Path) -> list[Block]:
        if self._converter is None:
            self._converter = self._build()

        document = self._converter.build_document(str(path))
        found: list[Block] = []
        for block in marker_document_to_blocks(document):
            text, _ = strip_unmapped_glyphs(block.text)
            if not text:
                continue
            found.append(replace(block, text=text))
        return found


def marker_document_to_blocks(document: object) -> list[Block]:
    """Blocks from a Marker document object, testable without Marker installed.

    Marker's `raw_text` needs the document root to resolve its own block ids, so
    the pages are walked with it carried along. A page id of zero is the first
    page in every Marker schema this was checked against, and Block pages are one
    based, which is why the increment happens here rather than at the caller.
    """
    kinds = MarkerConverter.KINDS
    found: list[Block] = []
    for page in getattr(document, "pages", []):
        for item in getattr(page, "children", []):
            kind = kinds.get(str(getattr(item, "block_type", "")).split(".")[-1])
            if kind is None:
                continue
            text = str(item.raw_text(document) or "").strip()
            if not text:
                continue
            found.append(Block(text=text, page=item.page_id + 1, kind=kind))
    return found


class MinerUConverter:
    """MinerU: layout analysis plus OCR, built for scanned and mixed documents.

    The pipeline backend runs per-page layout detection, then OCR where the text
    layer does not carry the page. It writes its results as files, including a
    content list whose entries name their own page, so the adapter drives the
    installed command rather than private internals: MinerU's Python surface has
    been rewritten between minor versions, and the content list is the only part
    of it documented as an output contract.

    Like Marker's, its output kinds are mapped onto the shared vocabulary.
    """

    name = "mineru"
    version = "3.4.5"

    def blocks(self, path: Path) -> list[Block]:
        executable = shutil.which("mineru")
        if executable is None:
            raise RuntimeError(
                f"{self.name} is not installed in this environment; convert through "
                "its own environment with python -m dastavez.offline_convert"
            )

        with tempfile.TemporaryDirectory(prefix="dastavez-mineru-") as out:
            completed = subprocess.run(
                [
                    executable,
                    "-p", str(path),
                    "-o", out,
                    "-b", "pipeline",
                    "-m", "auto",
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"mineru failed on {path.stem} ({completed.returncode}): "
                    f"{completed.stderr[-2000:]}"
                )
            rows = _mineru_content_list(Path(out))
            if rows is None:
                raise RuntimeError(f"mineru wrote no content list for {path.stem}")

        unmapped = 0
        found: list[Block] = []
        for block in content_list_to_blocks(rows):
            text, dropped = strip_unmapped_glyphs(block.text)
            unmapped += dropped
            if not text:
                continue
            found.append(replace(block, text=text))
        if unmapped:
            logger.warning(
                "convert.unmapped_glyphs",
                extra={"document": path.stem, "converter": self.name, "count": unmapped},
            )
        return found


def _mineru_content_list(out_dir: Path) -> list[dict] | None:
    """The one content list under the run directory, however MinerU nested it."""
    candidates = sorted(out_dir.rglob("*_content_list.json"))
    candidates = [c for c in candidates if "v2" not in c.stem]
    for candidate in candidates:
        try:
            rows = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return None


def content_list_to_blocks(rows: Iterable[dict]) -> list[Block]:
    """Blocks from a MinerU content list, testable without MinerU installed.

    Entries typed `image` are skipped: this project cites text against pages, and
    a picture path in the index would be a citation pointing at a file. Tables
    arrive as HTML in `table_body`; the tags are stripped rather than kept because
    markup tokens pollute lexical retrieval and no query contains them.
    """
    found: list[Block] = []
    for row in rows:
        kind = row.get("type")
        page = row.get("page_idx")
        if not isinstance(page, int) or page < 0:
            logger.warning("convert.block_without_page", extra={"type": kind})
            continue

        if kind == "table":
            body = re.sub(r"<[^>]+>", " ", str(row.get("table_body", "")))
            text = re.sub(r"\s+", " ", body).strip()
            kind_name = "TableItem"
            level = None
        elif kind == "text":
            text = str(row.get("text", "")).strip()
            raw_level = row.get("text_level")
            level = int(raw_level) if isinstance(raw_level, (int, float)) else None
            kind_name = "SectionHeaderItem" if level is not None else "TextItem"
        else:
            continue

        if text:
            found.append(Block(text=text, page=page + 1, kind=kind_name, level=level))
    return found


def converters() -> Iterator[Converter]:
    """Everything the ablation can vary along the converter axis.

    Marker and MinerU need an environment of their own (see the module docstring).
    Against this environment they read the cache written by
    `python -m dastavez.offline_convert`, and they say so when asked to convert
    live without their dependency. They are never stubbed: a stub that returns
    nothing is indistinguishable in a results table from a converter that failed.
    """
    yield PyPdfConverter()
    yield DoclingConverter(ocr=True)
    yield DoclingConverter(ocr=False)
    yield RoutedDoclingConverter()
    yield MarkerConverter()
    yield MinerUConverter()

"""Turning blocks into chunks, without losing the page they came from.

Two splitters, and they are an ablation axis rather than a choice made here. The
replicated Portuguese study found that chunking contributed more than the converter,
which is the hypothesis M6 tests on Indic documents, so both have to exist and be
comparable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from dastavez.chunks import Block, Chunk

# Characters, not tokens. A token count would need a tokenizer at ingestion time, and
# would tie chunk boundaries to whichever model happened to be current, which is
# exactly the kind of hidden coupling that makes an ablation unreproducible a year
# later. Characters are stable, and the retriever is what cares about token budget.
TARGET_CHARS = 1200
OVERLAP_CHARS = 150

# Docling's own type names for blocks that open a section. Used by the
# section-respecting splitter to decide where a chunk may not be cut.
HEADING_KINDS = frozenset({"SectionHeaderItem", "TitleItem"})


def fixed_size(document_id: str, blocks: Iterable[Block]) -> Iterator[Chunk]:
    """Pack blocks to a character target, ignoring structure entirely.

    This is the baseline arm. It will split a table down the middle and separate a
    heading from the rule it introduces, and that is the point: the section-respecting
    arm has to beat something.
    """
    yield from _pack(document_id, list(blocks), respect_sections=False)


def section_respecting(document_id: str, blocks: Iterable[Block]) -> Iterator[Chunk]:
    """Pack to the same target, but never across a heading, and never split a table.

    A converter that labels nothing makes this identical to `fixed_size`, which is
    not a bug. It is the interaction the ablation exists to show: the better splitter
    is unavailable to the floor converter, so converter and splitter are not
    independent and a table reporting them as independent would mislead.
    """
    yield from _pack(document_id, list(blocks), respect_sections=True)


# Paragraph first, then sentence, then a hard cut. A hard cut mid-word is ugly and is
# still better than a chunk three times the target, which is what happens without it.
_PARAGRAPH = "\n\n"
# Devanagari danda as well as the Latin sentence enders, because half this corpus is
# Hindi and a splitter that only knows about full stops will treat a Hindi paragraph
# as one unsplittable sentence.
_SENTENCE = re.compile(r"(?<=[.!?।])\s+")


def _divide(block: Block, *, respect_sections: bool) -> list[Block]:
    """Split one oversized block, keeping its page.

    Needed because the packer only ever splits between blocks. The floor converter
    emits exactly one block per page, so without this every page became one chunk no
    matter what the target said: measured at a 2341 character mean against a 1200
    target, with a 3425 character maximum.

    That is not merely coarse. It confounds two axes of the ablation. The floor
    converter would have scored badly for producing page-sized chunks rather than for
    extracting text badly, and the results table would have attributed a chunking
    failure to the converter.
    """
    if len(block.text) <= TARGET_CHARS:
        return [block]

    # A table is atomic to the section-respecting arm and is not to the naive one.
    # That difference is the ablation, not an inconsistency: cutting a table puts the
    # header row in one chunk and the numbers in another, and naive fixed-size
    # splitting is exactly the strategy that does that. Making both arms protect
    # tables would delete the contrast the study is trying to measure.
    if respect_sections and block.kind == "TableItem":
        return [block]

    pieces: list[str] = []
    for paragraph in block.text.split(_PARAGRAPH):
        if len(paragraph) <= TARGET_CHARS:
            pieces.append(paragraph)
            continue
        current = ""
        for sentence in _SENTENCE.split(paragraph):
            if current and len(current) + len(sentence) > TARGET_CHARS:
                pieces.append(current)
                current = sentence
            elif len(sentence) > TARGET_CHARS:
                for start in range(0, len(sentence), TARGET_CHARS):
                    pieces.append(sentence[start : start + TARGET_CHARS])
                current = ""
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)

    return [
        Block(text=piece.strip(), page=block.page, kind=block.kind, level=block.level)
        for piece in pieces
        if piece.strip()
    ]


def _pack(document_id: str, blocks: list[Block], *, respect_sections: bool) -> Iterator[Chunk]:
    ordinal = 0
    buffer: list[Block] = []
    size = 0

    def flush() -> Iterator[Chunk]:
        nonlocal ordinal, buffer, size
        if not buffer:
            return
        text = "\n\n".join(b.text for b in buffer).strip()
        if text:
            yield Chunk(
                document_id=document_id,
                ordinal=ordinal,
                text=text,
                page=min(b.page for b in buffer),
                end_page=max(b.page for b in buffer),
                kinds=tuple(sorted({b.kind for b in buffer if b.kind})),
            )
            ordinal += 1
        buffer, size = [], 0

    divided = [p for b in blocks for p in _divide(b, respect_sections=respect_sections)]
    for block in divided:
        opens_section = respect_sections and block.kind in HEADING_KINDS
        is_table = block.kind == "TableItem"

        if opens_section and buffer:
            yield from flush()

        # A table is emitted whole even when it busts the target. A table cut in half
        # is worse than a long chunk: the header row lands in one chunk and the
        # numbers in another, and neither answers a question about either.
        if respect_sections and is_table:
            yield from flush()
            buffer, size = [block], len(block.text)
            yield from flush()
            continue

        if size + len(block.text) > TARGET_CHARS and buffer:
            tail = buffer[-1] if OVERLAP_CHARS and len(buffer[-1].text) <= OVERLAP_CHARS else None
            yield from flush()
            if tail is not None:
                buffer, size = [tail], len(tail.text)

        buffer.append(block)
        size += len(block.text)

    yield from flush()


SPLITTERS = {"fixed": fixed_size, "section": section_respecting}

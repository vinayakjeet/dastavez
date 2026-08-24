"""Parsing quality, scored the way OmniDocBench defines it.

Three numbers per converter per page, each against the hand-curated gold page:

- text similarity: 1 minus the Levenshtein edit distance between the page's
  texts, normalized by the gold length. OmniDocBench scores text with a
  normalized edit distance (CVPR 2025, "OmniDocBench", text metric); this is
  the same distance reported as a similarity so every metric reads the same
  way, higher is better.
- table TEDS: tree edit distance similarity over the table's structure tree,
  from PubTabNet (Zhong et al., 2019) and adopted by OmniDocBench for tables.
  Cells compare on text as well as position.
- reading order: Kendall tau between the gold-order ranks of the converter's
  blocks, matched by text. OmniDocBench evaluates reading order over block
  sequences; this is a rank-correlation formulation of the same idea, stated
  here rather than implied.

What these scores do not predict: downstream quality. A page can parse at 0.98
and still retrieve badly, and the Portuguese admin-docs study plus "OCR Hinders
RAG" (arXiv 2412.02592) both found parsing scores a weak predictor of answer
quality, which is why M6 measures the pipeline end to end instead of stopping
here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Cell text shorter than this is compared exactly; longer text is compared by
# prefix so a truncated cell still matches its gold.
_CELL_MATCH_CHARS = 40


@dataclass(frozen=True)
class GoldTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def width(self) -> int:
        return len(self.header)


def normalize(text: str) -> str:
    """What a character-level comparison should treat as the same character.

    NFC because the corpus mixes decomposed and composed Devanagari. Whitespace
    is layout, not content. Dashes are left alone: the gold keeps the source's
    en dashes, and a converter that turns them into hyphens should pay for it.
    """
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def text_similarity(output: str, gold: str) -> float:
    """1 - normalized edit distance, so 1.0 is a perfect transcription."""
    a, b = normalize(output), normalize(gold)
    if not b:
        return 1.0 if not a else 0.0
    return 1.0 - levenshtein(a, b) / len(b)


def parse_markdown_table(text: str) -> GoldTable | None:
    """A GFM pipe table from block text, or None when the text is not one."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    rows = [line for line in lines if line.startswith("|")]
    if len(rows) < 2:
        return None
    if not re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", rows[1]):
        return None

    def cells(line: str) -> tuple[str, ...]:
        parts = line.strip().strip("|").split("|")
        return tuple(normalize(cell.replace("<br>", " ")) for cell in parts)

    header = cells(rows[0])
    body = [cells(line) for line in rows[2:]]
    return GoldTable(header=header, rows=tuple(body))


def _table_nodes(table: GoldTable) -> list[tuple[str, str]]:
    """The tree as (label, text) nodes in document order: table, rows, cells.

    TEDS compares trees, and a flat list in order is the tree for a table whose
    only nesting is table over row over cell, which is what both the gold
    format and every converter's markdown output produce.
    """
    nodes = [("table", "")]
    nodes.append(("tr", ""))
    for cell in table.header:
        nodes.append(("td", normalize(cell)))
    for row in table.rows:
        nodes.append(("tr", ""))
        for cell in row:
            nodes.append(("td", normalize(cell)))
    return nodes


def teds(output: str, gold: str) -> float:
    """Tree-edit-distance similarity between two markdown tables.

    Per PubTabNet: TEDS = 1 - edit_distance / max(nodes on either side), so an
    inserted or deleted row costs its cells and a changed cell costs one.
    """
    predicted, reference = parse_markdown_table(output), parse_markdown_table(gold)
    if predicted is None or reference is None:
        return 0.0

    a, b = _table_nodes(predicted), _table_nodes(reference)
    distance = _node_sequence_distance(a, b)
    return 1.0 - distance / max(len(a), len(b))


def _node_sequence_distance(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> int:
    """Edit distance over the node sequences, where a node matches on label and
    cell text. For table-shaped trees this equals the tree edit distance: every
    insertion or deletion of a node in sequence is an insertion or deletion in
    the tree, and no cross-order mapping can do better because both sequences
    are in document order."""
    previous = list(range(len(b) + 1))
    for i, node_a in enumerate(a, start=1):
        current = [i]
        for j, node_b in enumerate(b, start=1):
            if node_a == node_b or node_a[0] == node_b[0] and _cell_close(node_a[1], node_b[1]):
                cost = 0
            else:
                cost = 1
            current.append(
                min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def _cell_close(a: str, b: str) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return a.startswith(b[:_CELL_MATCH_CHARS]) or b.startswith(a[:_CELL_MATCH_CHARS])


def split_gold_blocks(markdown: str) -> list[str]:
    """Gold page as blocks: headings, paragraphs, and whole tables.

    Blank lines separate paragraphs the way they do in every converter's
    output; a table is one block however many rows it has, because a table cut
    into row-blocks would let reading order look better than the converter
    deserves.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_table = False

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append("\n".join(current))
            current = []

    for line in markdown.splitlines():
        stripped = line.strip()
        is_table = stripped.startswith("|")
        if is_table != in_table:
            flush()
            in_table = is_table
        if stripped:
            current.append(line)
        else:
            flush()
    flush()
    return [block for block in blocks if normalize(block)]


def reading_order_score(output_blocks: list[str], gold_blocks: list[str]) -> float | None:
    """Kendall tau between gold ranks of matched blocks, or None if unmeasurable.

    A block matches the first gold block whose normalized text it starts or
    ends with, or vice versa. Fewer than two concordant/discordant pairs is not
    an order measurement, and None says so rather than inventing a number.
    """
    normalized_gold = [normalize(block) for block in gold_blocks]
    ranks: list[int] = []
    for block in output_blocks:
        text = normalize(block)
        if not text:
            continue
        for index, gold_text in enumerate(normalized_gold):
            if not gold_text:
                continue
            if text.startswith(gold_text[:_CELL_MATCH_CHARS]) or gold_text.startswith(
                text[:_CELL_MATCH_CHARS]
            ):
                ranks.append(index)
                break

    concordant = discordant = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[i] < ranks[j]:
                concordant += 1
            elif ranks[i] > ranks[j]:
                discordant += 1

    pairs = concordant + discordant
    if pairs == 0:
        return None
    return (concordant - discordant) / pairs

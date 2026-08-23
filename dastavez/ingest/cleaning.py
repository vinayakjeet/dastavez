"""The cleaning axis: repeated furniture out, split words rejoined.

Two arms. `none` passes blocks through untouched and is the honest baseline. `strip`
removes the header and footer lines a document repeats on every page, and rejoins
words broken across a line by a hyphen.

Both operations are worth measuring rather than assuming. A repeated footer is real
text that a retriever will happily match, so the same boilerplate appearing in every
chunk pulls unrelated pages toward every query. De-hyphenation matters more than it
looks in this corpus: a scheme name split across a line break becomes two tokens that
BM25 will never match and that an embedding sees as noise.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from dastavez.chunks import Block

# A line has to repeat on this fraction of pages before it counts as furniture. Set
# from the shape of the problem rather than tuned: a header appears on nearly every
# page by definition, and a threshold low enough to catch a heading that happens to
# recur three times would delete content.
FURNITURE_SHARE = 0.6

# Only short lines are candidates. A long line repeating across pages is more likely a
# standard clause the document genuinely restates, and deleting a clause because it is
# repeated is exactly the kind of cleaning that quietly changes an answer.
FURNITURE_MAX_CHARS = 80

# A hyphen at end of line, followed by a lowercase continuation. Restricted to
# lowercase because "PM-" at a line end is a scheme name, not a broken word.
HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*([a-zऀ-ॿ])")


def no_cleaning(blocks: Iterable[Block]) -> list[Block]:
    return list(blocks)


def strip_furniture(blocks: Iterable[Block]) -> list[Block]:
    rows = list(blocks)
    if not rows:
        return rows

    furniture = _repeated_lines(rows)
    cleaned: list[Block] = []
    for block in rows:
        text = "\n".join(
            line for line in block.text.splitlines() if line.strip() not in furniture
        )
        text = HYPHEN_BREAK.sub(r"\1\2", text).strip()
        if text:
            cleaned.append(
                Block(text=text, page=block.page, kind=block.kind, level=block.level)
            )
    return cleaned


def _repeated_lines(blocks: list[Block]) -> set[str]:
    """Lines that appear on enough distinct pages to be page furniture.

    Counted per page rather than per occurrence. A line appearing five times on one
    page is a list marker; the same line appearing once on twenty pages is a header,
    and only the second is furniture.
    """
    pages = {b.page for b in blocks}
    if len(pages) < 3:
        # Too few pages to tell a header from a coincidence. Doing nothing is the
        # correct answer rather than guessing on a two page document.
        return set()

    seen: Counter[str] = Counter()
    for page in pages:
        lines = {
            line.strip()
            for b in blocks
            if b.page == page
            for line in b.text.splitlines()
            if line.strip() and len(line.strip()) <= FURNITURE_MAX_CHARS
        }
        seen.update(lines)

    threshold = max(3, int(len(pages) * FURNITURE_SHARE))
    return {line for line, count in seen.items() if count >= threshold}


CLEANERS = {"none": no_cleaning, "strip": strip_furniture}

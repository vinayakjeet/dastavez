"""What a chunk is, and the one thing that must never be lost.

Page provenance is the invariant. Every claim this system makes has to resolve to a
page of a named document, and a chunk that has forgotten which page it came from can
never get that back: the text is in the index, the citation is not recoverable, and
the failure only shows up as a citation that points at the wrong place.

So `page` is not optional on this type. A converter that cannot say which page a
block came from fails at ingestion rather than producing chunks that are useful for
retrieval and useless for citation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Block:
    """One unit of content as a converter saw it, before chunking.

    `kind` is the converter's own label (heading, paragraph, table, caption, and so
    on). It is the thing that makes section-respecting splitting possible rather than
    guessing at newlines, and it is why the converter axis and the splitting axis are
    separate axes in the ablation: a converter that loses block types makes the
    better splitter unavailable.

    Converters that cannot label blocks report `kind=None`. That is honest and it is
    also the floor baseline's whole character.
    """

    text: str
    page: int
    kind: str | None = None
    level: int | None = None


@dataclass(frozen=True)
class Chunk:
    document_id: str
    ordinal: int
    text: str
    page: int
    end_page: int
    kinds: tuple[str, ...] = ()
    section_path: str | None = None
    scheme: str | None = None
    doc_kind: str | None = None

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"{self.document_id}#{self.ordinal}: page must be 1 or greater")
        if self.end_page < self.page:
            raise ValueError(f"{self.document_id}#{self.ordinal}: end_page precedes page")
        if not self.text.strip():
            raise ValueError(f"{self.document_id}#{self.ordinal}: empty chunk")

    @property
    def spans_pages(self) -> bool:
        """A chunk crossing a page boundary cites a range, not a page.

        Worth knowing per chunk rather than computing later: a citation that says
        "page 7" for text that started on page 6 is wrong in the way a reader
        notices immediately.
        """
        return self.end_page > self.page


@dataclass
class IngestRun:
    """One pass of one configuration over one document set.

    The config hash is what makes two runs comparable, and it is recorded here rather
    than derived later, because the ablation's entire value is being able to say which
    configuration produced which number.
    """

    run_id: str
    config_hash: str
    converter: str
    splitter: str
    cleaning: str
    metadata: str
    started: str
    documents: list[str] = field(default_factory=list)

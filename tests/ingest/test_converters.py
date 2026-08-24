from __future__ import annotations

from types import SimpleNamespace

import pytest

from dastavez.ingest.chunking import fixed_size, section_respecting
from dastavez.ingest.converters import (
    content_list_to_blocks,
    docling_items_to_blocks,
    marker_document_to_blocks,
)


def make_docling_item(name: str, page: int, *, text: str = "", export=None):
    """A stand-in whose class name matches the Docling type being faked.

    `docling_items_to_blocks` labels blocks by `type(item).__name__`, so the kind
    under test only exists if the stand-in carries the real name.
    """

    def __init__(self) -> None:
        self.text = text
        self.prov = [SimpleNamespace(page_no=page)]
        self.data = object()
        self._export = export

    body = {
        "__init__": __init__,
        "export_to_markdown": lambda self, root: (
            self._export(root) if callable(self._export) else self._export
        ),
    }
    return type(name, (), body)


def test_docling_keeps_a_table_whose_text_is_empty():
    """A TableItem carries no `.text`; its content only exists through export.

    The loop that read `.text` alone dropped every table in the corpus without an
    error, which is how the section-respecting splitter came to have no real
    TableItem to protect until M1.2.
    """
    markdown = "| Sl | Particulars |\n|---|---|\n| 1 | Rates |\n"
    table = make_docling_item("TableItem", 9, text="", export=lambda root: markdown)

    blocks = docling_items_to_blocks([(table(), None)], document=None, document_id="d")

    assert len(blocks) == 1
    assert blocks[0].kind == "TableItem"
    assert blocks[0].page == 9
    assert "Rates" in blocks[0].text


def test_docling_still_drops_an_item_without_a_page():
    picture = make_docling_item("PictureItem", 0)
    blocks = docling_items_to_blocks([(picture(), None)], document=None, document_id="d")
    assert blocks == []


def test_docling_survives_a_table_export_that_fails():
    """One unreadable table must not cost the rest of the page its blocks."""
    broken = make_docling_item(
        "TableItem", 4, export=lambda root: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    after = make_docling_item("TextItem", 4, text="next")

    blocks = docling_items_to_blocks(
        [(broken(), None), (after(), None)], document=None, document_id="d"
    )

    assert [b.text for b in blocks] == ["next"]


class FakeMarkerBlock:
    def __init__(self, block_type: str, page_id: int, text: str):
        self.block_type = block_type
        self.page_id = page_id
        self._text = text

    def raw_text(self, document: object) -> str:
        return self._text


def test_marker_takes_content_blocks_and_leaves_containers():
    """Lines and spans duplicate what their parents already carry; indexing both
    would put every sentence in the index several times over."""
    document = SimpleNamespace(
        pages=[
            SimpleNamespace(
                children=[
                    FakeMarkerBlock("SectionHeader", 0, "Eligibility"),
                    FakeMarkerBlock("Text", 0, "Farmers holding land."),
                    FakeMarkerBlock("Line", 0, "Farmers holding land."),
                    FakeMarkerBlock("Span", 0, "Farmers holding land."),
                ]
            ),
            SimpleNamespace(children=[FakeMarkerBlock("ListItem", 1, "i) first")]),
        ]
    )

    blocks = marker_document_to_blocks(document)

    assert [(b.kind, b.page, b.text) for b in blocks] == [
        ("SectionHeaderItem", 1, "Eligibility"),
        ("TextItem", 1, "Farmers holding land."),
        ("ListItem", 2, "i) first"),
    ]


def test_marker_pages_are_one_based():
    document = SimpleNamespace(
        pages=[
            SimpleNamespace(children=[FakeMarkerBlock("Text", 8, "later")]),
            SimpleNamespace(children=[FakeMarkerBlock("Table", 9, "| a | b |")]),
        ]
    )

    blocks = marker_document_to_blocks(document)

    assert [b.page for b in blocks] == [9, 10]
    assert blocks[1].kind == "TableItem"


def test_mineru_maps_headings_tables_and_plain_text():
    rows = [
        {"type": "text", "text": "Operational Guidelines", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "The scheme pays Rs 6000.", "page_idx": 0},
        {
            "type": "table",
            "table_body": "<table><tr><td>Instalment</td></tr></table>",
            "page_idx": 2,
        },
        {"type": "image", "img_path": "images/figure.jpg", "page_idx": 2},
    ]

    blocks = content_list_to_blocks(rows)

    assert [(b.kind, b.level) for b in blocks] == [
        ("SectionHeaderItem", 1),
        ("TextItem", None),
        ("TableItem", None),
    ]
    assert "Instalment" in blocks[2].text
    assert "<" not in blocks[2].text


def test_mineru_drops_an_entry_without_a_page_rather_than_guessing():
    rows = [{"type": "text", "text": "orphaned"}]

    assert content_list_to_blocks(rows) == []


@pytest.mark.parametrize("splitter", [fixed_size, section_respecting], ids=["fixed", "section"])
def test_mineru_blocks_keep_pages_through_chunking(splitter):
    rows = [{"type": "text", "text": f"paragraph {n}", "page_idx": n} for n in range(3)]

    chunks = list(splitter("doc", content_list_to_blocks(rows)))

    assert {c.page for c in chunks} >= {1}

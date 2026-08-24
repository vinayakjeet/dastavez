from __future__ import annotations

import pytest

from dastavez.parsing import (
    levenshtein,
    normalize,
    parse_markdown_table,
    reading_order_score,
    split_gold_blocks,
    teds,
    text_similarity,
)


def test_levenshtein_counts_the_obvious_edits():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0


def test_text_similarity_is_one_for_identical_text_and_falls_with_edits():
    gold = "The scheme pays Rs.6000/- per year."
    assert text_similarity(gold, gold) == 1.0
    assert 0.0 < text_similarity("The scheme pays Rs.6000 per year.", gold) < 1.0
    assert text_similarity("unrelated", gold) < text_similarity(
        "The scheme pays Rs.6000 per year.", gold
    )


def test_text_similarity_ignores_whitespace_and_unicode_form_but_not_dashes():
    assert (
        text_similarity("a  b\n\tc", "a b c") == 1.0
    )
    assert text_similarity("Yojana \u2013 Urban", "Yojana - Urban") < 1.0


def test_parse_markdown_table_reads_header_and_rows_and_rejects_prose():
    table = parse_markdown_table(
        "| Age | Rate |\n|---|---|\n| 18 | 55 |\n| 19 | 58 |"
    )
    assert table is not None
    assert table.header == ("Age", "Rate")
    assert table.rows == (("18", "55"), ("19", "58"))

    assert parse_markdown_table("just a paragraph") is None
    assert parse_markdown_table("| a | b |") is None


def test_teds_is_one_for_the_same_table_and_penalises_a_lost_row():
    gold = "| Age | Rate |\n|---|---|\n| 18 | 55 |\n| 19 | 58 |"

    assert teds(gold, gold) == 1.0

    lost_row = "| Age | Rate |\n|---|---|\n| 18 | 55 |"
    assert 0.0 < teds(lost_row, gold) < 1.0

    assert teds("| a | b |\n|---|---|\n| 1 | 2 |", gold) < teds(lost_row, gold)


def test_teds_tolerates_a_truncated_cell_but_not_a_missing_column():
    gold = "| Particulars | Page No |\n|---|---|\n"
    gold += "| Application by prospective developers for registration of AHP projects | 7 |\n"
    gold += "| Process for Beneficiary Application under AHP vertical | 9 |"

    truncated = "| Particulars | Page No |\n|---|---|\n"
    truncated += (
        "| Application by prospective developers for registration of AHP projects"
        " on the Unified Web Portal | 7 |\n"
    )
    truncated += "| Process for Beneficiary Application under AHP vertical | 9 |"
    assert teds(truncated, gold) == 1.0

    narrow = "| Particulars |\n|---|\n| Application by prospective developers |"
    assert teds(narrow, gold) < 1.0


def test_teds_is_zero_when_either_side_has_no_table():
    assert teds("not a table", "| a |\n|---|") == 0.0
    assert teds("| a |\n|---|", "not a table") == 0.0


def test_split_gold_blocks_keeps_a_table_whole_and_paragraphs_apart():
    page = (
        "# Title\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
        "\n"
        "Last paragraph.\n"
    )

    blocks = split_gold_blocks(page)

    assert len(blocks) == 4
    assert blocks[2].count("|") == 12


def test_reading_order_is_one_for_identical_order_and_falls_with_a_swap():
    gold = ["Alpha.", "Beta.", "Gamma.", "Delta."]

    assert reading_order_score(gold, gold) == 1.0

    swapped = ["Alpha.", "Beta.", "Delta.", "Gamma."]
    assert reading_order_score(swapped, gold) == pytest.approx(2 / 3)

    reversed_blocks = list(reversed(gold))
    assert reading_order_score(reversed_blocks, gold) == -1.0


def test_reading_order_is_none_when_nothing_matches():
    assert reading_order_score(["totally different"], ["Alpha.", "Beta."]) is None
    assert reading_order_score([], ["Alpha."]) is None


def test_normalize_folds_whitespace_and_unicode_form_only():
    assert normalize("  a\u0062\tc  ") == "ab c"
    assert normalize("a\u2013b") == "a\u2013b"

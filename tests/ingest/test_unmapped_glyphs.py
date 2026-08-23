from __future__ import annotations

from dastavez.ingest.converters import strip_unmapped_glyphs


def test_a_null_inside_a_word_is_removed_and_counted():
    """pypdf emits NULL where a ligature maps to no codepoint, so "operation"
    arrives as opera<NUL>on and reads as a typo in the source rather than a defect
    in the extraction."""
    text, count = strip_unmapped_glyphs("long-term opera" + chr(0) + "on and maintenance")

    assert chr(0) not in text
    assert count == 1
    assert text == "long-term operaon and maintenance"


def test_clean_text_is_returned_unchanged_and_counts_zero():
    original = "All Persons who paid Income Tax in last assessment year."
    text, count = strip_unmapped_glyphs(original)

    assert text is original
    assert count == 0


def test_the_count_is_returned_rather_than_swallowed():
    """A high rate is a converter failing on a document, which is the signal the
    converter axis exists to expose. Silently cleaning it hides that."""
    _, count = strip_unmapped_glyphs(chr(0).join(["a", "b", "c", "d"]))

    assert count == 3


def test_devanagari_survives():
    text, count = strip_unmapped_glyphs("प्रधानमंत्री किसान सम्मान निधि")

    assert count == 0
    assert "किसान" in text

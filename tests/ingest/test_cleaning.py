from __future__ import annotations

from dastavez.chunks import Block
from dastavez.ingest.cleaning import no_cleaning, strip_furniture


def paged(*pages: str) -> list[Block]:
    return [Block(text=t, page=n, kind=None) for n, t in enumerate(pages, start=1)]


def test_a_header_repeated_on_every_page_is_removed():
    blocks = paged(
        "Ministry of Agriculture\nEligibility rules follow.",
        "Ministry of Agriculture\nExclusions follow.",
        "Ministry of Agriculture\nGrievances follow.",
        "Ministry of Agriculture\nAnnexures follow.",
    )
    cleaned = strip_furniture(blocks)

    assert all("Ministry of Agriculture" not in b.text for b in cleaned)
    assert "Eligibility rules follow." in cleaned[0].text


def test_content_that_merely_repeats_a_few_times_is_kept():
    """A heading recurring three times is not furniture, and deleting a clause
    because it is repeated is exactly the cleaning that quietly changes an answer."""
    blocks = paged(
        "Note\nalpha",
        "Note\nbeta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
    )
    cleaned = strip_furniture(blocks)

    assert any("Note" in b.text for b in cleaned)


def test_a_long_repeated_line_is_kept():
    """A long line recurring across pages is more likely a standard clause the
    document genuinely restates than page furniture."""
    clause = (
        "The competent authority shall verify the particulars furnished by the "
        "applicant before sanctioning any benefit under this scheme."
    )
    blocks = paged(*[f"{clause}\npage {n} body" for n in range(1, 6)])
    cleaned = strip_furniture(blocks)

    assert all(clause in b.text for b in cleaned)


def test_a_word_broken_across_a_line_is_rejoined():
    """A scheme name split by a line break becomes two tokens BM25 never matches."""
    blocks = paged("the bene-\nficiary must apply", "x", "y")
    cleaned = strip_furniture(blocks)

    assert "beneficiary must apply" in cleaned[0].text


def test_a_hyphenated_scheme_name_is_not_rejoined():
    """`PM-` at a line end is a name, not a broken word, so the rule requires a
    lowercase continuation."""
    blocks = paged("under PM-\nKISAN the amount", "x", "y")
    cleaned = strip_furniture(blocks)

    assert "PM-\nKISAN" in cleaned[0].text or "PM-" in cleaned[0].text


def test_too_few_pages_to_tell_a_header_from_a_coincidence():
    blocks = paged("Header\nalpha", "Header\nbeta")
    assert strip_furniture(blocks) == no_cleaning(blocks)


def test_the_none_arm_changes_nothing():
    blocks = paged("Header\nalpha", "Header\nbeta", "Header\ngamma", "Header\ndelta")
    assert no_cleaning(blocks) == blocks

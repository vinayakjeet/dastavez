"""The citation check, and the two kinds of damage it had to learn about.

The check itself is simple: do the numbers a gold answer asserts appear on the
page it cites. What is worth testing is the two corrections that took it from
eight failures to one, because both are the difference between measuring the
labels and measuring the extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_citations import claim_tokens, normalise, supported


class TestWhatCountsAsAClaim:
    def test_short_numbers_are_not_distinctive_enough_to_check(self) -> None:
        """Every page has a 1 and a 2 on it somewhere, so matching on them would
        pass any citation against any page."""
        assert claim_tokens("three equal installments of 3 each") == []

    def test_amounts_and_dates_are_the_claim(self) -> None:
        assert claim_tokens("Rs. 6000 per annum from 01.12.2018") == ["01.12.2018", "6000"]

    def test_thousands_separators_do_not_create_a_different_number(self) -> None:
        """A page writing 10,000 and an answer writing 10000 assert the same thing."""
        assert claim_tokens("Rs. 10,000 or more") == ["10000"]


class TestExtractionDamage:
    def test_a_space_inside_a_date_still_matches(self) -> None:
        """pmkisan-faq-revised page 1 carries "1 .6.2019" because pypdf put a space
        in it. Five gold answers citing that page were flagged as unsupported for
        that reason alone, which is the checker measuring the extraction rather
        than the label."""
        page = "The Scheme was later on revised w.e.f. 1 .6.2019 and extended to all"
        ok, missing = supported("Revised with effect from 1.6.2019.", page)
        assert ok, f"still missing {missing}"

    def test_normalise_only_closes_gaps_between_digits(self) -> None:
        """It must not glue together numbers that are genuinely separate, or a page
        listing 55 and 200 would appear to contain 55200."""
        assert normalise("pays 55 and the total is 110") == "pays 55 and the total is 110"
        assert normalise("1 .6.2019") == "1.6.2019"

    def test_a_genuinely_absent_number_is_still_caught(self) -> None:
        """The corrections must not make the check toothless."""
        ok, missing = supported("The benefit is Rs. 9999 per annum.", "The benefit is Rs. 6000.")
        assert not ok
        assert missing == ["9999"]


class TestNumbersTheQuestionSupplied:
    def test_a_number_from_the_question_is_not_a_claim_about_the_page(self) -> None:
        """"A pension of Rs. 8,000 does not trigger the exclusion" is grounded by a
        page stating the 10,000 threshold. The 8,000 came from the asker, and
        demanding the page contain it would fail every hypothetical question in the
        set."""
        question = "A retired official draws a monthly pension of Rs. 8,000. Does it apply?"
        page = "All superannuated pensioners whose monthly pension is Rs.10,000 or more"
        ok, missing = supported(
            "No. The exclusion applies at Rs. 10,000 per month or more, so Rs. 8,000 does not.",
            page,
            question,
        )
        assert ok, f"still missing {missing}"

    def test_the_answers_own_claim_is_still_checked(self) -> None:
        """Excluding question numbers must not exclude the answer's own assertion."""
        ok, missing = supported(
            "No, the threshold is Rs. 10,000.", "the threshold is Rs. 5000", "Is Rs. 8,000 enough?"
        )
        assert not ok
        assert missing == ["10000"]

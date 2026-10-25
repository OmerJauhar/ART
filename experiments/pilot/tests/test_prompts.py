"""
Tests for Answer Extraction and Prompts
=======================================
"""

import pytest

from experiments.pilot.prompts import extract_answer, format_question
from experiments.pilot.tests.helpers import make_question

LETTERS = ["A", "B", "C", "D"]


class TestExtractAnswer:
    """Test cases for extract_answer."""

    @pytest.mark.parametrize("text, expected", [
        ("Reasoning here.\nANSWER: B", "B"),
        ("ANSWER: C.", "C"),
        ("**ANSWER:** D", "D"),
        ("Answer: (A)", "A"),
        ("Final answer: B", "B"),
        ("  ANSWER:   C  ", "C"),
        ("ANSWER: **B**", "B"),
    ])
    def test_formats(self, text: str, expected: str) -> None:
        """Test accepted answer-line formats."""
        assert extract_answer(text, LETTERS) == expected

    def test_last_answer_line_wins(self) -> None:
        """Test that the last ANSWER line is used."""
        text = "ANSWER: A\nOn reflection that is wrong.\nANSWER: C"

        assert extract_answer(text, LETTERS) == "C"

    @pytest.mark.parametrize("text", [
        None,
        "",
        "I think it is B.",
        "ANSWER: Both are possible",
        "ANSWER: E",
        "ANSWER: <letter>",
        "The ANSWER: B is in the middle of a sentence",
    ])
    def test_unparseable(self, text: str | None) -> None:
        """Test that missing, invalid or out-of-range answers give None."""
        assert extract_answer(text, LETTERS) is None

    def test_lowercase_letter_not_accepted(self) -> None:
        """Test that 'ANSWER: a ...' is not read as option A."""
        assert extract_answer("ANSWER: a primer pair", LETTERS) is None


class TestFormatQuestion:
    """Test cases for format_question."""

    def test_contains_options_and_instruction(self) -> None:
        """Test that the prompt lists lettered options and the answer format."""
        question = make_question(1)

        text = format_question(question)

        assert question.stem in text
        assert "A. option 1-0" in text
        assert "D. option 1-3" in text
        assert '"ANSWER: <letter>"' in text
        assert "one of A, B, C, D" in text
        # The instruction itself must not parse as an answer
        assert extract_answer(text, question.letters) is None

"""
Pilot Prompts
=============

Question formatting, answer extraction, and the fusion prompt.
"""

from __future__ import annotations

import re

from experiments.pilot.data import LETTERS, Question

# One line holding the final answer: "ANSWER: B", "**ANSWER:** B", "Final answer: (B)".
# The letter must be uppercase and stand alone, so "ANSWER: Both" does not count.
_ANSWER_LINE = re.compile(
    r"^[ \t>*_#-]*(?:final[ \t]+)?answer[ \t*_]*[:：][ \t*_(\[]*([A-Z])\b",
    re.IGNORECASE | re.MULTILINE,
)


def format_question(question: Question) -> str:
    """
    Full question text sent to every model and agent.

    It carries the options and the answer-format instruction, so ART's revision step, which
    sees only this text, keeps the same format.
    """
    options = "\n".join(
        f"{letter}. {option}" for letter, option in zip(LETTERS, question.options)
    )
    letters = ", ".join(question.letters)
    return (
        f"{question.stem}\n\n"
        f"Options:\n{options}\n\n"
        "Reason briefly, then end your reply with a final line of the form "
        f'"ANSWER: <letter>", where <letter> is one of {letters}.'
    )


def extract_answer(text: str | None, letters: list[str]) -> str | None:
    """
    Return the letter on the last "ANSWER: X" line, or None if there is no valid one.

    Args:
        text: Model output
        letters: Valid letters for the question

    Returns:
        The answer letter, or None when unparseable
    """
    if not text:
        return None
    matches = [m.group(1) for m in _ANSWER_LINE.finditer(text) if m.group(1).isupper()]
    if not matches:
        return None
    letter = matches[-1]
    return letter if letter in letters else None


FUSION_SYSTEM = (
    "You are an expert adjudicator. Several assistants answered the same multiple-choice "
    "question. Read the question and their responses, decide which option is correct, and "
    "give a final answer of your own. The assistants may all be wrong."
)


def fusion_prompt(question_text: str, responses: list[str]) -> str:
    """User message for the fusion method: the question plus every response."""
    blocks = "\n\n".join(
        f"--- Response {i} ---\n{response}" for i, response in enumerate(responses, start=1)
    )
    return (
        f"QUESTION:\n{question_text}\n\n"
        f"RESPONSES:\n{blocks}\n\n"
        'Reason briefly, then end your reply with a final line of the form "ANSWER: <letter>".'
    )

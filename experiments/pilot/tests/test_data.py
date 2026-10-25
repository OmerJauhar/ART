"""
Tests for Sampling and the Calibration/Test Split
=================================================
"""

from collections import Counter
from typing import Any

import pytest

from experiments.pilot.data import (
    LETTERS,
    allocate,
    build_stem,
    is_text_only,
    sample_questions,
    shuffle_options,
)


def synthetic_rows(sizes: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Rows shaped like LAB-Bench records."""
    return {
        subset: [
            {
                "id": f"{subset}-{i:03d}",
                "question": f"{subset} question {i}",
                "ideal": f"right {i}",
                "distractors": [f"wrong {i}-{k}" for k in range(3)],
                "subtask": "synthetic",
                "protocol": "Step 1: do it",
            }
            for i in range(n)
        ]
        for subset, n in sizes.items()
    }


class TestAllocate:
    """Test cases for allocate."""

    def test_equal_split(self) -> None:
        """Test an even split when every group is large enough."""
        assert allocate(30, {"a": 50, "b": 50, "c": 50}) == {"a": 10, "b": 10, "c": 10}

    def test_caps_small_group(self) -> None:
        """Test that a small group gives its share to the others."""
        assert allocate(100, {"CloningScenarios": 32, "ProtocolQA": 108, "SeqQA": 440}) == {
            "CloningScenarios": 32, "ProtocolQA": 34, "SeqQA": 34,
        }

    def test_too_few(self) -> None:
        """Test that asking for more than available raises."""
        with pytest.raises(ValueError):
            allocate(10, {"a": 3, "b": 3})


class TestSample:
    """Test cases for sample_questions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.rows = synthetic_rows({"CloningScenarios": 32, "ProtocolQA": 108, "SeqQA": 440})

    def test_split_is_first_30_then_70(self) -> None:
        """Test that the first 30 questions are calibration and the last 70 are test."""
        questions = sample_questions(self.rows, seed=0)

        assert len(questions) == 100
        assert [q.split for q in questions] == ["calibration"] * 30 + ["test"] * 70
        assert [q.index for q in questions] == list(range(100))
        assert len({q.id for q in questions}) == 100

    def test_stratified(self) -> None:
        """Test subset counts in the sample and in each split."""
        questions = sample_questions(self.rows, seed=0)

        assert Counter(q.subset for q in questions) == {
            "CloningScenarios": 32, "ProtocolQA": 34, "SeqQA": 34,
        }
        assert Counter(q.subset for q in questions if q.split == "calibration") == {
            "CloningScenarios": 10, "ProtocolQA": 10, "SeqQA": 10,
        }

    def test_deterministic(self) -> None:
        """Test that the same seed gives the same sample and option order."""
        first = sample_questions(self.rows, seed=0)
        second = sample_questions(self.rows, seed=0)
        other = sample_questions(self.rows, seed=1)

        assert [q.to_dict() for q in first] == [q.to_dict() for q in second]
        assert [q.id for q in first] != [q.id for q in other]

    def test_correct_letter_points_at_ideal(self) -> None:
        """Test that the recorded letter is where the ideal answer landed."""
        for question in sample_questions(self.rows, seed=0):
            index = LETTERS.index(question.correct_letter)
            assert question.options[index].startswith("right ")

    def test_options_are_shuffled(self) -> None:
        """Test that the ideal answer is not always first."""
        letters = Counter(q.correct_letter for q in sample_questions(self.rows, seed=0))

        assert len(letters) == 4


class TestFilters:
    """Test cases for the text-only filter and stems."""

    def test_seqqa_gene_lookup_subtasks_excluded(self) -> None:
        """Test that SeqQA subtasks naming a gene without its sequence are dropped."""
        assert not is_text_only("SeqQA", {"subtask": "PCR-gene-enzprimers-v1-public"})
        assert is_text_only("SeqQA", {"subtask": "ORF-seq-AAid-v1-public"})

    def test_cloning_requires_embedded_sequence(self) -> None:
        """Test that CloningScenarios needs the sequence in the question."""
        assert is_text_only("CloningScenarios", {"question": "Plasmid " + "ACGT" * 12})
        assert not is_text_only("CloningScenarios", {"question": "Use Addgene 131001."})

    def test_protocol_inlined(self) -> None:
        """Test that ProtocolQA questions carry their protocol text."""
        stem = build_stem("ProtocolQA", {"question": "What went wrong?", "protocol": "Step 1"})

        assert stem.startswith("Protocol:\nStep 1")
        assert stem.endswith("Question: What went wrong?")

    def test_shuffle_options(self) -> None:
        """Test that option shuffling is seeded per question."""
        row = {"id": "x", "ideal": "right", "distractors": ["a", "b", "c"]}

        options, letter = shuffle_options(row, seed=0)

        assert sorted(options) == ["a", "b", "c", "right"]
        assert options[LETTERS.index(letter)] == "right"
        assert shuffle_options(row, seed=0) == (options, letter)

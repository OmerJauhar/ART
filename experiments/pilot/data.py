"""
Pilot Data
==========

Loads LAB-Bench from HuggingFace at a pinned revision, keeps only questions answerable from
their own text, and draws the fixed stratified sample and calibration/test split.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from experiments.pilot.settings import DATA_DIR

DATASET = "futurehouse/lab-bench"
# HuggingFace commit of futurehouse/lab-bench main, pinned 2026-10-25
REVISION = "5c77cec648430f30611808808861eb86f81d5eaa"
PARQUET_URL = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{subset}/"
    "train-00000-of-00001.parquet"
)

USED_SUBSETS = ("SeqQA", "ProtocolQA", "CloningScenarios")

EXCLUDED_SUBSETS = {
    "FigQA": "needs a figure image",
    "TableQA": "needs table images",
    "LitQA2": "needs literature retrieval (recent papers)",
    "SuppQA": "needs a paper's supplementary material",
    "DbQA": "needs database lookups (e.g. DisGeNet, OMIM)",
}

# These SeqQA subtasks name an E. coli gene without giving its sequence, so they need a lookup
EXCLUDED_SEQQA_SUBTASKS = (
    "PCR-gene-enzprimers-v1-public",
    "PCR-gene-gibshindprimers-v1-public",
    "PCR-gene-gibssmaprimers-v1-public",
    "PCR-geneprimers-enz-v1-public",
)

# CloningScenarios questions embed their plasmid sequences; one instead cites Addgene catalog
# numbers. A question with no run of 40+ nucleotides is treated as needing retrieval.
_EMBEDDED_SEQUENCE = re.compile(r"[ACGTacgt]{40,}")

SAMPLE_SIZE = 100
CALIBRATION_SIZE = 30
LETTERS = string.ascii_uppercase


@dataclass
class Question:
    """One multiple-choice question in the pilot sample."""

    id: str
    subset: str
    subtask: str
    stem: str
    options: list[str]
    correct_letter: str
    split: str
    index: int

    @property
    def letters(self) -> list[str]:
        """Valid answer letters for this question."""
        return list(LETTERS[: len(self.options)])

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


def is_text_only(subset: str, row: dict[str, Any]) -> bool:
    """Return True if the row can be answered from the text supplied with it."""
    if subset == "SeqQA":
        return row["subtask"] not in EXCLUDED_SEQQA_SUBTASKS
    if subset == "CloningScenarios":
        return bool(_EMBEDDED_SEQUENCE.search(row["question"]))
    return subset == "ProtocolQA"


def build_stem(subset: str, row: dict[str, Any]) -> str:
    """Question text shown to models; ProtocolQA gets its protocol inlined."""
    question = str(row["question"]).strip()
    if subset == "ProtocolQA":
        return f"Protocol:\n{str(row['protocol']).strip()}\n\nQuestion: {question}"
    return question


def download_subset(subset: str, data_dir: Path = DATA_DIR) -> Path:
    """Download one subset's parquet file at the pinned revision (cached on disk)."""
    path = data_dir / "lab-bench" / REVISION[:12] / f"{subset}.parquet"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = PARQUET_URL.format(dataset=DATASET, revision=REVISION, subset=subset)
    for attempt in range(4):
        try:
            with httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=True) as c:
                response = c.get(url)
                response.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 3:
                raise
    tmp = path.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.rename(path)
    return path


def load_rows(data_dir: Path = DATA_DIR) -> dict[str, list[dict[str, Any]]]:
    """Load the text-only rows of each used subset, sorted by ID."""
    import pandas as pd

    rows: dict[str, list[dict[str, Any]]] = {}
    for subset in USED_SUBSETS:
        frame = pd.read_parquet(download_subset(subset, data_dir))
        records = []
        for record in frame.to_dict("records"):
            record["distractors"] = [str(d) for d in record["distractors"]]
            if is_text_only(subset, record):
                records.append(record)
        rows[subset] = sorted(records, key=lambda r: r["id"])
    return rows


def allocate(total: int, available: dict[str, int]) -> dict[str, int]:
    """
    Split total as evenly as possible across groups, capped by what each group has.

    Leftover units go to groups in name order, so the result is deterministic.
    """
    if total > sum(available.values()):
        raise ValueError(f"Need {total} items but only {sum(available.values())} available")
    counts = {name: 0 for name in available}
    remaining = total
    open_groups = sorted(available)
    while remaining > 0:
        share, extra = divmod(remaining, len(open_groups))
        for i, name in enumerate(open_groups):
            take = min(share + (1 if i < extra else 0), available[name] - counts[name])
            counts[name] += take
            remaining -= take
        open_groups = [n for n in open_groups if counts[n] < available[n]]
    return counts


def shuffle_options(row: dict[str, Any], seed: int) -> tuple[list[str], str]:
    """Shuffle the ideal answer in among the distractors; return options and correct letter."""
    options = [str(row["ideal"])] + list(row["distractors"])
    order = list(range(len(options)))
    random.Random(f"{seed}:options:{row['id']}").shuffle(order)
    shuffled = [options[i] for i in order]
    return shuffled, LETTERS[order.index(0)]


def sample_questions(
    rows: dict[str, list[dict[str, Any]]],
    seed: int = 0,
    sample_size: int = SAMPLE_SIZE,
    calibration_size: int = CALIBRATION_SIZE,
) -> list[Question]:
    """
    Draw the stratified sample and split.

    Subsets get as equal a share of the sample as their sizes allow, and the same holds for
    the calibration split, so every subset appears in calibration. The first
    calibration_size questions returned are the calibration split; the rest are the test
    split.

    Args:
        rows: Text-only rows per subset
        seed: Random seed
        sample_size: Total questions to sample
        calibration_size: Questions in the calibration split

    Returns:
        Questions in sample order (calibration first)
    """
    sample_counts = allocate(sample_size, {s: len(r) for s, r in rows.items()})
    calibration_counts = allocate(calibration_size, sample_counts)

    calibration: list[tuple[str, dict[str, Any]]] = []
    test: list[tuple[str, dict[str, Any]]] = []
    for subset in sorted(rows):
        picked = random.Random(f"{seed}:sample:{subset}").sample(
            rows[subset], sample_counts[subset]
        )
        n_cal = calibration_counts[subset]
        calibration += [(subset, r) for r in picked[:n_cal]]
        test += [(subset, r) for r in picked[n_cal:]]
    random.Random(f"{seed}:order:calibration").shuffle(calibration)
    random.Random(f"{seed}:order:test").shuffle(test)

    questions = []
    for index, (subset, row) in enumerate(calibration + test):
        options, correct = shuffle_options(row, seed)
        questions.append(Question(
            id=str(row["id"]),
            subset=subset,
            subtask=str(row["subtask"]),
            stem=build_stem(subset, row),
            options=options,
            correct_letter=correct,
            split="calibration" if index < calibration_size else "test",
            index=index,
        ))
    return questions


def select_questions(
    questions: list[Question],
    subsets: list[str] | None = None,
    test_per_subset: int | None = None,
) -> list[Question]:
    """
    Restrict the drawn sample without resampling.

    Questions keep their sample order, index and split, so every request they make is the
    same as in a run over the whole sample and earlier cached calls are reused.

    Args:
        questions: The sample, in sample order
        subsets: Keep only these subsets (None keeps all)
        test_per_subset: Keep every calibration question plus the first N test questions of
            each subset in sample order (None keeps all test questions)

    Returns:
        The selected questions in sample order
    """
    if subsets is not None:
        unknown = sorted(set(subsets) - {q.subset for q in questions})
        if unknown:
            raise ValueError(f"Unknown or unsampled subsets: {', '.join(unknown)}")
    if test_per_subset is not None and test_per_subset < 0:
        raise ValueError("test_per_subset must be non-negative")
    selected = []
    test_taken: dict[str, int] = {}
    for question in questions:
        if subsets is not None and question.subset not in subsets:
            continue
        if question.split == "test" and test_per_subset is not None:
            if test_taken.get(question.subset, 0) >= test_per_subset:
                continue
            test_taken[question.subset] = test_taken.get(question.subset, 0) + 1
        selected.append(question)
    return selected


def usable_counts(rows: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Number of text-only questions per subset."""
    return {subset: len(r) for subset, r in rows.items()}

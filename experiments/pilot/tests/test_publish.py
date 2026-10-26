"""
Tests for Publishing Results
============================

publish.py on a dry run (fake provider, synthetic questions): the published folder must hold
no question text, option or model output, and no local path. No network.
"""

import asyncio
import csv
import json
import sys
from pathlib import Path

import pytest

from experiments.pilot import run as run_module
from experiments.pilot.publish import COPIED, check_published, markdown_table, publish
from experiments.pilot.settings import PILOT_DIR
from experiments.pilot.tests.helpers import make_question

CANARY = "TEST CANARY GUID 0000"


@pytest.fixture
def dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A finished dry run on synthetic questions; returns its results folder."""
    questions = [make_question(i, "calibration" if i < 4 else "test",
                               ["SeqQA", "ProtocolQA"][i % 2], "ABCD"[i % 4])
                 for i in range(8)]
    monkeypatch.setattr(run_module, "load_sample",
                        lambda seed: (questions, {"SeqQA": 4, "ProtocolQA": 4}))
    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path / "results")
    argv = ["--dry-run", "--run-name", "dry", "--cache", str(tmp_path / "c.sqlite"),
            "--skip-methods", "art_r2"]
    # As launched from the command line: config.json records the absolute script path
    monkeypatch.setattr(sys, "argv", [str(PILOT_DIR / "run.py"), *argv])
    assert asyncio.run(run_module.run(run_module.parse_args(argv))) == 0
    return tmp_path / "results"


class TestPublish:
    """Test cases for the published copy of a run."""

    def test_nothing_leaks(self, dry_run: Path, tmp_path: Path) -> None:
        """Test that no question text, option or model output reaches a published file."""
        out = publish("dry", results_dir=dry_run, published_dir=tmp_path / "published",
                      canary=CANARY)

        published = {p.name: p.read_bytes() for p in out.iterdir()}
        forbidden = []
        for line in (dry_run / "dry" / "questions.jsonl").read_text().splitlines():
            question = json.loads(line)
            forbidden += [question["stem"], *question["options"]]
        for line in (dry_run / "dry" / "per_question.jsonl").read_text().splitlines():
            forbidden += [a["content"] for a in json.loads(line)["answers"].values()
                          if a["content"]]
        assert len(forbidden) > 40
        for name, content in published.items():
            for text in forbidden:
                assert text.encode() not in content, f"{name} leaks {text!r}"
            assert str(Path.home()).encode() not in content, name

        assert set(published) == {*COPIED, "README.md", "config.json", "per_question.csv",
                                  "elo_router.json"}
        config = json.loads(published["config.json"])
        assert config["command"].startswith("python -m experiments.pilot.run --dry-run ")
        assert " --cache c.sqlite " in config["command"]  # outside the repo: name only

        readme = published["README.md"].decode()
        assert "## Data and attribution" in readme and CANARY in readme
        assert "arXiv:2407.10362" in readme and "CC BY-SA 4.0" in readme
        assert "questions.jsonl" not in readme
        rows = list(csv.DictReader((out / "per_question.csv").open()))
        assert len(rows) == 8
        assert list(rows[0])[:3] == ["question_id", "subset", "split"]
        assert {"art_r1.correct", "art_r1.cost_usd", "art_r1.calls",
                "elo_router.routed_to"} <= set(rows[0])
        assert {r["art_r1.correct"] for r in rows} <= {"0", "1"}
        router = json.loads(published["elo_router.json"])
        assert set(router) == {"ratings", "top_model", "trajectory"}
        assert markdown_table(out / "summary.csv").count("\n") == 2 + len(
            list(csv.DictReader((out / "summary.csv").open()))
        ) - 1

    def test_check_refuses_a_leak(self, tmp_path: Path) -> None:
        """Test that the built-in check rejects a file holding a forbidden string or the key."""
        (tmp_path / "notes.md").write_text("Synthetic question number 3: which option is right?")
        with pytest.raises(ValueError, match="notes.md"):
            check_published(tmp_path, ["Synthetic question number 3"], None)
        with pytest.raises(ValueError, match="API key"):
            check_published(tmp_path, [], "number 3")
        check_published(tmp_path, ["something else"], "sk-or-test")

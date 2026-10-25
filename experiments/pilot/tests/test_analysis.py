"""
Tests for Tie-Breaking, Cost Attribution and Tau Selection
==========================================================
"""

from typing import Any

import pytest

from experiments.pilot.analysis import (
    NEVER_ACCEPT_TAU,
    Analysis,
    brier_score,
    majority_vote,
    method_usage,
    reliability_table,
    risk_coverage,
    select_tau,
    tau_sweep,
)
from experiments.pilot.settings import load_models
from experiments.pilot.tests.helpers import make_record


class TestMajorityVote:
    """Test cases for majority_vote."""

    def test_clear_majority(self) -> None:
        """Test a plain majority with no tie."""
        assert majority_vote(["A", "A", "B", None], 0, "q1") == ("A", False, {"A": 2, "B": 1})

    def test_unparseable_do_not_vote(self) -> None:
        """Test that None answers are ignored and all-None gives None."""
        assert majority_vote(["C", None, None, None], 0, "q1")[0] == "C"
        assert majority_vote([None] * 4, 0, "q1") == (None, False, {})

    def test_tie_is_seeded_and_reported(self) -> None:
        """Test that ties are broken the same way for the same seed and question."""
        first = majority_vote(["A", "B", "A", "B"], 0, "q1")

        assert first[1] is True
        assert first[0] in ("A", "B")
        assert majority_vote(["B", "A", "B", "A"], 0, "q1") == first

    def test_tie_break_varies_by_question(self) -> None:
        """Test that different questions do not all break ties the same way."""
        picks = {majority_vote(["A", "B"], 0, f"q{i}")[0] for i in range(30)}

        assert picks == {"A", "B"}


class TestMethodUsage:
    """Test cases for cost attribution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.records = {
            "a1": make_record("a1", 0.01, latency=100),
            "a2": make_record("a2", 0.02, latency=300),
            "c1": make_record("c1", 0.005, latency=50),
            "c2": make_record("c2", 0.005, latency=80),
        }

    def test_union_and_latency(self) -> None:
        """Test cost over distinct calls and latency as sum of slowest per group."""
        usage = method_usage([["a1", "a2"], ["c1", "c2"]], self.records)

        assert usage.cost_usd == pytest.approx(0.04)
        assert usage.calls == 4
        assert usage.tokens == 40
        assert usage.latency_ms == pytest.approx(300 + 80)

    def test_shared_call_counted_once_per_method(self) -> None:
        """Test that a call appearing twice in one method is paid once."""
        usage = method_usage([["a1"], ["a1", "c1"]], self.records)

        assert usage.cost_usd == pytest.approx(0.015)
        assert usage.calls == 2

    def test_shared_call_counted_for_every_method(self) -> None:
        """Test that two methods sharing a call each pay for it in full."""
        first = method_usage([["a1"], ["c1"]], self.records)
        second = method_usage([["a1"], ["c2"]], self.records)

        assert first.cost_usd == pytest.approx(0.015)
        assert second.cost_usd == pytest.approx(0.015)

    def test_empty_keys_ignored(self) -> None:
        """Test that missing calls (empty keys) add nothing."""
        assert method_usage([["", "a1"], []], self.records).calls == 1


class TestTauSelection:
    """Test cases for select_tau and tau_sweep."""

    def test_picks_best_accuracy(self) -> None:
        """Test that the threshold separating good and bad cheap answers is chosen."""
        ps = [0.9, 0.8, 0.3, 0.2]
        cheap = [True, True, False, False]
        escalated = [True, True, True, True]

        # Accepting only p >= 0.8 keeps both right cheap answers and escalates the rest
        assert select_tau(ps, cheap, escalated) == 0.8

    def test_ties_go_to_higher_coverage(self) -> None:
        """Test that among equally accurate taus the lowest (most accepting) wins."""
        ps = [0.9, 0.6, 0.4]
        cheap = [True, True, True]
        escalated = [True, True, True]

        assert select_tau(ps, cheap, escalated) == 0.0

    def test_always_escalate_when_cheap_is_worse(self) -> None:
        """Test that the never-accept threshold is chosen when escalation always wins."""
        ps = [0.9, 0.5]
        cheap = [False, False]
        escalated = [True, True]

        assert select_tau(ps, cheap, escalated) == NEVER_ACCEPT_TAU

    def test_sweep(self) -> None:
        """Test accuracy, cost and coverage along a sweep."""
        rows = tau_sweep(
            ps=[0.9, 0.2],
            cheap_correct=[True, False],
            escalated_correct=[False, True],
            accept_cost=[1.0, 1.0],
            escalate_cost=[10.0, 10.0],
            taus=[0.0, 0.5, 1.0],
        )

        assert rows[0] == {"tau": 0.0, "accuracy": 0.5, "mean_cost_usd": 1.0, "coverage": 1.0}
        assert rows[1] == {"tau": 0.5, "accuracy": 1.0, "mean_cost_usd": 5.5, "coverage": 0.5}
        assert rows[2] == {"tau": 1.0, "accuracy": 0.5, "mean_cost_usd": 10.0, "coverage": 0.0}


class TestGateCalibration:
    """Test cases for the reliability table, Brier score and risk-coverage."""

    def test_reliability_bins(self) -> None:
        """Test decile binning, including p == 1 in the top bin."""
        rows = reliability_table([0.05, 0.07, 0.95, 1.0], [False, True, True, True])

        assert rows[0]["count"] == 2
        assert rows[0]["frac_correct"] == 0.5
        assert rows[9]["count"] == 2
        assert sum(r["count"] for r in rows) == 4

    def test_brier(self) -> None:
        """Test the Brier score."""
        assert brier_score([1.0, 0.0], [True, False]) == 0.0
        assert brier_score([0.5, 0.5], [True, False]) == 0.25

    def test_risk_coverage(self) -> None:
        """Test that the most confident answers are accepted first."""
        rows = risk_coverage([0.9, 0.1, 0.5], [True, False, True])

        assert [r["coverage"] for r in rows] == pytest.approx([1 / 3, 2 / 3, 1.0])
        assert [r["risk"] for r in rows] == pytest.approx([0.0, 0.0, 1 / 3])


def raw_record(
    qid: str,
    split: str,
    p_llm: float,
    cheap_letter: str | None,
    art_letter: str | None,
) -> dict[str, Any]:
    """A minimal raw per-question record using call keys prefixed with the question ID."""
    models = load_models()
    cheap = models.roles["cheap"]
    answers = {
        m: {"letter": cheap_letter if m == cheap else "A", "call": f"{qid}-ans-{m}"}
        for m in models.ids
    }
    answer_group = [f"{qid}-ans-{m}" for m in models.ids]
    art_groups = [answer_group, [f"{qid}-crit-{i}" for i in range(12)]]
    simple = {"answer": "A", "groups": [answer_group]}
    methods = {f"single:{m.label}": {"answer": "A", "groups": [[f"{qid}-ans-{m.id}"]]}
               for m in models.models}
    methods.update({
        "art_r1": {"answer": art_letter, "groups": art_groups},
        "art_r2": {"answer": art_letter, "groups": art_groups},
        "judge": simple,
        "fusion": simple,
        "jev_pick": {**simple, "orderings_disagree": False},
        "cascade_llm": {
            "cheap_answer": cheap_letter,
            "p": p_llm,
            "accept_groups": [[f"{qid}-ans-{cheap}"], [f"{qid}-gate"]],
        },
        "cascade_jev": {
            "cheap_answer": cheap_letter,
            "p": p_llm,
            "accept_groups": [[f"{qid}-ans-{cheap}"], [f"{qid}-jev"]],
        },
    })
    return {
        "id": qid, "index": 0, "subset": "SeqQA", "subtask": "s", "split": split,
        "correct_letter": "B", "answers": answers, "methods": methods,
    }


def records_for(raw: list[dict[str, Any]]) -> dict[str, Any]:
    """CallRecords for every key in the raw records: answers $0.01, critiques $0.002..."""
    keys = set()
    for record in raw:
        for method in record["methods"].values():
            for group in method.get("groups", []) + method.get("accept_groups", []):
                keys |= set(group)
    costs = {"ans": 0.01, "crit": 0.002, "gate": 0.003, "jev": 0.0001}
    return {k: make_record(k, costs[k.split("-")[1]]) for k in keys}


class TestCascadeAttribution:
    """Test cases for cascade cost and tau fitting inside Analysis."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.models = load_models()
        self.raw = [
            # calibration: cheap right when p high, wrong when p low; ART always right
            raw_record("c1", "calibration", 0.9, "B", "B"),
            raw_record("c2", "calibration", 0.2, "C", "B"),
            # test
            raw_record("t1", "test", 0.95, "B", "B"),
            raw_record("t2", "test", 0.1, "D", "B"),
        ]
        self.analysis = Analysis(self.raw, records_for(self.raw), self.models, None)

    def test_tau_fitted_on_calibration_only(self) -> None:
        """Test that the threshold comes from calibration questions."""
        assert self.analysis.taus["cascade_llm"] == 0.9

    def test_accepted_cost(self) -> None:
        """Test that an accepted question pays the cheap answer plus the gate."""
        outcome = self.analysis.outcomes["t1"]["cascade_llm"]

        assert outcome["accepted"] is True
        assert outcome["answer"] == "B"
        assert outcome["cost_usd"] == pytest.approx(0.01 + 0.003)
        assert outcome["calls"] == 2

    def test_escalated_cost(self) -> None:
        """Test that an escalated question also pays art_r1, counting the cheap answer once."""
        outcome = self.analysis.outcomes["t2"]["cascade_llm"]

        assert outcome["accepted"] is False
        assert outcome["answer"] == "B"
        # 4 answers (cheap answer shared, counted once) + 12 critiques + gate
        assert outcome["cost_usd"] == pytest.approx(4 * 0.01 + 12 * 0.002 + 0.003)
        assert outcome["calls"] == 4 + 12 + 1

    def test_jev_cascade_cost(self) -> None:
        """Test the Jev cascade's accepted cost."""
        outcome = self.analysis.outcomes["t1"]["cascade_jev"]

        assert outcome["cost_usd"] == pytest.approx(0.01 + 0.0001)

    def test_summary_on_test_split(self) -> None:
        """Test that the summary uses only test questions and reports coverage."""
        rows = {r["method"]: r for r in self.analysis.summary()}

        assert rows["cascade_llm"]["n"] == 2
        assert rows["cascade_llm"]["accuracy"] == 1.0
        assert rows["cascade_llm"]["coverage"] == 0.5
        assert rows["art_r1"]["mean_usd_per_question"] == pytest.approx(4 * 0.01 + 12 * 0.002)

    def test_best_single_calibration_vs_oracle(self) -> None:
        """Test that both best-single rows exist and the oracle is labelled."""
        rows = {r["method"]: r for r in self.analysis.summary()}

        assert "best_single_calibration" in rows
        assert rows["best_single_oracle"]["notes"].startswith("ORACLE")

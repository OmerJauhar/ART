"""
Pilot Analysis
==============

Turns raw per-question records into method outcomes, fits cascade thresholds on the
calibration split, and computes the test-split summary, tau sweeps and gate calibration.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.pilot.gateway import CallRecord, failure_cause
from experiments.pilot.settings import PilotModels

# A threshold above every possible p: the cascade always escalates
NEVER_ACCEPT_TAU = 1.01

CASCADES = ("cascade_llm", "cascade_jev")

# The cascades escalate to art_r1 and the ELO router is fitted from art_r1's tournaments
NEEDS_ART_R1 = ("cascade_llm", "cascade_jev", "elo_router")

FAILURE_CAUSES = ("refused", "unparseable", "other")


def all_methods(models: PilotModels) -> list[str]:
    """Every method name in report order."""
    return [f"single:{m.label}" for m in models.models] + [
        "majority_vote", "art_r1", "art_r2", "judge", "fusion", "elo_router",
        "best_single_calibration", "best_single_oracle", "jev_pick", *CASCADES,
    ]


def check_skip(skip: set[str], models: PilotModels) -> set[str]:
    """
    Validate a set of methods to skip.

    The four answers are always made because nearly every method builds on them; skipping a
    single:<model> method only leaves it out of the report.

    Raises:
        ValueError: On an unknown name, or when a kept method needs art_r1 but it is skipped
    """
    unknown = sorted(skip - set(all_methods(models)))
    if unknown:
        raise ValueError(
            f"Unknown methods: {', '.join(unknown)}. Choose from: {', '.join(all_methods(models))}"
        )
    if "art_r1" in skip:
        needing = [m for m in NEEDS_ART_R1 if m not in skip]
        if needing:
            raise ValueError(f"art_r1 is needed by {', '.join(needing)}; skip those too")
    return skip


@dataclass
class Usage:
    """Cost, calls, tokens and idealized latency of one method on one question."""

    cost_usd: float
    calls: int
    tokens: int
    latency_ms: float


def method_usage(groups: list[list[str]], records: dict[str, CallRecord]) -> Usage:
    """
    Usage of a method from its call groups, counted as if the method ran alone.

    Each distinct call is counted once at the cost recorded when it was made, even if
    other methods shared it. Latency assumes calls in a group run in parallel and groups
    run one after another.
    """
    keys = {k for group in groups for k in group if k}
    latency = 0.0
    for group in groups:
        group_latencies = [records[k].latency_ms for k in group if k]
        if group_latencies:
            latency += max(group_latencies)
    return Usage(
        cost_usd=sum(records[k].cost_usd for k in keys),
        calls=len(keys),
        tokens=sum(records[k].total_tokens for k in keys),
        latency_ms=latency,
    )


def majority_vote(
    letters: list[str | None],
    seed: int,
    question_id: str,
) -> tuple[str | None, bool, dict[str, int]]:
    """
    Majority over the parseable letters; ties broken by a seeded per-question draw.

    Returns:
        (answer, whether there was a tie, vote counts)
    """
    votes = Counter(x for x in letters if x)
    if not votes:
        return None, False, {}
    top = max(votes.values())
    leaders = sorted(x for x, n in votes.items() if n == top)
    if len(leaders) == 1:
        return leaders[0], False, dict(votes)
    rng = random.Random(f"{seed}:tie:{question_id}")
    return rng.choice(leaders), True, dict(votes)


def select_tau(
    ps: list[float],
    cheap_correct: list[bool],
    escalated_correct: list[bool],
) -> float:
    """
    Threshold maximizing cascade accuracy; ties go to the lowest tau (highest coverage).

    A question is accepted (cheap answer kept) when p >= tau, else escalated.

    Args:
        ps: Gate probability per calibration question
        cheap_correct: Whether the cheap answer was correct
        escalated_correct: Whether the escalation method (art_r1) was correct

    Returns:
        The chosen tau; NEVER_ACCEPT_TAU means always escalate
    """
    if not ps:
        return NEVER_ACCEPT_TAU
    candidates = sorted({0.0, *ps, NEVER_ACCEPT_TAU})
    best_tau, best_accuracy = NEVER_ACCEPT_TAU, -1.0
    for tau in candidates:
        correct = [c if p >= tau else e for p, c, e in zip(ps, cheap_correct, escalated_correct)]
        accuracy = sum(correct) / len(correct)
        if accuracy > best_accuracy + 1e-12:
            best_tau, best_accuracy = tau, accuracy
    return best_tau


def tau_sweep(
    ps: list[float],
    cheap_correct: list[bool],
    escalated_correct: list[bool],
    accept_cost: list[float],
    escalate_cost: list[float],
    taus: list[float] | None = None,
) -> list[dict[str, float]]:
    """Cascade accuracy, mean cost and coverage at each tau (default 0 to 1 in steps of 0.01)."""
    taus = taus if taus is not None else [round(t, 2) for t in np.linspace(0, 1, 101)]
    rows = []
    n = len(ps)
    for tau in taus:
        accept = [p >= tau for p in ps]
        correct = [c if a else e for a, c, e in zip(accept, cheap_correct, escalated_correct)]
        cost = [ac if a else ec for a, ac, ec in zip(accept, accept_cost, escalate_cost)]
        rows.append({
            "tau": tau,
            "accuracy": sum(correct) / n if n else float("nan"),
            "mean_cost_usd": sum(cost) / n if n else float("nan"),
            "coverage": sum(accept) / n if n else float("nan"),
        })
    return rows


def bootstrap_ci(
    values: list[bool] | list[float],
    n_resamples: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    if not values:
        return float("nan"), float("nan")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(n_resamples, len(data)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def reliability_table(ps: list[float], correct: list[bool], bins: int = 10) -> list[dict[str, Any]]:
    """Decile bins of p with count, mean p and observed accuracy."""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, c) for p, c in zip(ps, correct)
            if lo <= p < hi or (b == bins - 1 and p == hi)
        ]
        rows.append({
            "bin": f"[{lo:.1f}, {hi:.1f}{']' if b == bins - 1 else ')'}",
            "bin_low": lo,
            "bin_high": hi,
            "count": len(members),
            "mean_p": float(np.mean([p for p, _ in members])) if members else float("nan"),
            "frac_correct": float(np.mean([c for _, c in members])) if members else float("nan"),
        })
    return rows


def brier_score(ps: list[float], correct: list[bool]) -> float:
    """Mean squared error between p and the 0/1 outcome."""
    if not ps:
        return float("nan")
    return float(np.mean([(p - float(c)) ** 2 for p, c in zip(ps, correct)]))


def risk_coverage(ps: list[float], correct: list[bool]) -> list[dict[str, float]]:
    """Accept the k most confident answers for k = 1..n; risk is the error rate among them."""
    order = sorted(range(len(ps)), key=lambda i: -ps[i])
    rows = []
    errors = 0
    for k, i in enumerate(order, start=1):
        errors += 0 if correct[i] else 1
        rows.append({"coverage": k / len(ps), "risk": errors / k, "threshold": ps[i]})
    return rows


class Analysis:
    """Method outcomes, fitted thresholds and summaries for a set of raw records."""

    def __init__(
        self,
        raw: list[dict[str, Any]],
        records: dict[str, CallRecord],
        models: PilotModels,
        elo_router: dict[str, Any] | None,
        seed: int = 0,
        skip: set[str] | None = None,
    ) -> None:
        """
        Compute all outcomes.

        Methods in skip are left out of the outcomes; the runner made no calls for them.
        """
        self.raw = raw
        self.records = records
        self.models = models
        self.seed = seed
        self.elo_router = elo_router
        self.skip = set(skip or ())
        self.cascades = [name for name in CASCADES if name not in self.skip]
        self.labels = {m.id: m.label for m in models.models}

        self.outcomes: dict[str, dict[str, dict[str, Any]]] = {}
        for record in raw:
            self.outcomes[record["id"]] = self._base_outcomes(record)

        self.calibration = [r for r in raw if r["split"] == "calibration"]
        self.test = [r for r in raw if r["split"] == "test"]
        if self.test:
            self.eval_records, self.eval_split = self.test, "test"
        else:
            self.eval_records, self.eval_split = raw, "all (no test questions in this run)"

        self.taus = {name: self._fit_tau(name) for name in self.cascades}
        for record in raw:
            outcomes = self.outcomes[record["id"]]
            for name in self.cascades:
                outcomes[name] = self._cascade(record, name, self.taus[name])
            if elo_router and elo_router.get("top_model"):
                top = elo_router["top_model"].get(record["subset"])
                if top:
                    single = outcomes[f"single:{self.labels[top]}"]
                    outcomes["elo_router"] = {**single, "routed_to": top}

        self.best_single_calibration = self._best_single(self.calibration)
        self.best_single_oracle = self._best_single(self.eval_records)
        for record in raw:
            outcomes = self.outcomes[record["id"]]
            if self.best_single_calibration:
                outcomes["best_single_calibration"] = dict(
                    outcomes[f"single:{self.labels[self.best_single_calibration]}"]
                )
            if self.best_single_oracle:
                outcomes["best_single_oracle"] = dict(
                    outcomes[f"single:{self.labels[self.best_single_oracle]}"]
                )
            for name in self.skip:
                outcomes.pop(name, None)

    def _outcome(
        self,
        record: dict[str, Any],
        answer: str | None,
        groups: list[list[str]],
        error: str | None = None,
    ) -> dict[str, Any]:
        """
        Answer, correctness and usage for one method on one question.

        A missing answer is "refused" when the method failed on a provider refusal and
        "unparseable" otherwise; either way it counts as wrong.
        """
        usage = method_usage(groups, self.records)
        refused = answer is None and failure_cause(error) == "refused"
        return {
            "answer": answer,
            "correct": answer is not None and answer == record["correct_letter"],
            "unparseable": answer is None and not refused,
            "refused": refused,
            "cost_usd": usage.cost_usd,
            "calls": usage.calls,
            "tokens": usage.tokens,
            "latency_ms": usage.latency_ms,
            "groups": groups,
        }

    def _base_outcomes(self, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Outcomes for every method that needs no fitted parameter."""
        methods = record["methods"]
        outcomes = {}
        for name in [f"single:{m.label}" for m in self.models.models] + [
            "art_r1", "art_r2", "judge", "fusion", "jev_pick"
        ]:
            if name in methods:
                outcomes[name] = self._outcome(
                    record, methods[name]["answer"], methods[name]["groups"],
                    methods[name].get("error"),
                )

        if "majority_vote" not in self.skip:
            letters = [record["answers"][m]["letter"] for m in self.models.ids]
            answer, tie, votes = majority_vote(letters, self.seed, record["id"])
            answer_group = [record["answers"][m]["call"] for m in self.models.ids]
            outcomes["majority_vote"] = {
                **self._outcome(record, answer, [[k for k in answer_group if k]]),
                "tie": tie,
                "votes": votes,
            }
        if "jev_pick" in outcomes:
            outcomes["jev_pick"]["orderings_disagree"] = methods["jev_pick"]["orderings_disagree"]
        return outcomes

    def _cascade_inputs(self, records: list[dict[str, Any]], name: str) -> dict[str, list[Any]]:
        """Gate p, cheap/escalated correctness and both possible costs per question."""
        inputs: dict[str, list[Any]] = {
            "p": [], "cheap_correct": [], "escalated_correct": [],
            "accept_cost": [], "escalate_cost": [],
        }
        for record in records:
            cascade = record["methods"][name]
            art = record["methods"]["art_r1"]
            inputs["p"].append(float(cascade["p"]))
            inputs["cheap_correct"].append(cascade["cheap_answer"] == record["correct_letter"])
            inputs["escalated_correct"].append(art["answer"] == record["correct_letter"])
            accept = cascade["accept_groups"]
            inputs["accept_cost"].append(method_usage(accept, self.records).cost_usd)
            inputs["escalate_cost"].append(
                method_usage(accept + art["groups"], self.records).cost_usd
            )
        return inputs

    def _fit_tau(self, name: str) -> float:
        """Fit a cascade threshold on the calibration split only."""
        inputs = self._cascade_inputs(self.calibration, name)
        return select_tau(inputs["p"], inputs["cheap_correct"], inputs["escalated_correct"])

    def _cascade(self, record: dict[str, Any], name: str, tau: float) -> dict[str, Any]:
        """Cascade outcome at threshold tau."""
        cascade = record["methods"][name]
        accepted = float(cascade["p"]) >= tau
        groups = list(cascade["accept_groups"])
        if accepted:
            answer = cascade["cheap_answer"]
            error = record["answers"][self.models.roles["cheap"]].get("error")
        else:
            answer = record["methods"]["art_r1"]["answer"]
            error = record["methods"]["art_r1"].get("error")
            groups += record["methods"]["art_r1"]["groups"]
        return {
            **self._outcome(record, answer, groups, error),
            "p": cascade["p"],
            "tau": tau,
            "accepted": accepted,
            "gate_error": cascade.get("gate_error"),
        }

    def _best_single(self, records: list[dict[str, Any]]) -> str | None:
        """Most accurate single model on these records; ties go to the cheaper model."""
        if not records:
            return None
        def key(model_id: str) -> tuple[float, float]:
            label = self.labels[model_id]
            outcomes = [self.outcomes[r["id"]][f"single:{label}"] for r in records]
            accuracy = np.mean([o["correct"] for o in outcomes])
            return float(accuracy), -self.models.get(model_id).blended_price
        return max(self.models.ids, key=key)

    def method_names(self) -> list[str]:
        """Methods in report order."""
        present: set[str] = set()
        for outcomes in self.outcomes.values():
            present |= set(outcomes)
        return [n for n in all_methods(self.models) if n in present]

    def summary(self) -> list[dict[str, Any]]:
        """One row per method on the evaluation split."""
        rows = []
        for name in self.method_names():
            outs = [self.outcomes[r["id"]][name] for r in self.eval_records
                    if name in self.outcomes[r["id"]]]
            if not outs:
                continue
            correct = [o["correct"] for o in outs]
            lo, hi = bootstrap_ci(correct, seed=self.seed)
            row: dict[str, Any] = {
                "method": name,
                "n": len(outs),
                "accuracy": float(np.mean(correct)),
                "ci_low": lo,
                "ci_high": hi,
                "mean_usd_per_question": float(np.mean([o["cost_usd"] for o in outs])),
                "mean_calls_per_question": float(np.mean([o["calls"] for o in outs])),
                "mean_tokens_per_question": float(np.mean([o["tokens"] for o in outs])),
                "mean_latency_ms": float(np.mean([o["latency_ms"] for o in outs])),
                "unparseable": sum(o["unparseable"] for o in outs),
                "refused": sum(o["refused"] for o in outs),
                "coverage": "",
                "tau": "",
                "notes": "",
            }
            if name in self.cascades:
                row["coverage"] = float(np.mean([o["accepted"] for o in outs]))
                row["tau"] = self.taus[name]
                row["notes"] = "tau fitted on calibration split"
            elif name == "majority_vote":
                row["notes"] = f"ties: {sum(o['tie'] for o in outs)}"
            elif name == "jev_pick":
                row["notes"] = (
                    f"orderings disagree: {sum(o['orderings_disagree'] for o in outs)}"
                )
            elif name == "elo_router":
                routed = Counter(self.labels[o["routed_to"]] for o in outs)
                row["notes"] = "routes: " + ", ".join(f"{k}={v}" for k, v in sorted(routed.items()))
            elif name == "best_single_calibration" and self.best_single_calibration:
                row["notes"] = f"{self.labels[self.best_single_calibration]} (fixed on calibration)"
            elif name == "best_single_oracle" and self.best_single_oracle:
                oracle = self.labels[self.best_single_oracle]
                row["notes"] = f"ORACLE: {oracle} picked on this split"
            rows.append(row)
        return rows

    def dropped_critiques(self) -> list[dict[str, Any]]:
        """
        Critiques dropped on the evaluation split, counted by method, critic and cause.

        art_r1 and art_r2 count the peer critiques their tournaments dropped (every round);
        judge counts the judge model's critiques that failed.
        """
        counts: dict[tuple[str, str], Counter[str]] = {}
        for record in self.eval_records:
            methods = record["methods"]
            for name in ("art_r1", "art_r2"):
                for round_record in methods.get(name, {}).get("rounds", []):
                    for drop in round_record.get("dropped_critiques", []):
                        counts.setdefault((name, drop["critic"]), Counter())[drop["cause"]] += 1
            judge = methods.get("judge")
            if judge:
                critic = self.models.roles["judge"]
                for target in judge["per_target"].values():
                    if target["overall"] is None:
                        cause = failure_cause(target["error"])
                        counts.setdefault(("judge", critic), Counter())[cause] += 1
        order = {m: i for i, m in enumerate(["art_r1", "art_r2", "judge"])}
        rows = []
        for (method, critic), by_cause in sorted(
            counts.items(), key=lambda kv: (order[kv[0][0]], self.models.ids.index(kv[0][1]))
        ):
            row: dict[str, Any] = {"method": method, "critic": self.labels[critic]}
            row.update({cause: by_cause[cause] for cause in FAILURE_CAUSES})
            row["total"] = sum(by_cause.values())
            rows.append(row)
        return rows

    def sweeps(self) -> dict[str, list[dict[str, float]]]:
        """Tau sweep 0..1 on the evaluation split for each cascade (not a tuned result)."""
        result = {}
        for name in self.cascades:
            inputs = self._cascade_inputs(self.eval_records, name)
            result[name] = tau_sweep(
                inputs["p"], inputs["cheap_correct"], inputs["escalated_correct"],
                inputs["accept_cost"], inputs["escalate_cost"],
            )
        return result

    def gate_calibration(self) -> dict[str, dict[str, Any]]:
        """Reliability table, Brier score and risk-coverage curve per gate on the eval split."""
        result = {}
        for name, gate in (("cascade_llm", "llm_judge"), ("cascade_jev", "jev")):
            if name not in self.cascades:
                continue
            inputs = self._cascade_inputs(self.eval_records, name)
            result[gate] = {
                "reliability": reliability_table(inputs["p"], inputs["cheap_correct"]),
                "brier": brier_score(inputs["p"], inputs["cheap_correct"]),
                "risk_coverage": risk_coverage(inputs["p"], inputs["cheap_correct"]),
                "n": len(inputs["p"]),
            }
        return result

    def unique_cost(self, record: dict[str, Any]) -> float:
        """Cost of every distinct call made for one question (all methods together)."""
        keys = set()
        for outcome in self.outcomes[record["id"]].values():
            keys |= {k for group in outcome["groups"] for k in group if k}
        return sum(self.records[k].cost_usd for k in keys)

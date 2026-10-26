"""
Tests for the Run Selection Flags, Skipped Methods, Cost Estimate and Partial Runs
==================================================================================

All synthetic or fake-provider; no network.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from experiments.pilot import run as run_module
from experiments.pilot.analysis import Analysis, check_skip
from experiments.pilot.budget import BudgetExceeded
from experiments.pilot.cache import CallCache
from experiments.pilot.data import Question, select_questions
from experiments.pilot.estimate import (
    JEV,
    CostModel,
    Observation,
    StageFit,
    answer_dependents,
    estimate_cost,
    estimate_from_cache,
    fit_cost_model,
    plan_calls,
    replay,
)
from experiments.pilot.fake_provider import FakeTransport
from experiments.pilot.gateway import QUESTION_ID, ApiError
from experiments.pilot.jev import JevClient
from experiments.pilot.methods import QuestionRunner
from experiments.pilot.prompts import format_question
from experiments.pilot.settings import load_models
from experiments.pilot.tests.helpers import make_gateway, make_question

SAMPLE_IDS = Path(__file__).resolve().parents[1] / "sample_ids.json"


def sample_like() -> list[Question]:
    """Lightweight questions with the committed sample's IDs, subsets, order and split."""
    payload = json.loads(SAMPLE_IDS.read_text())
    return [
        Question(
            id=qid, subset=subset, subtask="", stem="", options=["a", "b", "c", "d"],
            correct_letter="A",
            split="calibration" if i < payload["calibration_size"] else "test", index=i,
        )
        for i, (qid, subset) in enumerate(zip(payload["ids"], payload["subsets"]))
    ]


def mixed_questions() -> list[Question]:
    """Two calibration and three test questions per subset, interleaved."""
    questions = []
    for i in range(10):
        split = "calibration" if i < 4 else "test"
        questions.append(make_question(i, split, ["SeqQA", "ProtocolQA"][i % 2], "ABCD"[i % 4]))
    return questions


class TestSelectQuestions:
    """Test cases for select_questions."""

    def test_no_flags_keeps_everything(self) -> None:
        """Test that no selection returns the sample unchanged."""
        questions = mixed_questions()

        assert select_questions(questions) == questions

    def test_subset_filter(self) -> None:
        """Test that only the chosen subsets remain, in order."""
        selected = select_questions(mixed_questions(), subsets=["ProtocolQA"])

        assert [q.index for q in selected] == [1, 3, 5, 7, 9]
        assert {q.subset for q in selected} == {"ProtocolQA"}

    def test_test_per_subset(self) -> None:
        """Test that all calibration questions stay and the first N tests per subset."""
        selected = select_questions(mixed_questions(), test_per_subset=1)

        assert [q.index for q in selected] == [0, 1, 2, 3, 4, 5]
        assert [q.split for q in selected] == ["calibration"] * 4 + ["test"] * 2

    def test_keeps_questions_unchanged(self) -> None:
        """Test that split and index are not reassigned (no resampling)."""
        questions = mixed_questions()
        selected = select_questions(questions, subsets=["SeqQA"], test_per_subset=2)

        assert all(q is questions[q.index] for q in selected)

    def test_unknown_subset(self) -> None:
        """Test that a misspelt subset is an error, not an empty run."""
        with pytest.raises(ValueError, match="Seqqa"):
            select_questions(mixed_questions(), subsets=["Seqqa"])

    def test_pilot_v1_selection_on_committed_sample(self) -> None:
        """Test the planned 50-question run against sample_ids.json."""
        sample = sample_like()
        selected = select_questions(sample, ["SeqQA", "ProtocolQA"], 15)

        counts = run_module.split_counts(selected)
        assert len(selected) == 50
        assert counts["calibration"] == {"SeqQA": 10, "ProtocolQA": 10}
        assert counts["test"] == {"SeqQA": 15, "ProtocolQA": 15}
        assert [q.index for q in selected] == sorted(q.index for q in selected)
        assert all(q.index < 30 for q in selected if q.split == "calibration")
        # The smoke run's questions (the first 5 of the sample) minus CloningScenarios
        assert {q.index for q in selected} & set(range(5)) == {
            q.index for q in sample[:5] if q.subset != "CloningScenarios"
        }
        # Test questions are the first 15 per subset in sample order
        for subset in ("SeqQA", "ProtocolQA"):
            tests = [q for q in sample if q.split == "test" and q.subset == subset]
            assert [q for q in selected if q.split == "test" and q.subset == subset] == tests[:15]


class TestSkipMethods:
    """Test cases for check_skip and skipped methods in the runner."""

    def test_unknown_method(self) -> None:
        """Test that an unknown method name is rejected."""
        with pytest.raises(ValueError, match="art_r3"):
            check_skip({"art_r3"}, load_models())

    def test_art_r1_dependency(self) -> None:
        """Test that art_r1 cannot be skipped while a method still needs it."""
        models = load_models()
        with pytest.raises(ValueError, match="cascade_llm"):
            check_skip({"art_r1"}, models)
        assert check_skip(
            {"art_r1", "cascade_llm", "cascade_jev", "elo_router"}, models
        ) == {"art_r1", "cascade_llm", "cascade_jev", "elo_router"}

    async def _run(self, tmp_path: Path, skip: set[str]) -> tuple[Any, list[dict[str, Any]], Any]:
        """Run two fake questions with the given methods skipped."""
        models = load_models()
        questions = [make_question(0, "calibration"), make_question(1, "test", correct="C")]
        transport = FakeTransport(questions, models)
        gateway = make_gateway(tmp_path, transport, models=models)
        jev = JevClient(gateway, models.jev.model, models.jev.fallback_model)
        runner = QuestionRunner(gateway, models, jev, seed=0, skip=skip)
        raw = [await runner.run(q) for q in questions]
        return transport, raw, gateway

    @pytest.mark.asyncio
    async def test_skip_art_r2_makes_no_calls(self, tmp_path: Path) -> None:
        """Test that skipping art_r2 drops its 16 calls and its outputs."""
        full_transport, _, _ = await self._run(tmp_path / "full", set())
        transport, raw, gateway = await self._run(tmp_path / "skip", {"art_r2"})

        # Per question: 37 calls in all, of which art_r2 alone needs 4 revisions + 12 critiques
        assert full_transport.calls == 2 * 37
        assert transport.calls == 2 * 21
        assert not any("revision" in r.stages for r in gateway.records.values())
        assert all("art_r2" not in r["methods"] for r in raw)

        analysis = Analysis(raw, gateway.records, load_models(), None, seed=0, skip={"art_r2"})
        names = {row["method"] for row in analysis.summary()}
        assert "art_r2" not in names
        assert {"art_r1", "judge", "fusion", "jev_pick", "cascade_llm", "cascade_jev"} <= names
        assert all("art_r2" not in outcomes for outcomes in analysis.outcomes.values())

    @pytest.mark.asyncio
    async def test_skip_jev_methods_makes_no_jev_calls(self, tmp_path: Path) -> None:
        """Test that skipping jev_pick and cascade_jev sends no Decisions request."""
        _, raw, gateway = await self._run(tmp_path, {"jev_pick", "cascade_jev"})

        assert not any(r.endpoint == "decisions" for r in gateway.records.values())
        assert all("cascade_llm" in r["methods"] for r in raw)

    @pytest.mark.asyncio
    async def test_skip_judge_keeps_llm_gate(self, tmp_path: Path) -> None:
        """Test that cascade_llm still gets its gate when judge is skipped."""
        models = load_models()
        transport, raw, _ = await self._run(tmp_path, {"judge", "fusion"})

        # The gate is one of art_r1's critiques, so only the judge's self-critique (1 call)
        # and fusion (1 call) are saved
        assert transport.calls == 2 * (37 - 2)
        for record in raw:
            assert "judge" not in record["methods"]
            gate = record["methods"]["cascade_llm"]
            assert gate["accept_groups"][1], "LLM gate call missing"
            assert gate["cheap_answer"] == record["answers"][models.roles["cheap"]]["letter"]

    @pytest.mark.asyncio
    async def test_skip_matches_estimate_plan(self, tmp_path: Path) -> None:
        """Test that the estimator plans exactly the calls the runner makes."""
        models = load_models()
        for skip in (set(), {"art_r2"}, {"art_r2", "judge", "jev_pick"}):
            transport, _, _ = await self._run(tmp_path / "-".join(sorted(skip)), skip)
            planned = set().union(*plan_calls(models, skip).values())
            assert transport.calls == 2 * len(planned), skip


def fit_fixture() -> tuple[CostModel, Question]:
    """A cost model with round numbers and a 1000-character question."""
    models = load_models()
    question = make_question(0)
    question.stem = "x" * (1000 - len(format_question(question)) + len(question.stem))
    assert len(format_question(question)) == 1000
    stages = {}
    for model in models.ids:
        for kind in ("answer", "critique", "revision", "fusion"):
            stages[(kind, model)] = StageFit(n=1, completion_tokens=100.0, overhead_tokens=0.0)
    stages[("jev_pick", JEV)] = StageFit(n=1, completion_tokens=0.0, overhead_tokens=0.0)
    stages[("jev_gate", JEV)] = StageFit(n=1, completion_tokens=0.0, overhead_tokens=0.0)
    tokens_per_char = {(m, "*"): 0.5 for m in [*models.ids, JEV]}
    cost_model = CostModel(models, stages, tokens_per_char, {}, n_calls=1)
    return cost_model, question


class TestEstimate:
    """Test cases for the measured-cost estimate."""

    def test_fit_recovers_parameters(self) -> None:
        """Test completion, overhead, tokens per character and the price multiplier."""
        models = load_models()
        question = make_question(0)
        chars = len(format_question(question))
        model = models.models[2]
        list_answer = model.cost(chars // 2, 300)
        list_critique = model.cost(chars // 2 + 700, 500)
        obs = [
            Observation("answer", model.id, question.id, chars // 2, 300, 2 * list_answer),
            Observation("critique", model.id, question.id, chars // 2 + 700, 500,
                        2 * list_critique),
        ]

        fit = fit_cost_model(obs, {question.id: question}, models)

        assert fit.tokens_per_char[(model.id, "SeqQA")] == pytest.approx((chars // 2) / chars)
        assert fit.stages[("critique", model.id)].overhead_tokens == pytest.approx(700)
        assert fit.stages[("critique", model.id)].completion_tokens == pytest.approx(500)
        assert fit.multiplier[model.id] == pytest.approx(2.0)
        # In sample, the fitted model reproduces the billed cost
        assert fit.call_cost("critique", model.id, question) == pytest.approx(2 * list_critique)

    def test_fit_needs_answers(self) -> None:
        """Test that an empty cache gives a clear error instead of a number."""
        with pytest.raises(ValueError, match="smoke"):
            fit_cost_model([], {}, load_models())

    def test_total_counts_shared_calls_once(self) -> None:
        """Test the run total against the per-call costs."""
        cost_model, question = fit_fixture()
        models = cost_model.models
        cost = {
            m: cost_model.call_cost("answer", m, question) for m in models.ids
        }  # every chat stage costs the same per model in the fixture
        jev = cost_model.call_cost("jev_gate", JEV, question)

        full = estimate_cost(cost_model, [question], set())
        no_r2 = estimate_cost(cost_model, [question], {"art_r2"})

        # Each model: 1 answer, 3 round-1 critiques, 1 revision, 3 round-2 critiques; the
        # judge model also critiques itself (judge) and fuses; Jev: 2 picks and 1 gate
        each = sum(cost.values())
        expected = 8 * each + 2 * cost[models.roles["judge"]] + 3 * jev
        assert full.total_usd == pytest.approx(expected)
        # Skipping art_r2 removes its 4 revisions and 12 round-2 critiques
        assert no_r2.total_usd == pytest.approx(expected - 4 * each)
        assert "art_r2" not in {row["method"] for row in no_r2.methods}
        assert no_r2.by_stage["revision"] == 0.0

    def test_cached_questions_cost_nothing_new(self) -> None:
        """Test that fully cached questions are excluded from new spend."""
        cost_model, question = fit_fixture()
        other = make_question(1)

        estimate = estimate_cost(cost_model, [question, other], {"art_r2"}, cached={question.id})

        assert estimate.cached_questions == [question.id]
        assert estimate.new_usd == pytest.approx(estimate.total_usd - estimate.cached_estimated_usd)
        assert 0 < estimate.new_usd < estimate.total_usd

    @pytest.mark.asyncio
    async def test_estimate_from_cache_makes_no_calls(self, tmp_path: Path) -> None:
        """Test the cache replay: hits are labelled, misses are refused and counted."""
        models = load_models()
        done, new = make_question(0, "calibration"), make_question(1, "test")
        transport = FakeTransport([done, new], models)
        gateway = make_gateway(tmp_path, transport, models=models)
        runner = QuestionRunner(gateway, models, JevClient(gateway, models.jev.model), seed=0)
        await runner.run(done)
        calls_made = transport.calls
        cache = CallCache(tmp_path / "cache.sqlite")

        records, missing = await replay([done, new], models, cache, 0, set())
        estimate = await estimate_from_cache([done, new], [done, new], models, cache, 0,
                                             {"art_r2"})

        assert transport.calls == calls_made
        assert len(cache) == calls_made
        assert not missing[done.id] and missing[new.id]
        assert {r.stage for r in records.values()} == {
            "answer", "critique", "revision", "fusion", "jev"
        }
        assert estimate.cached_questions == [done.id]
        assert estimate.cost_model.n_calls == calls_made
        assert estimate.new_usd > 0
        # Billed cost of the cached question counts only the calls the kept methods use
        check = make_gateway(tmp_path, FakeTransport([], models), models=models)
        await QuestionRunner(check, models, JevClient(check, models.jev.model), seed=0,
                             skip={"art_r2"}).run(done)
        assert check.live_calls == 0
        assert estimate.cached_billed_usd == pytest.approx(
            sum(r.cost_usd for r in check.records.values())
        )
        assert estimate.cached_billed_usd < sum(r.cost_usd for r in records.values())
        # A question with nothing cached is priced whole
        assert estimate.new_usd == pytest.approx(estimate.partial_whole_usd)


class TestMissingCalls:
    """New spend prices only the missing calls, not whole questions."""

    async def cached_question(self, tmp_path: Path) -> tuple[Question, Any, CallCache]:
        """
        Two questions run in full through the fake provider; the first, its gateway (call
        records of the first question only) and the cache. The second keeps every stage
        fitted when calls of the first are deleted.
        """
        models = load_models()
        question, other = make_question(0), make_question(1)
        for q in (other, question):
            gateway = make_gateway(tmp_path, FakeTransport([question, other], models),
                                   models=models)
            await QuestionRunner(gateway, models, JevClient(gateway, models.jev.model),
                                 seed=0, skip={"art_r2"}).run(q)
        self.sample = [question, other]
        return question, gateway, CallCache(tmp_path / "cache.sqlite")

    @staticmethod
    def drop(cache: CallCache, key: str) -> None:
        """Delete one call from a test cache file."""
        with sqlite3.connect(cache.path) as conn:
            conn.execute("DELETE FROM calls WHERE key = ?", (key,))

    async def estimate(self, question: Question, cache: CallCache) -> Any:
        """The estimate for a skip-art_r2 run of this one question."""
        return await estimate_from_cache(self.sample, [question], load_models(), cache, 0,
                                         {"art_r2"})

    @pytest.mark.asyncio
    async def test_fully_cached(self, tmp_path: Path) -> None:
        """Test that a fully cached question needs no new spend."""
        question, _, cache = await self.cached_question(tmp_path)

        estimate = await self.estimate(question, cache)

        assert estimate.new_usd == 0.0
        assert estimate.missing_calls == 0
        assert estimate.cached_questions == [question.id]

    @pytest.mark.asyncio
    async def test_one_missing_critique(self, tmp_path: Path) -> None:
        """Test that one missing critique costs one critique, not the whole question."""
        question, gateway, cache = await self.cached_question(tmp_path)
        models = load_models()
        critic = next(m for m in models.ids if m != models.roles["judge"])
        record = next(r for r in gateway.records.values()
                      if r.stage == "critique" and r.model == critic)
        self.drop(cache, record.key)

        estimate = await self.estimate(question, cache)

        assert estimate.missing_calls == 1
        assert estimate.partial_questions == [question.id]
        assert estimate.new_usd == pytest.approx(
            estimate.cost_model.call_cost("critique", critic, question)
        )
        assert estimate.new_usd < estimate.partial_whole_usd / 10

    @pytest.mark.asyncio
    async def test_missing_answer_prices_its_dependents(self, tmp_path: Path) -> None:
        """Test that a missing answer also prices the calls built on it."""
        question, gateway, cache = await self.cached_question(tmp_path)
        models = load_models()
        lost = next(m for m in models.ids if m not in models.roles.values())
        record = next(r for r in gateway.records.values()
                      if r.stage == "answer" and r.model == lost)
        self.drop(cache, record.key)

        estimate = await self.estimate(question, cache)

        run_calls = set().union(*plan_calls(models, {"art_r2"}).values())
        dependents = answer_dependents(run_calls, models, {lost})
        # Its answer, the 3 critiques of it (the judge's is one of them), its 3 critiques,
        # fusion and both Jev picks; never the other answers or the critiques among them
        assert len(dependents) == 10
        assert ("answer", lost, "") in dependents
        assert not any(c[0] == "answer" and c[1] != lost for c in dependents)
        expected = sum(estimate.cost_model.call_cost(c[0], c[1], question) for c in dependents)
        assert estimate.new_usd == pytest.approx(expected)
        assert 0 < estimate.new_usd < estimate.partial_whole_usd


class TestGateway402:
    """Test that running out of OpenRouter credits stops the run cleanly."""

    @pytest.mark.asyncio
    async def test_insufficient_credits(self, tmp_path: Path) -> None:
        """Test that HTTP 402 becomes a budget stop and marks the question incomplete."""
        class Broke:
            async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
                raise ApiError(402, "Insufficient credits")

        gateway = make_gateway(tmp_path, Broke())
        QUESTION_ID.set("q9")
        with pytest.raises(BudgetExceeded, match="insufficient credits"):
            await gateway.chat({"model": "deepseek/deepseek-v4-pro", "messages": [],
                                "max_tokens": 10})

        assert gateway.budget.exhausted
        assert gateway.budget.reserved == 0.0
        assert gateway.refused_questions == {"q9"}


class TestPartialRun:
    """End-to-end CLI run (fake provider) stopped part-way by the budget guard."""

    @pytest.mark.asyncio
    async def test_partial_outputs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that every output is written for the completed questions."""
        questions = [make_question(i, "calibration" if i < 4 else "test",
                                   ["SeqQA", "ProtocolQA"][i % 2], "ABCD"[i % 4])
                     for i in range(10)]
        monkeypatch.setattr(run_module, "load_sample",
                            lambda seed: (questions, {"SeqQA": 5, "ProtocolQA": 5}))
        monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path / "results")
        args = run_module.parse_args([
            "--dry-run", "--run-name", "partial", "--cache", str(tmp_path / "c.sqlite"),
            "--max-usd", "0.6", "--question-concurrency", "1", "--test-per-subset", "2",
            "--skip-methods", "art_r2",
        ])

        code = await run_module.run(args)

        out = tmp_path / "results" / "partial"
        config = json.loads((out / "config.json").read_text())
        status = config["status"]
        assert code == 1
        assert status["state"] == "partial"
        assert status["requested_questions"] == 8
        completed = status["completed_questions"]
        assert 0 < completed < 8
        # Calibration runs first, so the completed questions are a prefix of it
        per_question = [json.loads(line) for line in (out / "per_question.jsonl").open()]
        assert len(per_question) == completed
        assert [r["index"] for r in per_question] == [0, 1, 2, 3, 4, 5, 6, 7][:completed]
        assert f"partial: {completed} of 8 questions" in (out / "README.md").read_text()
        for name in ("summary.csv", "accuracy_vs_cost.png", "accuracy_vs_cost.svg",
                     "gate_reliability.png", "calls.jsonl", "questions.jsonl"):
            assert (out / name).exists(), name
        assert "art_r2" not in (out / "summary.csv").read_text()
        assert config["selection"]["skip_methods"] == ["art_r2"]
        assert config["selection"]["test_per_subset"] == 2
        assert config["projection"]["questions"] == 8

"""
Tests for Provider Refusals
===========================

The refusal classifier, cached refusals and --retry-refusals, the refused column and the
dropped-critique counts. Fake or mocked transports only; no network.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from experiments.pilot import run as run_module
from experiments.pilot.analysis import Analysis
from experiments.pilot.cache import CallCache, request_key
from experiments.pilot.fake_provider import FakeTransport
from experiments.pilot.gateway import (
    REFUSAL_KEY,
    ApiError,
    HttpTransport,
    ProviderRefusal,
    failure_cause,
    failure_text,
    is_provider_refusal,
)
from experiments.pilot.jev import JevClient
from experiments.pilot.methods import QuestionRunner
from experiments.pilot.settings import load_models
from experiments.pilot.tests.helpers import CountingTransport, make_gateway, make_question

FLAGGED = (
    "This content was flagged for possible biological risk. If this seems wrong, try "
    "rephrasing your request."
)
BODY = {"model": "anthropic/claude-sonnet-5.5", "messages": [{"role": "user", "content": "hi"}]}


class RefusingTransport(CountingTransport):
    """Refuses every request with a 403 until told to answer."""

    def __init__(self, status: int | str = 403, message: str = FLAGGED) -> None:
        """Create the transport."""
        super().__init__()
        self.status = status
        self.message = message
        self.refuse = True

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Refuse, or answer once refuse is off."""
        if self.refuse:
            self.calls += 1
            await asyncio.sleep(0)
            raise ApiError(self.status, self.message)
        return await super().post(url, body)


class TestClassifier:
    """Test cases for is_provider_refusal and failure_cause."""

    def test_flagged_403_is_a_refusal(self) -> None:
        """Test that a 403 saying the content was flagged is a refusal, as int or string."""
        assert is_provider_refusal(403, FLAGGED)
        assert is_provider_refusal("403", '{"error": {"message": "Content FLAGGED"}}')

    def test_other_errors_are_not(self) -> None:
        """Test that other statuses, and 403s without "flagged", are not refusals."""
        assert not is_provider_refusal(403, "Key limit exceeded")
        assert not is_provider_refusal(400, FLAGGED)
        assert not is_provider_refusal(500, FLAGGED)
        assert not is_provider_refusal(None, FLAGGED)

    def test_failure_cause(self) -> None:
        """Test the three causes recorded for a failed call or critique."""
        assert failure_cause(failure_text(ProviderRefusal(403, FLAGGED))) == "refused"
        unparseable = ValueError("Critique missing scores for: ACCURACY")
        assert failure_cause(failure_text(unparseable)) == "unparseable"
        assert failure_cause(failure_text(ValueError("Empty completion from x"))) == "other"
        assert failure_cause(failure_text(ApiError(403, "Key limit exceeded"))) == "other"
        assert failure_cause(None) == "other"


class TestCachedRefusals:
    """Test cases for refusals in the gateway and the cache."""

    @pytest.mark.asyncio
    async def test_refusal_cached_at_zero_cost(self, tmp_path: Path) -> None:
        """Test that a refusal is stored at cost 0 and replayed without another call."""
        transport = RefusingTransport()
        gateway = make_gateway(tmp_path, transport)

        with pytest.raises(ProviderRefusal):
            await gateway.chat(BODY)

        stored = gateway.cache.get(request_key("chat", BODY))
        assert stored is not None
        assert stored.cost_usd == 0.0
        assert stored.response[REFUSAL_KEY]["status"] == 403
        assert gateway.budget.spent == 0.0
        assert gateway.records == {}
        assert len(gateway.refusals) == 1

        # A rerun (new gateway, same cache file) replays the refusal
        rerun = make_gateway(tmp_path, transport)
        with pytest.raises(ProviderRefusal):
            await rerun.chat(BODY)
        assert transport.calls == 1

    @pytest.mark.asyncio
    async def test_other_403_not_cached(self, tmp_path: Path) -> None:
        """Test that a 403 without "flagged" is an ordinary error and is not cached."""
        transport = RefusingTransport(message="Key limit exceeded")
        gateway = make_gateway(tmp_path, transport)

        with pytest.raises(ApiError) as exc_info:
            await gateway.chat(BODY)

        assert not isinstance(exc_info.value, ProviderRefusal)
        assert len(gateway.cache) == 0

    @pytest.mark.asyncio
    async def test_retry_refusals(self, tmp_path: Path) -> None:
        """Test that --retry-refusals sends a cached refusal once and caches the new outcome."""
        transport = RefusingTransport()
        with pytest.raises(ProviderRefusal):
            await make_gateway(tmp_path, transport).chat(BODY)
        transport.refuse = False

        gateway = make_gateway(tmp_path, transport)
        gateway.retry_refusals = True
        records = await asyncio.gather(gateway.chat(BODY), gateway.chat(BODY))
        again = await gateway.chat(BODY)

        assert transport.calls == 2  # the original refusal and one retry
        assert all(r.response["choices"][0]["message"]["content"] == "ANSWER: A"
                   for r in [*records, again])
        assert REFUSAL_KEY not in gateway.cache.get(request_key("chat", BODY)).response

    @pytest.mark.asyncio
    async def test_http_403_through_real_transport(self, tmp_path: Path) -> None:
        """Test that OpenRouter's 403 body, through HttpTransport, becomes a cached refusal."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"code": 403, "message": FLAGGED}})

        transport = HttpTransport("test-key", backoff_base=0.0)
        transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = make_gateway(tmp_path, transport)

        with pytest.raises(ProviderRefusal):
            await gateway.chat(BODY)
        await transport.close()

        assert CallCache(tmp_path / "cache.sqlite").total_cost() == 0.0
        assert gateway.live_calls == 1


class RefusingFake(FakeTransport):
    """Fake provider whose GPT model refuses its answers on chosen questions and all critiques."""

    refuse_answers: set[str] = set()

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Refuse GPT requests as described; otherwise answer as the fake provider."""
        if body.get("model") == gpt_id():
            system = body["messages"][0]["content"]
            user = body["messages"][-1]["content"]
            question = self._find(user)
            if "evaluator" in system or (question and question.id in self.refuse_answers):
                self.calls += 1
                raise ApiError(403, FLAGGED)
        return await super().post(url, body)


def gpt_id() -> str:
    """The GPT model's ID."""
    return next(m for m in load_models().ids if "gpt" in m)


class TestRefusalsInResults:
    """Test cases for refusals in method outcomes, summary.csv and the README."""

    @pytest.mark.asyncio
    async def test_dropped_critiques_by_cause(self, tmp_path: Path) -> None:
        """Test that a tournament records each dropped critique with its critic and cause."""
        models = load_models()
        question = make_question(0)
        transport = RefusingFake([question], models)
        gateway = make_gateway(tmp_path, transport, models=models)
        runner = QuestionRunner(gateway, models, JevClient(gateway, models.jev.model),
                                skip={"art_r2"})

        record = await runner.run(question)

        [round_record] = record["methods"]["art_r1"]["rounds"]
        dropped = round_record["dropped_critiques"]
        assert len(dropped) == round_record["failed_critiques"]
        # The fake provider's other models sometimes write unparseable critiques
        by_gpt = [d for d in dropped if d["critic"] == gpt_id()]
        assert len(by_gpt) == 3 and {d["cause"] for d in by_gpt} == {"refused"}
        assert {d["cause"] for d in dropped if d["critic"] != gpt_id()} <= {"unparseable"}
        rows = Analysis([record], gateway.records, models, None, skip={"art_r2"}) \
            .dropped_critiques()
        gpt_label = models.get(gpt_id()).label
        assert {"method": "art_r1", "critic": gpt_label, "refused": 3, "unparseable": 0,
                "other": 0, "total": 3} in rows

    @pytest.mark.asyncio
    async def test_cli_refused_column(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the refused column and tables, and that a rerun replays every refusal."""
        questions = [make_question(i, "calibration" if i < 4 else "test",
                                   ["SeqQA", "ProtocolQA"][i % 2], "ABCD"[i % 4])
                     for i in range(8)]
        transports: list[RefusingFake] = []

        def make_transport(*args: Any) -> RefusingFake:
            transport = RefusingFake(*args)
            transport.refuse_answers = {"q5", "q6"}
            transports.append(transport)
            return transport

        monkeypatch.setattr(run_module, "load_sample",
                            lambda seed: (questions, {"SeqQA": 4, "ProtocolQA": 4}))
        monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(run_module, "FakeTransport", make_transport)
        argv = ["--dry-run", "--cache", str(tmp_path / "c.sqlite"), "--skip-methods", "art_r2"]

        assert await run_module.run(run_module.parse_args([*argv, "--run-name", "a"])) == 0
        assert await run_module.run(run_module.parse_args([*argv, "--run-name", "b"])) == 0

        assert transports[0].calls > 0
        assert transports[1].calls == 0  # refusals replayed from the cache
        gpt_label = load_models().get(gpt_id()).label
        outputs = []
        for name in ("a", "b"):
            out = tmp_path / "results" / name
            rows = {r["method"]: r for r in csv.DictReader((out / "summary.csv").open())}
            outputs.append(rows)
            gpt = rows[f"single:{gpt_label}"]
            assert gpt["refused"] == "2"
            assert gpt["unparseable"] == "0"
            readme = (out / "README.md").read_text()
            assert "| unparseable | refused |" in readme
            assert "## Provider refusals" in readme
            assert "## Dropped critiques (test split)" in readme
            config = json.loads((out / "config.json").read_text())
            assert {"model": gpt_label, "stage": "answer", "calibration": 0, "test": 2} in \
                config["refusals"]["by_model_stage"]
            art = {r["critic"]: r for r in config["dropped_critiques"]["rows"]
                   if r["method"] == "art_r1"}
            assert art[gpt_label]["refused"] > 0
            assert sum(r["refused"] for r in art.values()) == art[gpt_label]["refused"]
        # A refused answer still counts as wrong, and a replay changes nothing
        assert outputs[0] == outputs[1]
        assert outputs[0][f"single:{gpt_label}"]["accuracy"] <= str(2 / 4)


class TestBackfillLog:
    """Test cases for attributing run.log failure lines to tournaments (backfill_refusals)."""

    def test_log_failures(self, tmp_path: Path) -> None:
        """Test tournament attribution, refusal detection and the per-question cutoff."""
        from experiments.pilot.backfill_refusals import log_failures, read_log

        stamp = "2026-10-25 04:42:15.097 | {:<8} | art_framework.tournament.engine:f:1 - {}"
        refused = f"API error 403: {FLAGGED}"
        lines = [
            stamp.format("INFO", "Starting tournament A for query: Protocol:"),
            "Shared prefix...",
            stamp.format("INFO", "Starting tournament B for query: Other question..."),
            stamp.format("ERROR", f"Agent m1 failed to generate response: {refused}"),
            stamp.format("WARNING", f"Critique from m1 to m2 failed: {refused}"),
            stamp.format("WARNING", "Critique from m3 to m2 failed: Critique missing scores"),
            stamp.format("WARNING", "Round 1: 2 of 12 critiques failed and were dropped"),
            stamp.format("INFO", "Tournament B completed. Winner: m2"),
            stamp.format("INFO", "Tournament A completed. Winner: m2"),
            # The ELO router fit replays tournament A's question afterwards
            stamp.format("INFO", "Starting tournament C for query: Protocol:"),
            "Shared prefix...",
            stamp.format("WARNING", "Critique from m1 to m2 failed: client has been closed."),
            stamp.format("INFO", "Tournament C completed. Winner: m2"),
        ]
        path = tmp_path / "run.log"
        path.write_text("\n".join(lines) + "\n")

        failures, prefixes = log_failures(read_log(path), n_tournaments=2)

        assert prefixes == {"A": "Protocol:\nShared prefix", "B": "Other question"}
        answer, critique, unparseable = failures
        assert (answer.stage, answer.model, answer.tournaments) == ("answer", "m1", {"A", "B"})
        assert (critique.model, critique.target, critique.tournaments) == ("m1", "m2", {"B"})
        assert answer.refused and critique.refused and not unparseable.refused
        assert critique.message == FLAGGED
        assert answer.record.line == 4

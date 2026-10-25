"""
Tests for the Pilot Agent and the End-to-End Pipeline
=====================================================

The pipeline test runs every method on synthetic questions through the fake provider.
"""

from pathlib import Path

import pytest

from experiments.pilot.agents import PilotAgent
from experiments.pilot.analysis import Analysis
from experiments.pilot.fake_provider import FakeTransport
from experiments.pilot.jev import JevClient
from experiments.pilot.methods import QuestionRunner
from experiments.pilot.prompts import format_question
from experiments.pilot.run import fit_elo_router
from experiments.pilot.settings import load_models
from experiments.pilot.tests.helpers import CountingTransport, make_gateway, make_question


class TestPilotAgent:
    """Test cases for PilotAgent."""

    @pytest.mark.asyncio
    async def test_request_body(self, tmp_path: Path) -> None:
        """Test temperature 0, max_tokens and the reasoning setting in the body."""
        models = load_models()
        transport = CountingTransport(content="Because.\nANSWER: B", cost=0.003)
        gateway = make_gateway(tmp_path, transport, models=models)
        spec = models.get("google/gemini-3.8-flash")
        agent = PilotAgent(spec, gateway, models)

        response = await agent.generate_response("Question?", "q1")

        body = transport.bodies[0]
        assert body["temperature"] == 0
        assert body["max_tokens"] == models.max_tokens
        assert body["reasoning"] == {"effort": "low"}
        assert body["model"] == "google/gemini-3.8-flash"
        assert response.content.endswith("ANSWER: B")
        assert response.metadata["usage"]["cost"] == pytest.approx(0.003)
        assert agent.state.llm_calls == 1

    @pytest.mark.asyncio
    async def test_empty_completion_raises(self, tmp_path: Path) -> None:
        """Test that LLMAgent's empty-completion check still applies."""
        models = load_models()
        gateway = make_gateway(tmp_path, CountingTransport(content="  "), models=models)
        agent = PilotAgent(models.models[0], gateway, models)

        with pytest.raises(ValueError, match="Empty completion"):
            await agent._call_llm([{"role": "user", "content": "hi"}])


class TestPipeline:
    """End-to-end test with the fake provider."""

    @pytest.mark.asyncio
    async def test_all_methods_run(self, tmp_path: Path) -> None:
        """Test that every method produces an outcome and costs are consistent."""
        models = load_models()
        questions = [
            make_question(0, "calibration", "SeqQA", "A"),
            make_question(1, "calibration", "ProtocolQA", "B"),
            make_question(2, "test", "SeqQA", "C"),
            make_question(3, "test", "ProtocolQA", "D"),
        ]
        transport = FakeTransport(questions, models)
        gateway = make_gateway(tmp_path, transport, models=models)
        jev = JevClient(gateway, models.jev.model, models.jev.fallback_model)
        runner = QuestionRunner(gateway, models, jev, seed=0)

        raw = [await runner.run(q) for q in questions]
        calls_after_questions = transport.calls
        elo_router = await fit_elo_router(runner, questions[:2], models)

        # The ELO router replays cached art_r1 tournaments: no new calls
        assert transport.calls == calls_after_questions
        assert set(elo_router["top_model"]) == {"SeqQA", "ProtocolQA"}

        analysis = Analysis(raw, gateway.records, models, elo_router, seed=0)
        names = {row["method"] for row in analysis.summary()}
        assert {
            "majority_vote", "art_r1", "art_r2", "judge", "fusion", "elo_router",
            "best_single_calibration", "best_single_oracle", "jev_pick",
            "cascade_llm", "cascade_jev",
        } <= names
        assert sum(n.startswith("single:") for n in names) == 4

        for record in raw:
            out = analysis.outcomes[record["id"]]
            # art_r1: 4 answers + 12 critiques (fewer if a fake critique was unparseable
            # but the call still happened)
            assert out["art_r1"]["calls"] == 16
            # art_r2 adds 4 revisions and up to 12 new critiques
            assert out["art_r2"]["calls"] > out["art_r1"]["calls"]
            assert out["art_r2"]["cost_usd"] > out["art_r1"]["cost_usd"]
            # judge: 4 answers + 4 critiques; fusion: 4 answers + 1
            assert out["judge"]["calls"] == 8
            assert out["fusion"]["calls"] == 5
            assert out["jev_pick"]["calls"] == 6
            # The LLM gate is the judge critiquing the cheap answer, which is also one of
            # art_r1's critiques, so an escalated cascade costs exactly art_r1's calls
            cascade = out["cascade_llm"]
            expected = 2 if cascade["accepted"] else out["art_r1"]["calls"]
            assert cascade["calls"] == expected
            jev_cascade = out["cascade_jev"]
            expected = 2 if jev_cascade["accepted"] else out["art_r1"]["calls"] + 1
            assert jev_cascade["calls"] == expected
            single = out[f"single:{models.models[0].label}"]
            assert single["calls"] == 1

        rounds = raw[0]["methods"]["art_r2"]["rounds"]
        assert len(rounds) == 2
        assert set(rounds[0]["elo_before"]) == set(models.ids)
        assert rounds[1]["elo_before"] == rounds[0]["elo_after"]
        assert {r["confidence_source"] for r in rounds[1]["responses"].values()} == {
            "peer_score"
        }

    @pytest.mark.asyncio
    async def test_rerun_is_free(self, tmp_path: Path) -> None:
        """Test that running the same question again makes no live calls."""
        models = load_models()
        question = make_question(0)
        transport = FakeTransport([question], models)
        gateway = make_gateway(tmp_path, transport, models=models)
        runner = QuestionRunner(gateway, models, JevClient(gateway, models.jev.model), seed=0)
        await runner.run(question)

        rerun_transport = FakeTransport([question], models)
        rerun_gateway = make_gateway(tmp_path, rerun_transport, models=models)
        rerun = QuestionRunner(
            rerun_gateway, models, JevClient(rerun_gateway, models.jev.model), seed=0
        )
        await rerun.run(question)

        assert rerun_transport.calls == 0
        assert rerun_gateway.live_spend == 0.0

    def test_question_text_reaches_fake(self) -> None:
        """Test that the fake provider can match prompts back to questions."""
        models = load_models()
        question = make_question(5, correct="C")
        transport = FakeTransport([question], models)

        assert transport._find("prefix " + format_question(question)) is question

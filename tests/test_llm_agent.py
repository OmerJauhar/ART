"""
Tests for LLM Agent
===================

All HTTP traffic is served by httpx.MockTransport or a patched _call_llm, so no
network calls are made.
"""

import json
from typing import Any, Callable

import httpx
import pytest

from art_framework.agents.base import AgentConfig, AgentResponse, Critique
from art_framework.agents.llm_agent import LLMAgent
from art_framework.tournament.engine import TournamentConfig, TournamentEngine

VALID_CRITIQUE = """ACCURACY: 80
COHERENCE: 70
COMPLETENESS: 60
RELEVANCE: 90
OVERALL: 75

PROS:
- Clear

CONS:
- Short

SUGGESTIONS:
- Add examples
"""


def make_agent(
    handler: Callable[[httpx.Request], httpx.Response],
    temperature: float = 0.7,
) -> LLMAgent:
    """Create an LLMAgent whose HTTP client is served by a mock transport."""
    agent = LLMAgent(
        AgentConfig(
            name="Test",
            model_name="test-model",
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
        )
    )
    agent._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent._is_initialized = True
    return agent


def completion(
    content: str | None,
    usage: dict[str, Any] | None = None,
) -> httpx.Response:
    """Build a chat completions response."""
    body: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


class TestCallLLM:
    """Test cases for request building and response handling."""

    @pytest.mark.asyncio
    async def test_temperature_zero_is_sent(self) -> None:
        """Test that temperature 0 is not replaced by a default."""
        payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return completion("ok")

        agent = make_agent(handler, temperature=0.0)
        await agent._call_llm([{"role": "user", "content": "hi"}])
        await agent._call_llm([{"role": "user", "content": "hi"}], temperature=0.0)

        assert payloads[0]["temperature"] == 0
        assert payloads[1]["temperature"] == 0

    @pytest.mark.asyncio
    async def test_temperature_override_zero(self) -> None:
        """Test that an explicit temperature of 0 overrides the config value."""
        payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return completion("ok")

        agent = make_agent(handler, temperature=0.9)
        await agent._call_llm([{"role": "user", "content": "hi"}], temperature=0.0)

        assert payloads[0]["temperature"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", [None, "", "   \n"])
    async def test_empty_completion_raises(self, content: str | None) -> None:
        """Test that empty or missing completions raise ValueError."""
        agent = make_agent(lambda request: completion(content))

        with pytest.raises(ValueError, match="Empty completion from test-model"):
            await agent._call_llm([{"role": "user", "content": "hi"}])


class TestUsageTracking:
    """Test cases for token and cost tracking."""

    @pytest.mark.asyncio
    async def test_usage_accumulates(self) -> None:
        """Test that usage accumulates across calls and is attached to results."""
        usages = iter([
            {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001},
            {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.0005},
        ])
        agent = make_agent(lambda request: completion("Answer", next(usages)))

        first = await agent.generate_response("Q?", "q1")
        second = await agent.generate_response("Q?", "q1")

        assert first.metadata["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "cost": 0.001,
        }
        assert second.metadata["usage"]["prompt_tokens"] == 5
        assert agent.state.llm_calls == 2
        assert agent.state.prompt_tokens == 15
        assert agent.state.completion_tokens == 27
        assert agent.state.cost_usd == pytest.approx(0.0015)
        assert agent.to_dict()["cost_usd"] == pytest.approx(0.0015)

    @pytest.mark.asyncio
    async def test_missing_cost_defaults_to_zero(self) -> None:
        """Test that a response without usage.cost records zero cost."""
        agent = make_agent(
            lambda request: completion("Answer", {"prompt_tokens": 3, "completion_tokens": 4})
        )

        response = await agent.generate_response("Q?", "q1")

        assert response.metadata["usage"]["cost"] == 0.0
        assert agent.state.cost_usd == 0.0
        assert agent.state.prompt_tokens == 3

    @pytest.mark.asyncio
    async def test_usage_on_critique(self) -> None:
        """Test that critique usage is stored on the Critique."""
        agent = make_agent(
            lambda request: completion(
                VALID_CRITIQUE, {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.01}
            )
        )
        target = AgentResponse(agent_id="other", query_id="q1", content="A", confidence=0.5)

        critique = await agent.critique_response("Q?", target)

        assert critique.usage == {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.01}
        assert critique.score == 75.0


class TestImproveResponse:
    """Test cases for the revision prompt."""

    @pytest.mark.asyncio
    async def test_original_question_in_prompt(self) -> None:
        """Test that improve_response sends the original question to the model."""
        payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return completion("Revised answer")

        agent = make_agent(handler)
        original = AgentResponse(
            agent_id=agent.id, query_id="q1", content="Old answer", confidence=0.5
        )
        critique = Critique(
            critic_agent_id="other",
            target_agent_id=agent.id,
            response_id="r1",
            score=60.0,
            accuracy_score=60.0,
            coherence_score=60.0,
            completeness_score=60.0,
            relevance_score=60.0,
            cons=["Missing detail"],
        )

        improved = await agent.improve_response(
            original, [critique], original_query="What is the capital of France?"
        )

        user_message = payloads[0]["messages"][-1]["content"]
        assert "What is the capital of France?" in user_message
        assert "Old answer" in user_message
        assert "Missing detail" in user_message
        assert improved.content == "Revised answer"
        assert improved.confidence == 0.5
        assert improved.metadata["confidence_source"] == "unscored"


class TestCritiqueParser:
    """Test cases for critique score parsing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.agent = LLMAgent(AgentConfig(name="Parser", model_name="test-model"))

    def test_plain_format(self) -> None:
        """Test plain 'FIELD: score' lines."""
        scores = self.agent._parse_critique_scores(VALID_CRITIQUE)

        assert scores["accuracy"] == 80.0
        assert scores["coherence"] == 70.0
        assert scores["completeness"] == 60.0
        assert scores["relevance"] == 90.0
        assert scores["overall"] == 75.0
        assert scores["pros"] == ["Clear"]
        assert scores["cons"] == ["Short"]
        assert scores["suggestions"] == ["Add examples"]

    def test_bold_format(self) -> None:
        """Test markdown bold labels."""
        content = (
            "**ACCURACY:** 85\n**COHERENCE:** 80\n**COMPLETENESS:** 75\n"
            "**RELEVANCE**: 70\n**OVERALL:** 78\n"
        )
        scores = self.agent._parse_critique_scores(content)

        assert scores["accuracy"] == 85.0
        assert scores["relevance"] == 70.0
        assert scores["overall"] == 78.0

    def test_out_of_100_and_variants(self) -> None:
        """Test '/100' suffixes, decimals, dashes, mixed case and clamping."""
        content = (
            "ACCURACY: 85/100\nCoherence - 72.5\ncompleteness:   90\n"
            "RELEVANCE: 140\nOVERALL: 80/100\n"
        )
        scores = self.agent._parse_critique_scores(content)

        assert scores["accuracy"] == 85.0
        assert scores["coherence"] == 72.5
        assert scores["completeness"] == 90.0
        assert scores["relevance"] == 100.0
        assert scores["overall"] == 80.0

    def test_first_match_wins(self) -> None:
        """Test that the first occurrence of a field is used."""
        content = VALID_CRITIQUE + "\nACCURACY: 10\n"

        assert self.agent._parse_critique_scores(content)["accuracy"] == 80.0

    def test_bullet_styles(self) -> None:
        """Test that '-', '*' and '•' bullets are all collected."""
        content = VALID_CRITIQUE.replace("- Clear", "- Clear\n* Concise\n• Correct")

        assert self.agent._parse_critique_scores(content)["pros"] == [
            "Clear",
            "Concise",
            "Correct",
        ]

    def test_missing_overall_raises(self) -> None:
        """Test that a missing field raises instead of defaulting."""
        content = VALID_CRITIQUE.replace("OVERALL: 75\n", "")

        with pytest.raises(ValueError, match="OVERALL"):
            self.agent._parse_critique_scores(content)

    def test_unparseable_raises(self) -> None:
        """Test that free text with no scores raises and lists every field."""
        with pytest.raises(ValueError) as exc_info:
            self.agent._parse_critique_scores("This answer looks fine to me.")

        for name in ("ACCURACY", "COHERENCE", "COMPLETENESS", "RELEVANCE", "OVERALL"):
            assert name in str(exc_info.value)


def uniform_critique(score: float) -> str:
    """Build a critique that gives the same score on every field."""
    fields = ("ACCURACY", "COHERENCE", "COMPLETENESS", "RELEVANCE", "OVERALL")
    return "\n".join(f"{name}: {score}" for name in fields)


def scripted_agent(name: str, critique_scores: dict[str, float]) -> LLMAgent:
    """
    Create an LLMAgent with a patched _call_llm.

    Answers contain the agent's name; critiques score each answer by the name of the
    agent that wrote it, using critique_scores. A missing name yields free text.
    """
    agent = LLMAgent(AgentConfig(name=name, model_name="test-model"))
    agent._is_initialized = True

    async def fake_call_llm(messages: list[dict[str, str]], **kwargs: Any) -> str:
        if "evaluator" not in messages[0]["content"]:
            return f"Answer from {name}"
        for target, score in critique_scores.items():
            if f"Answer from {target}" in messages[-1]["content"]:
                return uniform_critique(score)
        return "I cannot score this."

    agent._call_llm = fake_call_llm  # type: ignore[method-assign]
    return agent


class TestEngineWithLLMAgents:
    """Test cases for engine behaviour with LLM agents."""

    @pytest.mark.asyncio
    async def test_confidence_from_peer_scores(self) -> None:
        """Test that scored responses take confidence from their round score."""
        scores = {"A": 90.0, "B": 60.0, "C": 30.0}
        agents = [scripted_agent(name, scores) for name in scores]
        engine = TournamentEngine(
            agents=agents,
            config=TournamentConfig(rounds_per_tournament=1),
        )

        result = await engine.run_tournament("Q?")
        round_result = result.rounds[0]

        assert round_result.failed_critiques == 0
        for agent in agents:
            response = round_result.responses[agent.id]
            assert round_result.scores[agent.id] == pytest.approx(scores[agent.name])
            assert response.confidence == pytest.approx(round_result.scores[agent.id] / 100)
            assert response.metadata["confidence_source"] == "peer_score"

    @pytest.mark.asyncio
    async def test_failed_critiques_counted(self) -> None:
        """Test that unparseable critiques are dropped and counted."""
        agents = [
            scripted_agent("A", {"B": 50.0, "C": 50.0}),
            scripted_agent("B", {"A": 50.0, "C": 50.0}),
            scripted_agent("C", {"A": 50.0}),  # cannot score B
        ]
        engine = TournamentEngine(
            agents=agents,
            config=TournamentConfig(rounds_per_tournament=1),
        )

        result = await engine.run_tournament("Q?")
        round_result = result.rounds[0]

        assert round_result.failed_critiques == 1
        assert round_result.to_dict()["failed_critiques"] == 1
        assert len(round_result.critiques[agents[1].id]) == 1

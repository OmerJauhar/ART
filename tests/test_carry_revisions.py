"""
Tests for Carrying Revisions Between Rounds
===========================================

Uses in-process fake agents that count calls; no network or sleeps.
"""

from typing import Any

import pytest

from art_framework.agents.base import Agent, AgentConfig, AgentResponse, Critique
from art_framework.tournament.engine import TournamentConfig, TournamentEngine

QUALITY = {"A": 90.0, "B": 70.0, "C": 50.0}


class CountingAgent(Agent):
    """
    Fake agent that counts calls and versions its answers.

    Answers read "<name> v<k>": generation gives v0 and each revision increments k.
    Critiques score every field with the target author's QUALITY value.
    """

    def __init__(
        self,
        name: str,
        fail_first_generation: bool = False,
        fail_revisions: bool = False,
    ) -> None:
        """Initialize the counting agent."""
        super().__init__(AgentConfig(name=name, model_name="fake"))
        self.fail_first_generation = fail_first_generation
        self.fail_revisions = fail_revisions
        self.generate_calls = 0
        self.critique_calls = 0
        self.revise_calls = 0
        self.critiqued_contents: list[str] = []
        self.revised_from: list[tuple[str, int, str | None]] = []

    async def initialize(self) -> None:
        """Initialize the agent."""
        self._is_initialized = True

    async def generate_response(
        self,
        query: str,
        query_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate version 0 of this agent's answer."""
        self.generate_calls += 1
        if self.fail_first_generation and self.generate_calls == 1:
            raise RuntimeError("generation failed")
        return AgentResponse(
            agent_id=self.id, query_id=query_id, content=f"{self.name} v0", confidence=0.5
        )

    async def critique_response(
        self,
        original_query: str,
        target_response: AgentResponse,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        """Score the target by its author's fixed quality."""
        self.critique_calls += 1
        self.critiqued_contents.append(target_response.content)
        score = QUALITY[target_response.content.split()[0]]
        return Critique(
            critic_agent_id=self.id,
            target_agent_id=target_response.agent_id,
            response_id=f"{self.name}-{self.critique_calls}",
            score=score,
            accuracy_score=score,
            coherence_score=score,
            completeness_score=score,
            relevance_score=score,
        )

    async def improve_response(
        self,
        original_response: AgentResponse,
        critiques: list[Critique],
        *,
        original_query: str | None = None,
    ) -> AgentResponse:
        """Return the next version of the answer."""
        self.revise_calls += 1
        self.revised_from.append((original_response.content, len(critiques), original_query))
        if self.fail_revisions:
            raise RuntimeError("revision failed")
        version = int(original_response.content.rsplit("v", 1)[1]) + 1
        return AgentResponse(
            agent_id=self.id,
            query_id=original_response.query_id,
            content=f"{self.name} v{version}",
            confidence=0.5,
        )


def make_engine(agents: list[CountingAgent], **config: Any) -> TournamentEngine:
    """Create an engine over the given agents."""
    return TournamentEngine(agents=agents, config=TournamentConfig(**config))


class TestCarryRevisions:
    """Test cases for carrying revised answers into later rounds."""

    @pytest.mark.asyncio
    async def test_generate_called_once_per_agent(self) -> None:
        """Test that only the first round generates answers."""
        agents = [CountingAgent(name) for name in QUALITY]

        await make_engine(agents, rounds_per_tournament=4).run_tournament("Q?")

        assert [a.generate_calls for a in agents] == [1, 1, 1]

    @pytest.mark.asyncio
    async def test_later_rounds_critique_revised_content(self) -> None:
        """Test that round r+1 scores the answers revised from round r."""
        agents = [CountingAgent(name) for name in QUALITY]

        result = await make_engine(agents, rounds_per_tournament=3).run_tournament("Q?")

        for r, round_result in enumerate(result.rounds):
            for agent in agents:
                assert round_result.responses[agent.id].content == f"{agent.name} v{r}"
        # Each revision starts from the answer scored in the previous round, with its
        # critiques and the original question
        for agent in agents:
            assert agent.revised_from == [
                (f"{agent.name} v0", 2, "Q?"),
                (f"{agent.name} v1", 2, "Q?"),
            ]
        # Each answer version is critiqued by the two other agents
        critiqued = sorted(c for a in agents for c in a.critiqued_contents)
        expected = sorted(f"{n} v{k}" for n in QUALITY for k in range(3) for _ in range(2))
        assert critiqued == expected

    @pytest.mark.asyncio
    async def test_call_counts_match_formula(self) -> None:
        """Test N generations + R*N*(N-1) critiques + (R-1)*N revisions for N=3, R=3."""
        n, r = 3, 3
        agents = [CountingAgent(name) for name in QUALITY]

        await make_engine(agents, rounds_per_tournament=r).run_tournament("Q?")

        assert sum(a.generate_calls for a in agents) == n
        assert sum(a.critique_calls for a in agents) == r * n * (n - 1)
        assert sum(a.revise_calls for a in agents) == (r - 1) * n

    @pytest.mark.asyncio
    async def test_best_response_scored_in_last_round(self) -> None:
        """Test that the best response is one critiqued and scored in the final round."""
        agents = [CountingAgent(name) for name in QUALITY]

        result = await make_engine(agents, rounds_per_tournament=3).run_tournament("Q?")
        last = result.rounds[-1]

        assert result.best_response is last.responses[result.winner_agent_id]
        assert last.critiques[result.winner_agent_id]
        assert result.winner_agent_id in last.scores
        assert result.best_response.content == "A v2"
        assert result.best_response.metadata["confidence_source"] == "peer_score"

    @pytest.mark.asyncio
    async def test_no_improvement_carries_answers_unchanged(self) -> None:
        """Test that disabling improvement re-scores the same answers without regenerating."""
        agents = [CountingAgent(name) for name in QUALITY]

        result = await make_engine(
            agents, rounds_per_tournament=3, enable_response_improvement=False
        ).run_tournament("Q?")

        assert [a.generate_calls for a in agents] == [1, 1, 1]
        assert [a.revise_calls for a in agents] == [0, 0, 0]
        for round_result in result.rounds:
            assert round_result.responses[agents[0].id].content == "A v0"
        # Carried answers are copies, so each round keeps its own confidence
        assert result.rounds[0].responses[agents[0].id] is not (
            result.rounds[1].responses[agents[0].id]
        )

    @pytest.mark.asyncio
    async def test_failed_revision_falls_back_to_previous_answer(self) -> None:
        """Test that a failed revision re-scores the previous answer instead of regenerating."""
        agents = [CountingAgent("A", fail_revisions=True), CountingAgent("B"), CountingAgent("C")]

        result = await make_engine(agents, rounds_per_tournament=3).run_tournament("Q?")

        assert agents[0].generate_calls == 1
        assert [r.responses[agents[0].id].content for r in result.rounds] == ["A v0"] * 3

    @pytest.mark.asyncio
    async def test_missing_answer_generated_for_that_agent_only(self) -> None:
        """Test that an agent with nothing to carry gets a fresh answer, alone."""
        agents = [
            CountingAgent("A", fail_first_generation=True),
            CountingAgent("B"),
            CountingAgent("C"),
        ]

        result = await make_engine(agents, rounds_per_tournament=2).run_tournament("Q?")

        assert [a.generate_calls for a in agents] == [2, 1, 1]
        assert agents[0].id not in result.rounds[0].responses
        assert result.rounds[1].responses[agents[0].id].content == "A v0"
        assert result.rounds[1].responses[agents[1].id].content == "B v1"

    @pytest.mark.asyncio
    async def test_championship_rounds_score_revisions(self) -> None:
        """Test that championship rounds revise and re-score the finalists' answers."""
        agents = [CountingAgent(name) for name in QUALITY]

        result = await make_engine(
            agents,
            rounds_per_tournament=2,
            enable_nested_tournaments=True,
            top_agents_for_finals=2,
            championship_rounds=2,
        ).run_tournament("Q?")

        assert [a.generate_calls for a in agents] == [1, 1, 1]
        assert len(result.rounds) == 4
        finals = result.rounds[-1].responses
        assert set(finals) == {agents[0].id, agents[1].id}
        assert finals[agents[0].id].content == "A v3"
        # The eliminated agent is not revised for rounds it does not play
        assert agents[2].revise_calls == 1

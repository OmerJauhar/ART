"""
Tests for Tournament Engine
===========================
"""

import pytest
from art_framework.agents.base import AgentConfig
from art_framework.agents.mock_agent import MockAgent
from art_framework.tournament.engine import (
    TournamentEngine,
    TournamentConfig,
    TournamentStatus,
)


class TestTournamentEngine:
    """Test cases for the Tournament Engine."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.agents = [
            MockAgent(
                AgentConfig(name="Agent1", model_name="model-1"),
                response_quality=0.8,
                deterministic=True,
            ),
            MockAgent(
                AgentConfig(name="Agent2", model_name="model-2"),
                response_quality=0.7,
                deterministic=True,
            ),
            MockAgent(
                AgentConfig(name="Agent3", model_name="model-3"),
                response_quality=0.6,
                deterministic=True,
            ),
        ]
        self.config = TournamentConfig(
            num_tournaments=1,
            rounds_per_tournament=2,
        )
        self.engine = TournamentEngine(
            agents=self.agents,
            config=self.config,
        )
    
    def test_engine_initialization(self) -> None:
        """Test tournament engine initialization."""
        assert len(self.engine.agents) == 3
        assert self.engine.config.rounds_per_tournament == 2
    
    def test_minimum_agents_required(self) -> None:
        """Test that at least 2 agents are required."""
        with pytest.raises(ValueError):
            TournamentEngine(
                agents=[self.agents[0]],
                config=self.config,
            )
    
    @pytest.mark.asyncio
    async def test_initialize_agents(self) -> None:
        """Test agent initialization."""
        await self.engine.initialize_agents()
        
        for agent in self.engine.agents.values():
            assert agent._is_initialized is True
    
    @pytest.mark.asyncio
    async def test_run_tournament(self) -> None:
        """Test running a complete tournament."""
        result = await self.engine.run_tournament(
            query="What is machine learning?",
            query_id="test-tournament-1",
        )
        
        assert result.tournament_id is not None
        assert result.query == "What is machine learning?"
        assert result.status == TournamentStatus.COMPLETED
        assert len(result.rounds) == 2
        assert result.winner_agent_id is not None
        assert result.best_response is not None
    
    @pytest.mark.asyncio
    async def test_elo_updates_after_tournament(self) -> None:
        """Test that ELO scores are updated after tournament."""
        initial_elos = {
            agent.id: agent.elo_score
            for agent in self.engine.agents.values()
        }
        
        await self.engine.run_tournament(
            query="Explain quantum computing",
            query_id="test-tournament-2",
        )
        
        # At least one agent should have changed ELO
        elo_changed = False
        for agent in self.engine.agents.values():
            if agent.elo_score != initial_elos[agent.id]:
                elo_changed = True
                break
        
        assert elo_changed is True
    
    @pytest.mark.asyncio
    async def test_tournament_rankings(self) -> None:
        """Test that tournament produces proper rankings."""
        result = await self.engine.run_tournament(
            query="What is artificial intelligence?",
            query_id="test-tournament-3",
        )
        
        rankings = result.final_rankings
        
        # Rankings should be sorted by ELO descending
        for i in range(len(rankings) - 1):
            assert rankings[i][1] >= rankings[i + 1][1]
    
    @pytest.mark.asyncio
    async def test_leaderboard(self) -> None:
        """Test leaderboard generation."""
        await self.engine.run_tournament(
            query="Test query",
            query_id="test-4",
        )
        
        leaderboard = self.engine.get_leaderboard()
        
        assert len(leaderboard) == 3
        assert leaderboard[0]["rank"] == 1
        assert "agent_id" in leaderboard[0]
        assert "elo_score" in leaderboard[0]
    
    @pytest.mark.asyncio
    async def test_tournament_history(self) -> None:
        """Test tournament history tracking."""
        await self.engine.run_tournament(
            query="Query 1",
            query_id="history-test-1",
        )
        await self.engine.run_tournament(
            query="Query 2",
            query_id="history-test-2",
        )
        
        history = self.engine.get_tournament_history()
        
        assert len(history) == 2
        assert history[0]["query"] == "Query 1"
        assert history[1]["query"] == "Query 2"
    
    def test_add_agent(self) -> None:
        """Test adding a new agent."""
        new_agent = MockAgent(
            AgentConfig(name="Agent4", model_name="model-4"),
            response_quality=0.9,
        )
        
        self.engine.add_agent(new_agent)
        
        assert len(self.engine.agents) == 4
        assert new_agent.id in self.engine.agents
    
    def test_remove_agent(self) -> None:
        """Test removing an agent."""
        agent_id = list(self.engine.agents.keys())[0]
        
        self.engine.remove_agent(agent_id)
        
        assert len(self.engine.agents) == 2
        assert agent_id not in self.engine.agents


class TestTournamentConfig:
    """Test cases for TournamentConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TournamentConfig()
        
        assert config.num_tournaments == 1
        assert config.rounds_per_tournament == 3
        assert config.k_factor == 32.0
        assert "accuracy" in config.scoring_weights
    
    def test_invalid_num_tournaments(self) -> None:
        """Test that invalid num_tournaments raises error."""
        with pytest.raises(ValueError):
            TournamentConfig(num_tournaments=0)
    
    def test_invalid_rounds(self) -> None:
        """Test that invalid rounds_per_tournament raises error."""
        with pytest.raises(ValueError):
            TournamentConfig(rounds_per_tournament=0)
    
    def test_weight_normalization(self) -> None:
        """Test that scoring weights are normalized."""
        config = TournamentConfig(
            scoring_weights={
                "accuracy": 2.0,
                "coherence": 2.0,
                "completeness": 2.0,
                "relevance": 2.0,
            }
        )
        
        total = sum(config.scoring_weights.values())
        assert total == pytest.approx(1.0, abs=0.001)

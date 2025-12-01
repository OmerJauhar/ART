"""
Tests for Mock Agent
====================
"""

import pytest
from art_framework.agents.base import AgentConfig, AgentResponse
from art_framework.agents.mock_agent import MockAgent


class TestMockAgent:
    """Test cases for the Mock Agent."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = AgentConfig(
            name="TestAgent",
            model_name="test-model",
            temperature=0.7,
            max_tokens=1024,
        )
        self.agent = MockAgent(
            config=self.config,
            response_quality=0.8,
            deterministic=True,
        )
    
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Test agent initialization."""
        await self.agent.initialize()
        assert self.agent._is_initialized is True
    
    @pytest.mark.asyncio
    async def test_generate_response(self) -> None:
        """Test response generation."""
        await self.agent.initialize()
        
        response = await self.agent.generate_response(
            query="What is the capital of France?",
            query_id="test-query-1",
        )
        
        assert isinstance(response, AgentResponse)
        assert response.agent_id == self.agent.id
        assert response.query_id == "test-query-1"
        assert len(response.content) > 0
        assert 0 <= response.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_critique_response(self) -> None:
        """Test critiquing another agent's response."""
        await self.agent.initialize()
        
        target_response = AgentResponse(
            agent_id="other-agent",
            query_id="test-query-1",
            content="The capital of France is Paris.",
            confidence=0.9,
        )
        
        critique = await self.agent.critique_response(
            original_query="What is the capital of France?",
            target_response=target_response,
        )
        
        assert critique.critic_agent_id == self.agent.id
        assert critique.target_agent_id == "other-agent"
        assert 0 <= critique.score <= 100
        assert 0 <= critique.accuracy_score <= 100
        assert len(critique.pros) > 0
        assert len(critique.cons) > 0
    
    @pytest.mark.asyncio
    async def test_improve_response(self) -> None:
        """Test response improvement based on critiques."""
        await self.agent.initialize()
        
        original_response = await self.agent.generate_response(
            query="Explain quantum physics",
            query_id="test-query-2",
        )
        
        # Create mock critiques
        from art_framework.agents.base import Critique
        critiques = [
            Critique(
                critic_agent_id="critic-1",
                target_agent_id=self.agent.id,
                response_id="resp-1",
                score=75,
                accuracy_score=80,
                coherence_score=70,
                completeness_score=75,
                relevance_score=75,
                pros=["Good structure"],
                cons=["Needs more examples"],
                suggestions=["Add examples"],
            ),
        ]
        
        improved = await self.agent.improve_response(original_response, critiques)
        
        assert improved.agent_id == self.agent.id
        assert len(improved.content) > len(original_response.content)
        assert improved.confidence >= original_response.confidence
    
    def test_agent_state(self) -> None:
        """Test agent state tracking."""
        assert self.agent.state.wins == 0
        assert self.agent.state.losses == 0
        assert self.agent.state.draws == 0
        
        self.agent.record_win()
        assert self.agent.state.wins == 1
        assert self.agent.state.win_rate == 1.0
        
        self.agent.record_loss()
        assert self.agent.state.losses == 1
        assert self.agent.state.win_rate == 0.5
        
        self.agent.record_draw()
        assert self.agent.state.draws == 1
        assert self.agent.state.total_matches == 3
    
    def test_context_management(self) -> None:
        """Test context history management."""
        assert len(self.agent.state.context_history) == 0
        
        self.agent.add_to_context({"role": "user", "content": "Hello"})
        assert len(self.agent.state.context_history) == 1
        
        self.agent.add_to_context({"role": "assistant", "content": "Hi there"})
        assert len(self.agent.state.context_history) == 2
        
        self.agent.clear_context()
        assert len(self.agent.state.context_history) == 0
    
    def test_agent_to_dict(self) -> None:
        """Test agent serialization."""
        agent_dict = self.agent.to_dict()
        
        assert "id" in agent_dict
        assert agent_dict["name"] == "TestAgent"
        assert agent_dict["model_name"] == "test-model"
        assert "elo_score" in agent_dict
        assert "wins" in agent_dict
        assert "losses" in agent_dict
        assert "draws" in agent_dict


class TestAgentConfig:
    """Test cases for AgentConfig."""
    
    def test_valid_config(self) -> None:
        """Test valid configuration creation."""
        config = AgentConfig(
            name="ValidAgent",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2048,
        )
        
        assert config.name == "ValidAgent"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
    
    def test_invalid_temperature(self) -> None:
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError):
            AgentConfig(
                name="BadAgent",
                model_name="gpt-4",
                temperature=3.0,  # Invalid: > 2.0
            )
    
    def test_invalid_max_tokens(self) -> None:
        """Test that invalid max_tokens raises error."""
        with pytest.raises(ValueError):
            AgentConfig(
                name="BadAgent",
                model_name="gpt-4",
                max_tokens=0,  # Invalid: < 1
            )

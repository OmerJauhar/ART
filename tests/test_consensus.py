"""
Tests for Consensus Engine
==========================
"""

import pytest
from art_framework.agents.base import AgentResponse
from art_framework.consensus.engine import ConsensusEngine, FusionStrategy


class TestConsensusEngine:
    """Test cases for the Consensus Engine."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = ConsensusEngine(
            default_strategy=FusionStrategy.WEIGHTED_VOTING,
            top_k_agents=3,
        )
        
        self.responses = {
            "agent_a": AgentResponse(
                agent_id="agent_a",
                query_id="test-query",
                content="Response from Agent A with high quality content.",
                confidence=0.9,
            ),
            "agent_b": AgentResponse(
                agent_id="agent_b",
                query_id="test-query",
                content="Response from Agent B with medium quality.",
                confidence=0.7,
            ),
            "agent_c": AgentResponse(
                agent_id="agent_c",
                query_id="test-query",
                content="Response from Agent C with lower quality.",
                confidence=0.6,
            ),
        }
        
        self.elo_scores = {
            "agent_a": 1600,
            "agent_b": 1500,
            "agent_c": 1400,
        }
    
    def test_top_response_fusion(self) -> None:
        """Test top response fusion strategy."""
        result = self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
            strategy=FusionStrategy.TOP_RESPONSE,
        )
        
        assert result.strategy == FusionStrategy.TOP_RESPONSE
        assert "agent_a" in result.contributing_agents
        assert "Agent A" in result.final_response
    
    def test_weighted_voting_fusion(self) -> None:
        """Test weighted voting fusion strategy."""
        result = self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
            strategy=FusionStrategy.WEIGHTED_VOTING,
        )
        
        assert result.strategy == FusionStrategy.WEIGHTED_VOTING
        assert len(result.contributing_agents) <= self.engine.top_k_agents
        assert result.confidence > 0
    
    def test_contextual_aggregation(self) -> None:
        """Test contextual aggregation strategy."""
        result = self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
            strategy=FusionStrategy.CONTEXTUAL_AGGREGATION,
        )
        
        assert result.strategy == FusionStrategy.CONTEXTUAL_AGGREGATION
        assert "Aggregated" in result.final_response or "Perspective" in result.final_response
    
    def test_hybrid_synthesis(self) -> None:
        """Test hybrid synthesis strategy."""
        result = self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
            strategy=FusionStrategy.HYBRID_SYNTHESIS,
        )
        
        assert result.strategy == FusionStrategy.HYBRID_SYNTHESIS
        assert len(result.contributing_agents) > 0
    
    def test_weight_calculation(self) -> None:
        """Test that weights are properly calculated."""
        weights = self.engine._calculate_weights(
            responses=self.responses,
            elo_scores=self.elo_scores,
        )
        
        # Weights should sum to 1
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.001)
        
        # Higher ELO should have higher weight
        assert weights["agent_a"] > weights["agent_c"]
    
    def test_confidence_calculation(self) -> None:
        """Test consensus confidence calculation."""
        weights = self.engine._calculate_weights(
            responses=self.responses,
            elo_scores=self.elo_scores,
        )
        
        confidence = self.engine._calculate_consensus_confidence(
            responses=self.responses,
            weights=weights,
        )
        
        assert 0 <= confidence <= 1
    
    def test_empty_responses(self) -> None:
        """Test handling of empty responses."""
        result = self.engine.generate_consensus(
            responses={},
            elo_scores={},
        )
        
        assert "No responses available" in result.final_response or result.confidence == 0
    
    def test_confidence_threshold_filtering(self) -> None:
        """Test that low confidence responses are filtered."""
        engine = ConsensusEngine(
            default_strategy=FusionStrategy.WEIGHTED_VOTING,
            min_confidence_threshold=0.8,
        )
        
        result = engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
        )
        
        # Only agent_a has confidence >= 0.8
        # Result should still work with fallback
        assert result.final_response is not None
    
    def test_history_tracking(self) -> None:
        """Test consensus history is tracked."""
        self.engine.clear_history()
        
        self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
        )
        self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
        )
        
        history = self.engine.get_history()
        assert len(history) == 2
    
    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        result = self.engine.generate_consensus(
            responses=self.responses,
            elo_scores=self.elo_scores,
        )
        
        result_dict = result.to_dict()
        
        assert "consensus_id" in result_dict
        assert "strategy" in result_dict
        assert "final_response" in result_dict
        assert "confidence" in result_dict
        assert "contributing_agents" in result_dict

"""
Tests for ELO Calculator
========================
"""

import pytest
from art_framework.tournament.elo import EloCalculator, MatchResult


class TestEloCalculator:
    """Test cases for the ELO calculator."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = EloCalculator(k_factor=32.0)
    
    def test_expected_score_equal_ratings(self) -> None:
        """Test expected score when ratings are equal."""
        expected = self.calculator.expected_score(1500, 1500)
        assert expected == pytest.approx(0.5, abs=0.001)
    
    def test_expected_score_higher_rating(self) -> None:
        """Test expected score when player A has higher rating."""
        expected = self.calculator.expected_score(1700, 1500)
        assert expected > 0.5
        assert expected < 1.0
    
    def test_expected_score_lower_rating(self) -> None:
        """Test expected score when player A has lower rating."""
        expected = self.calculator.expected_score(1300, 1500)
        assert expected < 0.5
        assert expected > 0.0
    
    def test_calculate_new_rating_win(self) -> None:
        """Test rating calculation after a win."""
        new_rating = self.calculator.calculate_new_rating(
            current_rating=1500,
            expected=0.5,
            actual=1.0,
        )
        assert new_rating > 1500
    
    def test_calculate_new_rating_loss(self) -> None:
        """Test rating calculation after a loss."""
        new_rating = self.calculator.calculate_new_rating(
            current_rating=1500,
            expected=0.5,
            actual=0.0,
        )
        assert new_rating < 1500
    
    def test_calculate_new_rating_draw(self) -> None:
        """Test rating calculation after a draw with equal ratings."""
        new_rating = self.calculator.calculate_new_rating(
            current_rating=1500,
            expected=0.5,
            actual=0.5,
        )
        assert new_rating == pytest.approx(1500, abs=0.001)
    
    def test_process_match_clear_win(self) -> None:
        """Test processing a match with a clear winner."""
        update_a, update_b = self.calculator.process_match(
            agent_a_id="agent_a",
            agent_a_rating=1500,
            agent_a_match_score=90,
            agent_b_id="agent_b",
            agent_b_rating=1500,
            agent_b_match_score=60,
        )
        
        assert update_a.match_result == MatchResult.WIN
        assert update_b.match_result == MatchResult.LOSS
        assert update_a.new_score > update_a.old_score
        assert update_b.new_score < update_b.old_score
    
    def test_process_match_draw(self) -> None:
        """Test processing a match that ends in a draw."""
        update_a, update_b = self.calculator.process_match(
            agent_a_id="agent_a",
            agent_a_rating=1500,
            agent_a_match_score=75,
            agent_b_id="agent_b",
            agent_b_rating=1500,
            agent_b_match_score=73,  # Within draw threshold
        )
        
        assert update_a.match_result == MatchResult.DRAW
        assert update_b.match_result == MatchResult.DRAW
    
    def test_process_multi_agent_match(self) -> None:
        """Test processing a match with more than two agents."""
        agent_scores = {
            "agent_a": (1500, 90),  # High score
            "agent_b": (1500, 70),  # Medium score
            "agent_c": (1500, 50),  # Low score
        }
        
        updates = self.calculator.process_multi_agent_match(agent_scores)
        
        assert len(updates) == 3
        
        # Agent A should have gained ELO
        agent_a_update = next(u for u in updates if u.agent_id == "agent_a")
        assert agent_a_update.change > 0
        
        # Agent C should have lost ELO
        agent_c_update = next(u for u in updates if u.agent_id == "agent_c")
        assert agent_c_update.change < 0
    
    def test_rating_bounds(self) -> None:
        """Test that ratings are bounded."""
        # Test minimum bound
        new_rating = self.calculator.calculate_new_rating(
            current_rating=150,
            expected=0.99,
            actual=0.0,
        )
        assert new_rating >= self.calculator.min_score
        
        # Test maximum bound
        new_rating = self.calculator.calculate_new_rating(
            current_rating=2950,
            expected=0.01,
            actual=1.0,
        )
        assert new_rating <= self.calculator.max_score
    
    def test_k_factor_override(self) -> None:
        """Test that K-factor can be overridden."""
        # Lower K-factor = smaller changes
        update_low_k, _ = self.calculator.process_match(
            agent_a_id="a",
            agent_a_rating=1500,
            agent_a_match_score=90,
            agent_b_id="b",
            agent_b_rating=1500,
            agent_b_match_score=60,
            k_factor=16,
        )
        
        # Higher K-factor = larger changes
        update_high_k, _ = self.calculator.process_match(
            agent_a_id="c",
            agent_a_rating=1500,
            agent_a_match_score=90,
            agent_b_id="d",
            agent_b_rating=1500,
            agent_b_match_score=60,
            k_factor=48,
        )
        
        assert abs(update_high_k.change) > abs(update_low_k.change)
    
    def test_get_history(self) -> None:
        """Test retrieving ELO update history."""
        self.calculator.clear_history()
        
        self.calculator.process_match(
            agent_a_id="agent_a",
            agent_a_rating=1500,
            agent_a_match_score=80,
            agent_b_id="agent_b",
            agent_b_rating=1500,
            agent_b_match_score=60,
        )
        
        history = self.calculator.get_history()
        assert len(history) == 2
        
        agent_a_history = self.calculator.get_history("agent_a")
        assert len(agent_a_history) == 1
        assert agent_a_history[0].agent_id == "agent_a"
    
    def test_get_statistics(self) -> None:
        """Test calculating statistics for an agent."""
        self.calculator.clear_history()
        
        # Simulate multiple matches
        for i in range(5):
            self.calculator.process_match(
                agent_a_id="test_agent",
                agent_a_rating=1500,
                agent_a_match_score=80,
                agent_b_id=f"opponent_{i}",
                agent_b_rating=1500,
                agent_b_match_score=60,
            )
        
        stats = self.calculator.get_statistics("test_agent")
        
        assert stats["agent_id"] == "test_agent"
        assert stats["matches"] == 5
        assert stats["wins"] == 5
        assert stats["win_rate"] == pytest.approx(1.0)

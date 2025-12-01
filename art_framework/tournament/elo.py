"""
ELO Calculator Module
=====================

Implements the ELO rating system for ranking agents based on
tournament performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MatchResult(str, Enum):
    """Result of a match between two agents."""
    
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class EloUpdate:
    """Record of an ELO score update."""
    
    agent_id: str
    old_score: float
    new_score: float
    change: float
    match_result: MatchResult
    opponent_id: str
    opponent_score: float
    k_factor: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "old_score": self.old_score,
            "new_score": self.new_score,
            "change": self.change,
            "match_result": self.match_result.value,
            "opponent_id": self.opponent_id,
            "opponent_score": self.opponent_score,
            "k_factor": self.k_factor,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MatchReport:
    """Report of a match between agents."""
    
    match_id: str
    agent_a_id: str
    agent_b_id: str
    agent_a_score: float
    agent_b_score: float
    winner_id: str | None  # None for draw
    result: MatchResult
    score_differential: float
    elo_updates: list[EloUpdate] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class EloCalculator:
    """
    ELO Rating Calculator for tournament-based agent ranking.
    
    The ELO system provides a method for calculating relative skill levels
    of players/agents in zero-sum games. This implementation supports:
    - Configurable K-factor for sensitivity adjustment
    - Partial wins with weighted scoring
    - Draw handling
    - Score bounds and normalization
    
    Attributes:
        k_factor: Base K-factor determining ELO sensitivity
        min_score: Minimum allowed ELO score
        max_score: Maximum allowed ELO score
        default_score: Default starting ELO score
        draw_threshold: Score difference threshold for draws
    """
    
    def __init__(
        self,
        k_factor: float = 32.0,
        min_score: float = 100.0,
        max_score: float = 3000.0,
        default_score: float = 1500.0,
        draw_threshold: float = 5.0,
    ) -> None:
        """
        Initialize the ELO calculator.
        
        Args:
            k_factor: Base K-factor for ELO calculations (default: 32)
            min_score: Minimum ELO score (default: 100)
            max_score: Maximum ELO score (default: 3000)
            default_score: Starting ELO score (default: 1500)
            draw_threshold: Score difference to consider as draw (default: 5)
        """
        self.k_factor = k_factor
        self.min_score = min_score
        self.max_score = max_score
        self.default_score = default_score
        self.draw_threshold = draw_threshold
        self._history: list[EloUpdate] = []
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Uses the standard ELO expected score formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        
        Args:
            rating_a: ELO rating of player A
            rating_b: ELO rating of player B
            
        Returns:
            Expected score for player A (0 to 1)
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))
    
    def calculate_new_rating(
        self,
        current_rating: float,
        expected: float,
        actual: float,
        k_factor: float | None = None,
    ) -> float:
        """
        Calculate new ELO rating based on match result.
        
        Formula: R_new = R_old + K * (S - E)
        Where:
        - R_old is the current rating
        - K is the K-factor
        - S is the actual score (1 for win, 0.5 for draw, 0 for loss)
        - E is the expected score
        
        Args:
            current_rating: Current ELO rating
            expected: Expected score (from expected_score())
            actual: Actual score (1=win, 0.5=draw, 0=loss, or weighted value)
            k_factor: Override K-factor (uses instance default if None)
            
        Returns:
            New ELO rating (bounded by min/max scores)
        """
        k = k_factor or self.k_factor
        change = k * (actual - expected)
        new_rating = current_rating + change
        
        # Bound the rating
        return max(self.min_score, min(self.max_score, new_rating))
    
    def process_match(
        self,
        agent_a_id: str,
        agent_a_rating: float,
        agent_a_match_score: float,
        agent_b_id: str,
        agent_b_rating: float,
        agent_b_match_score: float,
        k_factor: float | None = None,
    ) -> tuple[EloUpdate, EloUpdate]:
        """
        Process a match between two agents and calculate ELO updates.
        
        Args:
            agent_a_id: ID of agent A
            agent_a_rating: Current ELO rating of agent A
            agent_a_match_score: Match performance score of agent A (0-100)
            agent_b_id: ID of agent B
            agent_b_rating: Current ELO rating of agent B
            agent_b_match_score: Match performance score of agent B (0-100)
            k_factor: Override K-factor
            
        Returns:
            Tuple of (EloUpdate for A, EloUpdate for B)
        """
        k = k_factor or self.k_factor
        
        # Determine match result
        score_diff = agent_a_match_score - agent_b_match_score
        
        if abs(score_diff) <= self.draw_threshold:
            # Draw
            actual_a = 0.5
            actual_b = 0.5
            result_a = MatchResult.DRAW
            result_b = MatchResult.DRAW
        elif score_diff > 0:
            # Agent A wins - use weighted scoring for partial wins
            actual_a = min(1.0, 0.5 + (score_diff / 100.0) * 0.5)
            actual_b = 1.0 - actual_a
            result_a = MatchResult.WIN
            result_b = MatchResult.LOSS
        else:
            # Agent B wins
            actual_b = min(1.0, 0.5 + (-score_diff / 100.0) * 0.5)
            actual_a = 1.0 - actual_b
            result_a = MatchResult.LOSS
            result_b = MatchResult.WIN
        
        # Calculate expected scores
        expected_a = self.expected_score(agent_a_rating, agent_b_rating)
        expected_b = 1.0 - expected_a
        
        # Calculate new ratings
        new_rating_a = self.calculate_new_rating(
            agent_a_rating, expected_a, actual_a, k
        )
        new_rating_b = self.calculate_new_rating(
            agent_b_rating, expected_b, actual_b, k
        )
        
        # Create update records
        update_a = EloUpdate(
            agent_id=agent_a_id,
            old_score=agent_a_rating,
            new_score=new_rating_a,
            change=new_rating_a - agent_a_rating,
            match_result=result_a,
            opponent_id=agent_b_id,
            opponent_score=agent_b_rating,
            k_factor=k,
        )
        
        update_b = EloUpdate(
            agent_id=agent_b_id,
            old_score=agent_b_rating,
            new_score=new_rating_b,
            change=new_rating_b - agent_b_rating,
            match_result=result_b,
            opponent_id=agent_a_id,
            opponent_score=agent_a_rating,
            k_factor=k,
        )
        
        # Store in history
        self._history.extend([update_a, update_b])
        
        return update_a, update_b
    
    def process_multi_agent_match(
        self,
        agent_scores: dict[str, tuple[float, float]],
        k_factor: float | None = None,
    ) -> list[EloUpdate]:
        """
        Process a match with more than two agents.
        
        Uses a round-robin comparison where each agent is compared
        against every other agent.
        
        Args:
            agent_scores: Dict mapping agent_id to (current_elo, match_score)
            k_factor: Override K-factor
            
        Returns:
            List of EloUpdates for all agents
        """
        k = k_factor or self.k_factor
        # Reduce K-factor for multi-agent matches to prevent extreme swings
        adjusted_k = k / max(1, len(agent_scores) - 1)
        
        # Track cumulative ELO changes
        elo_changes: dict[str, float] = {agent_id: 0.0 for agent_id in agent_scores}
        update_details: dict[str, list[tuple[str, float, MatchResult]]] = {
            agent_id: [] for agent_id in agent_scores
        }
        
        # Round-robin comparison
        agent_ids = list(agent_scores.keys())
        for i, agent_a_id in enumerate(agent_ids):
            for agent_b_id in agent_ids[i + 1:]:
                rating_a, score_a = agent_scores[agent_a_id]
                rating_b, score_b = agent_scores[agent_b_id]
                
                update_a, update_b = self.process_match(
                    agent_a_id, rating_a, score_a,
                    agent_b_id, rating_b, score_b,
                    adjusted_k,
                )
                
                # Remove from history (we'll add aggregated updates later)
                self._history = self._history[:-2]
                
                elo_changes[agent_a_id] += update_a.change
                elo_changes[agent_b_id] += update_b.change
                
                update_details[agent_a_id].append(
                    (agent_b_id, rating_b, update_a.match_result)
                )
                update_details[agent_b_id].append(
                    (agent_a_id, rating_a, update_b.match_result)
                )
        
        # Create aggregated updates
        updates = []
        for agent_id, (current_rating, _) in agent_scores.items():
            new_rating = max(
                self.min_score,
                min(self.max_score, current_rating + elo_changes[agent_id])
            )
            
            # Determine overall result
            wins = sum(1 for _, _, r in update_details[agent_id] if r == MatchResult.WIN)
            losses = sum(1 for _, _, r in update_details[agent_id] if r == MatchResult.LOSS)
            
            if wins > losses:
                overall_result = MatchResult.WIN
            elif losses > wins:
                overall_result = MatchResult.LOSS
            else:
                overall_result = MatchResult.DRAW
            
            update = EloUpdate(
                agent_id=agent_id,
                old_score=current_rating,
                new_score=new_rating,
                change=elo_changes[agent_id],
                match_result=overall_result,
                opponent_id="multi-agent",
                opponent_score=0,
                k_factor=adjusted_k,
            )
            
            updates.append(update)
            self._history.append(update)
        
        return updates
    
    def get_history(self, agent_id: str | None = None) -> list[EloUpdate]:
        """
        Get ELO update history.
        
        Args:
            agent_id: Filter by agent ID (returns all if None)
            
        Returns:
            List of EloUpdate records
        """
        if agent_id:
            return [u for u in self._history if u.agent_id == agent_id]
        return self._history.copy()
    
    def clear_history(self) -> None:
        """Clear ELO update history."""
        self._history.clear()
    
    def get_statistics(self, agent_id: str) -> dict[str, Any]:
        """
        Get statistics for an agent based on history.
        
        Args:
            agent_id: Agent ID to get statistics for
            
        Returns:
            Dictionary of statistics
        """
        agent_history = self.get_history(agent_id)
        
        if not agent_history:
            return {
                "agent_id": agent_id,
                "matches": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "avg_change": 0.0,
                "total_change": 0.0,
                "max_gain": 0.0,
                "max_loss": 0.0,
            }
        
        wins = sum(1 for u in agent_history if u.match_result == MatchResult.WIN)
        losses = sum(1 for u in agent_history if u.match_result == MatchResult.LOSS)
        draws = sum(1 for u in agent_history if u.match_result == MatchResult.DRAW)
        changes = [u.change for u in agent_history]
        
        return {
            "agent_id": agent_id,
            "matches": len(agent_history),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(agent_history) if agent_history else 0.0,
            "avg_change": sum(changes) / len(changes) if changes else 0.0,
            "total_change": sum(changes),
            "max_gain": max(changes) if changes else 0.0,
            "max_loss": min(changes) if changes else 0.0,
        }

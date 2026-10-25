"""
Consensus Engine Module
=======================

Synthesizes final responses based on ELO rankings and agent confidence scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from art_framework.agents.base import AgentResponse


class FusionStrategy(str, Enum):
    """Strategies for fusing multiple responses."""
    
    WEIGHTED_VOTING = "weighted_voting"
    TOP_RESPONSE = "top_response"
    CONTEXTUAL_AGGREGATION = "contextual_aggregation"
    HYBRID_SYNTHESIS = "hybrid_synthesis"


@dataclass
class ConsensusResult:
    """Result of consensus generation."""
    
    consensus_id: str
    strategy: FusionStrategy
    final_response: str
    confidence: float
    contributing_agents: list[str]
    agent_weights: dict[str, float]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consensus_id": self.consensus_id,
            "strategy": self.strategy.value,
            "final_response": self.final_response,
            "confidence": self.confidence,
            "contributing_agents": self.contributing_agents,
            "agent_weights": self.agent_weights,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ConsensusEngine:
    """
    Consensus Engine for synthesizing final responses.
    
    Supports multiple fusion strategies:
    - Weighted Voting: Select response with highest weighted score
    - Top Response: Simply return the highest-ranked agent's response
    - Contextual Aggregation: Combine key points from multiple responses
    - Hybrid Synthesis: Generate new response using top responses as input
    
    The engine uses ELO rankings and confidence scores to weight
    agent contributions to the final response.
    """
    
    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_VOTING,
        min_confidence_threshold: float = 0.5,
        top_k_agents: int = 3,
    ) -> None:
        """
        Initialize the consensus engine.
        
        Args:
            default_strategy: Default fusion strategy to use
            min_confidence_threshold: Minimum confidence to include in consensus
            top_k_agents: Number of top agents to consider for consensus
        """
        self.default_strategy = default_strategy
        self.min_confidence_threshold = min_confidence_threshold
        self.top_k_agents = top_k_agents
        self._history: list[ConsensusResult] = []
    
    def generate_consensus(
        self,
        responses: dict[str, AgentResponse],
        elo_scores: dict[str, float],
        strategy: FusionStrategy | None = None,
    ) -> ConsensusResult:
        """
        Generate consensus from multiple agent responses.
        
        Args:
            responses: Dict mapping agent_id to AgentResponse
            elo_scores: Dict mapping agent_id to ELO score
            strategy: Fusion strategy to use (defaults to instance setting)
            
        Returns:
            ConsensusResult containing the synthesized response
        """
        import uuid
        
        strategy = strategy or self.default_strategy
        
        # Calculate weights based on ELO scores
        agent_weights = self._calculate_weights(responses, elo_scores)
        
        # Filter by confidence threshold; unscored responses carry only a placeholder
        # confidence, so they never pass
        filtered_responses = {
            agent_id: response
            for agent_id, response in responses.items()
            if response.confidence >= self.min_confidence_threshold
            and response.metadata.get("confidence_source") != "unscored"
        }
        
        if not filtered_responses:
            filtered_responses = responses  # Fall back to all responses
        
        # Apply fusion strategy
        if strategy == FusionStrategy.TOP_RESPONSE:
            result = self._top_response_fusion(filtered_responses, agent_weights)
        elif strategy == FusionStrategy.WEIGHTED_VOTING:
            result = self._weighted_voting_fusion(filtered_responses, agent_weights)
        elif strategy == FusionStrategy.CONTEXTUAL_AGGREGATION:
            result = self._contextual_aggregation(filtered_responses, agent_weights)
        elif strategy == FusionStrategy.HYBRID_SYNTHESIS:
            result = self._hybrid_synthesis(filtered_responses, agent_weights)
        else:
            result = self._top_response_fusion(filtered_responses, agent_weights)
        
        # Calculate overall confidence
        confidence = self._calculate_consensus_confidence(
            filtered_responses, agent_weights
        )
        
        consensus_result = ConsensusResult(
            consensus_id=str(uuid.uuid4()),
            strategy=strategy,
            final_response=result["content"],
            confidence=confidence,
            contributing_agents=result["agents"],
            agent_weights=agent_weights,
            reasoning=result["reasoning"],
        )
        
        self._history.append(consensus_result)
        logger.info(f"Generated consensus using {strategy.value}")
        
        return consensus_result
    
    def _calculate_weights(
        self,
        responses: dict[str, AgentResponse],
        elo_scores: dict[str, float],
    ) -> dict[str, float]:
        """Calculate normalized weights for each agent."""
        # Combine ELO scores with response confidence
        raw_weights = {}
        for agent_id, response in responses.items():
            elo = elo_scores.get(agent_id, 1500)
            # Normalize ELO to 0-1 range (assuming 1000-2000 typical range)
            elo_normalized = (elo - 1000) / 1000
            elo_normalized = max(0.1, min(1.0, elo_normalized))
            
            # Combine with confidence
            raw_weights[agent_id] = elo_normalized * 0.6 + response.confidence * 0.4
        
        # Normalize weights to sum to 1
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            return {
                agent_id: weight / total_weight
                for agent_id, weight in raw_weights.items()
            }
        return {agent_id: 1.0 / len(raw_weights) for agent_id in raw_weights}
    
    def _top_response_fusion(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """Simply return the highest-weighted response."""
        if not responses:
            return {
                "content": "No responses available",
                "agents": [],
                "reasoning": "No valid responses to select from",
            }
        
        # Find agent with highest weight
        best_agent_id = max(weights.keys(), key=lambda x: weights.get(x, 0))
        best_response = responses.get(best_agent_id)
        
        if not best_response:
            best_agent_id = list(responses.keys())[0]
            best_response = responses[best_agent_id]
        
        return {
            "content": best_response.content,
            "agents": [best_agent_id],
            "reasoning": f"Selected top-ranked agent ({best_agent_id}) response",
        }
    
    def _weighted_voting_fusion(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """
        Select response with highest weighted score.
        
        For text responses, this effectively becomes top-response
        but could be extended with semantic similarity clustering.
        """
        if not responses:
            return {
                "content": "No responses available",
                "agents": [],
                "reasoning": "No valid responses to select from",
            }
        
        # For now, weighted voting is similar to top response
        # but we track all contributing agents
        sorted_agents = sorted(
            weights.keys(),
            key=lambda x: weights.get(x, 0),
            reverse=True
        )
        
        top_k = sorted_agents[:self.top_k_agents]
        best_agent_id = top_k[0] if top_k else list(responses.keys())[0]
        best_response = responses.get(best_agent_id)
        
        if not best_response:
            return self._top_response_fusion(responses, weights)
        
        return {
            "content": best_response.content,
            "agents": top_k,
            "reasoning": (
                f"Weighted voting selected {best_agent_id} with weight "
                f"{weights.get(best_agent_id, 0):.3f}"
            ),
        }
    
    def _contextual_aggregation(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """
        Aggregate key points from multiple responses.
        
        Combines insights from top-weighted agents into a structured response.
        """
        sorted_agents = sorted(
            weights.keys(),
            key=lambda x: weights.get(x, 0),
            reverse=True
        )
        
        top_k = sorted_agents[:self.top_k_agents]
        
        # Build aggregated response
        sections = []
        contributing_agents = []
        
        for i, agent_id in enumerate(top_k):
            response = responses.get(agent_id)
            if response:
                weight = weights.get(agent_id, 0)
                sections.append(
                    f"[Perspective {i+1} (weight: {weight:.2f})]\n{response.content}"
                )
                contributing_agents.append(agent_id)
        
        if not sections:
            return self._top_response_fusion(responses, weights)
        
        aggregated_content = (
            "# Aggregated Consensus Response\n\n"
            "The following response synthesizes insights from multiple expert agents:\n\n"
            + "\n\n---\n\n".join(sections)
            + "\n\n## Summary\n\n"
            "The above perspectives have been weighted by their ELO ranking and "
            "response confidence to provide a comprehensive answer."
        )
        
        return {
            "content": aggregated_content,
            "agents": contributing_agents,
            "reasoning": f"Contextual aggregation of top {len(contributing_agents)} agents",
        }
    
    def _hybrid_synthesis(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """
        Create a hybrid synthesized response.
        
        For full implementation, this would use an LLM to synthesize
        a new response from the top responses. Currently provides
        a structured combination.
        """
        sorted_agents = sorted(
            weights.keys(),
            key=lambda x: weights.get(x, 0),
            reverse=True
        )
        
        top_k = sorted_agents[:self.top_k_agents]
        contributing_agents = []
        
        # Extract key content from top responses
        key_points = []
        for agent_id in top_k:
            response = responses.get(agent_id)
            if response:
                contributing_agents.append(agent_id)
                # Extract first paragraph or significant portion
                content = response.content
                first_para = content.split('\n\n')[0] if '\n\n' in content else content[:500]
                key_points.append(first_para)
        
        if not key_points:
            return self._top_response_fusion(responses, weights)
        
        # Synthesize response
        primary_response = responses.get(top_k[0])
        if primary_response:
            synthesis = (
                f"{primary_response.content}\n\n"
                "---\n\n"
                "**Additional Perspectives:**\n\n"
            )
            
            for i, point in enumerate(key_points[1:], start=2):
                synthesis += f"*Agent {i}:* {point[:200]}...\n\n"
        else:
            synthesis = "\n\n".join(key_points)
        
        return {
            "content": synthesis,
            "agents": contributing_agents,
            "reasoning": f"Hybrid synthesis combining {len(contributing_agents)} perspectives",
        }
    
    def _calculate_consensus_confidence(
        self,
        responses: dict[str, AgentResponse],
        weights: dict[str, float],
    ) -> float:
        """Calculate overall confidence in the consensus."""
        if not responses:
            return 0.0
        
        # Weighted average of individual confidences
        weighted_confidence = sum(
            response.confidence * weights.get(agent_id, 0)
            for agent_id, response in responses.items()
        )
        
        # Factor in agreement among top agents
        confidences = [r.confidence for r in responses.values()]
        variance = (
            sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences)
            / len(confidences)
        ) if len(confidences) > 1 else 0
        
        # Lower variance = higher agreement = higher confidence
        agreement_bonus = max(0, 0.1 - variance * 0.2)
        
        return min(1.0, weighted_confidence + agreement_bonus)
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get consensus generation history."""
        return [c.to_dict() for c in self._history]
    
    def clear_history(self) -> None:
        """Clear consensus history."""
        self._history.clear()

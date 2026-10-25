"""
Tournament Engine Module
========================

Core component for managing tournaments, rounds, and agent interactions.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from loguru import logger

from art_framework.agents.base import Agent, AgentResponse, Critique
from art_framework.tournament.elo import EloCalculator, EloUpdate, MatchResult


class TournamentStatus(str, Enum):
    """Status of a tournament."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class RoundStatus(str, Enum):
    """Status of a round within a tournament."""
    
    PENDING = "pending"
    RESPONSE_GENERATION = "response_generation"
    CROSS_EVALUATION = "cross_evaluation"
    SCORING = "scoring"
    COMPLETED = "completed"


@dataclass
class RoundResult:
    """Result of a single round."""
    
    round_id: str
    round_number: int
    responses: dict[str, AgentResponse]  # agent_id -> response
    critiques: dict[str, list[Critique]]  # agent_id -> critiques received
    scores: dict[str, float]  # agent_id -> round score
    elo_updates: list[EloUpdate]
    best_response_agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_id": self.round_id,
            "round_number": self.round_number,
            "scores": self.scores,
            "elo_updates": [u.to_dict() for u in self.elo_updates],
            "best_response_agent_id": self.best_response_agent_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class TournamentResult:
    """Result of a complete tournament."""
    
    tournament_id: str
    query: str
    query_id: str
    rounds: list[RoundResult]
    final_rankings: list[tuple[str, float]]  # [(agent_id, final_elo), ...]
    best_response: AgentResponse
    winner_agent_id: str
    status: TournamentStatus
    config: "TournamentConfig"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tournament_id": self.tournament_id,
            "query": self.query,
            "query_id": self.query_id,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_rankings": self.final_rankings,
            "best_response": {
                "agent_id": self.best_response.agent_id,
                "content": self.best_response.content,
                "confidence": self.best_response.confidence,
            },
            "winner_agent_id": self.winner_agent_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""
    
    num_tournaments: int = 1
    rounds_per_tournament: int = 3
    k_factor: float = 32.0
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.35,
        "coherence": 0.25,
        "completeness": 0.25,
        "relevance": 0.15,
    })
    enable_response_improvement: bool = True
    dynamic_agent_selection: bool = False
    top_agents_for_finals: int = 2
    enable_nested_tournaments: bool = False
    championship_rounds: int = 2
    parallel_responses: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_tournaments < 1:
            raise ValueError("num_tournaments must be at least 1")
        if self.rounds_per_tournament < 1:
            raise ValueError("rounds_per_tournament must be at least 1")
        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive")
        
        # Validate scoring weights
        total_weight = sum(self.scoring_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            # Normalize weights
            for key in self.scoring_weights:
                self.scoring_weights[key] /= total_weight


class TournamentEngine:
    """
    Tournament Engine for managing multi-agent competitions.
    
    Responsibilities:
    - Scheduling matches between agents across multiple rounds
    - Computing ELO scores after each round
    - Handling multi-agent interactions
    - Maintaining tournament state and histories
    - Supporting nested/championship tournament logic
    """
    
    def __init__(
        self,
        agents: list[Agent],
        config: TournamentConfig | None = None,
        elo_calculator: EloCalculator | None = None,
        on_round_complete: Callable[[RoundResult], None] | None = None,
        on_tournament_complete: Callable[[TournamentResult], None] | None = None,
    ) -> None:
        """
        Initialize the tournament engine.
        
        Args:
            agents: List of agents participating in tournaments
            config: Tournament configuration
            elo_calculator: ELO calculator instance
            on_round_complete: Callback when a round completes
            on_tournament_complete: Callback when a tournament completes
        """
        if len(agents) < 2:
            raise ValueError("At least 2 agents are required for a tournament")
        
        self.agents = {agent.id: agent for agent in agents}
        self.config = config or TournamentConfig()
        self.elo_calculator = elo_calculator or EloCalculator(
            k_factor=self.config.k_factor
        )
        self.on_round_complete = on_round_complete
        self.on_tournament_complete = on_tournament_complete
        
        self._current_tournament_id: str | None = None
        self._current_round: int = 0
        self._tournament_history: list[TournamentResult] = []
        self._active_agents: set[str] = set(self.agents.keys())
    
    async def initialize_agents(self) -> None:
        """Initialize all agents."""
        init_tasks = [agent.initialize() for agent in self.agents.values()]
        await asyncio.gather(*init_tasks)
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def run_tournament(
        self,
        query: str,
        query_id: str | None = None,
        config_override: TournamentConfig | None = None,
    ) -> TournamentResult:
        """
        Run a complete tournament for a given query.
        
        Args:
            query: The query/task for agents to respond to
            query_id: Optional unique identifier for the query
            config_override: Optional config to override instance config
            
        Returns:
            TournamentResult containing all rounds and final rankings
        """
        config = config_override or self.config
        query_id = query_id or str(uuid.uuid4())
        tournament_id = str(uuid.uuid4())
        
        logger.info(f"Starting tournament {tournament_id} for query: {query[:50]}...")
        
        self._current_tournament_id = tournament_id
        self._active_agents = set(self.agents.keys())
        
        start_time = datetime.utcnow()
        all_rounds: list[RoundResult] = []
        
        try:
            # Run main tournament rounds
            for round_num in range(1, config.rounds_per_tournament + 1):
                self._current_round = round_num
                round_result = await self._run_round(
                    query, query_id, round_num, config
                )
                all_rounds.append(round_result)
                
                if self.on_round_complete:
                    self.on_round_complete(round_result)
                
                # Dynamic agent selection - filter to top performers
                if config.dynamic_agent_selection and round_num < config.rounds_per_tournament:
                    self._filter_top_agents(config.top_agents_for_finals)
            
            # Run nested/championship tournament if enabled
            if config.enable_nested_tournaments:
                championship_rounds = await self._run_championship(
                    query, query_id, config
                )
                all_rounds.extend(championship_rounds)
            
            # Calculate final rankings
            final_rankings = self._get_rankings()
            
            # Determine winner and best response
            winner_id = final_rankings[0][0]
            best_response = all_rounds[-1].responses[winner_id]
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TournamentResult(
                tournament_id=tournament_id,
                query=query,
                query_id=query_id,
                rounds=all_rounds,
                final_rankings=final_rankings,
                best_response=best_response,
                winner_agent_id=winner_id,
                status=TournamentStatus.COMPLETED,
                config=config,
                duration_ms=duration_ms,
            )
            
            self._tournament_history.append(result)
            
            if self.on_tournament_complete:
                self.on_tournament_complete(result)
            
            logger.info(f"Tournament {tournament_id} completed. Winner: {winner_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Tournament {tournament_id} failed: {e}")
            raise
        finally:
            self._current_tournament_id = None
            self._current_round = 0
    
    async def _run_round(
        self,
        query: str,
        query_id: str,
        round_number: int,
        config: TournamentConfig,
    ) -> RoundResult:
        """Run a single round of the tournament."""
        round_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting round {round_number}")
        
        # Step 1: Generate responses
        responses = await self._generate_responses(query, query_id, config)
        
        # Step 2: Cross-evaluation
        critiques = await self._cross_evaluate(query, responses, config)
        
        # Step 3: Calculate scores
        scores = self._calculate_scores(critiques, config)
        
        # Step 4: Update ELO ratings
        elo_updates = self._update_elo_ratings(scores)
        
        # Step 5: Improve responses if enabled
        if config.enable_response_improvement and round_number < config.rounds_per_tournament:
            responses = await self._improve_responses(query, responses, critiques)
        
        # Determine best response
        best_agent_id = max(scores.items(), key=lambda x: x[1])[0]
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return RoundResult(
            round_id=round_id,
            round_number=round_number,
            responses=responses,
            critiques=critiques,
            scores=scores,
            elo_updates=elo_updates,
            best_response_agent_id=best_agent_id,
            duration_ms=duration_ms,
        )
    
    async def _generate_responses(
        self,
        query: str,
        query_id: str,
        config: TournamentConfig,
    ) -> dict[str, AgentResponse]:
        """Generate responses from all active agents."""
        responses: dict[str, AgentResponse] = {}
        
        active_agents = [
            self.agents[aid] for aid in self._active_agents
            if aid in self.agents
        ]
        
        if config.parallel_responses:
            # Generate responses in parallel
            tasks = [
                agent.generate_response(query, query_id)
                for agent in active_agents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent, result in zip(active_agents, results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent.id} failed to generate response: {result}")
                else:
                    responses[agent.id] = result
        else:
            # Generate responses sequentially
            for agent in active_agents:
                try:
                    response = await agent.generate_response(query, query_id)
                    responses[agent.id] = response
                except Exception as e:
                    logger.error(f"Agent {agent.id} failed: {e}")
        
        return responses
    
    async def _cross_evaluate(
        self,
        query: str,
        responses: dict[str, AgentResponse],
        config: TournamentConfig,
    ) -> dict[str, list[Critique]]:
        """Have each agent critique all other agents' responses."""
        critiques: dict[str, list[Critique]] = {
            agent_id: [] for agent_id in responses
        }
        
        # Each agent critiques every other agent's response
        tasks = []
        task_info = []  # Track which agent pair each task corresponds to
        
        for critic_id in responses:
            for target_id in responses:
                if critic_id != target_id:
                    critic_agent = self.agents[critic_id]
                    target_response = responses[target_id]
                    
                    tasks.append(
                        critic_agent.critique_response(query, target_response)
                    )
                    task_info.append((critic_id, target_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (critic_id, target_id), result in zip(task_info, results):
            if isinstance(result, Exception):
                logger.warning(f"Critique from {critic_id} to {target_id} failed: {result}")
            else:
                critiques[target_id].append(result)
        
        return critiques
    
    def _calculate_scores(
        self,
        critiques: dict[str, list[Critique]],
        config: TournamentConfig,
    ) -> dict[str, float]:
        """Calculate weighted scores for each agent based on critiques."""
        scores: dict[str, float] = {}
        weights = config.scoring_weights
        
        for agent_id, agent_critiques in critiques.items():
            if not agent_critiques:
                scores[agent_id] = 50.0  # Default score if no critiques
                continue
            
            # Average the critique scores
            accuracy_scores = [c.accuracy_score for c in agent_critiques]
            coherence_scores = [c.coherence_score for c in agent_critiques]
            completeness_scores = [c.completeness_score for c in agent_critiques]
            relevance_scores = [c.relevance_score for c in agent_critiques]
            
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            avg_completeness = sum(completeness_scores) / len(completeness_scores)
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            
            # Calculate weighted score
            weighted_score = (
                weights.get("accuracy", 0.25) * avg_accuracy +
                weights.get("coherence", 0.25) * avg_coherence +
                weights.get("completeness", 0.25) * avg_completeness +
                weights.get("relevance", 0.25) * avg_relevance
            )
            
            scores[agent_id] = weighted_score
        
        return scores
    
    def _update_elo_ratings(
        self,
        scores: dict[str, float],
    ) -> list[EloUpdate]:
        """Update ELO ratings based on round scores."""
        # Prepare agent scores for multi-agent match
        agent_scores = {
            agent_id: (self.agents[agent_id].elo_score, score)
            for agent_id, score in scores.items()
            if agent_id in self.agents
        }
        
        # Process the multi-agent match
        updates = self.elo_calculator.process_multi_agent_match(agent_scores)
        
        # Apply updates to agents
        for update in updates:
            if update.agent_id in self.agents:
                agent = self.agents[update.agent_id]
                agent.elo_score = update.new_score
                
                # Update win/loss/draw counts
                if update.match_result == MatchResult.WIN:
                    agent.record_win()
                elif update.match_result == MatchResult.LOSS:
                    agent.record_loss()
                else:
                    agent.record_draw()
        
        return updates
    
    async def _improve_responses(
        self,
        query: str,
        responses: dict[str, AgentResponse],
        critiques: dict[str, list[Critique]],
    ) -> dict[str, AgentResponse]:
        """Allow agents to improve their responses based on critiques."""
        improved_responses: dict[str, AgentResponse] = {}
        
        tasks = []
        agent_ids = []
        
        for agent_id, response in responses.items():
            agent_critiques = critiques.get(agent_id, [])
            if agent_critiques:
                agent = self.agents[agent_id]
                tasks.append(
                    agent.improve_response(response, agent_critiques, original_query=query)
                )
                agent_ids.append(agent_id)
            else:
                improved_responses[agent_id] = response
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {agent_id} failed to improve response: {result}")
                improved_responses[agent_id] = responses[agent_id]
            else:
                improved_responses[agent_id] = result
        
        return improved_responses
    
    async def _run_championship(
        self,
        query: str,
        query_id: str,
        config: TournamentConfig,
    ) -> list[RoundResult]:
        """Run championship rounds with top agents."""
        # Filter to top agents
        rankings = self._get_rankings()
        top_n = min(config.top_agents_for_finals, len(rankings))
        self._active_agents = set(agent_id for agent_id, _ in rankings[:top_n])
        
        logger.info(f"Running championship with top {top_n} agents")
        
        championship_rounds = []
        for round_num in range(1, config.championship_rounds + 1):
            round_result = await self._run_round(
                query, query_id,
                config.rounds_per_tournament + round_num,
                config,
            )
            championship_rounds.append(round_result)
        
        return championship_rounds
    
    def _filter_top_agents(self, top_n: int) -> None:
        """Filter to keep only top N agents based on ELO."""
        rankings = self._get_rankings()
        self._active_agents = set(
            agent_id for agent_id, _ in rankings[:top_n]
        )
        logger.info(f"Filtered to top {top_n} agents: {self._active_agents}")
    
    def _get_rankings(self) -> list[tuple[str, float]]:
        """Get current agent rankings by ELO score."""
        rankings = [
            (agent_id, self.agents[agent_id].elo_score)
            for agent_id in self._active_agents
            if agent_id in self.agents
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get the current leaderboard."""
        rankings = self._get_rankings()
        leaderboard = []
        
        for rank, (agent_id, elo) in enumerate(rankings, start=1):
            agent = self.agents[agent_id]
            leaderboard.append({
                "rank": rank,
                "agent_id": agent_id,
                "agent_name": agent.name,
                "elo_score": elo,
                "wins": agent.state.wins,
                "losses": agent.state.losses,
                "draws": agent.state.draws,
                "win_rate": agent.state.win_rate,
            })
        
        return leaderboard
    
    def get_tournament_history(self) -> list[dict[str, Any]]:
        """Get history of all tournaments."""
        return [t.to_dict() for t in self._tournament_history]
    
    def add_agent(self, agent: Agent) -> None:
        """Add a new agent to the tournament."""
        self.agents[agent.id] = agent
        self._active_agents.add(agent.id)
        logger.info(f"Added agent {agent.id} to tournament")
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the tournament."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        self._active_agents.discard(agent_id)
        logger.info(f"Removed agent {agent_id} from tournament")

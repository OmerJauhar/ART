"""
Data Models Module
==================

Defines Pydantic models for API requests/responses and data transfer objects.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryStatus(str, Enum):
    """Status of a query being processed."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Query(BaseModel):
    """Query submitted to the ART framework."""
    
    id: str | None = None
    content: str = Field(..., min_length=1, max_length=10000)
    context: dict[str, Any] | None = None
    num_tournaments: int = Field(default=1, ge=1, le=10)
    rounds_per_tournament: int = Field(default=3, ge=1, le=20)
    custom_scoring_weights: dict[str, float] | None = None
    dynamic_agent_selection: bool = False
    status: QueryStatus = QueryStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("custom_scoring_weights")
    @classmethod
    def validate_scoring_weights(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            required_keys = {"accuracy", "coherence", "completeness", "relevance"}
            if not required_keys.issubset(v.keys()):
                raise ValueError(f"Scoring weights must include: {required_keys}")
            if any(w < 0 for w in v.values()):
                raise ValueError("Scoring weights must be non-negative")
        return v
    
    model_config = {"extra": "forbid"}


class Response(BaseModel):
    """Response from an agent."""
    
    agent_id: str
    query_id: str
    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CritiqueScore(BaseModel):
    """Scores from a critique."""
    
    accuracy: float = Field(..., ge=0.0, le=100.0)
    coherence: float = Field(..., ge=0.0, le=100.0)
    completeness: float = Field(..., ge=0.0, le=100.0)
    relevance: float = Field(..., ge=0.0, le=100.0)
    overall: float = Field(..., ge=0.0, le=100.0)


class Critique(BaseModel):
    """Critique of a response."""
    
    critic_agent_id: str
    target_agent_id: str
    scores: CritiqueScore
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentScore(BaseModel):
    """ELO score and statistics for an agent."""
    
    agent_id: str
    agent_name: str
    model_name: str
    elo_score: float = Field(default=1500.0, ge=100.0, le=3000.0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    total_queries: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)


class RoundResult(BaseModel):
    """Result of a tournament round."""
    
    round_id: str
    round_number: int = Field(..., ge=1)
    scores: dict[str, float]
    elo_changes: dict[str, float]
    best_response_agent_id: str
    duration_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TournamentConfig(BaseModel):
    """Configuration for a tournament."""
    
    num_tournaments: int = Field(default=1, ge=1, le=10)
    rounds_per_tournament: int = Field(default=3, ge=1, le=20)
    k_factor: float = Field(default=32.0, gt=0.0, le=100.0)
    scoring_weights: dict[str, float] = Field(default_factory=lambda: {
        "accuracy": 0.35,
        "coherence": 0.25,
        "completeness": 0.25,
        "relevance": 0.15,
    })
    enable_response_improvement: bool = True
    dynamic_agent_selection: bool = False
    top_agents_for_finals: int = Field(default=2, ge=1)
    enable_nested_tournaments: bool = False
    championship_rounds: int = Field(default=2, ge=1)


class TournamentResult(BaseModel):
    """Result of a complete tournament."""
    
    tournament_id: str
    query_id: str
    query: str
    status: str
    rounds: list[RoundResult]
    final_rankings: list[tuple[str, float]]
    best_response: Response
    winner_agent_id: str
    duration_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntry(BaseModel):
    """Entry in the leaderboard."""
    
    rank: int = Field(..., ge=1)
    agent_id: str
    agent_name: str
    elo_score: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0


class Leaderboard(BaseModel):
    """Leaderboard of agent rankings."""
    
    entries: list[LeaderboardEntry]
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    total_agents: int
    total_tournaments: int = 0
    total_rounds: int = 0


class SystemMetrics(BaseModel):
    """System-wide performance metrics."""
    
    total_queries: int = 0
    total_tournaments: int = 0
    total_rounds: int = 0
    total_agents: int = 0
    avg_response_latency_ms: float = 0.0
    avg_round_duration_ms: float = 0.0
    avg_tournament_duration_ms: float = 0.0
    queries_per_minute: float = 0.0
    uptime_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(default_factory=dict)


# Request/Response models for API

class QuerySubmitRequest(BaseModel):
    """Request to submit a query."""
    
    content: str = Field(..., min_length=1, max_length=10000)
    context: dict[str, Any] | None = None
    config: TournamentConfig | None = None


class QuerySubmitResponse(BaseModel):
    """Response after submitting a query."""
    
    query_id: str
    status: str
    message: str
    estimated_duration_ms: float | None = None


class QueryResultResponse(BaseModel):
    """Response containing query results."""
    
    query_id: str
    query: str
    status: str
    result: TournamentResult | None = None
    error: str | None = None


class AgentRegistrationRequest(BaseModel):
    """Request to register a new agent."""
    
    name: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1, max_length=100)
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)


class AgentRegistrationResponse(BaseModel):
    """Response after registering an agent."""
    
    agent_id: str
    name: str
    model_name: str
    elo_score: float
    status: str

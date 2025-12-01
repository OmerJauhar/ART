"""
API Routes Module
=================

Defines RESTful API endpoints for the ART framework.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from art_framework.agents.base import AgentConfig
from art_framework.agents.mock_agent import MockAgent
from art_framework.consensus.engine import ConsensusEngine, FusionStrategy
from art_framework.data.models import (
    Query,
    QuerySubmitRequest,
    QuerySubmitResponse,
    QueryResultResponse,
    TournamentConfig,
    TournamentResult,
    AgentRegistrationRequest,
    AgentRegistrationResponse,
    Leaderboard,
    LeaderboardEntry,
    SystemMetrics,
    HealthCheck,
    AgentScore,
)
from art_framework.tournament.engine import (
    TournamentEngine,
    TournamentConfig as TournamentEngineConfig,
)

router = APIRouter()

# In-memory storage for demo purposes
_queries: dict[str, dict[str, Any]] = {}
_agents: dict[str, MockAgent] = {}
_tournaments: dict[str, Any] = {}
_tournament_engine: TournamentEngine | None = None
_metrics = {
    "total_queries": 0,
    "total_tournaments": 0,
    "total_rounds": 0,
    "start_time": datetime.utcnow(),
}


def _get_or_create_tournament_engine() -> TournamentEngine:
    """Get or create the tournament engine with registered agents."""
    global _tournament_engine
    
    if _tournament_engine is None:
        if len(_agents) < 2:
            # Create default mock agents
            default_agents = [
                MockAgent(
                    AgentConfig(name="Agent-Alpha", model_name="mock-gpt-4"),
                    response_quality=0.85,
                ),
                MockAgent(
                    AgentConfig(name="Agent-Beta", model_name="mock-gpt-3.5"),
                    response_quality=0.75,
                ),
                MockAgent(
                    AgentConfig(name="Agent-Gamma", model_name="mock-claude"),
                    response_quality=0.80,
                ),
            ]
            for agent in default_agents:
                _agents[agent.id] = agent
        
        _tournament_engine = TournamentEngine(
            agents=list(_agents.values()),
            config=TournamentEngineConfig(),
        )
    
    return _tournament_engine


# Health Check Endpoints

@router.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check() -> HealthCheck:
    """Check the health status of the API."""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        components={
            "api": "healthy",
            "tournament_engine": "healthy" if _tournament_engine else "not_initialized",
            "agents": f"{len(_agents)} registered",
        },
    )


@router.get("/metrics", response_model=SystemMetrics, tags=["Health"])
async def get_metrics() -> SystemMetrics:
    """Get system-wide performance metrics."""
    uptime = (datetime.utcnow() - _metrics["start_time"]).total_seconds()
    qpm = _metrics["total_queries"] / max(1, uptime / 60)
    
    return SystemMetrics(
        total_queries=_metrics["total_queries"],
        total_tournaments=_metrics["total_tournaments"],
        total_rounds=_metrics["total_rounds"],
        total_agents=len(_agents),
        queries_per_minute=qpm,
        uptime_seconds=uptime,
    )


# Query Endpoints

@router.post(
    "/queries",
    response_model=QuerySubmitResponse,
    tags=["Queries"],
    summary="Submit a query for tournament processing",
)
async def submit_query(
    request: QuerySubmitRequest,
    background_tasks: BackgroundTasks,
) -> QuerySubmitResponse:
    """
    Submit a query to be processed through the tournament system.
    
    The query will be distributed to all registered agents, who will
    generate responses and cross-evaluate each other over multiple rounds.
    """
    query_id = str(uuid.uuid4())
    
    # Store query
    _queries[query_id] = {
        "id": query_id,
        "content": request.content,
        "context": request.context,
        "config": request.config,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": datetime.utcnow(),
    }
    
    _metrics["total_queries"] += 1
    
    # Process in background
    background_tasks.add_task(process_query, query_id)
    
    return QuerySubmitResponse(
        query_id=query_id,
        status="pending",
        message="Query submitted successfully. Processing will begin shortly.",
        estimated_duration_ms=5000.0,
    )


async def process_query(query_id: str) -> None:
    """Background task to process a query through the tournament."""
    try:
        query_data = _queries.get(query_id)
        if not query_data:
            return
        
        query_data["status"] = "processing"
        
        # Get tournament engine
        engine = _get_or_create_tournament_engine()
        await engine.initialize_agents()
        
        # Configure tournament
        config_data = query_data.get("config")
        if config_data:
            config = TournamentEngineConfig(
                num_tournaments=config_data.num_tournaments,
                rounds_per_tournament=config_data.rounds_per_tournament,
                k_factor=config_data.k_factor,
            )
        else:
            config = TournamentEngineConfig()
        
        # Run tournament
        result = await engine.run_tournament(
            query=query_data["content"],
            query_id=query_id,
            config_override=config,
        )
        
        query_data["status"] = "completed"
        query_data["result"] = result.to_dict()
        
        _metrics["total_tournaments"] += 1
        _metrics["total_rounds"] += len(result.rounds)
        
        logger.info(f"Query {query_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}")
        if query_id in _queries:
            _queries[query_id]["status"] = "failed"
            _queries[query_id]["error"] = str(e)


@router.get(
    "/queries/{query_id}",
    response_model=QueryResultResponse,
    tags=["Queries"],
    summary="Get query result",
)
async def get_query_result(query_id: str) -> QueryResultResponse:
    """Get the result of a previously submitted query."""
    if query_id not in _queries:
        raise HTTPException(status_code=404, detail="Query not found")
    
    query_data = _queries[query_id]
    
    return QueryResultResponse(
        query_id=query_id,
        query=query_data["content"],
        status=query_data["status"],
        result=query_data.get("result"),
        error=query_data.get("error"),
    )


@router.get(
    "/queries",
    tags=["Queries"],
    summary="List all queries",
)
async def list_queries(
    status: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    """List all submitted queries with optional filtering."""
    queries = list(_queries.values())
    
    if status:
        queries = [q for q in queries if q["status"] == status]
    
    total = len(queries)
    queries = queries[offset:offset + limit]
    
    return {
        "queries": queries,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# Tournament Endpoints

@router.post(
    "/tournaments",
    tags=["Tournaments"],
    summary="Run a tournament directly",
)
async def run_tournament(
    query: str,
    config: TournamentConfig | None = None,
) -> dict[str, Any]:
    """
    Run a tournament directly without background processing.
    
    This endpoint blocks until the tournament completes.
    """
    engine = _get_or_create_tournament_engine()
    await engine.initialize_agents()
    
    engine_config = TournamentEngineConfig(
        num_tournaments=config.num_tournaments if config else 1,
        rounds_per_tournament=config.rounds_per_tournament if config else 3,
    )
    
    result = await engine.run_tournament(
        query=query,
        config_override=engine_config,
    )
    
    _metrics["total_tournaments"] += 1
    _metrics["total_rounds"] += len(result.rounds)
    
    return result.to_dict()


@router.get(
    "/tournaments/{tournament_id}",
    tags=["Tournaments"],
    summary="Get tournament details",
)
async def get_tournament(tournament_id: str) -> dict[str, Any]:
    """Get details of a specific tournament."""
    engine = _get_or_create_tournament_engine()
    
    for t in engine.get_tournament_history():
        if t["tournament_id"] == tournament_id:
            return t
    
    raise HTTPException(status_code=404, detail="Tournament not found")


@router.get(
    "/tournaments",
    tags=["Tournaments"],
    summary="List tournament history",
)
async def list_tournaments() -> list[dict[str, Any]]:
    """Get history of all tournaments."""
    engine = _get_or_create_tournament_engine()
    return engine.get_tournament_history()


# Agent Endpoints

@router.post(
    "/agents",
    response_model=AgentRegistrationResponse,
    tags=["Agents"],
    summary="Register a new agent",
)
async def register_agent(request: AgentRegistrationRequest) -> AgentRegistrationResponse:
    """
    Register a new LLM agent for tournament participation.
    
    The agent will be added to the tournament pool and can participate
    in future tournaments.
    """
    global _tournament_engine
    
    # Create agent config
    config = AgentConfig(
        name=request.name,
        model_name=request.model_name,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        api_key=request.api_key,
        base_url=request.base_url,
    )
    
    # Create mock agent for now (would use LLMAgent with real API)
    agent = MockAgent(config, response_quality=0.75)
    _agents[agent.id] = agent
    
    # Reset tournament engine to include new agent
    _tournament_engine = None
    
    logger.info(f"Registered agent: {agent.name} ({agent.id})")
    
    return AgentRegistrationResponse(
        agent_id=agent.id,
        name=agent.name,
        model_name=config.model_name,
        elo_score=agent.elo_score,
        status="active",
    )


@router.get(
    "/agents",
    tags=["Agents"],
    summary="List all registered agents",
)
async def list_agents() -> list[dict[str, Any]]:
    """Get list of all registered agents."""
    return [agent.to_dict() for agent in _agents.values()]


@router.get(
    "/agents/{agent_id}",
    response_model=AgentScore,
    tags=["Agents"],
    summary="Get agent details",
)
async def get_agent(agent_id: str) -> AgentScore:
    """Get details of a specific agent."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = _agents[agent_id]
    return AgentScore(
        agent_id=agent.id,
        agent_name=agent.name,
        model_name=agent.config.model_name,
        elo_score=agent.elo_score,
        wins=agent.state.wins,
        losses=agent.state.losses,
        draws=agent.state.draws,
        total_queries=agent.state.total_queries,
        win_rate=agent.state.win_rate,
        created_at=agent.state.created_at,
        last_active=agent.state.last_active,
    )


@router.delete(
    "/agents/{agent_id}",
    tags=["Agents"],
    summary="Remove an agent",
)
async def remove_agent(agent_id: str) -> dict[str, str]:
    """Remove an agent from the tournament pool."""
    global _tournament_engine
    
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del _agents[agent_id]
    _tournament_engine = None  # Reset engine
    
    return {"message": f"Agent {agent_id} removed successfully"}


# Leaderboard Endpoints

@router.get(
    "/leaderboard",
    response_model=Leaderboard,
    tags=["Leaderboard"],
    summary="Get agent leaderboard",
)
async def get_leaderboard() -> Leaderboard:
    """Get the current leaderboard of agent rankings."""
    engine = _get_or_create_tournament_engine()
    
    leaderboard_data = engine.get_leaderboard()
    entries = [
        LeaderboardEntry(
            rank=entry["rank"],
            agent_id=entry["agent_id"],
            agent_name=entry["agent_name"],
            elo_score=entry["elo_score"],
            wins=entry["wins"],
            losses=entry["losses"],
            draws=entry["draws"],
            win_rate=entry["win_rate"],
        )
        for entry in leaderboard_data
    ]
    
    return Leaderboard(
        entries=entries,
        total_agents=len(entries),
        total_tournaments=_metrics["total_tournaments"],
        total_rounds=_metrics["total_rounds"],
    )


# Consensus Endpoints

@router.post(
    "/consensus",
    tags=["Consensus"],
    summary="Generate consensus from responses",
)
async def generate_consensus(
    responses: dict[str, dict[str, Any]],
    elo_scores: dict[str, float],
    strategy: str = "weighted_voting",
) -> dict[str, Any]:
    """
    Generate a consensus response from multiple agent responses.
    
    Available strategies:
    - weighted_voting: Select based on weighted ELO scores
    - top_response: Simply return highest-ranked response
    - contextual_aggregation: Combine key points from top responses
    - hybrid_synthesis: Create synthesized response
    """
    from art_framework.agents.base import AgentResponse
    
    # Convert to AgentResponse objects
    agent_responses = {
        agent_id: AgentResponse(
            agent_id=agent_id,
            query_id=data.get("query_id", "unknown"),
            content=data["content"],
            confidence=data.get("confidence", 0.8),
        )
        for agent_id, data in responses.items()
    }
    
    # Get fusion strategy
    try:
        fusion_strategy = FusionStrategy(strategy)
    except ValueError:
        fusion_strategy = FusionStrategy.WEIGHTED_VOTING
    
    # Generate consensus
    consensus_engine = ConsensusEngine(default_strategy=fusion_strategy)
    result = consensus_engine.generate_consensus(
        responses=agent_responses,
        elo_scores=elo_scores,
        strategy=fusion_strategy,
    )
    
    return result.to_dict()


# Configuration Endpoints

@router.get(
    "/config",
    tags=["Configuration"],
    summary="Get current configuration",
)
async def get_config() -> dict[str, Any]:
    """Get the current tournament configuration."""
    engine = _get_or_create_tournament_engine()
    
    return {
        "num_tournaments": engine.config.num_tournaments,
        "rounds_per_tournament": engine.config.rounds_per_tournament,
        "k_factor": engine.config.k_factor,
        "scoring_weights": engine.config.scoring_weights,
        "enable_response_improvement": engine.config.enable_response_improvement,
        "dynamic_agent_selection": engine.config.dynamic_agent_selection,
        "top_agents_for_finals": engine.config.top_agents_for_finals,
        "enable_nested_tournaments": engine.config.enable_nested_tournaments,
    }


@router.put(
    "/config",
    tags=["Configuration"],
    summary="Update configuration",
)
async def update_config(config: TournamentConfig) -> dict[str, Any]:
    """Update the tournament configuration."""
    global _tournament_engine
    
    # Reset tournament engine with new config
    engine_config = TournamentEngineConfig(
        num_tournaments=config.num_tournaments,
        rounds_per_tournament=config.rounds_per_tournament,
        k_factor=config.k_factor,
        scoring_weights=config.scoring_weights,
        enable_response_improvement=config.enable_response_improvement,
        dynamic_agent_selection=config.dynamic_agent_selection,
        top_agents_for_finals=config.top_agents_for_finals,
        enable_nested_tournaments=config.enable_nested_tournaments,
        championship_rounds=config.championship_rounds,
    )
    
    if _agents:
        _tournament_engine = TournamentEngine(
            agents=list(_agents.values()),
            config=engine_config,
        )
    else:
        _tournament_engine = None
    
    return {"message": "Configuration updated successfully"}

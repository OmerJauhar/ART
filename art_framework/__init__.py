"""
ART (Adaptive Response Tuning) Framework
==========================================

A high-performance, multi-agent framework designed to optimize outputs from
Large Language Models (LLMs) via tournament-style ELO ranking and iterative reasoning.

Features:
- Multi-agent reasoning with cross-evaluation
- Tournament-style ELO ranking system
- Response optimization workflow
- Consensus generation
- RESTful API interface
"""

__version__ = "1.0.0"
__author__ = "ART Framework Team"

from art_framework.agents.base import Agent, AgentConfig, AgentState
from art_framework.tournament.engine import TournamentEngine
from art_framework.tournament.elo import EloCalculator
from art_framework.consensus.engine import ConsensusEngine
from art_framework.data.models import (
    Query,
    Response,
    TournamentConfig,
    TournamentResult,
    AgentScore,
    Leaderboard,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentState",
    "TournamentEngine",
    "EloCalculator",
    "ConsensusEngine",
    "Query",
    "Response",
    "TournamentConfig",
    "TournamentResult",
    "AgentScore",
    "Leaderboard",
]

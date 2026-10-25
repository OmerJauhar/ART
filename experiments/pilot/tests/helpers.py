"""
Test Helpers
============

Synthetic questions and in-memory transports for the pilot tests. No network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.pilot.budget import BudgetGuard
from experiments.pilot.cache import CallCache
from experiments.pilot.data import Question
from experiments.pilot.gateway import CallRecord, Gateway
from experiments.pilot.settings import PilotModels, load_models


def make_question(
    index: int,
    split: str = "test",
    subset: str = "SeqQA",
    correct: str = "B",
) -> Question:
    """A synthetic four-option question."""
    return Question(
        id=f"q{index}",
        subset=subset,
        subtask="synthetic",
        stem=f"Synthetic question number {index}: which option is right?",
        options=[f"option {index}-{k}" for k in range(4)],
        correct_letter=correct,
        split=split,
        index=index,
    )


def make_record(key: str, cost: float, latency: float = 100.0, tokens: int = 10) -> CallRecord:
    """A CallRecord with the given cost and latency."""
    return CallRecord(
        key=key,
        endpoint="chat",
        model="m",
        model_version="m-1",
        stage="answer",
        response={},
        prompt_tokens=tokens,
        completion_tokens=0,
        cost_usd=cost,
        latency_ms=latency,
        cached=False,
    )


class CountingTransport:
    """Transport that returns a canned chat completion and counts calls."""

    def __init__(self, content: str = "ANSWER: A", cost: float = 0.001) -> None:
        """Create the transport."""
        self.content = content
        self.cost = cost
        self.calls = 0
        self.bodies: list[dict[str, Any]] = []

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Return a completion."""
        self.calls += 1
        self.bodies.append(body)
        return {
            "model": f"{body.get('model')}-test",
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": self.cost},
        }


def make_gateway(
    tmp_path: Path,
    transport: Any,
    max_usd: float = 100.0,
    models: PilotModels | None = None,
) -> Gateway:
    """Gateway over a temporary cache file."""
    models = models or load_models()
    cache = CallCache(tmp_path / "cache.sqlite")
    return Gateway(transport, cache, BudgetGuard(max_usd), models, concurrency=4)

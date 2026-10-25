"""
Jev Client
==========

Thin client for OpenRouter's alpha Decisions API (POST /api/alpha/decisions) with TypeSafe's
Jev model. The API is labelled alpha, so everything that depends on its request and response
shape lives in this one class.

Request: {"model", "state", "questions": {name: {"type", "instructions", "criteria"}}}
  - choice: criteria is {option: description}; answer has choice, confidence, probabilities
  - noul: criteria is {"true": ..., "false": ...}; answer has noul = P(yes)
Response: {"id", "model" (dated snapshot), "provider", "answers", "usage": {input_tokens,
output_tokens, cost}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.pilot.gateway import STAGE, ApiError, Gateway


@dataclass
class JevAnswer:
    """One parsed Jev answer plus the call's metadata."""

    question_type: str
    model_version: str
    probabilities: dict[str, float] = field(default_factory=dict)
    choice: str | None = None
    confidence: float | None = None
    p_yes: float | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0


def build_choice_request(
    model: str,
    state: Any,
    instructions: str,
    criteria: dict[str, str],
    name: str = "pick",
) -> dict[str, Any]:
    """Request body with one Choice question."""
    return {
        "model": model,
        "state": state,
        "questions": {
            name: {"type": "choice", "instructions": instructions, "criteria": criteria}
        },
    }


def build_noul_request(
    model: str,
    state: Any,
    instructions: str,
    criteria: dict[str, str],
    name: str = "check",
) -> dict[str, Any]:
    """Request body with one Noul (yes/no) question; criteria needs "true" and "false"."""
    if set(criteria) != {"true", "false"}:
        raise ValueError("Noul criteria must have exactly the keys 'true' and 'false'")
    return {
        "model": model,
        "state": state,
        "questions": {
            name: {"type": "noul", "instructions": instructions, "criteria": criteria}
        },
    }


def parse_response(response: dict[str, Any], name: str) -> JevAnswer:
    """
    Parse one named answer out of a Decisions response.

    For a Choice answer without probabilities, the chosen option gets probability 1.

    Raises:
        ValueError: If the answer is missing or has an unexpected type
    """
    answers = response.get("answers") or {}
    if name not in answers:
        raise ValueError(f"Decisions response has no answer named {name!r}")
    answer = answers[name]
    usage = response.get("usage") or {}
    cost = usage.get("cost")
    parsed = JevAnswer(
        question_type=str(answer.get("type")),
        model_version=str(response.get("model", "")),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0,
        input_tokens=int(usage.get("input_tokens") or 0),
    )
    if parsed.question_type == "choice":
        parsed.choice = answer.get("choice")
        parsed.confidence = answer.get("confidence")
        probabilities = answer.get("probabilities")
        if probabilities:
            parsed.probabilities = {str(k): float(v) for k, v in probabilities.items()}
        elif parsed.choice is not None:
            parsed.probabilities = {str(parsed.choice): 1.0}
    elif parsed.question_type == "noul":
        parsed.p_yes = float(answer["noul"])
    else:
        raise ValueError(f"Unexpected Jev answer type {parsed.question_type!r}")
    return parsed


class JevClient:
    """Sends Jev questions through the Gateway, falling back to an older model if needed."""

    def __init__(self, gateway: Gateway, model: str, fallback_model: str | None = None) -> None:
        """Create the client."""
        self.gateway = gateway
        self.model = model
        self.fallback_model = fallback_model
        self._checked = False

    async def _send(self, build: Any, name: str) -> JevAnswer:
        """Send a request built for the current model; fall back once if it is unavailable."""
        token = STAGE.set("jev")
        try:
            try:
                record = await self.gateway.decide(build(self.model))
            except ApiError as e:
                unavailable = e.status in (400, 404) and "model" in str(e).lower()
                if self._checked or not self.fallback_model or not unavailable:
                    raise
                self.model = self.fallback_model
                record = await self.gateway.decide(build(self.model))
            self._checked = True
        finally:
            STAGE.reset(token)
        return parse_response(record.response, name)

    async def choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> JevAnswer:
        """Ask one Choice question."""
        return await self._send(
            lambda model: build_choice_request(model, state, instructions, criteria), "pick"
        )

    async def noul(self, state: Any, instructions: str, criteria: dict[str, str]) -> JevAnswer:
        """Ask one Noul question; JevAnswer.p_yes is P(yes)."""
        return await self._send(
            lambda model: build_noul_request(model, state, instructions, criteria), "check"
        )

"""
Pilot Agents
============

LLMAgent subclass that sends its calls through the pilot Gateway (cache + budget guard).
Prompts, critique parsing and usage accounting are LLMAgent's own.
"""

from __future__ import annotations

from typing import Any

from art_framework.agents.base import AgentConfig, AgentResponse, Critique
from art_framework.agents.llm_agent import LLMAgent
from experiments.pilot.gateway import STAGE, Gateway
from experiments.pilot.settings import ModelSpec, PilotModels


class PilotAgent(LLMAgent):
    """LLMAgent whose _call_llm goes through the Gateway instead of its own HTTP client."""

    def __init__(self, spec: ModelSpec, gateway: Gateway, models: PilotModels) -> None:
        """Create an agent for one pilot model."""
        super().__init__(
            AgentConfig(
                name=spec.label,
                model_name=spec.id,
                temperature=models.temperature,
                max_tokens=models.max_tokens,
                base_url=models.base_url,
            )
        )
        self.spec = spec
        self.gateway = gateway
        self._is_initialized = True
        # A stable ID (instead of a random UUID) keeps the engine's agent ordering, and so
        # every prompt, identical across runs; that is what lets reruns hit the cache
        self.state.agent_id = spec.id

    async def initialize(self) -> None:
        """Nothing to set up: the Gateway owns the HTTP client."""
        self._is_initialized = True

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Same request and response handling as LLMAgent._call_llm, sent via the Gateway.

        The only addition to the request body is the model's OpenRouter reasoning setting.

        Raises:
            ValueError: If the model returns an empty completion
        """
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if self.spec.reasoning:
            payload["reasoning"] = dict(self.spec.reasoning)

        record = await self.gateway.chat(payload)
        data = record.response
        content = data["choices"][0]["message"].get("content")
        if content is None or not content.strip():
            raise ValueError(f"Empty completion from {self.config.model_name}")
        self._last_usage = self._record_usage(data.get("usage") or {})
        return content

    async def generate_response(
        self,
        query: str,
        query_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate an answer, tagging the call as an answer."""
        token = STAGE.set("answer")
        try:
            return await super().generate_response(query, query_id, context)
        finally:
            STAGE.reset(token)

    async def critique_response(
        self,
        original_query: str,
        target_response: AgentResponse,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        """Critique an answer, tagging the call as a critique."""
        token = STAGE.set("critique")
        try:
            return await super().critique_response(original_query, target_response, context)
        finally:
            STAGE.reset(token)

    async def improve_response(
        self,
        original_response: AgentResponse,
        critiques: list[Critique],
        *,
        original_query: str | None = None,
    ) -> AgentResponse:
        """
        Revise an answer, tagging the call as a revision.

        Critiques arrive in task-completion order; sorting them by critic keeps the revision
        prompt identical across runs. The prompt template itself is LLMAgent's.
        """
        token = STAGE.set("revision")
        try:
            ordered = sorted(critiques, key=lambda c: c.critic_agent_id)
            return await super().improve_response(
                original_response, ordered, original_query=original_query
            )
        finally:
            STAGE.reset(token)

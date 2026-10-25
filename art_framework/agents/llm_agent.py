"""
LLM Agent Implementation
========================

Real LLM agent implementation using OpenAI-compatible APIs.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
from loguru import logger

from art_framework.agents.base import (
    Agent,
    AgentConfig,
    AgentResponse,
    Critique,
    CritiqueStrategy,
)


class LLMAgent(Agent):
    """
    LLM Agent that interfaces with OpenAI-compatible APIs.
    
    Supports any OpenAI-compatible endpoint including:
    - OpenAI (GPT-4, GPT-3.5)
    - Azure OpenAI
    - OpenRouter
    - Local LLMs (LMStudio, Ollama, etc.)
    """
    
    def __init__(self, config: AgentConfig) -> None:
        """Initialize the LLM agent."""
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._default_headers: dict[str, str] = {}
        self._last_usage: dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the agent and verify API connectivity."""
        if self._is_initialized:
            return
        
        self._default_headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api_key:
            self._default_headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers=self._default_headers,
        )
        
        self._is_initialized = True
        logger.info(f"Agent '{self.name}' initialized with model '{self.config.model_name}'")
    
    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Make a call to the LLM API.
        
        Args:
            messages: List of message dictionaries
            temperature: Override temperature setting
            max_tokens: Override max tokens setting
            
        Returns:
            Generated text response. Token usage and cost for the call are
            stored on ``self._last_usage`` and added to the agent's totals.
            
        Raises:
            ValueError: If the model returns an empty completion
        """
        if not self._is_initialized or self._client is None:
            await self.initialize()
        
        base_url = self.config.base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        
        try:
            response = await self._client.post(url, json=payload)  # type: ignore
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content")
            if content is None or not content.strip():
                raise ValueError(f"Empty completion from {self.config.model_name}")
            self._last_usage = self._record_usage(data.get("usage") or {})
            return content
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _record_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        """
        Add one call's token usage and cost to the agent's running totals.
        
        OpenRouter returns ``usage.cost`` (in USD-denominated credits) on every
        chat completion without any request option. Other OpenAI-compatible
        providers usually omit it, in which case cost is recorded as 0.0.
        
        Args:
            usage: The ``usage`` object from the chat completions response
            
        Returns:
            Per-call dict with prompt_tokens, completion_tokens and cost
        """
        # TODO: compute cost from per-token prices for providers that do not return usage.cost
        cost = usage.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            cost = 0.0
        call_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "cost": float(cost),
        }
        
        self.state.llm_calls += 1
        self.state.prompt_tokens += call_usage["prompt_tokens"]
        self.state.completion_tokens += call_usage["completion_tokens"]
        self.state.cost_usd += call_usage["cost"]
        return call_usage
    
    async def generate_response(
        self,
        query: str,
        query_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate a response to the given query."""
        start_time = time.time()
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant participating in a multi-agent "
                    "reasoning tournament. Provide accurate, well-reasoned, and "
                    "comprehensive responses. Support your answers with clear "
                    "reasoning and evidence when applicable."
                ),
            },
        ]
        
        # Add context from previous rounds if available
        if context and "history" in context:
            for entry in context["history"][-5:]:  # Last 5 context entries
                messages.append({
                    "role": "user" if entry.get("role") == "user" else "assistant",
                    "content": entry.get("content", ""),
                })
        
        messages.append({"role": "user", "content": query})
        
        try:
            content = await self._call_llm(messages)
            # Read immediately: no await between _call_llm returning and this line,
            # so concurrent calls on the same agent cannot overwrite it.
            usage = dict(self._last_usage)
            latency_ms = (time.time() - start_time) * 1000
            
            self.state.total_queries += 1
            self.add_to_context({"role": "user", "content": query})
            self.add_to_context({"role": "assistant", "content": content})
            
            return AgentResponse(
                agent_id=self.id,
                query_id=query_id,
                content=content,
                confidence=0.85,  # Could be calculated based on model output
                reasoning=f"Generated using {self.config.model_name}",
                metadata={"model": self.config.model_name, "usage": usage},
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise
    
    async def critique_response(
        self,
        original_query: str,
        target_response: AgentResponse,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        """Critique another agent's response."""
        critique_prompts = self._get_critique_prompt(
            original_query,
            target_response.content,
            self.config.critique_strategy,
        )
        
        messages = [
            {"role": "system", "content": critique_prompts["system"]},
            {"role": "user", "content": critique_prompts["user"]},
        ]
        
        try:
            content = await self._call_llm(messages)
            usage = dict(self._last_usage)
            scores = self._parse_critique_scores(content)
            
            self.state.total_critiques_given += 1
            
            return Critique(
                critic_agent_id=self.id,
                target_agent_id=target_response.agent_id,
                response_id=str(uuid.uuid4()),
                score=scores["overall"],
                accuracy_score=scores["accuracy"],
                coherence_score=scores["coherence"],
                completeness_score=scores["completeness"],
                relevance_score=scores["relevance"],
                pros=scores.get("pros", []),
                cons=scores.get("cons", []),
                suggestions=scores.get("suggestions", []),
                reasoning=content,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Critique generation failed: {e}")
            raise
    
    def _get_critique_prompt(
        self,
        query: str,
        response: str,
        strategy: CritiqueStrategy,
    ) -> dict[str, str]:
        """Get critique prompt based on strategy."""
        base_system = (
            "You are an expert evaluator in a multi-agent reasoning tournament. "
            "Your task is to critically evaluate a response to a query. "
            "Provide scores from 0-100 for each criterion and overall."
        )
        
        if strategy == CritiqueStrategy.FACT_CHECK:
            system = base_system + (
                " Focus primarily on factual accuracy and verifiable claims."
            )
        elif strategy == CritiqueStrategy.LOGIC_EVALUATION:
            system = base_system + (
                " Focus primarily on logical consistency and reasoning quality."
            )
        elif strategy == CritiqueStrategy.PROS_CONS:
            system = base_system + (
                " Provide a balanced evaluation of strengths and weaknesses."
            )
        else:  # COMPREHENSIVE
            system = base_system + (
                " Provide a comprehensive evaluation covering all aspects."
            )
        
        user = f"""Evaluate the following response:

ORIGINAL QUERY: {query}

RESPONSE TO EVALUATE: {response}

Please evaluate using these criteria:
1. ACCURACY (0-100): Factual correctness and reliability
2. COHERENCE (0-100): Logical flow and clarity
3. COMPLETENESS (0-100): Coverage of the topic
4. RELEVANCE (0-100): How well it addresses the query

Format your response as:
ACCURACY: [score]
COHERENCE: [score]
COMPLETENESS: [score]
RELEVANCE: [score]
OVERALL: [score]

PROS:
- [pro 1]
- [pro 2]

CONS:
- [con 1]
- [con 2]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]

REASONING:
[Your detailed analysis]
"""
        
        return {"system": system, "user": user}
    
    def _parse_critique_scores(self, content: str) -> dict[str, Any]:
        """Parse critique scores from LLM response."""
        scores: dict[str, Any] = {
            "accuracy": 70.0,
            "coherence": 70.0,
            "completeness": 70.0,
            "relevance": 70.0,
            "overall": 70.0,
            "pros": [],
            "cons": [],
            "suggestions": [],
        }
        
        lines = content.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "ACCURACY:" in line.upper():
                try:
                    scores["accuracy"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "COHERENCE:" in line.upper():
                try:
                    scores["coherence"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "COMPLETENESS:" in line.upper():
                try:
                    scores["completeness"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "RELEVANCE:" in line.upper():
                try:
                    scores["relevance"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "OVERALL:" in line.upper():
                try:
                    scores["overall"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "PROS:" in line.upper():
                current_section = "pros"
            elif "CONS:" in line.upper():
                current_section = "cons"
            elif "SUGGESTIONS:" in line.upper():
                current_section = "suggestions"
            elif "REASONING:" in line.upper():
                current_section = None
            elif line.startswith("-") and current_section:
                scores[current_section].append(line[1:].strip())
        
        return scores
    
    async def improve_response(
        self,
        original_response: AgentResponse,
        critiques: list[Critique],
        *,
        original_query: str | None = None,
    ) -> AgentResponse:
        """Improve a response based on received critiques."""
        critique_summary = "\n".join([
            f"Critic feedback (score: {c.score}):\n"
            f"- Pros: {', '.join(c.pros)}\n"
            f"- Cons: {', '.join(c.cons)}\n"
            f"- Suggestions: {', '.join(c.suggestions)}"
            for c in critiques
        ])
        
        question_section = (
            f"ORIGINAL QUESTION:\n{original_query}\n\n" if original_query else ""
        )
        user_content = (
            f"{question_section}"
            f"YOUR PREVIOUS ANSWER:\n{original_response.content}\n\n"
            f"CRITIQUES:\n{critique_summary}\n\n"
            "Write a revised, complete answer to the original question. Address the "
            "critiques while keeping the strengths they identified. Focus on accuracy, "
            "completeness, and clarity. Reply with the revised answer only."
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant. Your previous response has "
                    "been evaluated by other experts. Use their feedback to "
                    "provide an improved response."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        
        start_time = time.time()
        content = await self._call_llm(messages)
        usage = dict(self._last_usage)
        latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            agent_id=self.id,
            query_id=original_response.query_id,
            content=content,
            confidence=min(0.95, original_response.confidence + 0.1),
            reasoning="Improved based on peer critiques",
            metadata={
                "model": self.config.model_name,
                "critiques_incorporated": len(critiques),
                "usage": usage,
            },
            latency_ms=latency_ms,
        )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._is_initialized = False

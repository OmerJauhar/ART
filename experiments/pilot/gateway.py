"""
Pilot Gateway
=============

Every model call (chat completions and Jev decisions) goes through one Gateway, which adds
the persistent cache, the budget guard, a concurrency limit, retries with exponential backoff,
and a per-call record used for cost attribution.

A provider refusal (see is_provider_refusal) is a deterministic outcome: it is cached at cost 0
and replayed as a ProviderRefusal, unless the gateway is told to retry refusals.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from experiments.pilot.budget import BudgetExceeded, BudgetGuard
from experiments.pilot.cache import CachedCall, CallCache, request_key
from experiments.pilot.settings import PilotModels

# Conservative characters-per-token used only to size budget reservations
_RESERVE_CHARS_PER_TOKEN = 2.0

# Key of the cached response body that stands for a provider refusal
REFUSAL_KEY = "pilot_refusal"

# Start of LLMAgent's error for a critique without every score (see _parse_critique_scores)
UNPARSEABLE_CRITIQUE = "Critique missing scores"


class ApiError(RuntimeError):
    """A non-retryable API error, or retries exhausted."""

    def __init__(self, status: int | str | None, message: str) -> None:
        """Store the status alongside the message."""
        super().__init__(f"API error {status}: {message}")
        self.status = status
        self.message = message


class ProviderRefusal(ApiError):
    """The provider declined the request on content grounds (see is_provider_refusal)."""


def is_provider_refusal(status: int | str | None, message: str) -> bool:
    """
    Whether an API error is a provider refusal.

    Only the one case seen in the pilot counts: an HTTP 403 whose body says the content was
    flagged (OpenAI's "This content was flagged for possible biological risk...", passed
    through by OpenRouter). Other providers' wording is not guessed at.
    """
    return str(status) == "403" and "flagged" in message.lower()


def failure_text(error: BaseException) -> str:
    """How a failed call or method is recorded: exception type, then message."""
    return f"{type(error).__name__}: {error}"


def failure_cause(text: str | None) -> str:
    """
    Classify a failure recorded with failure_text.

    Returns:
        "refused" (provider refusal), "unparseable" (a critique without every score) or
        "other"
    """
    text = text or ""
    if text.startswith(f"{ProviderRefusal.__name__}:"):
        return "refused"
    if text.startswith(f"ValueError: {UNPARSEABLE_CRITIQUE}"):
        return "unparseable"
    return "other"


class Transport(Protocol):
    """Sends one POST and returns the decoded JSON body."""

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Send body to url."""
        ...


class HttpTransport:
    """Real HTTP transport to OpenRouter with retries on 429 and 5xx."""

    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}

    def __init__(
        self,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 6,
        backoff_base: float = 2.0,
    ) -> None:
        """Create the client. The key is only placed in the Authorization header."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "ART pilot",
            },
        )
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff; raise ApiError on failure."""
        last_error = "no attempts"
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                response = await self._client.post(url, json=body)
            except httpx.TimeoutException:
                last_error = "timeout"
            except httpx.TransportError as e:
                last_error = f"transport error: {type(e).__name__}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    error = data.get("error") if isinstance(data, dict) else None
                    if not error:
                        return data
                    code = error.get("code") if isinstance(error, dict) else None
                    message = str(error.get("message", error) if isinstance(error, dict) else error)
                    if code not in self.RETRY_STATUS:
                        raise ApiError(code, message[:500])
                    last_error = f"{code}: {message[:200]}"
                elif response.status_code in self.RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    header = response.headers.get("retry-after")
                    if header:
                        try:
                            retry_after = float(header)
                        except ValueError:
                            retry_after = None
                else:
                    raise ApiError(response.status_code, response.text[:500])
            if attempt == self.max_retries:
                break
            delay = retry_after if retry_after is not None else self.backoff_base * 2**attempt
            await asyncio.sleep(min(delay, 120.0) + random.uniform(0, self.backoff_base / 2))
        raise ApiError("retries_exhausted", last_error)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


@dataclass
class CallRecord:
    """One call as seen by a method: the response plus its usage and cost."""

    key: str
    endpoint: str
    model: str
    model_version: str
    stage: str
    response: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool
    question_ids: set[str] = field(default_factory=set)
    stages: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def to_log(self) -> dict[str, Any]:
        """Row for calls.jsonl."""
        return {
            "prompt_hash": self.key,
            "endpoint": self.endpoint,
            "model": self.model,
            "model_version": self.model_version,
            "stages": sorted(self.stages),
            "question_ids": sorted(self.question_ids),
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost": self.cost_usd,
            },
            "latency_ms": round(self.latency_ms, 1),
            "served_from_cache": self.cached,
            "response": self.response,
        }


# Calls made while a trace list is set are appended to it; methods use this to find out
# exactly which calls they needed.
TRACE: contextvars.ContextVar[list[CallRecord] | None] = contextvars.ContextVar(
    "TRACE", default=None
)
# What the current call is for ("answer", "critique", "revision", "fusion", "jev", ...)
STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("STAGE", default="unknown")
QUESTION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("QUESTION_ID", default="")


def _usage_numbers(endpoint: str, response: dict[str, Any]) -> tuple[int, int, float | None]:
    """Prompt tokens, completion tokens and reported cost from a response."""
    usage = response.get("usage") or {}
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        cost = None
    if endpoint == "decisions":
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), cost
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), cost


class Gateway:
    """Cache, budget guard, concurrency limit and call records for all API traffic."""

    def __init__(
        self,
        transport: Transport,
        cache: CallCache,
        budget: BudgetGuard,
        models: PilotModels,
        concurrency: int = 8,
        retry_refusals: bool = False,
    ) -> None:
        """
        Create the gateway.

        With retry_refusals, each cached refusal is sent again once in this gateway's life
        and the new outcome replaces it in the cache.
        """
        self.transport = transport
        self.cache = cache
        self.budget = budget
        self.models = models
        self._semaphore = asyncio.Semaphore(concurrency)
        self._inflight: dict[str, asyncio.Future[tuple[CachedCall, bool]]] = {}
        self.records: dict[str, CallRecord] = {}
        self.live_calls = 0
        self.live_spend = 0.0
        # Questions with at least one call refused by the budget guard (incomplete)
        self.refused_questions: set[str] = set()
        self.retry_refusals = retry_refusals
        self._retried: set[str] = set()
        # Provider refusals met by this gateway (live or cached), keyed by call
        self.refusals: dict[str, dict[str, Any]] = {}

    def chat_url(self) -> str:
        """Chat completions URL."""
        return f"{self.models.base_url.rstrip('/')}/chat/completions"

    def _worst_case(self, endpoint: str, model: str, body: dict[str, Any]) -> float:
        """Upper bound on a call's cost, used for the budget reservation."""
        import json

        prompt_tokens = int(len(json.dumps(body)) / _RESERVE_CHARS_PER_TOKEN) + 1
        if endpoint == "decisions":
            return self.models.jev.cost(prompt_tokens)
        spec = self.models.get(model)
        max_tokens = int(body.get("max_tokens") or self.models.max_tokens)
        return spec.cost(prompt_tokens, max_tokens)

    def _list_price_cost(self, endpoint: str, model: str, prompt: int, completion: int) -> float:
        """Cost from token counts at list price, for responses without usage.cost."""
        if endpoint == "decisions":
            return self.models.jev.cost(prompt)
        return self.models.get(model).cost(prompt, completion)

    async def chat(self, body: dict[str, Any]) -> CallRecord:
        """Send a chat completions request."""
        return await self._call("chat", self.chat_url(), body)

    async def decide(self, body: dict[str, Any]) -> CallRecord:
        """Send a Decisions API request."""
        return await self._call("decisions", self.models.jev.decisions_url, body)

    async def _call(self, endpoint: str, url: str, body: dict[str, Any]) -> CallRecord:
        """Serve from cache or make the live call, then record it."""
        key = request_key(endpoint, body)
        model = str(body.get("model", ""))

        cached = self.cache.get(key)
        if cached is not None and self._retry(key, cached):
            cached = None
        served_from_cache = cached is not None
        if cached is None:
            try:
                cached, served_from_cache = await self._fetch(key, endpoint, url, model, body)
            except BudgetExceeded:
                if QUESTION_ID.get():
                    self.refused_questions.add(QUESTION_ID.get())
                raise

        refusal = cached.response.get(REFUSAL_KEY)
        if refusal is not None:
            # Not a CallRecord: like any failed call, it is in no trace and costs nothing
            entry = self.refusals.setdefault(key, {
                "model": model, "stage": STAGE.get(), "question_ids": set(),
            })
            if QUESTION_ID.get():
                entry["question_ids"].add(QUESTION_ID.get())
            raise ProviderRefusal(refusal.get("status"), str(refusal.get("message", "")))

        prompt, completion, _ = _usage_numbers(endpoint, cached.response)
        record = self.records.get(key)
        if record is None:
            record = CallRecord(
                key=key,
                endpoint=endpoint,
                model=model,
                model_version=str(cached.response.get("model", "")),
                stage=STAGE.get(),
                response=cached.response,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cost_usd=cached.cost_usd,
                latency_ms=cached.latency_ms,
                cached=served_from_cache,
            )
            self.records[key] = record
        record.stages.add(STAGE.get())
        if QUESTION_ID.get():
            record.question_ids.add(QUESTION_ID.get())

        trace = TRACE.get()
        if trace is not None:
            trace.append(record)
        return record

    def _retry(self, key: str, cached: CachedCall) -> bool:
        """Whether to send a cached refusal again instead of replaying it."""
        if not self.retry_refusals or REFUSAL_KEY not in cached.response:
            return False
        if key in self._retried and key not in self._inflight:
            # Already retried in this run; the cache holds the new outcome
            return False
        self._retried.add(key)
        return True

    async def _fetch(
        self,
        key: str,
        endpoint: str,
        url: str,
        model: str,
        body: dict[str, Any],
    ) -> tuple[CachedCall, bool]:
        """
        Make the live call, or wait for an identical one already in flight.

        Returns:
            (stored call, whether it came from another caller's call)
        """
        future = self._inflight.get(key)
        if future is not None:
            cached, _ = await asyncio.shield(future)
            return cached, True
        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            cached = await self._live_call(key, endpoint, url, model, body)
            future.set_result((cached, False))
            return cached, False
        except BaseException as e:
            future.set_exception(e)
            # Mark retrieved so an unawaited future does not warn
            future.exception()
            raise
        finally:
            self._inflight.pop(key, None)

    async def _live_call(
        self,
        key: str,
        endpoint: str,
        url: str,
        model: str,
        body: dict[str, Any],
    ) -> CachedCall:
        """Make a billed call under the budget guard and store it."""
        worst_case = self._worst_case(endpoint, model, body)
        await self.budget.reserve(worst_case)
        start = time.perf_counter()
        try:
            async with self._semaphore:
                start = time.perf_counter()
                response = await self.transport.post(url, body)
                latency_ms = (time.perf_counter() - start) * 1000
        except ApiError as e:
            await self.budget.release(worst_case)
            if str(e.status) == "402":
                # Out of OpenRouter credits: stop the run as if the budget guard had
                self.budget.exhausted = True
                raise BudgetExceeded(f"OpenRouter reports insufficient credits: {e}") from e
            if not is_provider_refusal(e.status, e.message):
                raise
            # A refusal is not billed; store it so reruns replay it instead of asking again
            self.live_calls += 1
            refusal = {REFUSAL_KEY: {"status": e.status, "message": e.message}}
            latency_ms = (time.perf_counter() - start) * 1000
            self.cache.put(key, endpoint, model, body, refusal, latency_ms, 0.0)
            stored = self.cache.get(key)
            assert stored is not None
            return stored
        except BaseException:
            await self.budget.release(worst_case)
            raise

        # The fake provider reports a simulated latency instead of real elapsed time
        fake_latency = response.pop("_fake_latency_ms", None)
        if fake_latency is not None:
            latency_ms = float(fake_latency)

        prompt, completion, cost = _usage_numbers(endpoint, response)
        if cost is None:
            cost = self._list_price_cost(endpoint, model, prompt, completion)
        await self.budget.settle(worst_case, cost)
        self.live_calls += 1
        self.live_spend += cost

        self.cache.put(key, endpoint, model, body, response, latency_ms, cost)
        stored = self.cache.get(key)
        assert stored is not None
        return stored

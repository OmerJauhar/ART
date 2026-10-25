"""
Tests for the Cache, Budget Guard and Gateway
=============================================
"""

import asyncio
from pathlib import Path

import pytest

from experiments.pilot.budget import BudgetExceeded, BudgetGuard
from experiments.pilot.cache import CallCache, request_key
from experiments.pilot.gateway import TRACE, ApiError, HttpTransport
from experiments.pilot.tests.helpers import CountingTransport, make_gateway

BODY = {"model": "deepseek/deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0, "max_tokens": 100}


class TestCache:
    """Test cases for CallCache."""

    def test_key_ignores_dict_order(self) -> None:
        """Test that the key depends on content, not key order."""
        reordered = {"max_tokens": 100, "temperature": 0.0, "messages": BODY["messages"],
                     "model": BODY["model"]}

        assert request_key("chat", BODY) == request_key("chat", reordered)
        assert request_key("chat", BODY) != request_key("decisions", BODY)
        assert request_key("chat", BODY) != request_key("chat", {**BODY, "temperature": 0.5})

    def test_round_trip_and_persistence(self, tmp_path: Path) -> None:
        """Test that stored calls survive reopening the file."""
        cache = CallCache(tmp_path / "c.sqlite")
        cache.put("k", "chat", "m", BODY, {"answer": 1}, 12.5, 0.25)
        cache.close()

        reopened = CallCache(tmp_path / "c.sqlite")
        stored = reopened.get("k")

        assert stored is not None
        assert stored.response == {"answer": 1}
        assert stored.latency_ms == 12.5
        assert reopened.total_cost() == 0.25
        assert len(reopened) == 1
        assert reopened.get("missing") is None


class TestBudgetGuard:
    """Test cases for BudgetGuard."""

    @pytest.mark.asyncio
    async def test_reserve_and_settle(self) -> None:
        """Test that reservations count until settled at actual cost."""
        guard = BudgetGuard(1.0)

        await guard.reserve(0.4)
        assert guard.remaining == pytest.approx(0.6)
        await guard.settle(0.4, 0.1)

        assert guard.spent == pytest.approx(0.1)
        assert guard.reserved == 0.0

    @pytest.mark.asyncio
    async def test_refuses_call_that_would_exceed(self) -> None:
        """Test that a reservation past the cap raises and marks the guard exhausted."""
        guard = BudgetGuard(1.0, already_spent=0.7)
        await guard.reserve(0.2)

        with pytest.raises(BudgetExceeded):
            await guard.reserve(0.2)
        assert guard.exhausted is True

    @pytest.mark.asyncio
    async def test_release(self) -> None:
        """Test that a failed call's reservation is released without spending."""
        guard = BudgetGuard(1.0)
        await guard.reserve(0.5)
        await guard.release(0.5)

        assert guard.spent == 0.0
        assert guard.remaining == 1.0


class TestGateway:
    """Test cases for the Gateway."""

    @pytest.mark.asyncio
    async def test_second_call_served_from_cache(self, tmp_path: Path) -> None:
        """Test that a repeated request makes one live call and the rerun is free."""
        transport = CountingTransport(cost=0.002)
        gateway = make_gateway(tmp_path, transport)

        first = await gateway.chat(BODY)
        second = await gateway.chat(BODY)

        assert transport.calls == 1
        assert first.cached is False
        assert second.cost_usd == pytest.approx(0.002)
        assert gateway.live_spend == pytest.approx(0.002)

        # A fresh gateway over the same cache file spends nothing
        rerun_transport = CountingTransport()
        rerun = make_gateway(tmp_path, rerun_transport)
        record = await rerun.chat(BODY)
        assert rerun_transport.calls == 0
        assert record.cached is True
        assert record.cost_usd == pytest.approx(0.002)
        assert rerun.live_spend == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_identical_requests_share_one_call(self, tmp_path: Path) -> None:
        """Test that identical in-flight requests are deduplicated."""
        transport = CountingTransport()
        gateway = make_gateway(tmp_path, transport)

        await asyncio.gather(*[gateway.chat(BODY) for _ in range(5)])

        assert transport.calls == 1

    @pytest.mark.asyncio
    async def test_budget_stops_before_sending(self, tmp_path: Path) -> None:
        """Test that a call whose worst case exceeds the budget is never sent."""
        transport = CountingTransport()
        gateway = make_gateway(tmp_path, transport, max_usd=1e-9)

        with pytest.raises(BudgetExceeded):
            await gateway.chat(BODY)
        assert transport.calls == 0
        assert gateway.budget.exhausted is True

    @pytest.mark.asyncio
    async def test_cost_falls_back_to_list_price(self, tmp_path: Path) -> None:
        """Test that a response without usage.cost is priced from tokens."""
        class NoCost(CountingTransport):
            async def post(self, url: str, body: dict) -> dict:
                data = await super().post(url, body)
                del data["usage"]["cost"]
                return data

        gateway = make_gateway(tmp_path, NoCost())
        record = await gateway.chat(BODY)
        spec = gateway.models.get(BODY["model"])

        assert record.cost_usd == pytest.approx(spec.cost(10, 5))

    @pytest.mark.asyncio
    async def test_trace_collects_calls(self, tmp_path: Path) -> None:
        """Test that calls made with a trace set are appended to it."""
        gateway = make_gateway(tmp_path, CountingTransport())
        trace: list = []
        token = TRACE.set(trace)
        try:
            await gateway.chat(BODY)
            await gateway.chat(BODY)
        finally:
            TRACE.reset(token)

        assert len(trace) == 2
        assert trace[0].key == trace[1].key


class TestHttpTransport:
    """Test cases for retries in the real transport (with a mocked HTTP layer)."""

    @pytest.mark.asyncio
    async def test_retries_429_then_succeeds(self) -> None:
        """Test that a 429 is retried with backoff."""
        import httpx

        responses = iter([
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ])
        transport = HttpTransport("test-key", backoff_base=0.0)
        transport._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: next(responses))
        )

        data = await transport.post("https://example.invalid/chat", {"x": 1})

        assert data["choices"][0]["message"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_client_error_not_retried(self) -> None:
        """Test that a 400 raises immediately."""
        import httpx

        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400, text="bad model")

        transport = HttpTransport("test-key", backoff_base=0.0)
        transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with pytest.raises(ApiError) as exc_info:
            await transport.post("https://example.invalid/chat", {"x": 1})
        assert exc_info.value.status == 400
        assert len(calls) == 1

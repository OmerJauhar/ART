"""
Tests for the Pilot Runner's Shutdown
=====================================

The transport must stay open until every use of the gateway is over: the ELO router fit
replays the calibration tournaments after the per-question pass, and any call missing from
the cache then goes to the transport. Fake provider only; no network.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.pilot import run as run_module
from experiments.pilot.cache import request_key
from experiments.pilot.fake_provider import FakeTransport
from experiments.pilot.gateway import ApiError
from experiments.pilot.tests.helpers import make_question


class FlakyCritiqueTransport(FakeTransport):
    """
    Fake provider that fails each critique by one model the first time it is asked, so the
    ELO router fit has to make those calls again; counts requests made after close.
    """

    critic = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create the transport."""
        super().__init__(*args, **kwargs)
        self.failed: set[str] = set()
        self.retried = 0
        self.after_close = 0
        self.close_calls = 0

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Fail first critiques by the critic model; otherwise answer as the fake provider."""
        if self.closed:
            self.after_close += 1
        if not url.rstrip("/").endswith("/decisions"):
            key = request_key("chat", body)
            system = body["messages"][0]["content"]
            if body["model"] == self.critic and "evaluator" in system:
                if key not in self.failed:
                    self.failed.add(key)
                    raise ApiError(500, "transient failure")
                self.retried += 1
        return await super().post(url, body)

    async def close(self) -> None:
        """Count the close."""
        self.close_calls += 1
        await super().close()


class TestShutdown:
    """The transport is closed exactly once, after the ELO router fit."""

    @pytest.mark.asyncio
    async def test_no_request_after_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the router fit's live calls happen before the single close."""
        questions = [make_question(i, "calibration" if i < 4 else "test",
                                   ["SeqQA", "ProtocolQA"][i % 2], "ABCD"[i % 4])
                     for i in range(8)]
        transports: list[FlakyCritiqueTransport] = []

        def make_transport(*args: Any) -> FlakyCritiqueTransport:
            transport = FlakyCritiqueTransport(*args)
            transport.critic = next(m for m in transport.models.ids if "gpt" in m)
            transports.append(transport)
            return transport

        monkeypatch.setattr(run_module, "load_sample",
                            lambda seed: (questions, {"SeqQA": 4, "ProtocolQA": 4}))
        monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(run_module, "FakeTransport", make_transport)
        args = run_module.parse_args([
            "--dry-run", "--run-name", "shutdown", "--cache", str(tmp_path / "c.sqlite"),
            "--skip-methods", "art_r2",
        ])

        code = await run_module.run(args)

        assert code == 0
        [transport] = transports
        # The router fit did go to the transport for the critiques that failed first time
        assert transport.retried > 0
        assert transport.after_close == 0
        assert transport.close_calls == 1
        config = json.loads((tmp_path / "results" / "shutdown" / "config.json").read_text())
        assert config["elo_router"]["top_model"]
        assert "client has been closed" not in (
            tmp_path / "results" / "shutdown" / "run.log"
        ).read_text()

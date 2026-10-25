"""
Tests for the Jev Client
========================

Parsing is checked against a fixture copied from the documented Decisions API response.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.pilot.gateway import ApiError
from experiments.pilot.jev import (
    JevClient,
    build_choice_request,
    build_noul_request,
    parse_response,
)
from experiments.pilot.tests.helpers import make_gateway

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "jev_response.json").read_text())


class TestRequests:
    """Test cases for request builders."""

    def test_choice_request(self) -> None:
        """Test the Choice request shape."""
        body = build_choice_request(
            "typesafe/jev-1.14", {"q": "x"}, "Which?", {"R1": "first", "R2": "second"}
        )

        assert body == {
            "model": "typesafe/jev-1.14",
            "state": {"q": "x"},
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {"R1": "first", "R2": "second"},
                }
            },
        }

    def test_noul_request(self) -> None:
        """Test the Noul request shape and its required criteria keys."""
        body = build_noul_request("m", "state", "Is it?", {"true": "yes", "false": "no"})

        assert body["questions"]["check"]["type"] == "noul"
        assert body["questions"]["check"]["criteria"] == {"true": "yes", "false": "no"}
        with pytest.raises(ValueError):
            build_noul_request("m", "state", "Is it?", {"yes": "y", "no": "n"})


class TestParse:
    """Test cases for parse_response against the fixture."""

    def test_choice(self) -> None:
        """Test parsing a Choice answer."""
        answer = parse_response(FIXTURE, "team")

        assert answer.question_type == "choice"
        assert answer.choice == "payments"
        assert answer.confidence == 0.67
        assert answer.probabilities == {"payments": 0.78, "frontend": 0.22, "account": 0.0}
        assert answer.model_version == "typesafe/jev-1.13-20260917"
        assert answer.cost_usd == pytest.approx(0.000019992)
        assert answer.input_tokens == 476

    def test_noul(self) -> None:
        """Test parsing a Noul answer."""
        answer = parse_response(FIXTURE, "is_bug")

        assert answer.question_type == "noul"
        assert answer.p_yes == 0.96

    def test_choice_without_probabilities(self) -> None:
        """Test that a Choice answer without probabilities puts all mass on the choice."""
        response = {
            "model": "typesafe/jev-1.14-20261001",
            "answers": {"pick": {"type": "choice", "choice": "R2"}},
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }

        answer = parse_response(response, "pick")

        assert answer.probabilities == {"R2": 1.0}
        assert answer.cost_usd == 0.0

    def test_missing_and_unsupported(self) -> None:
        """Test errors for a missing answer and an unsupported type."""
        with pytest.raises(ValueError, match="no answer named"):
            parse_response(FIXTURE, "absent")
        with pytest.raises(ValueError, match="Unexpected Jev answer type"):
            parse_response(FIXTURE, "urgency")


class DecisionsTransport:
    """Transport that serves Decisions responses, rejecting one model."""

    def __init__(self, reject_model: str | None = None) -> None:
        """Create the transport."""
        self.reject_model = reject_model
        self.models: list[str] = []

    async def post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Return a Noul answer, or raise for the rejected model."""
        self.models.append(body["model"])
        if body["model"] == self.reject_model:
            raise ApiError(404, f"model {body['model']} not found")
        return {
            "model": f"{body['model']}-20261001",
            "answers": {"check": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 100, "output_tokens": 1, "cost": 0.0000042},
        }


class TestClient:
    """Test cases for JevClient through the Gateway."""

    @pytest.mark.asyncio
    async def test_noul_records_version(self, tmp_path: Path) -> None:
        """Test that the returned dated model version is kept."""
        transport = DecisionsTransport()
        client = JevClient(make_gateway(tmp_path, transport), "typesafe/jev-1.14",
                           "typesafe/jev-1.13")

        answer = await client.noul("state", "Is it?", {"true": "y", "false": "n"})

        assert answer.p_yes == 0.7
        assert answer.model_version == "typesafe/jev-1.14-20261001"
        assert transport.models == ["typesafe/jev-1.14"]

    @pytest.mark.asyncio
    async def test_falls_back_when_model_unavailable(self, tmp_path: Path) -> None:
        """Test the fallback from jev-1.14 to jev-1.13."""
        transport = DecisionsTransport(reject_model="typesafe/jev-1.14")
        client = JevClient(make_gateway(tmp_path, transport), "typesafe/jev-1.14",
                           "typesafe/jev-1.13")

        answer = await client.noul("state", "Is it?", {"true": "y", "false": "n"})

        assert client.model == "typesafe/jev-1.13"
        assert answer.model_version == "typesafe/jev-1.13-20261001"
        assert transport.models == ["typesafe/jev-1.14", "typesafe/jev-1.13"]

"""
Pilot Settings
==============

Paths, model configuration from models.yaml, role assignment, and API key loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PILOT_DIR = Path(__file__).resolve().parent
MODELS_PATH = PILOT_DIR / "models.yaml"
DATA_DIR = PILOT_DIR / "data"
RESULTS_DIR = PILOT_DIR / "results"
ENV_PATH = PILOT_DIR / ".env"


@dataclass
class ModelSpec:
    """One chat model and its per-token prices."""

    id: str
    label: str
    family: str
    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float
    supports_temperature: bool = True
    reasoning: dict[str, Any] | None = None
    why: str = ""

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost in USD for the given token counts at list price."""
        return (
            prompt_tokens * self.prompt_usd_per_mtok
            + completion_tokens * self.completion_usd_per_mtok
        ) / 1e6

    @property
    def blended_price(self) -> float:
        """Price used to rank models: input plus output price per million tokens."""
        return self.prompt_usd_per_mtok + self.completion_usd_per_mtok


@dataclass
class JevSpec:
    """Jev decision model settings."""

    decisions_url: str
    model: str
    fallback_model: str
    input_usd_per_mtok: float

    def cost(self, input_tokens: int) -> float:
        """Cost in USD for the given input tokens at list price."""
        return input_tokens * self.input_usd_per_mtok / 1e6


@dataclass
class PilotModels:
    """All model settings for a pilot run."""

    base_url: str
    temperature: float
    max_tokens: int
    models: list[ModelSpec]
    jev: JevSpec
    fetched: str = ""
    roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the model list and derive roles from prices."""
        if len(self.models) != 4:
            raise ValueError(f"Expected 4 models, got {len(self.models)}")
        if len({m.id for m in self.models}) != 4:
            raise ValueError("Model IDs must be unique")
        if not self.roles:
            self.roles = assign_roles(self.models)

    def get(self, model_id: str) -> ModelSpec:
        """Look up a model by ID."""
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(model_id)

    @property
    def ids(self) -> list[str]:
        """Model IDs in configuration order."""
        return [m.id for m in self.models]


def assign_roles(models: list[ModelSpec]) -> dict[str, str]:
    """
    Assign the cheap and judge roles by price.

    cheap is the cheapest model and judge the second-cheapest, so the judge never grades
    its own answer in the cascades.

    Args:
        models: The four pilot models

    Returns:
        Dict with "cheap" and "judge" model IDs
    """
    ranked = sorted(models, key=lambda m: (m.blended_price, m.id))
    return {"cheap": ranked[0].id, "judge": ranked[1].id}


def load_models(path: Path = MODELS_PATH) -> PilotModels:
    """Load models.yaml."""
    raw = yaml.safe_load(path.read_text())
    return PilotModels(
        base_url=raw["base_url"],
        temperature=float(raw["temperature"]),
        max_tokens=int(raw["max_tokens"]),
        models=[ModelSpec(**m) for m in raw["models"]],
        jev=JevSpec(**raw["jev"]),
        fetched=str(raw.get("fetched", "")),
    )


def load_api_key(env_path: Path = ENV_PATH) -> str | None:
    """
    Read OPENROUTER_API_KEY from the environment, then from experiments/pilot/.env.

    The key is returned to the caller only; it is never printed or logged.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    if env_path.exists():
        from dotenv import dotenv_values

        value = dotenv_values(env_path).get("OPENROUTER_API_KEY")
        if value:
            return value.strip()
    return None

"""
Configuration Utilities
=======================

Manages application configuration using environment variables and defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="ART Framework")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # API Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=False)
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/art_framework"
    )
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_ttl: int = Field(default=3600)
    
    # LLM Settings
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    default_model: str = Field(default="gpt-3.5-turbo")
    default_temperature: float = Field(default=0.7)
    default_max_tokens: int = Field(default=2048)
    llm_timeout: float = Field(default=60.0)
    
    # Tournament Settings
    default_num_tournaments: int = Field(default=1)
    default_rounds_per_tournament: int = Field(default=3)
    default_k_factor: float = Field(default=32.0)
    max_concurrent_tournaments: int = Field(default=10)
    
    # Authentication
    secret_key: str = Field(default="your-secret-key-change-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: str | None = Field(default=None)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            app_name=os.getenv("ART_APP_NAME", "ART Framework"),
            app_version=os.getenv("ART_APP_VERSION", "1.0.0"),
            debug=os.getenv("ART_DEBUG", "false").lower() == "true",
            environment=os.getenv("ART_ENVIRONMENT", "development"),
            host=os.getenv("ART_HOST", "0.0.0.0"),
            port=int(os.getenv("ART_PORT", "8000")),
            workers=int(os.getenv("ART_WORKERS", "4")),
            reload=os.getenv("ART_RELOAD", "false").lower() == "true",
            openai_api_key=os.getenv("ART_OPENAI_API_KEY"),
            openai_base_url=os.getenv("ART_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            default_model=os.getenv("ART_DEFAULT_MODEL", "gpt-3.5-turbo"),
            default_temperature=float(os.getenv("ART_DEFAULT_TEMPERATURE", "0.7")),
            default_max_tokens=int(os.getenv("ART_DEFAULT_MAX_TOKENS", "2048")),
            llm_timeout=float(os.getenv("ART_LLM_TIMEOUT", "60.0")),
            default_num_tournaments=int(os.getenv("ART_DEFAULT_NUM_TOURNAMENTS", "1")),
            default_rounds_per_tournament=int(os.getenv("ART_DEFAULT_ROUNDS_PER_TOURNAMENT", "3")),
            default_k_factor=float(os.getenv("ART_DEFAULT_K_FACTOR", "32.0")),
            max_concurrent_tournaments=int(os.getenv("ART_MAX_CONCURRENT_TOURNAMENTS", "10")),
            secret_key=os.getenv("ART_SECRET_KEY", "your-secret-key-change-in-production"),
            algorithm=os.getenv("ART_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ART_ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            rate_limit_requests=int(os.getenv("ART_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("ART_RATE_LIMIT_WINDOW", "60")),
            log_level=os.getenv("ART_LOG_LEVEL", "INFO"),
            log_file=os.getenv("ART_LOG_FILE"),
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings instance
    """
    return Settings.from_env()


def get_tournament_config() -> dict[str, Any]:
    """
    Get default tournament configuration from settings.
    
    Returns:
        Dictionary of tournament configuration
    """
    settings = get_settings()
    return {
        "num_tournaments": settings.default_num_tournaments,
        "rounds_per_tournament": settings.default_rounds_per_tournament,
        "k_factor": settings.default_k_factor,
    }

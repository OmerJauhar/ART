"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from art_framework.api.routes import router
from art_framework.utils.config import get_settings
from art_framework.utils.logging import setup_logging


# Global state
_app_state: dict[str, Any] = {
    "start_time": None,
    "tournament_engine": None,
    "agents": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    # Startup
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    
    _app_state["start_time"] = datetime.utcnow()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ART Framework")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description=(
            "ART (Adaptive Response Tuning) Framework - "
            "A multi-agent LLM optimization system using tournament-style ELO ranking"
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


def get_app_state() -> dict[str, Any]:
    """Get application state."""
    return _app_state


# Create the application instance
app = create_app()

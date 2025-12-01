"""
API Module
==========

Contains FastAPI application and RESTful endpoints for the ART framework.
"""

from art_framework.api.app import create_app
from art_framework.api.routes import router

__all__ = [
    "create_app",
    "router",
]

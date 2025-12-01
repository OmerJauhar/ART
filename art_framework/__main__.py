"""
ART Framework - Main Entry Point
=================================

Run the ART Framework API server.

Usage:
    python -m art_framework
    
Or:
    uvicorn art_framework.api.app:app --reload
"""

import uvicorn
from art_framework.utils.config import get_settings


def main() -> None:
    """Run the ART Framework API server."""
    settings = get_settings()
    
    uvicorn.run(
        "art_framework.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

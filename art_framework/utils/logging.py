"""
Logging Utilities
=================

Configures structured logging for the ART framework.
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure logging for the ART framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        log_file: Path to log file (None for console only)
        rotation: Log file rotation policy
        retention: Log file retention policy
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
    
    logger.info(f"Logging configured at level {level}")


def get_logger(name: str) -> Any:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Name for the logger context
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)

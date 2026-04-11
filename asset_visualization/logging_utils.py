"""Logging helpers for ``asset_visualization``.

The goal of this module is to give the package a single, predictable logger
configuration without reconfiguring the root logger every time a file imports
it. That keeps the package polite when used from a larger application.
"""

from __future__ import annotations

import logging
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "asset_visualization", level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured package logger.

    This function adds a stream handler only once, so repeated imports do not
    duplicate log lines.

    Args:
        name: Logger name to retrieve.
        level: Logging level for the logger and its default handler.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = get_logger()

__all__ = ["get_logger", "logger", "DEFAULT_LOG_FORMAT", "DEFAULT_DATE_FORMAT"]

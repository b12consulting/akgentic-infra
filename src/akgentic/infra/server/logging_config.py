"""Logging configuration — explicit root logger setup for the akgentic-infra server."""

from __future__ import annotations

import logging

_THIRD_PARTY_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "httpx",
    "httpcore",
    "pydantic_ai",
)

_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str) -> None:
    """Configure the root logger with a human-readable StreamHandler.

    Uses explicit handler assignment (not ``basicConfig``) to guarantee
    deterministic behaviour regardless of import order.

    Args:
        level: Log level name (e.g. "DEBUG", "INFO").
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

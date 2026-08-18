"""Logging configuration — explicit root logger setup for the akgentic-infra server."""

from __future__ import annotations

import logging

_THIRD_PARTY_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "httpx",
    "httpcore",
    "pydantic_ai",
)

_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _DowngradeGracefulShutdownFilter(logging.Filter):
    """Downgrade Uvicorn's 'Cancel N running task(s)' from ERROR to WARNING.

    This message is expected when ``timeout_graceful_shutdown`` cancels
    WebSocket streaming tasks during shutdown (see ADR-015).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.ERROR and "timeout graceful shutdown exceeded" in record.msg:
            record.levelno = logging.WARNING
            record.levelname = "WARNING"
        return True


# The one handler configure_logging itself installed on the root logger.
# Tracking it (instead of clearing root.handlers wholesale) is what makes
# re-runs idempotent AND guarantees a handler someone else attached between
# calls — e.g. an OTel LoggingHandler installed by a tier's process hook —
# is never removed by a later call.
_managed_handler: logging.Handler | None = None


def configure_logging(level: str) -> None:
    """Configure the root logger with a human-readable StreamHandler.

    Idempotent by contract: a re-run with the same level is a no-op, a
    re-run never installs a second handler, and NO call removes a handler
    this function did not install — a foreign handler attached between
    calls survives. A level change updates the root level only.

    Args:
        level: Log level name (e.g. "DEBUG", "INFO").
    """
    global _managed_handler
    root = logging.getLogger()
    root.setLevel(level)

    if _managed_handler is None or _managed_handler not in root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
        _managed_handler = handler

    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    uvicorn_error = logging.getLogger("uvicorn.error")
    if not any(isinstance(f, _DowngradeGracefulShutdownFilter) for f in uvicorn_error.filters):
        uvicorn_error.addFilter(_DowngradeGracefulShutdownFilter())

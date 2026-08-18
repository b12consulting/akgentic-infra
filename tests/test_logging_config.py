"""Tests for configure_logging() utility."""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

from akgentic.infra.server import logging_config
from akgentic.infra.server.logging_config import (
    _DowngradeGracefulShutdownFilter,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging_state() -> Generator[None, None, None]:
    """Save and restore process-global logging state around each test.

    Also resets the module's managed-handler sentinel so every test starts
    from a never-configured state — earlier suite tests calling ``create_app``
    leave the (correctly idempotent) managed handler attached otherwise.
    """
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    original_managed = logging_config._managed_handler
    uvicorn_error = logging.getLogger("uvicorn.error")
    original_filters = list(uvicorn_error.filters)
    if original_managed is not None and original_managed in root.handlers:
        root.removeHandler(original_managed)
    logging_config._managed_handler = None
    yield
    logging_config._managed_handler = original_managed
    root.setLevel(original_level)
    root.handlers = original_handlers
    uvicorn_error.filters = original_filters


class TestConfigureLogging:
    """configure_logging() sets up root logger deterministically."""

    def test_sets_root_level_debug(self) -> None:
        """configure_logging('DEBUG') sets root logger to DEBUG."""
        configure_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_sets_root_level_info(self) -> None:
        """configure_logging('INFO') sets root logger to INFO."""
        configure_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_sets_format(self) -> None:
        """The one handler this call installs carries the expected format."""
        root = logging.getLogger()
        before = list(root.handlers)
        configure_logging("INFO")
        added = [h for h in root.handlers if h not in before]
        assert len(added) == 1
        formatter = added[0].formatter
        assert formatter is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_suppresses_third_party_loggers(self) -> None:
        """Third-party loggers are set to WARNING."""
        configure_logging("DEBUG")
        for name in (
            "uvicorn",
            "uvicorn.error",
            "httpx",
            "httpcore",
            "pydantic_ai",
        ):
            assert logging.getLogger(name).level == logging.WARNING


class TestConfigureLoggingIdempotence:
    """Story 57.7 (AC 4): re-runs are no-ops and never remove foreign handlers."""

    def test_same_level_rerun_is_noop(self) -> None:
        """A second call with the same level changes neither handlers nor level."""
        configure_logging("INFO")
        root = logging.getLogger()
        handlers_after_first = list(root.handlers)
        configure_logging("INFO")
        assert root.handlers == handlers_after_first
        assert root.level == logging.INFO

    def test_foreign_handler_survives_rerun(self) -> None:
        """An OTel-style handler installed between calls survives a re-run."""
        configure_logging("INFO")
        root = logging.getLogger()
        foreign = logging.Handler()
        root.addHandler(foreign)
        configure_logging("INFO")
        assert foreign in root.handlers

    def test_preexisting_handlers_survive_first_call(self) -> None:
        """configure_logging never removes a handler it did not install."""
        root = logging.getLogger()
        preexisting = logging.Handler()
        root.addHandler(preexisting)
        configure_logging("INFO")
        assert preexisting in root.handlers

    def test_level_change_updates_level_without_new_handler(self) -> None:
        """A different level moves the root level; the handler set is untouched."""
        configure_logging("INFO")
        root = logging.getLogger()
        handlers_after_first = list(root.handlers)
        configure_logging("DEBUG")
        assert root.handlers == handlers_after_first
        assert root.level == logging.DEBUG

    def test_rerun_adds_no_duplicate_downgrade_filter(self) -> None:
        """The uvicorn.error downgrade filter is installed at most once."""
        configure_logging("INFO")
        configure_logging("INFO")
        filters = logging.getLogger("uvicorn.error").filters
        assert sum(isinstance(f, _DowngradeGracefulShutdownFilter) for f in filters) == 1

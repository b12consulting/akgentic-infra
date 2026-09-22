"""Shared fixtures for server model tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_akgentic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``AKGENTIC_*`` variable so default assertions see the defaults.

    ``tests/integration/conftest.py`` loads the project ``.env`` into
    ``os.environ`` for the whole session, and ``pydantic-settings`` reads the
    environment on every construction — so any key a developer's ``.env`` sets
    silently overrides the default a test asserts. Tests that need a variable
    set it themselves, after this runs.
    """
    for key in list(os.environ):
        if key.startswith("AKGENTIC_"):
            monkeypatch.delenv(key)

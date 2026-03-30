"""E2E test fixtures — real running server, real LLM, real CLI binary.

Run locally with:
    source .env && uvicorn akgentic.infra.server.app:app --port 8010 &
    pytest tests/e2e/ -v

All E2E tests require:
  1. A running akgentic-infra server (default http://localhost:8010)
  2. OPENAI_API_KEY set in the environment (or loaded from .env)

Tests are skipped when either condition is not met.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def poll_until(
    predicate: Callable[[], bool],
    timeout: float = 60.0,
    interval: float = 1.0,
    message: str = "Condition not met within timeout",
) -> None:
    """Poll predicate until True or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(message)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv() -> None:
    """Load .env once per session (makes OPENAI_API_KEY available)."""
    load_dotenv(_PROJECT_ROOT / ".env")


@pytest.fixture(scope="session")
def e2e_api_key(_load_dotenv: None) -> str:
    """Return OPENAI_API_KEY or skip all E2E tests."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — required for E2E tests")
    return key


@pytest.fixture(scope="session")
def e2e_server_url() -> str:
    """Return the server URL (configurable via AKGENTIC_SERVER_URL)."""
    return os.environ.get("AKGENTIC_SERVER_URL", "http://localhost:8010")


@pytest.fixture(scope="session")
def e2e_ws_url(e2e_server_url: str) -> str:
    """Return the WebSocket URL derived from the server URL."""
    return e2e_server_url.replace("http://", "ws://").replace("https://", "wss://")


@pytest.fixture(scope="session")
def e2e_server_ready(e2e_server_url: str, e2e_api_key: str) -> str:
    """Verify the server is reachable, skip all E2E tests if not."""
    try:
        resp = httpx.get(f"{e2e_server_url}/teams/", timeout=5.0)
        if resp.status_code not in (200, 401, 403):
            pytest.skip(f"Server at {e2e_server_url} returned unexpected status {resp.status_code}")
    except httpx.ConnectError:
        pytest.skip(f"Server not reachable at {e2e_server_url} — start it before running E2E tests")
    return e2e_server_url


@pytest.fixture(scope="session")
def e2e_http_client(e2e_server_ready: str) -> Generator[httpx.Client, None, None]:
    """Session-scoped httpx.Client with base_url set to the running server."""
    client = httpx.Client(base_url=e2e_server_ready, timeout=30.0)
    yield client
    client.close()

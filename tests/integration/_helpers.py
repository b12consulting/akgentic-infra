"""Shared helpers for integration tests — real app, real LLM."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

CATALOG_ENTRY_ID = "test-team"
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 60.0


def create_team(client: TestClient) -> str:
    """POST /teams and return the team_id."""
    resp = client.post("/teams/", json={"catalog_entry_id": CATALOG_ENTRY_ID})
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "running"
    return data["team_id"]


def wait_for_llm_response(
    client: TestClient,
    team_id: str,
    timeout: float = POLL_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Poll GET /teams/{team_id}/events until @Manager responds."""
    deadline = time.monotonic() + timeout
    events: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        resp = client.get(f"/teams/{team_id}/events")
        assert resp.status_code == 200
        events = resp.json()["events"]
        if has_llm_content(events):
            return events
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"Timed out after {timeout}s waiting for LLM response "
        f"(got {len(events)} events, none with LLM content)"
    )


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    """Write a single YAML entry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def seed_integration_catalog(catalog_root: Path) -> None:
    """Seed YAML catalog with an LLM-capable agent for integration testing.

    Uses gpt-4o-mini for fast, cheap LLM calls.
    """
    _write_yaml(
        catalog_root / "agents" / "human-proxy.yaml",
        {
            "id": "human-proxy",
            "tool_ids": [],
            "card": {
                "role": "Human",
                "description": "Human user interface",
                "skills": [],
                "agent_class": "akgentic.agent.HumanProxy",
                "config": {"name": "@Human", "role": "Human"},
                "routes_to": ["@Manager"],
            },
        },
    )
    _write_yaml(
        catalog_root / "agents" / "manager.yaml",
        {
            "id": "manager",
            "tool_ids": [],
            "card": {
                "role": "Manager",
                "description": "Integration test manager agent",
                "skills": ["coordination"],
                "agent_class": "akgentic.agent.BaseAgent",
                "config": {
                    "name": "@Manager",
                    "role": "Manager",
                    "prompt": {
                        "template": (
                            "You are a helpful assistant. Reply concisely in one or two sentences."
                        ),
                    },
                    "model_cfg": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "temperature": 0.0,
                    },
                    "usage_limits": {
                        "request_limit": 5,
                        "total_tokens_limit": 10000,
                    },
                },
                "routes_to": [],
            },
        },
    )
    _write_yaml(
        catalog_root / "teams" / "test-team.yaml",
        {
            "id": "test-team",
            "name": "Integration Test Team",
            "entry_point": "human-proxy",
            "message_types": ["akgentic.agent.AgentMessage"],
            "members": [
                {"agent_id": "human-proxy"},
                {"agent_id": "manager"},
            ],
            "profiles": [],
        },
    )
    (catalog_root / "templates").mkdir(parents=True, exist_ok=True)
    (catalog_root / "tools").mkdir(parents=True, exist_ok=True)


def has_llm_content(events: list[dict[str, object]]) -> bool:
    """Check if any event contains LLM-generated content from @Manager."""
    for ev_wrapper in events:
        ev = ev_wrapper["event"]
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        sender = ev.get("sender")
        if not isinstance(sender, dict):
            continue
        if isinstance(content, str) and len(content) > 0 and sender.get("name") == "@Manager":
            return True
    return False

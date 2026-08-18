"""Shared test fixtures for akgentic-infra tests."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from akgentic.catalog import ENV_VAR as CATALOG_PREFIXES_ENV_VAR
from akgentic.catalog import reset_allowed_prefixes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked with ``llm`` when OPENAI_API_KEY is not set."""
    if not os.environ.get("OPENAI_API_KEY"):
        skip_llm = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    """Write a single YAML entry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _seed_catalog(catalog_root: Path) -> None:
    """Create minimal YAML catalog entries for a test team.

    Seeds the v2 per-namespace layout (``{catalog_root}/{namespace}/{kind}/{id}.yaml``)
    only. After Story 18.3 the community-tier wiring exposes a single unified
    ``Catalog`` — the legacy v1 per-kind layout is no longer consumed.
    Namespace ``test-team`` matches what tests post via
    ``catalog_namespace="test-team"``.

    Two more namespaces stand in for the failure modes ``POST /teams`` must tell
    apart from an absent namespace, both seeded on disk exactly like the valid
    one so the resolution they trigger is the real one:

    * ``broken-team`` — holds a team entry that does not resolve.
    * ``teamless`` — exists and holds an entry, but none of kind ``team``.
    """
    _seed_v2_namespace(catalog_root, namespace="test-team")
    _seed_invalid_v2_namespace(catalog_root, namespace="broken-team")
    _seed_teamless_v2_namespace(catalog_root, namespace="teamless")


_TEAM_CARD_TYPE = "akgentic.team.models.TeamCard"
_NAMESPACE_META_TYPE = "akgentic.catalog.models.namespace_meta.NamespaceMeta"

BROKEN_REF_MARKER: dict[str, object] = {
    "__ref__": "id_team_prompt",
    "params": {"tone": "formal"},
}
"""A ref marker carrying a forbidden sibling key — the field regression case.

``params`` sits next to ``__ref__``, which a marker may not carry: it is a pure
pointer. The catalog rejects the bundle at resolution time with a message naming
the marker and the offending key, and that message is what ``POST /teams`` must
carry to the client instead of collapsing it into "not found".
"""


def _team_payload() -> dict[str, Any]:
    """Build the minimal resolvable ``TeamCard`` payload used by the seeders.

    The ``TeamCard`` payload shape is taken from
    ``akgentic.team.models.TeamCard``; every agent_class / model_type
    string satisfies the v2 allowlist (``akgentic.*``).

    The member configs use plain ``akgentic.core.agent.Akgent`` (which
    expects ``BaseConfig``) because the v2 resolver hydrates
    ``AgentCard.config`` against the declared annotation (``BaseConfig``).
    Upgrading v2 to per-agent-class config types is tracked as part of
    Epic 19's v1 removal. Tests that only need the agents to *exist*
    and route messages by name work with the plain base class.
    """
    return {
        "name": "Test Team",
        "description": "v2 test team for infra tests",
        "entry_point": {
            "card": {
                "description": "Human user interface",
                "skills": [],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "@Human", "role": "Human"},
                "routes_to": ["@Manager"],
            },
            "headcount": 1,
            "members": [],
        },
        "members": [
            {
                "card": {
                    "description": "Test manager agent",
                    "skills": ["coordination"],
                    "agent_class": "akgentic.core.agent.Akgent",
                    "config": {"name": "@Manager", "role": "Manager"},
                    "routes_to": [],
                },
                "headcount": 1,
                "members": [],
            },
        ],
        "message_types": [{"__type__": "akgentic.core.messages.UserMessage"}],
        "agent_profiles": [],
    }


def _write_team_entry(catalog_root: Path, namespace: str, payload: dict[str, Any]) -> None:
    """Write ``payload`` as the ``kind="team"`` entry of ``namespace``."""
    _write_yaml(
        catalog_root / namespace / "team" / "team.yaml",
        {
            "id": "team",
            "kind": "team",
            "namespace": namespace,
            "model_type": _TEAM_CARD_TYPE,
            "description": "v2 test team namespace bundle",
            "payload": payload,
        },
    )


def _seed_v2_namespace(catalog_root: Path, namespace: str) -> None:
    """Write a minimal, resolvable v2 team-namespace bundle into ``catalog_root``."""
    _write_team_entry(catalog_root, namespace, _team_payload())


def _seed_invalid_v2_namespace(catalog_root: Path, namespace: str) -> None:
    """Write a namespace that exists and holds a team entry that does not resolve.

    The team entry is well-formed YAML and lists non-empty, so the namespace is
    unmistakably present; resolution fails only when the catalog walks the
    payload and finds the ref marker's forbidden sibling key. That ordering is
    what makes this a regression fixture rather than a mock: nothing here raises
    an exception on purpose.

    The marker's target prompt is seeded alongside, as it was in the field: a
    marker pointing at nothing fails earlier, with a different message, and
    would never exercise the rule that actually bit.
    """
    payload = _team_payload()
    payload["members"][0]["card"]["config"]["prompt"] = dict(BROKEN_REF_MARKER)
    _write_team_entry(catalog_root, namespace, payload)
    _write_yaml(
        catalog_root / namespace / "prompt" / "id_team_prompt.yaml",
        {
            "id": "id_team_prompt",
            "kind": "prompt",
            "namespace": namespace,
            "model_type": "akgentic.llm.PromptTemplate",
            "description": "target of the malformed ref marker",
            "payload": {"template": "hello {tone}", "params": {}},
        },
    )


def _seed_teamless_v2_namespace(catalog_root: Path, namespace: str) -> None:
    """Write a namespace holding a ``kind="meta"`` anchor and no team entry.

    Present, listable, and nothing about it is broken — there is simply no team
    to create from, which is a 404 that says so, not a 409.
    """
    _write_yaml(
        catalog_root / namespace / "meta" / "_meta.yaml",
        {
            "id": "_meta",
            "kind": "meta",
            "namespace": namespace,
            "model_type": _NAMESPACE_META_TYPE,
            "description": "v2 namespace anchor with no team entry",
            "payload": {"name": namespace, "description": "no team here"},
        },
    )


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy OPENAI_API_KEY so BaseAgent actors can initialise in unit tests."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")


@pytest.fixture(autouse=True)
def _isolate_catalog_model_type_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Keep the catalog's process-wide prefix allowlist from leaking between tests.

    The allowlist is a module global that *latches*: once resolved — by the
    first read, or by ``create_app`` applying the setting — the environment is
    never consulted again for the life of the process. Since ``create_app``
    resolves it, every app-building test in this suite would otherwise pin the
    policy for whatever runs next, and a test that sets the variable would pass
    or fail by collection order. Suite-wide and autouse so no test can forget.
    """
    monkeypatch.delenv(CATALOG_PREFIXES_ENV_VAR, raising=False)
    reset_allowed_prefixes()
    yield
    reset_allowed_prefixes()


@pytest.fixture()
def server_settings(tmp_path: Path) -> CommunitySettings:
    """Server settings with tmp_path-based workspaces."""
    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )


@pytest.fixture()
def seeded_settings(tmp_path: Path) -> CommunitySettings:
    """Server settings with pre-seeded catalog YAML files."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    _seed_catalog(settings.catalog_path)
    return settings


@pytest.fixture()
def community_services(
    seeded_settings: CommunitySettings,
) -> Generator[CommunityServices, None, None]:
    """Wired community services with seeded catalog data."""
    services = wire_community(seeded_settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def team_service(
    community_services: CommunityServices,
) -> TeamService:
    """The wired TeamService carried by the community services container."""
    assert community_services.team_service is not None
    return community_services.team_service


@pytest.fixture()
def app(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> Generator[FastAPI, None, None]:
    """FastAPI app via the two-arg factory."""
    application = create_app(community_services, seeded_settings)
    yield application
    application.state.services.actor_system.shutdown()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Sync HTTP test client."""
    return TestClient(app)

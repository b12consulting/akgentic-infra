"""Team-metadata models and catalog fixtures for the metadata HTTP-surface tests.

The models live in an importable module rather than inside a test function
because both ends of the persistence round-trip need a dotted path: the catalog
stores ``TeamCard.metadata_type`` as a ``__type__`` tag, and the event store
stores ``Process.metadata`` as a ``__model__``-tagged dict. A locally-defined
class would serialize fine and fail to import back.

Field names and values use ``acme`` / ``contoso`` placeholders (Golden Rule #9).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from akgentic.agent.config import AgentConfig
from akgentic.core.agent_card import AgentCard
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.metadata import TeamMetadata
from akgentic.tool import ToolCard
from pydantic import Field

ACME_METADATA_TYPE = "tests.fixtures.team_metadata.AcmeCaseMetadata"
"""Dotted path the seeded catalog entry declares as the team's metadata_type."""


class AcmeOwner(SerializableBaseModel):
    """A nested sub-model, so tests can prove the ``__model__`` recursion is real.

    Only *indexed* metadata fields must be scalars, so a metadata value may
    legitimately nest — and a nested model carries its own ``__model__`` tag.
    That is exactly the case a top-level-only scan or strip would miss.
    """

    email: str = Field(description="Owner contact address")
    squad: str | None = Field(default=None, description="Owning squad, when known")


class AcmeCaseMetadata(TeamMetadata):
    """Business metadata for a support-case team.

    ``tenant``, ``case`` and ``channel`` are indexed (filterable); the rest are
    not, and do not need to be. ``owner`` nests a sub-model and ``watchers``
    holds a *list* of them — the two shapes that carry a nested ``__model__``
    tag, so both the inbound scan and the outbound strip are exercised on real
    data rather than on a hand-built dict.

    ``channel`` is indexed *and optional*, which the two required indexed fields
    cannot stand in for: an indexed field that goes absent must take its index
    entry with it, so replace-vs-merge is only observable through the index when
    an entry can disappear rather than merely be overwritten.
    """

    tenant: str = Field(json_schema_extra={"indexed": True}, description="Owning tenant")
    case: str = Field(json_schema_extra={"indexed": True}, description="Case reference")
    channel: str | None = Field(
        default=None,
        json_schema_extra={"indexed": True},
        description="Intake channel, when known; indexed and optional",
    )
    owner: AcmeOwner | None = Field(default=None, description="Case owner, when assigned")
    watchers: list[AcmeOwner] = Field(
        default_factory=list, description="Additional watchers; not filterable"
    )
    note: str | None = Field(default=None, description="Free-text note; not filterable")


def make_metadata_body(**overrides: Any) -> dict[str, Any]:
    """Build a valid ``AcmeCaseMetadata`` request body from a real model instance.

    Follows the fixture-factory rule: the dict comes from ``model_dump`` of a
    real model, so it cannot drift from the schema. The ``__model__`` tag is
    stripped because a request body must never carry one — that is the very
    thing the route rejects.
    """
    defaults: dict[str, Any] = {"tenant": "acme", "case": "C-1234"}
    defaults.update(overrides)
    dumped = AcmeCaseMetadata(**defaults).model_dump(mode="json")
    return _strip_tags(dumped)


def _strip_tags(value: Any) -> Any:
    """Remove every ``__model__`` key so the result is a legal request body."""
    if isinstance(value, dict):
        return {k: _strip_tags(v) for k, v in value.items() if k != "__model__"}
    if isinstance(value, list):
        return [_strip_tags(item) for item in value]
    return value


def _manager_card_payload(tools: Sequence[ToolCard]) -> dict[str, Any]:
    """The Manager member's card, carrying *tools* when any are given.

    A member carrying tools must be a ``BaseAgent``, whose config annotation is
    ``AgentConfig``, so that ``tools`` hydrates; and each tool entry needs its
    ``__model__`` tag. Both come from dumping a real ``AgentCard`` rather than
    typing the YAML by hand. With no tools, the plain ``Akgent`` member this file
    always seeded is returned unchanged, so every existing namespace is
    byte-identical.
    """
    if not tools:
        return {
            "role": "Manager",
            "description": "Test manager agent",
            "skills": ["coordination"],
            "agent_class": "akgentic.core.agent.Akgent",
            "config": {"name": "@Manager", "role": "Manager"},
        }
    card = AgentCard(
        description="Test manager agent",
        skills=["coordination"],
        agent_class="akgentic.agent.agent.BaseAgent",
        config=AgentConfig(name="@Manager", role="Manager", tools=list(tools)),
    )
    return card.model_dump(mode="json")


def seed_metadata_namespace(
    catalog_root: Path,
    namespace: str,
    *,
    with_type: bool,
    tools: Sequence[ToolCard] = (),
) -> None:
    """Seed a v2 team namespace whose card declares (or omits) a metadata_type.

    Mirrors ``tests/conftest.py:_seed_v2_namespace`` and adds the one field this
    epic cares about. ``with_type=False`` produces the "team declares no metadata
    contract" card that AC #3 and AC #8 need. ``tools`` puts tool cards on the
    Manager member, which is how a seeded team comes to bind a real workspace
    when it is created through the wired service.
    """
    team_payload: dict[str, Any] = {
        "name": "Acme Case Team",
        "description": "v2 test team carrying business metadata",
        "entry_point": {
            "card": {
                "role": "Human",
                "description": "Human user interface",
                "skills": [],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "@Human", "role": "Human"},
            },
            "headcount": 1,
            "members": [],
        },
        "members": [
            {
                "card": _manager_card_payload(tools),
                "headcount": 1,
                "members": [],
            },
        ],
        "message_types": [{"__type__": "akgentic.core.messages.UserMessage"}],
        "agent_profiles": [],
    }
    if with_type:
        team_payload["metadata_type"] = {"__type__": ACME_METADATA_TYPE}

    path = catalog_root / namespace / "team" / "team.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            {
                "id": "team",
                "kind": "team",
                "namespace": namespace,
                "model_type": "akgentic.team.models.TeamCard",
                "description": "v2 team namespace bundle for metadata tests",
                "payload": team_payload,
            },
            default_flow_style=False,
        )
    )

"""Team-metadata models and catalog fixtures for the metadata HTTP-surface tests.

The models live in an importable module rather than inside a test function
because both ends of the persistence round-trip need a dotted path: the catalog
stores ``TeamCard.metadata_type`` as a ``__type__`` tag, and the event store
stores ``Process.metadata`` as a ``__model__``-tagged dict. A locally-defined
class would serialize fine and fail to import back.

Field names and values use ``acme`` / ``contoso`` placeholders (Golden Rule #9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.metadata import TeamMetadata
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

    ``tenant`` and ``case`` are indexed (filterable); the rest are not, and do
    not need to be. ``owner`` nests a sub-model and ``watchers`` holds a *list*
    of them — the two shapes that carry a nested ``__model__`` tag, so both the
    inbound scan and the outbound strip are exercised on real data rather than
    on a hand-built dict.
    """

    tenant: str = Field(json_schema_extra={"indexed": True}, description="Owning tenant")
    case: str = Field(json_schema_extra={"indexed": True}, description="Case reference")
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


def seed_metadata_namespace(catalog_root: Path, namespace: str, *, with_type: bool) -> None:
    """Seed a v2 team namespace whose card declares (or omits) a metadata_type.

    Mirrors ``tests/conftest.py:_seed_v2_namespace`` and adds the one field this
    epic cares about. ``with_type=False`` produces the "team declares no metadata
    contract" card that AC #3 and AC #8 need.
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
                "routes_to": ["@Manager"],
            },
            "headcount": 1,
            "members": [],
        },
        "members": [
            {
                "card": {
                    "role": "Manager",
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

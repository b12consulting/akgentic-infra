"""Fakes shared by the maintenance-sweep tests."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
from akgentic.core.agent_card import AgentCard
from akgentic.team.models import Process, TeamCard, TeamStatus
from akgentic.team.projection import derive_team_projection

from akgentic.infra.maintenance.models import ResourceKind, ResourceRef

_TEAM_CARD_PAYLOAD = {
    "name": "Sweep Team",
    "description": "maintenance-sweep test team",
    "entry_point": {
        "card": {
            "role": "Human",
            "description": "Human user interface",
            "skills": [],
            "agent_class": "akgentic.core.agent.Akgent",
            "config": {"name": "@Human", "role": "Human"},
            "routes_to": [],
        },
        "headcount": 1,
        "members": [],
    },
    "members": [],
}


def _default_cards() -> dict[str, AgentCard]:
    """Card map for the shared team payload, keyed by the hash refs use.

    Every ``make_process`` team is built from one payload, so one map resolves
    all of them — and the driver's claim lookup is exercised for real rather
    than stubbed away.
    """
    card = TeamCard.model_validate(_TEAM_CARD_PAYLOAD)
    projection = derive_team_projection(card)
    return {
        ref.card_hash: agent_card
        for ref, agent_card in zip(projection.agent_cards, projection.cards, strict=True)
    }


def make_process(team_id: uuid.UUID, status: TeamStatus = TeamStatus.RUNNING) -> Process:
    """Build a persisted process for *team_id* in *status*."""
    now = datetime.now(UTC)
    card = TeamCard.model_validate(_TEAM_CARD_PAYLOAD)
    projection = derive_team_projection(card)
    return Process(
        team_id=team_id,
        status=status,
        user_id="user-1",
        created_at=now,
        updated_at=now,
        team_name=projection.team_name,
        entry_point=projection.entry_point,
        supervisors=projection.supervisors,
        agent_cards=projection.agent_cards,
        message_types=projection.message_types,
    )


class FakeEventStore:
    """An ``EventStore`` stub exposing only what the sweep reads.

    Records the call order into a shared journal so the scan-before-live-read
    ordering can be asserted rather than assumed.
    """

    def __init__(
        self,
        processes: list[Process],
        journal: list[str] | None = None,
        cards: dict[str, AgentCard] | None = None,
        *,
        card_error: Exception | None = None,
    ) -> None:
        self.processes = processes
        self.journal = journal if journal is not None else []
        self.cards = cards if cards is not None else _default_cards()
        self._card_error = card_error
        self.calls = 0

    def list_teams(self) -> list[Process]:
        """Return every persisted process, live or deleted."""
        self.calls += 1
        self.journal.append("list_teams")
        return self.processes

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Resolve card hashes, omitting any the store does not hold."""
        self.journal.append("load_agent_cards")
        if self._card_error is not None:
            raise self._card_error
        return {h: self.cards[h] for h in hashes if h in self.cards}

    def set_card_error(self, error: Exception) -> None:
        """Make every subsequent ``load_agent_cards`` raise *error*."""
        self._card_error = error


class FakeReaper:
    """A ``TeamResourceReaper`` stub over a fixed reference list."""

    def __init__(
        self,
        refs: list[ResourceRef],
        journal: list[str] | None = None,
        *,
        scan_error: Exception | None = None,
        purge_error: Exception | None = None,
        kind: ResourceKind = ResourceKind.DOCKER,
    ) -> None:
        self.kind = kind
        self._refs = refs
        self.journal = journal if journal is not None else []
        self._scan_error = scan_error
        self._purge_error = purge_error
        self.purged: list[ResourceRef] = []
        self.closed = False

    def scan(self) -> list[ResourceRef]:
        """Return the fixed reference list, or raise the configured error."""
        self.journal.append("scan")
        if self._scan_error is not None:
            raise self._scan_error
        return list(self._refs)

    def purge(self, ref: ResourceRef) -> int:
        """Record the purge, or raise the configured error."""
        if self._purge_error is not None:
            raise self._purge_error
        self.purged.append(ref)
        return ref.size_hint or 1

    def close(self) -> None:
        """Mark the reaper closed."""
        self.closed = True


def make_ref(team_id: str, *, age_seconds: float | None = None, size: int = 1) -> ResourceRef:
    """Build a Docker-kind reference owned by *team_id*."""
    return ResourceRef(
        kind=ResourceKind.DOCKER,
        team_id=team_id,
        detail=f"cid-{team_id[:8]}",
        label=f"sandbox-{team_id}",
        size_hint=size,
        age_seconds=age_seconds,
    )


@pytest.fixture
def live_team_id() -> uuid.UUID:
    """A team id that exists and is running."""
    return uuid.uuid4()


@pytest.fixture
def dead_team_id() -> uuid.UUID:
    """A team id that no store knows about."""
    return uuid.uuid4()


def make_claiming_store(
    team_id: uuid.UUID,
    workspace_id: str,
    status: TeamStatus = TeamStatus.RUNNING,
) -> FakeEventStore:
    """Build a store holding one team that declares *workspace_id*.

    Uses a real ``WorkspaceTool`` card rather than a hand-shaped dict, and
    resolves the cards through the store exactly as the driver does — so the
    claim is discovered through the same nesting and the same round trip a
    deployment produces.
    """
    payload = json.loads(json.dumps(_TEAM_CARD_PAYLOAD))
    payload["members"] = [
        {
            "card": {
                "role": "Worker",
                "description": "Worker with a shared workspace",
                "skills": [],
                "agent_class": "akgentic.agent.agent.BaseAgent",
                "config": {
                    "__model__": "akgentic.agent.config.AgentConfig",
                    "name": "@Worker",
                    "role": "Worker",
                    "tools": [
                        {
                            "__model__": "akgentic.tool.workspace.card.WorkspaceTool",
                            "workspace_id": workspace_id,
                        }
                    ],
                },
                "routes_to": [],
            },
            "headcount": 1,
            "members": [],
        }
    ]
    card = TeamCard.model_validate(payload)
    projection = derive_team_projection(card)
    now = datetime.now(UTC)
    process = Process(
        team_id=team_id,
        status=status,
        user_id="user-1",
        created_at=now,
        updated_at=now,
        team_name=projection.team_name,
        entry_point=projection.entry_point,
        supervisors=projection.supervisors,
        agent_cards=projection.agent_cards,
        message_types=projection.message_types,
    )
    stored = {
        ref.card_hash: agent_card
        for ref, agent_card in zip(process.agent_cards, projection.cards, strict=True)
    }
    return FakeEventStore([process], cards=stored)

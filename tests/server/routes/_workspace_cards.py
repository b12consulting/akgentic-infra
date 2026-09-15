"""Builders for teams whose cards declare workspaces (Story 67.1).

The seeded catalog team carries plain ``BaseConfig`` members, so it declares no
workspace at all — which is now the correct answer to every ``?workspace_id=``.
Exercising the declared path needs cards whose ``AgentConfig`` really carries a
``WorkspaceTool`` — file-capable or shell-only — and these build them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from akgentic.agent.config import AgentConfig
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.models import AgentCardRef, AgentRef, Process, TeamStatus
from akgentic.team.ports import EventStore
from akgentic.team.projection import hash_agent_card
from akgentic.tool import ToolCard
from akgentic.tool.workspace import WorkspaceTool

__all__ = [
    "CaseMetadata",
    "RecordingCardStore",
    "bare_card",
    "declare_workspaces",
    "exec_only_workspace",
    "process_with_cards",
    "tool_card",
]


def exec_only_workspace(
    workspace_id: str | None = None, *, workspace_metadata_keys: list[str] | None = None
) -> WorkspaceTool:
    """A ``WorkspaceTool`` granting sandboxed execution and nothing else.

    The migration shape of the retired standalone exec card: ``workspace_exec``
    on, every file capability explicitly off. It still declares a workspace
    through the same two layout fields as any other card, which is what the
    resolution seam must keep honouring — a shell-only agent writes into a real
    directory.
    """
    return WorkspaceTool(
        workspace_id=workspace_id,
        workspace_metadata_keys=workspace_metadata_keys or [],
        workspace_exec=True,
        workspace_read=False,
        workspace_view=False,
        workspace_list=False,
        workspace_glob=False,
        workspace_grep=False,
        expand_media_refs=False,
        workspace_write=False,
        workspace_delete=False,
        workspace_edit=False,
        workspace_multi_edit=False,
        workspace_patch=False,
        workspace_mkdir=False,
    )


class CaseMetadata(SerializableBaseModel):
    """A team-metadata stand-in with the two fields a metadata card names."""

    customer_id: str = "ACME"
    case_id: str = "42"


class RecordingCardStore:
    """A card store recording every ``load_agent_cards`` call.

    Only ``load_agent_cards`` is exercised — it is the single method
    ``resolve_agent_cards`` reaches — so the rest of the ``EventStore``
    Protocol is deliberately absent rather than stubbed into existence.
    """

    def __init__(self, cards: list[AgentCard], *, missing: bool = False) -> None:
        self.calls: list[list[str]] = []
        self._by_hash = {} if missing else {hash_agent_card(c): c for c in cards}

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        self.calls.append(list(hashes))
        return {h: self._by_hash[h] for h in hashes if h in self._by_hash}


def tool_card(role: str, *tools: ToolCard) -> AgentCard:
    """An agent card whose ``AgentConfig`` carries *tools*."""
    return AgentCard(
        description=f"{role} agent",
        skills=[],
        agent_class="akgentic.agent.agent.BaseAgent",
        config=AgentConfig(name=f"@{role}", role=role, tools=list(tools)),
    )


def bare_card(role: str) -> AgentCard:
    """An agent card carrying a plain ``BaseConfig`` — no ``tools`` field at all."""
    return AgentCard(
        description=f"{role} agent",
        skills=[],
        agent_class="akgentic.core.agent.Akgent",
        config=BaseConfig(name=f"@{role}", role=role),
    )


def process_with_cards(
    cards: list[AgentCard],
    *,
    team_id: uuid.UUID | None = None,
    user_id: str = "alice",
    metadata: SerializableBaseModel | None = None,
) -> Process:
    """A ``Process`` whose ``agent_cards`` reference *cards*, entry point first."""
    refs = [AgentCardRef(role=c.role, card_hash=hash_agent_card(c)) for c in cards]
    now = datetime.now(UTC)
    return Process(
        team_id=team_id or uuid.uuid4(),
        user_id=user_id,
        status=TeamStatus.RUNNING,
        created_at=now,
        updated_at=now,
        entry_point=AgentRef(name=f"@{cards[0].role}", role=cards[0].role),
        agent_cards=refs,
        metadata=metadata,
    )


def declare_workspaces(
    store: EventStore,
    process: Process,
    *tools: WorkspaceTool,
    role: str = "WorkspaceHolder",
    metadata: SerializableBaseModel | None = None,
) -> Process:
    """Give a live team a card declaring *tools*, and persist the result.

    Appends a new role rather than replacing ``agent_cards`` wholesale, so the
    ``Process`` validators that require ``entry_point`` and every supervisor to
    resolve keep holding. The card is saved into the same content-addressed
    store the route resolves against, so nothing here is a mock: the route takes
    exactly the path it takes in production.
    """
    card = tool_card(role, *tools)
    store.save_agent_cards([card])
    updated = process.model_copy(
        update={
            "agent_cards": [
                *process.agent_cards,
                AgentCardRef(role=card.role, card_hash=hash_agent_card(card)),
            ],
            **({"metadata": metadata} if metadata is not None else {}),
        }
    )
    store.save_team(updated)
    return updated

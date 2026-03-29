"""Dependency injection containers for tiered service assembly."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from akgentic.catalog.services import (
    AgentCatalog,
    TeamCatalog,
    TemplateCatalog,
    ToolCatalog,
)
from akgentic.core import ActorSystem
from akgentic.infra.protocols.auth import AuthStrategy
from akgentic.infra.protocols.placement import PlacementStrategy
from akgentic.infra.protocols.team_handle import RuntimeCache
from akgentic.team.manager import TeamManager
from akgentic.team.ports import ServiceRegistry
from akgentic.team.repositories.yaml import YamlEventStore


class TierServices(BaseModel):
    """Base dependency container holding services common to all deployment tiers.

    This is a runtime DI container, NOT a serialization model — arbitrary_types_allowed
    is intentional (Golden Rule #1b exemption).

    Note: ``event_store`` uses the concrete ``YamlEventStore`` type because
    ``EventStore`` (from akgentic-team) is not ``@runtime_checkable`` and cannot
    be modified from this submodule.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    placement: PlacementStrategy = Field(
        description="Strategy for placing teams on workers"
    )
    service_registry: ServiceRegistry = Field(
        description="Service discovery registry for worker instances"
    )
    auth: AuthStrategy = Field(
        description="Authentication strategy for incoming requests"
    )
    event_store: YamlEventStore = Field(
        description="Persistence backend for team event sourcing"
    )
    runtime_cache: RuntimeCache = Field(
        description="Cache mapping team IDs to live TeamHandle instances"
    )


class CommunityServices(TierServices):
    """Community-tier service container with embedded TeamManager and catalogs.

    Extends TierServices with an in-process TeamManager and YAML-backed
    catalogs for single-process deployment.
    """

    actor_system: ActorSystem = Field(
        description="Actor system for managing agent lifecycle"
    )
    team_manager: TeamManager = Field(
        description="Team lifecycle manager (embedded, in-process)"
    )
    team_catalog: TeamCatalog = Field(
        description="Catalog service for team entry resolution"
    )
    agent_catalog: AgentCatalog = Field(
        description="Catalog service for agent entry resolution"
    )
    tool_catalog: ToolCatalog = Field(
        description="Catalog service for tool entry resolution"
    )
    template_catalog: TemplateCatalog = Field(
        description="Catalog service for template entry resolution"
    )

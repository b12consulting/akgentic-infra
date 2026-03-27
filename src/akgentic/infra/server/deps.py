"""Dependency injection containers for tiered service assembly."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from akgentic.infra.protocols.auth import AuthStrategy
from akgentic.infra.protocols.placement import PlacementStrategy
from akgentic.team.manager import TeamManager
from akgentic.team.ports import EventStore, ServiceRegistry


class TierServices(BaseModel):
    """Base dependency container holding services common to all deployment tiers.

    This is a runtime DI container, NOT a serialization model — arbitrary_types_allowed
    is intentional (Golden Rule #1b exemption). Protocol-typed fields use SkipValidation
    because non-runtime_checkable Protocols cannot be used with isinstance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    placement: SkipValidation[PlacementStrategy] = Field(
        description="Strategy for placing teams on workers"
    )
    service_registry: SkipValidation[ServiceRegistry] = Field(
        description="Service discovery registry for worker instances"
    )
    auth: SkipValidation[AuthStrategy] = Field(
        description="Authentication strategy for incoming requests"
    )
    event_store: SkipValidation[EventStore] = Field(
        description="Persistence backend for team event sourcing"
    )


class CommunityServices(TierServices):
    """Community-tier service container with embedded TeamManager.

    Extends TierServices with an in-process TeamManager for single-process deployment.
    """

    team_manager: SkipValidation[TeamManager] = Field(
        description="Team lifecycle manager (embedded, in-process)"
    )

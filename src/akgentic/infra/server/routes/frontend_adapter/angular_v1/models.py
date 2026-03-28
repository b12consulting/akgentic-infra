"""V1 response models for the Angular V1 frontend adapter.

Self-contained Pydantic models representing V1 API response shapes.
These are independent from V2 models — no shared response models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class V1ActorAddress(BaseModel):
    """V1 actor address representation."""

    name: str = Field(description="Actor display name")
    role: str = Field(description="Actor role from AgentCard")


class V1ProcessParams(BaseModel):
    """V1 process creation parameters."""

    type: str = Field(description="Team type / catalog entry ID")
    agents: list[V1ActorAddress] = Field(
        default_factory=list,
        description="List of actors in the team",
    )


class V1ProcessContext(BaseModel):
    """V1 team representation — maps from V2 Process model."""

    id: str = Field(description="UUID as string")
    type: str = Field(description="Team name / catalog entry")
    status: str = Field(description="Team lifecycle status")
    created_at: str = Field(description="ISO datetime string")
    updated_at: str = Field(description="ISO datetime string")
    params: dict[str, str] = Field(
        default_factory=dict,
        description="Empty dict for V2 compat",
    )


class V1MessageEntry(BaseModel):
    """V1 message history entry — maps from persisted events."""

    id: str = Field(description="Message UUID as string")
    sender: str = Field(description="Sender name")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="ISO datetime string")
    type: str = Field(description="Message type: user, agent, system")


class V1LlmContextEntry(BaseModel):
    """V1 LLM context entry — maps from persisted events."""

    role: str = Field(description="Message role: user, agent, system")
    content: str = Field(description="Message content")
    timestamp: str = Field(description="ISO datetime string")


class V1StateEntry(BaseModel):
    """V1 state change entry — maps from StateChangedMessage events."""

    agent: str = Field(description="Agent name that changed state")
    state: dict[str, object] = Field(description="Serialized agent state")
    timestamp: str = Field(description="ISO datetime string")


class V1ProcessList(BaseModel):
    """V1 list of processes."""

    processes: list[V1ProcessContext] = Field(
        default_factory=list,
        description="List of V1 process contexts",
    )

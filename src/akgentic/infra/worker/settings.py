"""WorkerSettings -- typed configuration for the akgentic-infra worker."""

from __future__ import annotations

import warnings

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from akgentic.team.models import DeleteMode

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# The one variable the record half of a deletion is read from, by the server
# and the worker alike — deliberately without the worker's prefix.
_TEAM_DELETE_MODE_ENV = "AKGENTIC_TEAM_DELETE_MODE"


class WorkerSettings(BaseSettings):
    """Tier-agnostic worker configuration loaded from environment variables.

    Contains only settings common to all deployment tiers.
    All fields can be overridden via environment variables prefixed with ``AKGENTIC_WORKER_``,
    except ``team_delete_mode``, which reads the un-prefixed ``AKGENTIC_TEAM_DELETE_MODE``.

    **There is no workspaces-root knob here, deliberately.** The tree's location
    is the tool's own ``AKGENTIC_WORKSPACES_ROOT``, read wherever a card binds.
    A field on this model would bind ``AKGENTIC_WORKER_WORKSPACES_ROOT`` — a
    differently named variable nothing reads — and making the tool observe it
    would mean writing ``os.environ`` from settings, in a module that has no
    process entrypoint to do it in. Both deployment tiers already export the
    tool's variable on every service from one source. An undeclared
    ``AKGENTIC_WORKER_``-prefixed variable left over in a tier's chart is simply
    ignored, which is what keeps removing the field safe.

    **No document-reader field either, for the same reason.** The reader is
    configured by akgentic-tool's own ``AKGENTIC_DOCUMENT_READER_*`` variables,
    which ``DocumentReader`` reads itself; the worker declares none of them.
    The README lists them.

    **``team_delete_mode`` binds the un-prefixed ``AKGENTIC_TEAM_DELETE_MODE``,
    on purpose.** It is the record half of a team deletion — ``hard`` purges the
    team's ``Process``, events and agent states, ``soft`` retires the record
    with status ``DELETED`` — and the event store the worker writes through must
    agree with the server's on it, so both processes read one variable rather
    than two that can drift. ``AKGENTIC_WORKER_TEAM_DELETE_MODE`` is not read.

    **The other half — the workspace trees — is ``workspace_delete_mode``
    (``AKGENTIC_WORKSPACE_DELETE_MODE``) on ``ServerSettings``, and there is no
    such field here,** because only the server removes trees. The two halves are
    independent by design: soft record + hard workspace means the record can be
    recovered but its files are gone; hard record + soft workspace leaves the
    trees on disk with no record pointing at them, and nothing reclaims them
    later.
    """

    model_config = SettingsConfigDict(env_prefix="AKGENTIC_WORKER_")

    host: str = Field(
        default="0.0.0.0",
        description="Bind address for the worker HTTP server",
    )
    port: int = Field(
        default=8001,
        description="Port number for the worker HTTP server",
    )
    log_level: str = Field(
        default="INFO",
        description="Application log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    shutdown_drain_timeout: int = Field(
        default=30,
        ge=0,
        description="Max seconds for stop_all() during graceful shutdown",
    )
    shutdown_pre_drain_delay: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds to wait after marking draining before teardown "
            "(0 for standalone, 5-10 for LB deployments)"
        ),
    )
    worker_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels for placement strategy matching (e.g. gpu=true, region=eu)",
    )
    # A plain string alias, not ``AliasChoices(..., "team_delete_mode")``:
    # pydantic-settings looks every alias choice up in the environment, so a
    # field-name choice would also read a bare ``TEAM_DELETE_MODE`` variable.
    # Construction by field name is kept by ``_accept_team_delete_mode_by_name``
    # below, not by ``validate_by_name``: with that flag on, pydantic-settings
    # also reads ``env_prefix + field_name`` — the very
    # ``AKGENTIC_WORKER_TEAM_DELETE_MODE`` this field must not read.
    team_delete_mode: DeleteMode = Field(
        default=DeleteMode.HARD,
        validation_alias=_TEAM_DELETE_MODE_ENV,
        description=(
            "What deleting a team does to its record (Process, events, agent "
            "states): 'hard' purges it, 'soft' retires it with status DELETED. "
            "Read from the un-prefixed AKGENTIC_TEAM_DELETE_MODE, the same "
            "variable the server reads. The workspace trees are the other half, "
            "controlled by the server's workspace_delete_mode "
            "(AKGENTIC_WORKSPACE_DELETE_MODE) — the worker has no such field "
            "because only the server removes trees. The two are independent by "
            "design: soft record + hard workspace means the record can be "
            "recovered but its files are gone; hard record + soft workspace "
            "leaves the trees on disk with no record pointing at them, and "
            "nothing reclaims them later."
        ),
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """Normalize to uppercase and fall back to INFO for invalid values."""
        upper = str(v).upper()
        if upper not in _VALID_LOG_LEVELS:
            warnings.warn(
                f"Invalid AKGENTIC_WORKER_LOG_LEVEL '{v}', falling back to INFO",
                UserWarning,
                stacklevel=1,
            )
            return "INFO"
        return upper

    @model_validator(mode="before")
    @classmethod
    def _accept_team_delete_mode_by_name(cls, data: object) -> object:
        """Let ``WorkerSettings(team_delete_mode=...)`` construct by field name.

        An init keyword wins over the environment, as it does for every other
        field: the value is moved onto the alias key, replacing whatever the
        environment put there.
        """
        if isinstance(data, dict) and "team_delete_mode" in data:
            data = dict(data)
            data[_TEAM_DELETE_MODE_ENV] = data.pop("team_delete_mode")
        return data

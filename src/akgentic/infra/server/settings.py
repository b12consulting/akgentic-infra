"""ServerSettings — typed configuration for the akgentic-infra server."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, TypeGuard

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from akgentic.catalog import parse_prefixes

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _is_string_sequence(v: object) -> TypeGuard[Sequence[str]]:
    """Report whether ``v`` is a non-string sequence whose items are all strings."""
    return isinstance(v, Sequence) and all(isinstance(item, str) for item in v)


class ServerSettings(BaseSettings):
    """Tier-agnostic server configuration loaded from environment variables.

    Contains only settings common to all deployment tiers.
    All fields can be overridden via environment variables prefixed with ``AKGENTIC_``.
    """

    model_config = SettingsConfigDict(env_prefix="AKGENTIC_")

    host: str = Field(
        default="0.0.0.0",
        description="Bind address for the HTTP server",
    )
    port: int = Field(
        default=8000,
        description="Port number for the HTTP server",
    )
    log_level: str = Field(
        default="INFO",
        description="Application log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """Normalize to uppercase and fall back to INFO for invalid values."""
        upper = str(v).upper()
        if upper not in _VALID_LOG_LEVELS:
            warnings.warn(
                f"Invalid AKGENTIC_LOG_LEVEL '{v}', falling back to INFO",
                UserWarning,
                stacklevel=1,
            )
            return "INFO"
        return upper

    frontend_adapter: str | None = Field(
        default=None,
        description="FQDN for frontend adapter plugin class",
    )
    # Community-tier permissive default. Department/enterprise tiers must
    # override with explicit origins in their environment configuration.
    # Set to [] to disable the CORS middleware entirely — useful when an
    # external gateway (e.g. Azure App Service) manages CORS.
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins for the HTTP server (empty list disables middleware)",
    )
    shutdown_drain_timeout: int = Field(
        default=30,
        ge=0,
        description="Max seconds for stop_all() to complete during graceful shutdown",
    )
    shutdown_pre_drain_delay: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds to wait after marking draining before starting teardown "
            "(0 for standalone, 5-10 for LB deployments)"
        ),
    )
    ws_reader_pool_size: int = Field(
        default=256,
        ge=1,
        description=(
            "Dedicated thread pool for WebSocket event-stream reader polling. "
            "Each active WS connection holds one thread per read_next(0.5) tick. "
            "Size for concurrent WS budget plus headroom for burst open/close "
            "cycles. Isolated from the default executor to prevent "
            "cross-subsystem starvation."
        ),
    )
    # ``NoDecode`` is load-bearing, not decoration: pydantic-settings
    # json-decodes complex fields in its env source *before* any mode="before"
    # validator runs, so a bare ``list[str]`` would make the comma form raise
    # SettingsError at construction and never reach the validator below.
    # ``cors_origins`` above is not a precedent — it is JSON-only.
    catalog_model_type_prefixes: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description=(
            "Extra module prefixes an Entry.model_type may name, on top of the "
            "always-allowed 'akgentic.'. Comma-separated (acme.,contoso.models.) "
            'or a JSON list (["acme."]). Must carry the same value in every '
            "process of a deployment (server, worker, CLI) — the policy is "
            "process-wide, and a server/worker mismatch makes an entry writable "
            "in one process and unresolvable in another."
        ),
    )

    @field_validator("catalog_model_type_prefixes", mode="before")
    @classmethod
    def _normalize_catalog_model_type_prefixes(cls, v: object) -> list[str]:
        """Normalize and validate through the catalog's own parser.

        Infra deliberately owns no splitting, stripping, trailing-dot, or
        identifier-shape logic — one parser, so the settings field and the
        catalog's own lazy environment read can never disagree. A malformed
        prefix raises ``ValueError`` here, which pydantic surfaces as a
        ``ValidationError`` at settings construction rather than at the first
        catalog write.

        ``v`` is typed ``object`` because a ``mode="before"`` validator is
        handed whatever the caller passed. The shape guard is not parsing: it
        exists because ``parse_prefixes`` assumes strings, so a programmatic
        caller passing ``5`` or ``[1, 2]`` would otherwise surface a bare
        ``TypeError``/``AttributeError`` from inside the catalog — pydantic
        converts neither, so that value would escape settings construction as
        something other than a ``ValidationError``.
        """
        if v is None or isinstance(v, str) or _is_string_sequence(v):
            return list(parse_prefixes(v))
        raise ValueError(
            f"invalid model_type prefix: {v!r} is not a string or a sequence of strings"
        )


class CommunitySettings(ServerSettings):
    """Community-tier settings extending base ServerSettings.

    Adds filesystem-backed workspace and catalog configuration
    specific to the community (single-process) deployment tier.
    """

    workspaces_root: Path = Field(
        default=Path("workspaces"),
        description="Root directory for team workspace storage",
    )
    event_store_path: Path = Field(
        default=Path("data/event_store"),
        description="Root directory for event store persistence",
    )
    catalog_path: Path = Field(
        default=Path("data/catalog"),
        description="Catalog directory for team/agent/tool/template definitions",
    )
    channel_registry_path: Path | None = Field(
        default=None,
        description=(
            "Path to channel registry YAML file; "
            "when unset, the channel registry is disabled (no-op lookups)"
        ),
    )
    auth_strategy: str = Field(
        default="noauth",
        description=(
            "Auth-strategy selector for community wiring. Defaults to 'noauth' "
            "(the anonymous community strategy — byte-identical and "
            "dependency-closure-clean: no entry-point lookup, no auth-library "
            "import). Any other value is resolved at wire time via the "
            "'akgentic.infra.auth.strategies' entry-point group and requires the "
            "matching extra to be installed; an unknown name fails loud."
        ),
    )

"""ServerSettings — typed configuration for the akgentic-infra server."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Typed server configuration loaded from environment variables.

    All fields can be overridden via environment variables prefixed with ``AKGENTIC_``.
    For example, ``AKGENTIC_HOST=127.0.0.1`` overrides the ``host`` field.
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
    frontend_adapter: str | None = Field(
        default=None,
        description="FQDN for frontend adapter plugin class",
    )
    workspaces_root: Path = Field(
        default=Path("workspaces"),
        description="Root directory for team workspace storage",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins for the HTTP server",
    )
    event_store: str = Field(
        default="yaml",
        description="Event store backend: 'yaml' or 'memory'",
    )
    catalog_backend: str = Field(
        default="yaml",
        description="Catalog backend: 'yaml'",
    )
    catalog_path: Path | None = Field(
        default=None,
        description="Catalog directory; defaults to workspaces_root / 'catalog'",
    )

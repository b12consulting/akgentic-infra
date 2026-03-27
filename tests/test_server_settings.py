"""Tests for ServerSettings model."""

from __future__ import annotations

import os
from pathlib import Path

from akgentic.infra.server.settings import ServerSettings


class TestServerSettingsDefaults:
    """AC4: ServerSettings provides typed configuration with defaults."""

    def test_default_host(self) -> None:
        """Default host is 0.0.0.0."""
        settings = ServerSettings()
        assert settings.host == "0.0.0.0"

    def test_default_port(self) -> None:
        """Default port is 8000."""
        settings = ServerSettings()
        assert settings.port == 8000

    def test_default_frontend_adapter(self) -> None:
        """Default frontend_adapter is None."""
        settings = ServerSettings()
        assert settings.frontend_adapter is None

    def test_default_workspaces_root(self) -> None:
        """Default workspaces_root is Path('workspaces')."""
        settings = ServerSettings()
        assert settings.workspaces_root == Path("workspaces")


class TestServerSettingsEnvOverride:
    """AC4: ServerSettings loads from AKGENTIC_ prefixed env vars."""

    def test_host_from_env(self) -> None:
        """AKGENTIC_HOST overrides host field."""
        os.environ["AKGENTIC_HOST"] = "127.0.0.1"
        try:
            settings = ServerSettings()
            assert settings.host == "127.0.0.1"
        finally:
            del os.environ["AKGENTIC_HOST"]

    def test_port_from_env(self) -> None:
        """AKGENTIC_PORT overrides port field."""
        os.environ["AKGENTIC_PORT"] = "9000"
        try:
            settings = ServerSettings()
            assert settings.port == 9000
        finally:
            del os.environ["AKGENTIC_PORT"]

    def test_frontend_adapter_from_env(self) -> None:
        """AKGENTIC_FRONTEND_ADAPTER overrides frontend_adapter field."""
        os.environ["AKGENTIC_FRONTEND_ADAPTER"] = "my.adapter.Class"
        try:
            settings = ServerSettings()
            assert settings.frontend_adapter == "my.adapter.Class"
        finally:
            del os.environ["AKGENTIC_FRONTEND_ADAPTER"]

    def test_workspaces_root_from_env(self) -> None:
        """AKGENTIC_WORKSPACES_ROOT overrides workspaces_root field."""
        os.environ["AKGENTIC_WORKSPACES_ROOT"] = "/tmp/ws"
        try:
            settings = ServerSettings()
            assert settings.workspaces_root == Path("/tmp/ws")
        finally:
            del os.environ["AKGENTIC_WORKSPACES_ROOT"]


class TestServerSettingsModel:
    """AC4: ServerSettings extends BaseSettings."""

    def test_is_base_settings_subclass(self) -> None:
        """ServerSettings is a pydantic_settings BaseSettings subclass."""
        from pydantic_settings import BaseSettings

        assert issubclass(ServerSettings, BaseSettings)

    def test_field_descriptions_present(self) -> None:
        """All fields have descriptions."""
        for name, field_info in ServerSettings.model_fields.items():
            assert field_info.description is not None, f"Field {name} missing description"

"""Tests for ServerSettings and CommunitySettings models."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from akgentic.infra.server.settings import CommunitySettings, ServerSettings


class TestServerSettingsDefaults:
    """ServerSettings provides tier-agnostic configuration with defaults."""

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

    def test_no_workspaces_root_on_base(self) -> None:
        """ServerSettings does not have workspaces_root (community-specific)."""
        assert "workspaces_root" not in ServerSettings.model_fields


class TestServerSettingsEnvOverride:
    """ServerSettings loads from AKGENTIC_ prefixed env vars."""

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


class TestServerSettingsLogLevel:
    """ServerSettings log_level field — validation and env override."""

    def test_default_log_level(self) -> None:
        """Default log_level is INFO."""
        settings = ServerSettings()
        assert settings.log_level == "INFO"

    def test_log_level_from_env(self) -> None:
        """AKGENTIC_LOG_LEVEL overrides log_level field."""
        os.environ["AKGENTIC_LOG_LEVEL"] = "DEBUG"
        try:
            settings = ServerSettings()
            assert settings.log_level == "DEBUG"
        finally:
            del os.environ["AKGENTIC_LOG_LEVEL"]

    def test_log_level_case_insensitive(self) -> None:
        """AKGENTIC_LOG_LEVEL is normalized to uppercase."""
        os.environ["AKGENTIC_LOG_LEVEL"] = "debug"
        try:
            settings = ServerSettings()
            assert settings.log_level == "DEBUG"
        finally:
            del os.environ["AKGENTIC_LOG_LEVEL"]

    def test_invalid_log_level_falls_back(self) -> None:
        """Invalid AKGENTIC_LOG_LEVEL falls back to INFO with a warning."""

        os.environ["AKGENTIC_LOG_LEVEL"] = "TRACE"
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                settings = ServerSettings()
                assert settings.log_level == "INFO"
                assert len(w) == 1
                assert "TRACE" in str(w[0].message)
        finally:
            del os.environ["AKGENTIC_LOG_LEVEL"]


class TestServerSettingsModel:
    """ServerSettings extends BaseSettings."""

    def test_is_base_settings_subclass(self) -> None:
        """ServerSettings is a pydantic_settings BaseSettings subclass."""
        from pydantic_settings import BaseSettings

        assert issubclass(ServerSettings, BaseSettings)

    def test_field_descriptions_present(self) -> None:
        """All fields have descriptions."""
        for name, field_info in ServerSettings.model_fields.items():
            assert field_info.description is not None, f"Field {name} missing description"


class TestCommunitySettingsDefaults:
    """CommunitySettings extends ServerSettings with community-specific fields."""

    def test_inherits_server_settings(self) -> None:
        """CommunitySettings is a subclass of ServerSettings."""
        assert issubclass(CommunitySettings, ServerSettings)

    def test_default_workspaces_root(self) -> None:
        """Default workspaces_root is Path('workspaces')."""
        settings = CommunitySettings()
        assert settings.workspaces_root == Path("workspaces")

    def test_default_catalog_path(self) -> None:
        """Default catalog_path is None."""
        settings = CommunitySettings()
        assert settings.catalog_path is None

    def test_inherits_base_fields(self) -> None:
        """CommunitySettings inherits host, port, cors_origins from ServerSettings."""
        settings = CommunitySettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.cors_origins == ["*"]

    def test_workspaces_root_from_env(self) -> None:
        """AKGENTIC_WORKSPACES_ROOT overrides workspaces_root field."""
        os.environ["AKGENTIC_WORKSPACES_ROOT"] = "/tmp/ws"
        try:
            settings = CommunitySettings()
            assert settings.workspaces_root == Path("/tmp/ws")
        finally:
            del os.environ["AKGENTIC_WORKSPACES_ROOT"]

    def test_default_channel_registry_path(self) -> None:
        """Default channel_registry_path is None."""
        settings = CommunitySettings()
        assert settings.channel_registry_path is None

    def test_channel_registry_path_from_env(self) -> None:
        """AKGENTIC_CHANNEL_REGISTRY_PATH overrides channel_registry_path field."""
        os.environ["AKGENTIC_CHANNEL_REGISTRY_PATH"] = "/tmp/reg.yaml"
        try:
            settings = CommunitySettings()
            assert settings.channel_registry_path == Path("/tmp/reg.yaml")
        finally:
            del os.environ["AKGENTIC_CHANNEL_REGISTRY_PATH"]

    def test_field_descriptions_present(self) -> None:
        """All CommunitySettings fields have descriptions."""
        for name, field_info in CommunitySettings.model_fields.items():
            assert field_info.description is not None, f"Field {name} missing description"

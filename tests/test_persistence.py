"""Tests for the community-tier persistence switch (YAML or PostgreSQL).

Covers the two settings, the store builders in ``akgentic.infra.persistence``
and the ``event_store`` / ``catalog_repository`` overrides on
``wire_community``. Nothing here opens a database connection: both Postgres
stores only record the DSN at construction, so these tests need the
``[postgres]`` extra installed but no running server.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from akgentic.catalog import YamlEntryRepository
from akgentic.team.models import DeleteMode
from akgentic.team.repositories.yaml import YamlEventStore
from pydantic import ValidationError

from akgentic.infra import persistence
from akgentic.infra.persistence import (
    PersistenceBackendUnavailableError,
    build_catalog_repository,
    build_event_store,
)
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend
from akgentic.infra.wiring import wire_community

DSN = "postgresql://akgentic:s3cret-pw@localhost:5432/akgentic"


def _yaml_settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
        **overrides,
    )


def _postgres_settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    return _yaml_settings(
        tmp_path,
        persistence_backend=PersistenceBackend.POSTGRES,
        postgres_conn_string=DSN,
        **overrides,
    )


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an operator's shell from leaking into the settings under test."""
    monkeypatch.delenv("AKGENTIC_PERSISTENCE_BACKEND", raising=False)
    monkeypatch.delenv("AKGENTIC_POSTGRES_CONN_STRING", raising=False)


@pytest.mark.usefixtures("clean_env")
class TestPersistenceSettings:
    """``persistence_backend`` and ``postgres_conn_string`` on CommunitySettings."""

    def test_default_backend_is_yaml_without_a_dsn(self, tmp_path: Path) -> None:
        settings = _yaml_settings(tmp_path)
        assert settings.persistence_backend is PersistenceBackend.YAML
        assert settings.postgres_conn_string is None

    def test_backend_and_dsn_read_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_PERSISTENCE_BACKEND", "postgres")
        monkeypatch.setenv("AKGENTIC_POSTGRES_CONN_STRING", DSN)
        settings = _yaml_settings(tmp_path)
        assert settings.persistence_backend is PersistenceBackend.POSTGRES
        assert settings.postgres_conn_string is not None
        assert settings.postgres_conn_string.get_secret_value() == DSN

    def test_postgres_without_a_dsn_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="AKGENTIC_POSTGRES_CONN_STRING"):
            _yaml_settings(tmp_path, persistence_backend=PersistenceBackend.POSTGRES)

    def test_postgres_with_an_empty_dsn_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="must start with 'postgresql://'"):
            _yaml_settings(
                tmp_path,
                persistence_backend=PersistenceBackend.POSTGRES,
                postgres_conn_string="",
            )

    def test_unknown_backend_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            _yaml_settings(tmp_path, persistence_backend="mongodb")

    @pytest.mark.parametrize(
        "dsn",
        [
            # Nagra dispatches on the exact ``postgresql://`` prefix and echoes
            # any other DSN, password included, in its error.
            "postgres://akgentic:s3cret-pw@localhost/akgentic",
            "mysql://akgentic:s3cret-pw@localhost/akgentic",
        ],
    )
    def test_non_postgresql_scheme_is_rejected_without_echoing_the_dsn(
        self, tmp_path: Path, dsn: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _yaml_settings(tmp_path, postgres_conn_string=dsn)
        assert "s3cret-pw" not in str(exc_info.value)

    def test_dsn_is_a_secret_in_repr(self, tmp_path: Path) -> None:
        settings = _postgres_settings(tmp_path)
        assert "s3cret-pw" not in repr(settings)
        assert "s3cret-pw" not in str(settings.model_dump())

    def test_dsn_is_allowed_alongside_the_yaml_backend(self, tmp_path: Path) -> None:
        """A DSN left in the environment must not stop a YAML deployment from booting."""
        settings = _yaml_settings(tmp_path, postgres_conn_string=DSN)
        assert settings.persistence_backend is PersistenceBackend.YAML


@pytest.mark.usefixtures("clean_env")
class TestBuildersYaml:
    """The default branch builds exactly what wire_community built before."""

    def test_event_store_is_yaml_at_the_configured_path(self, tmp_path: Path) -> None:
        store = build_event_store(_yaml_settings(tmp_path))
        assert isinstance(store, YamlEventStore)
        assert store._data_dir == tmp_path / "event_store"

    def test_event_store_carries_the_team_delete_mode(self, tmp_path: Path) -> None:
        store = build_event_store(_yaml_settings(tmp_path, team_delete_mode=DeleteMode.SOFT))
        assert store._delete_mode is DeleteMode.SOFT  # type: ignore[attr-defined]

    def test_catalog_repository_is_yaml_at_the_configured_path(self, tmp_path: Path) -> None:
        repo = build_catalog_repository(_yaml_settings(tmp_path))
        assert isinstance(repo, YamlEntryRepository)


@pytest.mark.usefixtures("clean_env")
class TestBuildersPostgres:
    """The postgres branch builds the Nagra stores without touching a database."""

    @pytest.fixture(autouse=True)
    def _needs_postgres_extra(self) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("psycopg_pool")

    def test_event_store_is_nagra_with_the_dsn(self, tmp_path: Path) -> None:
        from akgentic.team.repositories.postgres import NagraEventStore

        store = build_event_store(_postgres_settings(tmp_path))
        assert isinstance(store, NagraEventStore)
        assert store._conn_string == DSN

    def test_event_store_carries_the_team_delete_mode(self, tmp_path: Path) -> None:
        store = build_event_store(_postgres_settings(tmp_path, team_delete_mode=DeleteMode.SOFT))
        assert store._delete_mode is DeleteMode.SOFT  # type: ignore[attr-defined]

    def test_catalog_repository_is_postgres_with_the_dsn(self, tmp_path: Path) -> None:
        from akgentic.catalog.repositories.postgres import PostgresEntryRepository

        repo = build_catalog_repository(_postgres_settings(tmp_path))
        assert isinstance(repo, PostgresEntryRepository)
        assert repo._conn_string == DSN

    def test_postgres_backend_writes_nothing_under_the_yaml_paths(self, tmp_path: Path) -> None:
        settings = _postgres_settings(tmp_path)
        build_event_store(settings)
        build_catalog_repository(settings)
        assert not (tmp_path / "event_store").exists()
        assert not (tmp_path / "catalog").exists()


_ALL_POSTGRES_MODULES = ("nagra", "psycopg", "psycopg_pool")


def _hide_modules(monkeypatch: pytest.MonkeyPatch, hidden: tuple[str, ...]) -> None:
    """Make ``persistence``'s dependency probe see exactly ``hidden`` as missing.

    The probe answers for every Postgres module itself — missing if hidden,
    present otherwise — and never reaches the real import, so the result does not
    depend on what the running environment has installed: CI installs ``.[dev]``
    without the extra, a developer venv usually has it.
    """

    def fake_import_module(name: str) -> object:
        if name in hidden:
            raise ImportError(f"No module named {name!r}")
        if name in _ALL_POSTGRES_MODULES:
            return object()
        raise AssertionError(f"unexpected probe of {name!r}")

    monkeypatch.setattr(persistence.importlib, "import_module", fake_import_module)


@pytest.mark.usefixtures("clean_env")
class TestMissingPostgresExtra:
    """Selecting postgres without the extra fails at wire time, naming the fix."""

    @pytest.fixture()
    def without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide_modules(monkeypatch, _ALL_POSTGRES_MODULES)

    @pytest.mark.usefixtures("without_extra")
    @pytest.mark.parametrize("builder", [build_event_store, build_catalog_repository])
    def test_builder_raises_with_install_hint(self, tmp_path: Path, builder: Any) -> None:
        with pytest.raises(PersistenceBackendUnavailableError) as exc_info:
            builder(_postgres_settings(tmp_path))
        message = str(exc_info.value)
        assert "nagra, psycopg, psycopg_pool are not installed" in message
        assert "akgentic-infra[postgres]" in message
        assert "s3cret-pw" not in message

    def test_names_only_the_module_that_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-installed extra (Nagra present, no pool) is caught at wire time too."""
        _hide_modules(monkeypatch, ("psycopg_pool",))
        with pytest.raises(PersistenceBackendUnavailableError) as exc_info:
            build_event_store(_postgres_settings(tmp_path))
        assert "but psycopg_pool is not installed" in str(exc_info.value)

    @pytest.mark.usefixtures("without_extra")
    def test_wire_community_raises_before_starting_anything(self, tmp_path: Path) -> None:
        with pytest.raises(PersistenceBackendUnavailableError):
            wire_community(_postgres_settings(tmp_path))

    @pytest.mark.usefixtures("without_extra")
    def test_yaml_backend_does_not_need_the_extra(self, tmp_path: Path) -> None:
        assert isinstance(build_event_store(_yaml_settings(tmp_path)), YamlEventStore)


class _RecordingEventStore(YamlEventStore):
    """A real store with a distinct type, so the test can tell it was the one used."""


@pytest.mark.usefixtures("clean_env")
class TestWireCommunityOverrides:
    """Injected stores win over the setting, one store at a time."""

    @pytest.fixture()
    def injected(self, tmp_path: Path) -> Generator[tuple[CommunityServices, Any, Any], None, None]:
        store = _RecordingEventStore(data_dir=tmp_path / "injected_events")
        repo = YamlEntryRepository(root=tmp_path / "injected_catalog")
        services = wire_community(
            _yaml_settings(tmp_path), event_store=store, catalog_repository=repo
        )
        yield services, store, repo
        services.team_manager._actor_system.shutdown(timeout=5)

    def test_injected_event_store_reaches_container_and_team_manager(
        self, injected: tuple[CommunityServices, Any, Any]
    ) -> None:
        services, store, _ = injected
        assert services.event_store is store
        assert services.team_manager._event_store is store

    def test_injected_catalog_repository_backs_the_catalog(
        self, injected: tuple[CommunityServices, Any, Any]
    ) -> None:
        services, _, repo = injected
        assert services.catalog._repository is repo

    def test_injected_stores_win_over_a_postgres_setting(self, tmp_path: Path) -> None:
        """With both stores injected, postgres is never built — no extra, no DSN use."""
        store = _RecordingEventStore(data_dir=tmp_path / "injected_events")
        repo = YamlEntryRepository(root=tmp_path / "injected_catalog")
        services = wire_community(
            _postgres_settings(tmp_path), event_store=store, catalog_repository=repo
        )
        try:
            assert services.event_store is store
            assert services.catalog._repository is repo
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_without_overrides_the_setting_decides(self, tmp_path: Path) -> None:
        services = wire_community(_yaml_settings(tmp_path))
        try:
            assert type(services.event_store) is YamlEventStore
            assert isinstance(services.catalog._repository, YamlEntryRepository)
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

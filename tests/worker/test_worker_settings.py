"""Tests for WorkerSettings configuration model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from akgentic.infra.worker.settings import WorkerSettings

WORKSPACES_ROOT_ENV = "AKGENTIC_WORKER_WORKSPACES_ROOT"
"""The variable the retired ``workspaces_root`` field used to bind.

Named as a literal because no field declares it any more — which is the point.
``WorkerSettings`` carries ``env_prefix="AKGENTIC_WORKER_"``, while the tool
resolves the workspace tree from ``AKGENTIC_WORKSPACES_ROOT``, so this was never
the knob the tool reads: an operator who set it got silence. Both deployment
tiers still export it (enterprise's Helm worker template; department's compose),
so it outlives the field it bound.
"""


class TestDefaultValues:
    """WorkerSettings must have sensible defaults matching ADR-017 Decision 2."""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AKGENTIC_WORKER_HOST", raising=False)
        monkeypatch.delenv("AKGENTIC_WORKER_PORT", raising=False)
        monkeypatch.delenv("AKGENTIC_WORKER_LOG_LEVEL", raising=False)
        monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
        monkeypatch.delenv("AKGENTIC_WORKER_SHUTDOWN_DRAIN_TIMEOUT", raising=False)
        monkeypatch.delenv("AKGENTIC_WORKER_SHUTDOWN_PRE_DRAIN_DELAY", raising=False)
        monkeypatch.delenv("AKGENTIC_WORKER_WORKER_LABELS", raising=False)
        settings = WorkerSettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8001
        assert settings.log_level == "INFO"
        assert settings.shutdown_drain_timeout == 30
        assert settings.shutdown_pre_drain_delay == 0
        assert settings.worker_labels == {}


class TestEnvVarOverride:
    """WorkerSettings must load overrides from AKGENTIC_WORKER_ prefixed env vars."""

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AKGENTIC_WORKER_HOST", "127.0.0.1")
        monkeypatch.setenv("AKGENTIC_WORKER_PORT", "9999")
        monkeypatch.setenv("AKGENTIC_WORKER_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("AKGENTIC_WORKER_SHUTDOWN_DRAIN_TIMEOUT", "60")
        monkeypatch.setenv("AKGENTIC_WORKER_SHUTDOWN_PRE_DRAIN_DELAY", "5")
        monkeypatch.setenv(
            "AKGENTIC_WORKER_WORKER_LABELS", '{"gpu": "true", "region": "eu"}'
        )

        settings = WorkerSettings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 9999
        assert settings.log_level == "DEBUG"
        assert settings.shutdown_drain_timeout == 60
        assert settings.shutdown_pre_drain_delay == 5
        assert settings.worker_labels == {"gpu": "true", "region": "eu"}


class TestTheRetiredWorkspacesRootVariableIsHarmless:
    """Story 71.2 AC #6: removing the field must not crash a worker that still sets it.

    ``akgentic-infra-enterprise``'s ``EnterpriseWorkerSettings`` **inherits**
    this model and its Helm worker template sets ``AKGENTIC_WORKER_WORKSPACES_ROOT``
    on every worker pod, so after this story an enterprise worker starts in an
    environment that sets a prefixed variable no field binds. Removing the
    variable from either tier's chart is another repository's work (Golden
    Rule 4), which is exactly why the safety has to be asserted here.

    pydantic-settings' environment source collects values for **declared** fields
    only, so an undeclared prefixed variable should be ignored rather than
    tripping the model's ``extra`` behaviour. *Should* is not the guarantee to
    ship on a path that would crash two deployments at startup.
    """

    def test_constructing_with_the_variable_still_set_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The retired variable is the only one this spec sets. The others are
        # cleared for the same reason ``TestDefaultValues`` clears them: the
        # model reads the real environment, so an ambient ``AKGENTIC_WORKER_*``
        # would redden the one guard standing between two deployments and a
        # startup crash, for a reason that has nothing to do with AC #6.
        monkeypatch.delenv("AKGENTIC_WORKER_PORT", raising=False)
        monkeypatch.setenv(WORKSPACES_ROOT_ENV, "/data/workspaces")

        settings = WorkerSettings()

        assert settings.port == 8001
        assert not hasattr(settings, "workspaces_root")


class TestLogLevelNormalization:
    """Log level validator must normalize case and reject invalid values."""

    def test_log_level_normalizes_to_uppercase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_WORKER_LOG_LEVEL", "debug")
        settings = WorkerSettings()
        assert settings.log_level == "DEBUG"

    def test_log_level_invalid_falls_back_to_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_WORKER_LOG_LEVEL", "bogus")
        with pytest.warns(UserWarning, match="Invalid AKGENTIC_WORKER_LOG_LEVEL"):
            settings = WorkerSettings()
        assert settings.log_level == "INFO"


class TestConstraintValidation:
    """Constrained fields must reject invalid values."""

    def test_shutdown_drain_timeout_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="shutdown_drain_timeout"):
            WorkerSettings(shutdown_drain_timeout=-1)

    def test_shutdown_pre_drain_delay_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="shutdown_pre_drain_delay"):
            WorkerSettings(shutdown_pre_drain_delay=-1)


class TestModelStructure:
    """WorkerSettings model metadata and structure."""

    def test_is_base_settings_subclass(self) -> None:
        from pydantic_settings import BaseSettings

        assert issubclass(WorkerSettings, BaseSettings)

    def test_field_descriptions_present(self) -> None:
        for name, field_info in WorkerSettings.model_fields.items():
            assert field_info.description is not None, f"Field {name} missing description"

    def test_has_only_tier_agnostic_fields(self) -> None:
        """The exact field set — kept exact, because it is AC #5's guard.

        ``workspaces_root`` is absent by decision, not by oversight: it bound
        ``AKGENTIC_WORKER_WORKSPACES_ROOT`` while the tool reads
        ``AKGENTIC_WORKSPACES_ROOT``, so it was a differently-named knob nothing
        could ever have read, with exactly one reference in ``src/`` — its own
        declaration. Loosening this to a subset check would let it come back.
        """
        fields = set(WorkerSettings.model_fields.keys())
        expected = {
            "host",
            "port",
            "log_level",
            "shutdown_drain_timeout",
            "shutdown_pre_drain_delay",
            "worker_labels",
        }
        assert fields == expected, (
            f"WorkerSettings fields mismatch: got {fields}, expected {expected}"
        )

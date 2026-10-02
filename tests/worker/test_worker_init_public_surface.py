"""Story 28.1 / 39.1 / 80.2: ``akgentic.infra.worker`` public surface contract.

Post-Epic 28 the surviving exports were four symbols; Story 39.1 added the shared
``memory_diagnostics_router``, and Story 80.2 adds ``TeamDescriptionGenerator``,
making the contract six. The dead worker shell (``create_worker_app``,
``WorkerLifecycle``, the ``services/`` package) is gone — this test guards
against regression if a contributor reintroduces it.
"""

from __future__ import annotations

import importlib

import pytest

import akgentic.infra.worker as worker_pkg


class TestWorkerPublicSurface:
    """The six-symbol contract on ``akgentic.infra.worker``."""

    def test_canonical_six_symbol_import_succeeds(self) -> None:
        """Settings, services, the generator, and the three shared routers import."""
        from akgentic.infra.worker import (  # noqa: PLC0415
            TeamDescriptionGenerator,
            WorkerServices,
            WorkerSettings,
            memory_diagnostics_router,
            readiness_router,
            teams_router,
        )

        assert WorkerSettings is not None
        assert WorkerServices is not None
        assert TeamDescriptionGenerator is not None
        assert teams_router is not None
        assert readiness_router is not None
        assert memory_diagnostics_router is not None

    def test_all_equals_sorted_six_symbol_list(self) -> None:
        """``__all__`` is exactly the six symbols (sorted comparison)."""
        assert sorted(worker_pkg.__all__) == [
            "TeamDescriptionGenerator",
            "WorkerServices",
            "WorkerSettings",
            "memory_diagnostics_router",
            "readiness_router",
            "teams_router",
        ]

    def test_describing_team_handle_is_not_re_exported(self) -> None:
        """The wrapper is importable from its module but is not part of the contract."""
        from akgentic.infra.worker.description import DescribingTeamHandle  # noqa: PLC0415

        assert DescribingTeamHandle is not None
        assert "DescribingTeamHandle" not in worker_pkg.__all__

    def test_worker_lifecycle_is_not_an_attribute(self) -> None:
        """``WorkerLifecycle`` is gone — no attribute on ``akgentic.infra.worker``."""
        assert not hasattr(worker_pkg, "WorkerLifecycle")

    def test_create_worker_app_is_not_an_attribute(self) -> None:
        """``create_worker_app`` is gone — no attribute on ``akgentic.infra.worker``."""
        assert not hasattr(worker_pkg, "create_worker_app")

    def test_app_submodule_does_not_exist(self) -> None:
        """``akgentic.infra.worker.app`` is gone — the file was deleted."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("akgentic.infra.worker.app")

    def test_services_subpackage_does_not_exist(self) -> None:
        """``akgentic.infra.worker.services`` is gone — the package was deleted."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("akgentic.infra.worker.services")

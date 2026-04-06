"""Tests for LocalWorkerHandle adapter."""

from __future__ import annotations

import inspect
import uuid
from unittest.mock import MagicMock

from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.adapters.community.local_worker_handle import LocalWorkerHandle
from akgentic.infra.protocols.worker_handle import WorkerHandle


def _make_adapter() -> tuple[LocalWorkerHandle, MagicMock, MagicMock]:
    """Create a LocalWorkerHandle with mock dependencies, return adapter, team_manager, actor_system."""
    team_manager = MagicMock()
    service_registry = MagicMock()
    actor_system = MagicMock()
    return (
        LocalWorkerHandle(team_manager, service_registry, actor_system),
        team_manager,
        actor_system,
    )


class TestLocalWorkerHandleProtocolCompliance:
    """AC4: LocalWorkerHandle implements WorkerHandle protocol."""

    def test_satisfies_worker_handle_protocol(self) -> None:
        """LocalWorkerHandle structurally satisfies WorkerHandle."""
        adapter, _, _ = _make_adapter()
        assert isinstance(adapter, WorkerHandle)

    def test_has_all_protocol_methods(self) -> None:
        """LocalWorkerHandle exposes all 5 WorkerHandle methods."""
        adapter, _, _ = _make_adapter()
        for method in ("stop_team", "delete_team", "resume_team", "get_team", "stop_all"):
            assert callable(getattr(adapter, method))

    def test_stop_team_signature(self) -> None:
        """stop_team has team_id parameter."""
        sig = inspect.signature(LocalWorkerHandle.stop_team)
        assert "team_id" in sig.parameters

    def test_delete_team_signature(self) -> None:
        """delete_team has team_id parameter."""
        sig = inspect.signature(LocalWorkerHandle.delete_team)
        assert "team_id" in sig.parameters

    def test_resume_team_signature(self) -> None:
        """resume_team has team_id parameter."""
        sig = inspect.signature(LocalWorkerHandle.resume_team)
        assert "team_id" in sig.parameters

    def test_get_team_signature(self) -> None:
        """get_team has team_id parameter."""
        sig = inspect.signature(LocalWorkerHandle.get_team)
        assert "team_id" in sig.parameters

    def test_stop_all_signature(self) -> None:
        """stop_all takes no parameters (besides self)."""
        sig = inspect.signature(LocalWorkerHandle.stop_all)
        params = [p for p in sig.parameters if p != "self"]
        assert params == []


class TestLocalWorkerHandleBehavior:
    """AC4: LocalWorkerHandle delegates to TeamManager correctly."""

    def test_stop_team_delegates_to_team_manager(self) -> None:
        """stop_team calls TeamManager.stop_team with correct team_id."""
        adapter, tm, _ = _make_adapter()
        tid = uuid.uuid4()
        adapter.stop_team(tid)
        tm.stop_team.assert_called_once_with(tid)

    def test_delete_team_delegates_to_team_manager(self) -> None:
        """delete_team calls TeamManager.delete_team with correct team_id."""
        adapter, tm, _ = _make_adapter()
        tid = uuid.uuid4()
        adapter.delete_team(tid)
        tm.delete_team.assert_called_once_with(tid)

    def test_resume_team_delegates_to_team_manager(self) -> None:
        """resume_team calls TeamManager.resume_team with correct team_id."""
        adapter, tm, _ = _make_adapter()
        tid = uuid.uuid4()
        adapter.resume_team(tid)
        tm.resume_team.assert_called_once_with(tid)

    def test_resume_team_returns_local_team_handle(self) -> None:
        """resume_team wraps TeamManager result in LocalTeamHandle."""
        adapter, _, _ = _make_adapter()
        result = adapter.resume_team(uuid.uuid4())
        assert isinstance(result, LocalTeamHandle)

    def test_get_team_delegates_to_team_manager(self) -> None:
        """get_team calls TeamManager.get_team with correct team_id."""
        adapter, tm, _ = _make_adapter()
        tid = uuid.uuid4()
        adapter.get_team(tid)
        tm.get_team.assert_called_once_with(tid)

    def test_get_team_returns_team_manager_result(self) -> None:
        """get_team returns whatever TeamManager.get_team returns."""
        adapter, tm, _ = _make_adapter()
        sentinel = MagicMock()
        tm.get_team.return_value = sentinel
        result = adapter.get_team(uuid.uuid4())
        assert result is sentinel

    def test_get_team_returns_none_when_not_found(self) -> None:
        """get_team returns None when TeamManager returns None."""
        adapter, tm, _ = _make_adapter()
        tm.get_team.return_value = None
        result = adapter.get_team(uuid.uuid4())
        assert result is None


class TestStopAll:
    """AC #1-3, #8: stop_all() stops all teams and shuts down ActorSystem."""

    def test_stop_all_calls_stop_team_for_each_runtime(self) -> None:
        """stop_all() calls stop_team() for every team in _runtimes."""
        adapter, tm, _ = _make_adapter()
        tid1, tid2 = uuid.uuid4(), uuid.uuid4()
        tm._runtimes = {tid1: MagicMock(), tid2: MagicMock()}
        adapter.stop_all()
        tm.stop_team.assert_any_call(tid1)
        tm.stop_team.assert_any_call(tid2)
        assert tm.stop_team.call_count == 2

    def test_stop_all_logs_and_skips_on_failure(self) -> None:
        """stop_all() logs and skips when one stop_team() raises."""
        adapter, tm, actor_system = _make_adapter()
        tid1, tid2 = uuid.uuid4(), uuid.uuid4()
        tm._runtimes = {tid1: MagicMock(), tid2: MagicMock()}
        tm.stop_team.side_effect = [RuntimeError("boom"), None]
        adapter.stop_all()
        # Both were attempted
        assert tm.stop_team.call_count == 2
        # ActorSystem.shutdown() still called
        actor_system.shutdown.assert_called_once()

    def test_stop_all_calls_actor_system_shutdown_after_teams(self) -> None:
        """stop_all() calls ActorSystem.shutdown() after all teams processed."""
        adapter, tm, actor_system = _make_adapter()
        call_order: list[str] = []
        tm._runtimes = {uuid.uuid4(): MagicMock()}
        tm.stop_team.side_effect = lambda _tid: call_order.append("stop_team")
        actor_system.shutdown.side_effect = lambda: call_order.append("shutdown")
        adapter.stop_all()
        assert call_order == ["stop_team", "shutdown"]

    def test_stop_all_calls_shutdown_even_if_all_stop_team_fail(self) -> None:
        """ActorSystem.shutdown() is called even when every stop_team() raises."""
        adapter, tm, actor_system = _make_adapter()
        tid1, tid2 = uuid.uuid4(), uuid.uuid4()
        tm._runtimes = {tid1: MagicMock(), tid2: MagicMock()}
        tm.stop_team.side_effect = RuntimeError("all fail")
        adapter.stop_all()
        actor_system.shutdown.assert_called_once()

    def test_stop_all_with_no_runtimes(self) -> None:
        """stop_all() with empty _runtimes just calls ActorSystem.shutdown()."""
        adapter, tm, actor_system = _make_adapter()
        tm._runtimes = {}
        adapter.stop_all()
        tm.stop_team.assert_not_called()
        actor_system.shutdown.assert_called_once()

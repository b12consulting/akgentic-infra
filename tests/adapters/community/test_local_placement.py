"""Tests for LocalPlacement adapter."""

from __future__ import annotations

import inspect
import uuid
from pathlib import PurePosixPath
from unittest.mock import MagicMock

import pytest
from akgentic.infra.adapters.community.local_placement import LocalPlacement
from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.protocols.placement import (
    DeclaredWorkspaces,
    PlacementError,
    PlacementStrategy,
)
from akgentic.tool.workspace import METADATA_SCOPE

from tests.fixtures.team_metadata import AcmeCaseMetadata


def _make_adapter() -> LocalPlacement:
    """Create a LocalPlacement with mock dependencies."""
    team_manager = MagicMock()
    service_registry = MagicMock()
    return LocalPlacement(team_manager, service_registry)


class TestLocalPlacementProtocolCompliance:
    """AC5: LocalPlacement implements PlacementStrategy protocol."""

    def test_satisfies_placement_strategy_protocol(self) -> None:
        """LocalPlacement structurally satisfies PlacementStrategy."""
        adapter = _make_adapter()
        assert isinstance(adapter, PlacementStrategy)

    def test_has_create_team_method(self) -> None:
        """LocalPlacement exposes create_team with correct signature."""
        adapter = _make_adapter()
        assert callable(adapter.create_team)

    def test_create_team_signature_matches_protocol(self) -> None:
        """create_team has all parameters matching PlacementStrategy."""
        sig = inspect.signature(LocalPlacement.create_team)
        assert "team_card" in sig.parameters
        assert "user_id" in sig.parameters
        assert "user_email" in sig.parameters
        assert "team_id" in sig.parameters
        assert "catalog_namespace" in sig.parameters
        assert sig.parameters["catalog_namespace"].default is None
        # Optional and defaulting to None, so pre-metadata callers are unaffected.
        assert "metadata" in sig.parameters
        assert sig.parameters["metadata"].default is None
        # Story 68.2: the resolved workspaces, last, optional, defaulting to None.
        assert "workspaces" in sig.parameters
        assert sig.parameters["workspaces"].default is None
        assert list(sig.parameters)[-1] == "workspaces"

    def test_workspaces_parameter_matches_the_protocol(self) -> None:
        """AC #8: the protocol declares the same parameter with the same default.

        ``runtime_checkable`` checks that ``create_team`` exists and nothing
        about its shape, so the two signatures are held together by hand.
        """
        protocol = inspect.signature(PlacementStrategy.create_team).parameters
        adapter = inspect.signature(LocalPlacement.create_team).parameters
        assert "workspaces" in protocol
        assert protocol["workspaces"].default is None
        assert list(protocol)[-1] == "workspaces"
        assert list(protocol) == list(adapter)


class TestLocalPlacementBehavior:
    """AC5: LocalPlacement delegates to TeamManager and returns LocalTeamHandle."""

    def test_create_team_delegates_to_team_manager(self) -> None:
        """create_team calls TeamManager.create_team forwarding all args."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        team_card = MagicMock()
        adapter.create_team(team_card, "user-1")
        team_manager.create_team.assert_called_once_with(
            team_card, "user-1", user_email="", team_id=None, catalog_namespace=None, metadata=None
        )

    def test_create_team_forwards_catalog_namespace(self) -> None:
        """create_team forwards catalog_namespace to TeamManager.create_team."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        team_card = MagicMock()
        adapter.create_team(team_card, "user-1", catalog_namespace="ns-abc")
        team_manager.create_team.assert_called_once_with(
            team_card,
            "user-1",
            user_email="",
            team_id=None,
            catalog_namespace="ns-abc",
            metadata=None,
        )

    def test_create_team_forwards_user_email_and_team_id(self) -> None:
        """create_team forwards caller-supplied user_email and team_id verbatim."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        team_card = MagicMock()
        explicit_id = uuid.uuid4()
        adapter.create_team(team_card, "user-1", user_email="user@example.com", team_id=explicit_id)
        team_manager.create_team.assert_called_once_with(
            team_card,
            "user-1",
            user_email="user@example.com",
            team_id=explicit_id,
            catalog_namespace=None,
            metadata=None,
        )

    def test_create_team_forwards_validated_metadata(self) -> None:
        """A validated metadata model reaches TeamManager.create_team verbatim.

        Asserted on the *call*, not only on a persisted result: a MagicMock
        TeamManager accepts any kwargs silently, so a pass-through that dropped
        the value would otherwise go green.
        """
        team_manager = MagicMock()
        adapter = LocalPlacement(team_manager, MagicMock())
        team_card = MagicMock()
        metadata = AcmeCaseMetadata(tenant="acme", case="C-1234")

        adapter.create_team(team_card, "user-1", metadata=metadata)

        assert team_manager.create_team.call_args.kwargs["metadata"] is metadata

    def test_create_team_returns_local_team_handle(self) -> None:
        """create_team wraps TeamManager result in LocalTeamHandle."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        result = adapter.create_team(MagicMock(), "user-1")
        assert isinstance(result, LocalTeamHandle)

    def test_instance_id_is_stable(self) -> None:
        """instance_id does not change between calls."""
        adapter = _make_adapter()
        assert adapter.instance_id == adapter.instance_id

    def test_instance_id_is_uuid(self) -> None:
        """instance_id is a uuid.UUID."""
        adapter = _make_adapter()
        assert isinstance(adapter.instance_id, uuid.UUID)

    def test_different_instances_have_different_ids(self) -> None:
        """Two LocalPlacement instances have different instance_ids."""
        a = _make_adapter()
        b = _make_adapter()
        assert a.instance_id != b.instance_id


_TWO_META = DeclaredWorkspaces(
    shared={
        PurePosixPath(METADATA_SCOPE) / "customer_id-ACME",
        PurePosixPath(METADATA_SCOPE) / "case_id-42",
    }
)
"""The one shape a tier that routes must refuse — rule 3's unsatisfiable case."""


class TestLocalPlacementIgnoresWorkspaces:
    """Story 68.2, AC #6: one process is one worker, so the community tier refuses nothing.

    Every spec here proves the adapter forwards the same call to ``TeamManager``
    whatever the value carries, never consults the routing rule, and creates a
    team a multi-worker tier would refuse. None proves affinity: one worker,
    so this proves the community tier does not route or refuse, not that two
    teams land together.
    """

    def test_two_metadata_trees_are_delegated_with_todays_exact_call_shape(self) -> None:
        """The positive beside the negatives: the manager call is byte-identical.

        The ``assert_called_once_with`` is the same one the pre-metadata specs
        use, and it is the proof that nothing new is forwarded — a
        ``TeamManager.create_team`` has no ``workspaces`` parameter to receive.
        """
        team_manager = MagicMock()
        adapter = LocalPlacement(team_manager, MagicMock())
        team_card = MagicMock()

        result = adapter.create_team(team_card, "user-1", workspaces=_TWO_META)

        assert isinstance(result, LocalTeamHandle)
        team_manager.create_team.assert_called_once_with(
            team_card, "user-1", user_email="", team_id=None, catalog_namespace=None, metadata=None
        )
        assert "workspaces" not in team_manager.create_team.call_args.kwargs

    def test_the_routing_rule_is_never_consulted(self) -> None:
        """One worker; there is nothing to honour, so ``routing_key()`` is never called.

        A subclass records every call. Passing the two-``_meta/`` value makes
        the negative bite: a call would not merely be recorded, it would raise,
        so an adapter that consulted the rule could not also have delegated.
        """
        calls: list[str] = []

        class _Recording(DeclaredWorkspaces):
            def routing_key(self) -> PurePosixPath | None:
                calls.append("routing_key")
                return super().routing_key()

        team_manager = MagicMock()
        adapter = LocalPlacement(team_manager, MagicMock())
        value = _Recording(shared=_TWO_META.shared)

        result = adapter.create_team(MagicMock(), "user-1", workspaces=value)

        assert isinstance(result, LocalTeamHandle)
        assert calls == []
        team_manager.create_team.assert_called_once()

    def test_omitting_the_value_is_the_same_call(self) -> None:
        """``None`` and a value produce one manager call shape — the seam stops here."""
        team_manager = MagicMock()
        adapter = LocalPlacement(team_manager, MagicMock())
        team_card = MagicMock()

        adapter.create_team(team_card, "user-1")
        adapter.create_team(team_card, "user-1", workspaces=DeclaredWorkspaces())

        first, second = team_manager.create_team.call_args_list
        assert first == second


class TestLocalPlacementCreateFailure:
    """AC12: a TeamManager.create_team failure surfaces as a PlacementError."""

    def test_create_team_failure_raises_placement_error(self) -> None:
        """A delegate exception is wrapped in PlacementError (a ServerError)."""
        team_manager = MagicMock()
        team_manager.create_team.side_effect = RuntimeError("boom")
        adapter = LocalPlacement(team_manager, MagicMock())
        with pytest.raises(PlacementError) as exc_info:
            adapter.create_team(MagicMock(), "user-1")
        # Wrapped, not re-raised verbatim: carries the placement HTTP mapping.
        assert exc_info.value.status_code == 503
        assert exc_info.value.__cause__ is not None

    def test_create_team_passes_through_placement_error(self) -> None:
        """An already-typed PlacementError propagates unchanged (not re-wrapped)."""
        original = PlacementError("already typed")
        team_manager = MagicMock()
        team_manager.create_team.side_effect = original
        adapter = LocalPlacement(team_manager, MagicMock())
        with pytest.raises(PlacementError) as exc_info:
            adapter.create_team(MagicMock(), "user-1")
        assert exc_info.value is original

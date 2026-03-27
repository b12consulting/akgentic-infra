"""Tests for LocalPlacement adapter."""

from __future__ import annotations

import inspect
import uuid

from akgentic.infra.adapters.local_placement import LocalPlacement


class TestLocalPlacementProtocolCompliance:
    """AC1: LocalPlacement implements PlacementStrategy protocol."""

    def test_has_select_worker_method(self) -> None:
        """LocalPlacement exposes select_worker with correct signature."""
        adapter = LocalPlacement()
        assert callable(adapter.select_worker)

    def test_select_worker_signature_matches_protocol(self) -> None:
        """select_worker has team_id parameter matching PlacementStrategy."""
        sig = inspect.signature(LocalPlacement.select_worker)
        assert "team_id" in sig.parameters


class TestLocalPlacementBehavior:
    """AC1: LocalPlacement returns instance_id for all teams."""

    def test_select_worker_returns_instance_id(self) -> None:
        """select_worker always returns the adapter's instance_id."""
        adapter = LocalPlacement()
        team_id = uuid.uuid4()
        result = adapter.select_worker(team_id)
        assert result == adapter.instance_id

    def test_select_worker_returns_uuid(self) -> None:
        """select_worker returns a uuid.UUID."""
        adapter = LocalPlacement()
        result = adapter.select_worker(uuid.uuid4())
        assert isinstance(result, uuid.UUID)

    def test_instance_id_is_stable(self) -> None:
        """instance_id does not change between calls."""
        adapter = LocalPlacement()
        assert adapter.instance_id == adapter.instance_id

    def test_select_worker_same_for_different_teams(self) -> None:
        """All teams get placed on the same instance."""
        adapter = LocalPlacement()
        team1 = uuid.uuid4()
        team2 = uuid.uuid4()
        assert adapter.select_worker(team1) == adapter.select_worker(team2)

    def test_different_instances_have_different_ids(self) -> None:
        """Two LocalPlacement instances have different instance_ids."""
        a = LocalPlacement()
        b = LocalPlacement()
        assert a.instance_id != b.instance_id

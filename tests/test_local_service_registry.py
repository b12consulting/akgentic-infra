"""Tests for LocalServiceRegistry adapter."""

from __future__ import annotations

import uuid

from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.team.ports import ServiceRegistry


class TestLocalServiceRegistryProtocolCompliance:
    """AC3: LocalServiceRegistry implements ServiceRegistry protocol."""

    def test_satisfies_service_registry_protocol(self) -> None:
        """LocalServiceRegistry structurally satisfies ServiceRegistry."""
        registry = LocalServiceRegistry()
        assert isinstance(registry, ServiceRegistry)

    def test_has_all_six_methods(self) -> None:
        """LocalServiceRegistry exposes all 6 required methods."""
        registry = LocalServiceRegistry()
        assert callable(registry.register_instance)
        assert callable(registry.deregister_instance)
        assert callable(registry.register_team)
        assert callable(registry.deregister_team)
        assert callable(registry.find_team)
        assert callable(registry.get_active_instances)


class TestLocalServiceRegistryInstances:
    """AC3: Instance registration and deregistration."""

    def test_register_instance(self) -> None:
        """register_instance adds instance to active set."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        registry.register_instance(instance_id)
        assert instance_id in registry.get_active_instances()

    def test_deregister_instance(self) -> None:
        """deregister_instance removes instance from active set."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        registry.register_instance(instance_id)
        registry.deregister_instance(instance_id)
        assert instance_id not in registry.get_active_instances()

    def test_deregister_nonexistent_instance_is_noop(self) -> None:
        """deregister_instance on unknown ID does not raise."""
        registry = LocalServiceRegistry()
        registry.deregister_instance(uuid.uuid4())

    def test_register_instance_idempotent(self) -> None:
        """Registering the same instance twice does not duplicate."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        registry.register_instance(instance_id)
        registry.register_instance(instance_id)
        assert registry.get_active_instances().count(instance_id) == 1

    def test_get_active_instances_empty(self) -> None:
        """get_active_instances returns empty list initially."""
        registry = LocalServiceRegistry()
        assert registry.get_active_instances() == []


class TestLocalServiceRegistryTeams:
    """AC3: Team registration, deregistration, and lookup."""

    def test_register_team(self) -> None:
        """register_team associates team with instance."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        team_id = uuid.uuid4()
        registry.register_instance(instance_id)
        registry.register_team(instance_id, team_id)
        assert registry.find_team(team_id) == instance_id

    def test_deregister_team(self) -> None:
        """deregister_team removes team association."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        team_id = uuid.uuid4()
        registry.register_instance(instance_id)
        registry.register_team(instance_id, team_id)
        registry.deregister_team(instance_id, team_id)
        assert registry.find_team(team_id) is None

    def test_find_team_returns_none_for_unknown(self) -> None:
        """find_team returns None for unregistered team."""
        registry = LocalServiceRegistry()
        assert registry.find_team(uuid.uuid4()) is None

    def test_register_team_on_unregistered_instance_is_noop(self) -> None:
        """register_team on unknown instance does not raise."""
        registry = LocalServiceRegistry()
        registry.register_team(uuid.uuid4(), uuid.uuid4())

    def test_deregister_team_on_unregistered_instance_is_noop(self) -> None:
        """deregister_team on unknown instance does not raise."""
        registry = LocalServiceRegistry()
        registry.deregister_team(uuid.uuid4(), uuid.uuid4())

    def test_deregister_instance_removes_teams(self) -> None:
        """deregister_instance also removes all team associations."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        team_id = uuid.uuid4()
        registry.register_instance(instance_id)
        registry.register_team(instance_id, team_id)
        registry.deregister_instance(instance_id)
        assert registry.find_team(team_id) is None

    def test_multiple_teams_on_one_instance(self) -> None:
        """Multiple teams can be registered on the same instance."""
        registry = LocalServiceRegistry()
        instance_id = uuid.uuid4()
        team1 = uuid.uuid4()
        team2 = uuid.uuid4()
        registry.register_instance(instance_id)
        registry.register_team(instance_id, team1)
        registry.register_team(instance_id, team2)
        assert registry.find_team(team1) == instance_id
        assert registry.find_team(team2) == instance_id

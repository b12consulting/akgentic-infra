"""Tests for LocalRuntimeCache — community-tier RuntimeCache adapter."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from akgentic.infra.adapters.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.protocols.team_handle import RuntimeCache


class TestLocalRuntimeCacheProtocolConformance:
    """AC2: LocalRuntimeCache satisfies the RuntimeCache protocol."""

    def test_isinstance_check(self) -> None:
        """isinstance(LocalRuntimeCache(), RuntimeCache) returns True."""
        cache = LocalRuntimeCache()
        assert isinstance(cache, RuntimeCache)


class TestLocalRuntimeCacheStartsEmpty:
    """AC2: cache starts empty."""

    def test_get_returns_none_for_any_id(self) -> None:
        """get() returns None for any team_id before store()."""
        cache = LocalRuntimeCache()
        assert cache.get(uuid.uuid4()) is None

    def test_get_returns_none_for_specific_id(self) -> None:
        """get() returns None for a specific UUID that was never stored."""
        cache = LocalRuntimeCache()
        team_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert cache.get(team_id) is None


class TestLocalRuntimeCacheStoreCycle:
    """AC2: store/get/remove lifecycle."""

    def test_store_then_get_returns_handle(self) -> None:
        """store(team_id, handle) followed by get(team_id) returns the handle."""
        cache = LocalRuntimeCache()
        team_id = uuid.uuid4()
        handle = MagicMock()
        cache.store(team_id, handle)
        assert cache.get(team_id) is handle

    def test_store_overwrites_previous(self) -> None:
        """Storing a new handle for an existing team_id overwrites the old one."""
        cache = LocalRuntimeCache()
        team_id = uuid.uuid4()
        handle1 = MagicMock()
        handle2 = MagicMock()
        cache.store(team_id, handle1)
        cache.store(team_id, handle2)
        assert cache.get(team_id) is handle2

    def test_remove_then_get_returns_none(self) -> None:
        """remove(team_id) followed by get(team_id) returns None."""
        cache = LocalRuntimeCache()
        team_id = uuid.uuid4()
        handle = MagicMock()
        cache.store(team_id, handle)
        cache.remove(team_id)
        assert cache.get(team_id) is None

    def test_remove_unknown_id_is_noop(self) -> None:
        """remove() for an unknown team_id does not raise."""
        cache = LocalRuntimeCache()
        cache.remove(uuid.uuid4())  # should not raise

    def test_multiple_teams_independent(self) -> None:
        """Multiple teams stored independently; removing one doesn't affect others."""
        cache = LocalRuntimeCache()
        id1, id2 = uuid.uuid4(), uuid.uuid4()
        h1, h2 = MagicMock(), MagicMock()
        cache.store(id1, h1)
        cache.store(id2, h2)
        cache.remove(id1)
        assert cache.get(id1) is None
        assert cache.get(id2) is h2

"""Tests for TierServices dependency injection container."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest
from akgentic.catalog import Catalog
from akgentic.core.agent_card import AgentCard
from akgentic.team.models import (
    AgentCardEntry,
    AgentCardRef,
    AgentRef,
    AgentStateSnapshot,
    DescriptionOrigin,
    PersistedEvent,
    Process,
    TeamStatus,
)
from akgentic.team.ports import EventStore

from akgentic.infra.protocols.auth import AuthStrategy
from akgentic.infra.protocols.channels import ChannelRegistry
from akgentic.infra.protocols.event_stream import EventStream
from akgentic.infra.protocols.placement import PlacementStrategy
from akgentic.infra.protocols.runtime_cache import RuntimeCache
from akgentic.infra.protocols.worker_handle import WorkerHandle
from akgentic.infra.server.deps import TierServices


class FakeEventStore:
    """Minimal EventStore-shaped class satisfying the protocol via structural subtyping.

    Does NOT inherit from EventStore -- validates that Pydantic accepts
    protocol-typed fields through structural subtyping when
    arbitrary_types_allowed=True and SkipValidation is used.

    Team snapshots are the one thing it actually stores: the description write
    is conditional on what is already there, so a fake that remembered nothing
    could not honour the port's contract.
    """

    def __init__(self) -> None:
        self._teams: dict[uuid.UUID, Process] = {}

    def save_event(self, event: PersistedEvent) -> None:
        """No-op stub."""

    def load_events(
        self, team_id: uuid.UUID, after_event_id: uuid.UUID | None = None
    ) -> list[PersistedEvent]:
        """Return empty list."""
        return []

    def save_team(self, process: Process) -> None:
        """Store the snapshot under its team id."""
        self._teams[process.team_id] = process

    def update_team_description(
        self,
        team_id: uuid.UUID,
        description: str | None,
        origin: DescriptionOrigin,
    ) -> Process | None:
        """The port's conditional write: USER lands and latches, AUTO yields to USER.

        Derived by ``model_copy`` so a field this fake has never heard of
        survives the write, as it does on every real backend.
        """
        stored = self._teams.get(team_id)
        if stored is None:
            return None
        if origin is DescriptionOrigin.AUTO and stored.description_origin is DescriptionOrigin.USER:
            return stored
        updated = stored.model_copy(
            update={
                "team_description": description,
                "description_origin": origin,
                "updated_at": datetime.now(UTC),
            }
        )
        self._teams[team_id] = updated
        return updated

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Return the stored snapshot, or None for an unknown team."""
        return self._teams.get(team_id)

    def delete_team(self, team_id: uuid.UUID) -> None:
        """No-op stub."""

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        """No-op stub."""

    def list_teams(
        self,
        user_id: str | None = None,
        status: TeamStatus | None = None,
        metadata: Mapping[str, list[str]] | None = None,
    ) -> list[Process]:
        """Return empty list."""
        return []

    def get_max_sequence(self, team_id: uuid.UUID) -> int:
        """Return 0."""
        return 0

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Return empty list."""
        return []

    def load_agent_state(
        self, team_id: uuid.UUID, agent_id: uuid.UUID
    ) -> AgentStateSnapshot | None:
        """Return None."""
        return None

    def save_agent_cards(self, cards: list[AgentCard]) -> None:
        """No-op stub."""

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Return empty mapping."""
        return {}

    def list_agent_card_entries(self) -> list[AgentCardEntry]:
        """Return empty list."""
        return []


class TestTierServicesEventStoreProtocol:
    """AC6: TierServices accepts a MongoEventStore-shaped object via structural subtyping."""

    def test_tierservices_accepts_mongo_shaped_event_store(self) -> None:
        """TierServices construction succeeds with a fake EventStore implementation.

        The fake class does NOT inherit from EventStore -- it satisfies the
        protocol purely through structural subtyping, the same pattern used
        by MongoEventStore and YamlEventStore.
        """
        fake_store = FakeEventStore()

        services = TierServices(
            placement=MagicMock(spec=PlacementStrategy),
            worker_handle=MagicMock(spec=WorkerHandle),
            auth=MagicMock(spec=AuthStrategy),
            event_store=fake_store,
            runtime_cache=MagicMock(spec=RuntimeCache),
            event_stream=MagicMock(spec=EventStream),
            channel_registry=MagicMock(spec=ChannelRegistry),
            catalog=MagicMock(spec=Catalog),
        )

        assert services.event_store is fake_store


def _parameter_shape(method: Callable[..., object]) -> list[tuple[str, str, object]]:
    """Return (name, kind, default) per parameter of ``method``, excluding ``self``.

    Annotations are deliberately left out: both modules use
    ``from __future__ import annotations``, so ``Signature`` carries them as
    bare strings and ``"str | None"`` would not compare equal to a resolved
    type. Type equality is asserted separately via ``get_type_hints``.
    """
    return [
        (param.name, param.kind.name, param.default)
        for param in inspect.signature(method).parameters.values()
        if param.name != "self"
    ]


_PROTOCOL_METHODS = sorted(
    name for name, member in vars(EventStore).items() if callable(member) and name[0] != "_"
)


class TestFakeEventStoreProtocolShape:
    """The fake's own surface, held against the EventStore protocol it claims to satisfy.

    Nothing else in this package can notice when it drifts: the protocol is not
    ``@runtime_checkable``, ``TierServices.event_store`` is ``SkipValidation``-typed
    so Pydantic validates nothing, and CI type-checks and lints ``src/`` only —
    the test tree is executed, never analysed. That gap let ``list_teams`` sit on
    a two-widenings-stale signature and ``load_events`` on a one-widening-stale
    one, both unnoticed. These assertions are the gate that was missing.
    """

    def test_list_teams_accepts_every_protocol_call_shape(self) -> None:
        """Every way the protocol allows ``list_teams`` to be called works on the fake.

        The metadata shapes are term LISTS. The bare-``str`` calls these used to
        make were green only because the fake returns ``[]`` without consulting
        the value — a call shape the real contract now rejects with ``TypeError``.
        """
        store = FakeEventStore()

        assert store.list_teams() == []
        assert store.list_teams("u1") == []
        assert store.list_teams(user_id="u1") == []
        assert store.list_teams(status=TeamStatus.RUNNING) == []
        assert store.list_teams(user_id="u1", status=TeamStatus.RUNNING) == []
        assert store.list_teams(metadata={"tenant": ["acme"]}) == []
        assert store.list_teams(user_id="u1", metadata={"tenant": ["acme", "contoso"]}) == []

    @pytest.mark.parametrize("method_name", _PROTOCOL_METHODS)
    def test_method_signature_matches_the_protocol(self, method_name: str) -> None:
        """Parameter names, order, kinds, defaults and types match the port exactly.

        The call-shape test above cannot catch a swap of ``user_id`` and
        ``status`` — both are optional and both accept ``None``, so all five
        shapes still run — yet the order is load-bearing: it is what keeps
        existing positional callers of ``list_teams("u1")`` meaning what they
        say. Comparing against the live protocol rather than a transcribed
        signature is also what makes this test outlast the next widening.
        """
        fake_method = getattr(FakeEventStore, method_name)
        protocol_method = getattr(EventStore, method_name)

        assert _parameter_shape(fake_method) == _parameter_shape(protocol_method)
        assert get_type_hints(fake_method) == get_type_hints(protocol_method)

    def test_the_protocol_surface_was_actually_discovered(self) -> None:
        """The parametrized guard above is only real while the derivation finds methods.

        An empty parameter set is reported by pytest as SKIPPED, not as a
        failure, and the run still exits 0 — so the signature comparison could
        quietly stop covering anything and no gate would turn red. Break the
        derivation deliberately and this is the only test that notices.
        """
        assert _PROTOCOL_METHODS

    def test_the_fake_does_not_inherit_from_the_protocol(self) -> None:
        """Structural subtyping is what the fake exists to demonstrate.

        ``FakeEventStore``'s whole point is that a class which merely has the
        right shape is accepted for a protocol-typed field; making it a subclass
        would erase the thing under test while leaving every other assertion in
        this file green — the signature comparison included, since the overrides
        are unchanged. ``issubclass`` cannot be used here: ``EventStore`` is not
        ``@runtime_checkable``, so the check goes through the MRO instead.
        """
        assert EventStore not in FakeEventStore.__mro__


def _process(
    *,
    team_description: str | None = None,
    description_origin: DescriptionOrigin = DescriptionOrigin.AUTO,
    model_class: type[Process] = Process,
) -> Process:
    """A minimal persisted ``Process`` whose stamps sit safely in the past.

    ``updated_at`` is a day old so "the stamp moved" and "the stamp did not
    move" are both assertable without racing the clock. ``model_class`` lets the
    copy-not-rebuild guard construct a subclass through the same path.
    """
    then = datetime.now(UTC) - timedelta(days=1)
    return model_class(
        team_id=uuid.uuid4(),
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=then,
        updated_at=then,
        entry_point=AgentRef(name="@Manager", role="Manager"),
        agent_cards=[AgentCardRef(role="Manager", card_hash="0" * 64)],
        team_description=team_description,
        description_origin=description_origin,
    )


class _ProcessWithExtraField(Process):
    """A ``Process`` carrying a field the fake's write path has never heard of."""

    extra_field: str = "sentinel"


class TestFakeEventStoreDescriptionSemantics:
    """AC13: the fake honours the port's conditional-write contract, not just its shape.

    A fake that always wrote would satisfy the signature guard above and still
    let a service test pass against semantics the real stores do not have.
    """

    def test_a_user_write_lands_and_sets_the_origin_to_user(self) -> None:
        store = FakeEventStore()
        process = _process()
        store.save_team(process)

        updated = store.update_team_description(
            process.team_id, "Triage inbound acme cases", DescriptionOrigin.USER
        )

        assert updated is not None
        assert updated.team_description == "Triage inbound acme cases"
        assert updated.description_origin is DescriptionOrigin.USER
        assert updated.updated_at > process.updated_at
        assert store.load_team(process.team_id) == updated

    def test_an_auto_write_against_a_fresh_record_lands(self) -> None:
        store = FakeEventStore()
        process = _process()
        store.save_team(process)

        updated = store.update_team_description(
            process.team_id, "Generated summary", DescriptionOrigin.AUTO
        )

        assert updated is not None
        assert updated.team_description == "Generated summary"
        assert updated.description_origin is DescriptionOrigin.AUTO

    def test_an_auto_write_against_a_user_owned_record_is_a_no_op(self) -> None:
        store = FakeEventStore()
        owned = _process(team_description="Mine", description_origin=DescriptionOrigin.USER)
        store.save_team(owned)

        result = store.update_team_description(
            owned.team_id, "Generated summary", DescriptionOrigin.AUTO
        )

        assert result == owned
        assert result.team_description == "Mine"
        assert result.updated_at == owned.updated_at
        assert store.load_team(owned.team_id) == owned

    def test_a_user_clear_keeps_the_latch_so_a_later_auto_write_is_still_a_no_op(
        self,
    ) -> None:
        store = FakeEventStore()
        owned = _process(team_description="Mine", description_origin=DescriptionOrigin.USER)
        store.save_team(owned)

        cleared = store.update_team_description(owned.team_id, None, DescriptionOrigin.USER)
        assert cleared is not None
        assert cleared.team_description is None
        assert cleared.description_origin is DescriptionOrigin.USER

        later = store.update_team_description(
            owned.team_id, "Generated summary", DescriptionOrigin.AUTO
        )
        assert later == cleared
        assert later.team_description is None

    def test_an_unknown_team_id_returns_none(self) -> None:
        store = FakeEventStore()

        assert store.update_team_description(uuid.uuid4(), "x", DescriptionOrigin.USER) is None

    def test_the_write_survives_a_field_the_fake_has_never_heard_of(self) -> None:
        """The record is derived by copy, not rebuilt field by field.

        An enumerated reconstruction returns a plain ``Process`` and drops the
        sentinel; only ``model_copy(update=...)`` keeps both.
        """
        store = FakeEventStore()
        process = _process(model_class=_ProcessWithExtraField)
        store.save_team(process)

        updated = store.update_team_description(process.team_id, "x", DescriptionOrigin.USER)

        assert isinstance(updated, _ProcessWithExtraField)
        assert updated.extra_field == "sentinel"

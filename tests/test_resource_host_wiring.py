"""Specs for the one resource host ``wire_community`` creates, and for it running cold.

Four properties are under test here and each has a named falsifying case, because a
wiring spec's characteristic failure is passing whether or not the wiring happened:

- **exactly one host, and it answers.** A count of one is also what a process that leaked
  a host from an earlier test would show, and also what a wedged host would show. So the
  count is bracketed: an empty-registry assertion *before* wiring makes the count mean
  "this wiring made it", and a round trip *after* makes it mean "and it works".
- **the host exists before the first team of the process resumes.** ``warm`` resumes
  RUNNING teams inside ``wire_community``, so this is a statement-order constraint within
  one function. It is proved by observation — a wrapper on ``LocalRuntimeCache.warm``
  recording the host count as it is entered — rather than by reading the source.
- **the process is cold.** No store is registered, so the host never delivers a state.
  Proved twice: a spy that must see zero ``register_store`` calls, and a hosted actor
  whose ``init_state`` marker list must stay empty.
- **teardown leaves nothing, and wiring is repeatable.** The pykka registry is
  process-global and 30 tests in this suite wire a community in one process, so a host
  that survived its own teardown would make the next wiring produce two — which the
  orchestrator's forward refuses with a ``RuntimeError``, in a deployment, with nothing
  in this suite having ever seen it.

What is out of reach here: everything about a *second replica*. These are single-process
specs and no assertion in this file says anything about what another replica holds. The
host is per-process actor-runtime state, which is why it is on ``CommunityServices`` and
not on ``TierServices``; the field's absence from the tier model is asserted, but that is
a statement about the model, not evidence about a replica.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from akgentic.core import (
    ActorAddress,
    ActorSystem,
    Akgent,
    BaseConfig,
    BaseState,
    ResourceHost,
    ResourceStore,
)

from akgentic.infra.adapters.community.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community

if TYPE_CHECKING:
    from akgentic.core import StateDelta
    from akgentic.team.repositories.yaml import YamlEventStore

    from akgentic.infra.adapters.community.local_worker_handle import LocalWorkerHandle

TIMEOUT = 10.0

# How long a host may take to leave the process registry after shutdown. This absorbs only
# the hand-off between the stop request and pykka deregistering the ref. It is NOT a leak
# guard and cannot be one: ``ActorSystem.shutdown`` ends in ``pykka.ActorRegistry.stop_all()``,
# which is process-global, so any shutdown stops every host in the process — including one
# leaked by an earlier test — before this wait begins. What the wait-then-assert proves is
# that a stop actually deregisters the host; what would catch a leak is an assertion made
# BEFORE any shutdown runs, which is why every spec here asserts the empty registry first.
TEARDOWN_GRACE = 5.0


##
## Test doubles
##
class _ProbeState(BaseState):
    """State of the hosted probe. ``value`` is what a store would have restored."""

    value: str = "default"


class _ProbeActor(Akgent[BaseConfig, _ProbeState]):
    """Hosted actor recording its constructions and every state delivered to it.

    Shaped after core's own ``_CountingActor``. The ``restorations`` list is the
    behavioural half of the cold-host spec: the host calls ``init_state`` only when its
    store returned a state, so on a cold host this list stays empty and on a host with a
    store it does not.
    """

    constructions: ClassVar[list[uuid.UUID]] = []
    restorations: ClassVar[list[str]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = _ProbeState()
        _ProbeActor.constructions.append(self.agent_id)

    def init_state(self, state: _ProbeState) -> None:
        """Record the delivered state, then behave exactly as the base class does."""
        _ProbeActor.restorations.append(state.value)
        super().init_state(state)

    def ping(self) -> str:
        """A method a caller can reach through a proxy, to prove the address works."""
        return "pong"


class _PopulatingStore:
    """A store that always restores a populated state.

    Never registered by any spec here — it exists so mutation (c) has something to
    register, and so a reader can see what the cold assertions would look like if they
    were false. Its presence in this file changes no behaviour.
    """

    def load(self, actor_class: type[Akgent[Any, Any]], scope: str) -> BaseState | None:
        """Answer a populated state for every scope."""
        return _ProbeState(value="restored")

    def apply(self, actor_class: type[Akgent[Any, Any]], scope: str, delta: StateDelta) -> None:
        """Absorb nothing; the write-back half is not what these specs measure."""


##
## Helpers
##
def _settings(tmp_path: Path) -> CommunitySettings:
    """Community settings rooted entirely under *tmp_path*."""
    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )


def _hosts() -> list[ActorAddress]:
    """Every live ``ResourceHost`` in the process, by class lookup rather than by field."""
    return ActorSystem.find_by_class(ResourceHost)


def _hosted(services: CommunityServices, host: ActorAddress, name: str) -> ActorAddress:
    """Ask *host* directly for the probe named *name*, the way a card's bind will."""
    return services.actor_system.proxy_ask(host, ResourceHost, timeout=TIMEOUT).getResourceOrCreate(
        _ProbeActor, BaseConfig(name=name, role="probe")
    )


def _flush(services: CommunityServices, address: ActorAddress) -> None:
    """Block until everything already queued for *address* has been handled.

    Load-bearing wherever a spec asserts that something did **not** arrive. ``init_state``
    reaches a hosted actor by ``proxy_tell``, so an assertion made without this would read
    an empty list while the delivery was still in flight — and would then stay green under
    the very mutation it exists to catch. One FIFO mailbox serves proxy calls and tells
    alike, so a resolved ``ping`` proves every earlier envelope was processed.
    """
    assert services.actor_system.proxy_ask(address, _ProbeActor, timeout=TIMEOUT).ping() == "pong"


def _assert_no_host_within_grace() -> None:
    """Assert the process registry holds no host, allowing for deregistration lag."""
    deadline = time.monotonic() + TEARDOWN_GRACE
    while _hosts() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _hosts() == []


@pytest.fixture(autouse=True)
def _reset_probe() -> None:
    """Clear the probe's class-level records so one spec cannot read another's."""
    _ProbeActor.constructions.clear()
    _ProbeActor.restorations.clear()


##
## Specs
##
class TestOneHostPerProcess:
    """AC #1, AC #3: wiring creates exactly one live host and hands it to the container."""

    def test_wiring_creates_exactly_one_live_host_that_answers(self, tmp_path: Path) -> None:
        """One host after wiring, and it serves a get-or-create round trip.

        The pre-condition is not ceremony. Without it the count below is satisfied by a
        process that leaked a host from an earlier test and created none here.
        """
        assert _hosts() == [], "a host leaked into this process before wiring ran"

        services = wire_community(_settings(tmp_path))
        try:
            hosts = _hosts()
            assert len(hosts) == 1
            assert hosts[0].is_alive()

            # A count of one is also what a wedged host that answers nothing would give.
            hosted = _hosted(services, hosts[0], "#Probe-1")
            assert isinstance(hosted, ActorAddress)
            assert hosted.is_alive()
            _flush(services, hosted)
            assert len(_ProbeActor.constructions) == 1
        finally:
            services.actor_system.shutdown()

    def test_container_holds_the_same_host_the_lookup_finds(self, tmp_path: Path) -> None:
        """``services.resource_host`` is *the* host, not a second one."""
        assert _hosts() == []

        services = wire_community(_settings(tmp_path))
        try:
            hosts = _hosts()
            assert len(hosts) == 1
            assert services.resource_host.agent_id == hosts[0].agent_id
        finally:
            services.actor_system.shutdown()

    def test_host_field_is_community_tier_only(self) -> None:
        """The field is on the community container and deliberately not on the tier one.

        The negative is the assertion that matters and the only one a misplacement turns
        red: ``CommunityServices`` inherits ``TierServices``, so moving the field up leaves
        the positive assertion green while making every tier's *server* claim a host it
        does not have.
        """
        assert "resource_host" in CommunityServices.model_fields
        assert "resource_host" not in TierServices.model_fields


class TestHostExistsBeforeTeamsResume:
    """AC #2: the host is created before ``warm`` resumes anything."""

    def test_host_exists_when_runtime_cache_warm_is_entered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``warm`` finds a host already in the registry the moment it is entered.

        Patched on the class, not the instance: ``wire_community`` builds the
        ``LocalRuntimeCache`` itself, so no instance exists for a test to reach.

        This proves the host exists before ``warm`` is **entered**. It does not prove a
        team actually resumed after it, because no RUNNING team is seeded here.
        """
        observed: list[int] = []
        original = LocalRuntimeCache.warm

        def _recording_warm(
            cache: LocalRuntimeCache, worker_handle: LocalWorkerHandle, event_store: YamlEventStore
        ) -> None:
            observed.append(len(_hosts()))
            original(cache, worker_handle, event_store)

        monkeypatch.setattr(LocalRuntimeCache, "warm", _recording_warm)

        assert _hosts() == []
        services = wire_community(_settings(tmp_path))
        try:
            assert observed == [1], (
                "warm must be entered exactly once, with the host already created"
            )
        finally:
            services.actor_system.shutdown()


class TestProcessIsCold:
    """AC #4: no store is registered, and the host therefore restores nothing."""

    def test_wiring_registers_no_store_and_restores_no_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both halves of cold: the spy sees no call, and no state is ever delivered.

        The spy alone would be satisfied by a store attached some other way; the marker
        list alone is vacuous on a cold host, since with nothing stored there is nothing
        that *could* arrive. Together they fail whenever a store appears.
        """
        registered: list[ResourceStore] = []
        original = ResourceHost.register_store

        def _recording_register(host: ResourceHost, store: ResourceStore) -> None:
            registered.append(store)
            original(host, store)

        monkeypatch.setattr(ResourceHost, "register_store", _recording_register)

        assert _hosts() == [], "a host leaked into this process before wiring ran"
        services = wire_community(_settings(tmp_path))
        try:
            assert registered == []

            hosted = _hosted(services, _hosts()[0], "#Probe-cold")
            _flush(services, hosted)
            assert _ProbeActor.restorations == []
            # Deliberately NOT asserted here: that the probe's state is the class
            # default. On a cold host nothing could have replaced it, so that assertion
            # is true whatever the wiring does — decoration, not a guard.
        finally:
            services.actor_system.shutdown()

    def test_a_registered_store_does_restore_state(self, tmp_path: Path) -> None:
        """Positive control: the empty marker list above is a result, not a certainty.

        The cold spec asserts that a list stays empty, and a list stays empty for many
        uninteresting reasons — a probe that never runs, a marker that is never appended,
        a hosted actor the host never created. This registers a store on the very host
        ``wire_community`` built and shows the same list filling, which is the reachable
        falsifying case the cold assertion needs in order to mean anything.

        The store is registered **here, by the test**. No ``src/`` file registers one, and
        this changes nothing about the tier: the community process still runs cold.
        """
        assert _hosts() == [], "a host leaked into this process before wiring ran"
        services = wire_community(_settings(tmp_path))
        try:
            host = _hosts()[0]
            services.actor_system.proxy_ask(host, ResourceHost, timeout=TIMEOUT).register_store(
                _PopulatingStore()
            )

            hosted = _hosted(services, host, "#Probe-warm")
            _flush(services, hosted)
            assert _ProbeActor.restorations == ["restored"]
        finally:
            services.actor_system.shutdown()


class TestTeardownAndRepeatability:
    """AC #5: shutdown takes the host with it, and wiring twice yields one host, not two."""

    def test_second_wiring_after_shutdown_yields_one_host(self, tmp_path: Path) -> None:
        """Wire, tear down, wire again — never two hosts in one process.

        A host that survived its own teardown would make the second wiring produce two,
        and two hosts is the defect the whole design exists to remove.
        """
        assert _hosts() == []

        first = wire_community(_settings(tmp_path / "one"))
        assert len(_hosts()) == 1
        first.actor_system.shutdown()
        _assert_no_host_within_grace()

        second = wire_community(_settings(tmp_path / "two"))
        try:
            assert len(_hosts()) == 1, "the first host outlived its own actor system"
        finally:
            second.actor_system.shutdown()
        _assert_no_host_within_grace()

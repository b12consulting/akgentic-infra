"""Tests for InteractionChannelDispatcher — one shared, team-agnostic instance."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.messages.message import UserMessage
from akgentic.core.messages.orchestrator import ReceivedMessage, SentMessage, StartMessage
from akgentic.core.orchestrator import EventSubscriber

from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.shared.channel_dispatcher import InteractionChannelDispatcher
from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

# ---------------------------------------------------------------------------
# Stub adapters satisfying InteractionChannelAdapter protocol (structural)
# ---------------------------------------------------------------------------


class _MatchingAdapter:
    """Adapter stub that always matches."""

    def __init__(self) -> None:
        self.matches_called = False
        self.deliver_called = False
        self.stop_called = False
        self.stop_team_id: uuid.UUID | None = None
        self.matches_msg: SentMessage | None = None
        self.deliver_msg: SentMessage | None = None
        self.matches_binding: ChannelBinding | None = None
        self.deliver_binding: ChannelBinding | None = None
        self.notices: list[tuple[ChannelAddress, str]] = []

    def matches(self, msg: SentMessage, binding: ChannelBinding) -> bool:
        self.matches_called = True
        self.matches_msg = msg
        self.matches_binding = binding
        return True

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        self.deliver_called = True
        self.deliver_msg = msg
        self.deliver_binding = binding

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        self.notices.append((address, text))

    def on_stop(self, team_id: uuid.UUID) -> None:
        self.stop_called = True
        self.stop_team_id = team_id


class _NonMatchingAdapter:
    """Adapter stub that never matches."""

    def __init__(self) -> None:
        self.matches_called = False
        self.deliver_called = False
        self.stop_called = False
        self.stop_team_id: uuid.UUID | None = None

    def matches(self, msg: SentMessage, binding: ChannelBinding) -> bool:
        self.matches_called = True
        return False

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        self.deliver_called = True

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        pass

    def on_stop(self, team_id: uuid.UUID) -> None:
        self.stop_called = True
        self.stop_team_id = team_id


# ---------------------------------------------------------------------------
# Registry double — records which face the dispatcher uses
# ---------------------------------------------------------------------------


class _StubChannelRegistry:
    """Records what the dispatcher asks of the registry, and on which face.

    ``trap_awaits`` makes every async method raise the instant it is awaited.
    ``on_message`` runs on a Pykka actor thread with no event loop, so any await
    on that path wedges the thread — the trap is what notices, now that the
    dispatcher's annotation is the full ``ChannelRegistry`` and no longer can.
    """

    def __init__(
        self,
        bindings: dict[tuple[uuid.UUID, str], ChannelBinding] | None = None,
        *,
        trap_awaits: bool = False,
    ) -> None:
        self._bindings = dict(bindings or {})
        self._trap_awaits = trap_awaits
        self.sync_lookups: list[tuple[uuid.UUID, str]] = []
        self.awaited: list[str] = []
        self.deregistered_teams: list[uuid.UUID] = []
        self.deregister_team_raises = False

    def _record_await(self, method: str) -> None:
        self.awaited.append(method)
        if self._trap_awaits:
            msg = f"{method} was awaited — this path must never block"
            raise RuntimeError(msg)

    def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
        self.sync_lookups.append((team_id, agent_name))
        return self._bindings.get((team_id, agent_name))

    async def register(self, binding: ChannelBinding) -> None:
        self._record_await("register")

    async def find_binding(self, address: ChannelAddress) -> ChannelBinding | None:
        self._record_await("find_binding")
        return None

    async def deregister(self, address: ChannelAddress) -> None:
        self._record_await("deregister")

    async def deregister_team(self, team_id: uuid.UUID) -> None:
        self._record_await("deregister_team")
        self.deregistered_teams.append(team_id)
        if self.deregister_team_raises:
            msg = "registry unavailable"
            raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEAM_A = uuid.uuid4()
TEAM_B = uuid.uuid4()

AGENT_A = "@HumanProxy_0"
AGENT_B = "human_support"


def _make_addr(
    team_id: uuid.UUID,
    name: str = AGENT_A,
    *,
    is_user_proxy: bool = True,
) -> ActorAddressProxy:
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "akgentic.core.actor_address_impl.ActorAddressProxy",
            "agent_id": str(uuid.uuid4()),
            "name": name,
            "role": "tester",
            "team_id": str(team_id),
            "squad_id": str(uuid.uuid4()),
            "user_message": False,
            "is_user_proxy": is_user_proxy,
        }
    )


def _make_sent_message(
    team_id: uuid.UUID = TEAM_A,
    name: str = AGENT_A,
    *,
    is_user_proxy: bool = True,
    content: str = "Hello from the agent!",
) -> SentMessage:
    recipient = _make_addr(team_id, name, is_user_proxy=is_user_proxy)
    sender = _make_addr(team_id, "assistant", is_user_proxy=False)
    inner = UserMessage(content=content, sender=sender)
    return SentMessage(message=inner, recipient=recipient, sender=sender)


def _binding(
    team_id: uuid.UUID = TEAM_A,
    agent_name: str = AGENT_A,
    channel: str = "telegram",
    channel_user_id: str = "987654321",
) -> ChannelBinding:
    return ChannelBinding(
        channel=channel,
        channel_user_id=channel_user_id,
        team_id=team_id,
        agent_name=agent_name,
    )


def _registry_for(*bindings: ChannelBinding, trap_awaits: bool = False) -> _StubChannelRegistry:
    return _StubChannelRegistry(
        {(b.team_id, b.agent_name): b for b in bindings},
        trap_awaits=trap_awaits,
    )


# ---------------------------------------------------------------------------
# AC 1: one instance, many teams — no team_id in the constructor, no asserts
# ---------------------------------------------------------------------------


class TestOneDispatcherManyTeams:
    """AC 1: a second team is ordinary traffic, not an ``AssertionError``.

    This replaces ``TestOnStopAssertsTeamId``, which asserted the opposite. The
    dispatcher is shared across every team in the process, so a lifecycle hook
    carrying an unfamiliar team id is the normal case.
    """

    def test_delivers_for_two_teams_through_one_instance(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A), _binding(TEAM_B))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))
        assert adapter.deliver_binding is not None
        assert adapter.deliver_binding.team_id == TEAM_A

        dispatcher.on_message(_make_sent_message(TEAM_B))
        assert adapter.deliver_binding.team_id == TEAM_B

    def test_on_stop_for_an_unfamiliar_team_does_not_raise(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_stop(TEAM_B)

        assert adapter.stop_called
        assert adapter.stop_team_id == TEAM_B

    def test_set_restoring_for_an_unfamiliar_team_does_not_raise(self) -> None:
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.set_restoring(TEAM_B, True)

        assert TEAM_B in dispatcher._restoring


# ---------------------------------------------------------------------------
# AC 2: set_restoring is per-team and suppresses only that team
# ---------------------------------------------------------------------------


class TestRestoreMode:
    """AC 2: restore suppression is scoped to the replaying team."""

    def test_restoring_team_a_suppresses_team_a(self) -> None:
        """A replaying team is dropped before the registry is consulted.

        The lookup assertion is what pins the *order* AC 3 states. Without it
        the replay check can sink below ``find_binding_sync`` and every spec in
        this file stays green — verified by mutation, which is the only way to
        tell a guarded ordering from an asserted one.
        """
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A), _binding(TEAM_B))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.set_restoring(TEAM_A, True)
        dispatcher.on_message(_make_sent_message(TEAM_A))

        assert registry.sync_lookups == []
        assert not adapter.matches_called
        assert not adapter.deliver_called

    def test_restoring_team_a_does_not_suppress_team_b(self) -> None:
        """One team's replay must not mute every other team on the instance."""
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A), _binding(TEAM_B))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.set_restoring(TEAM_A, True)
        dispatcher.on_message(_make_sent_message(TEAM_B))

        assert adapter.deliver_called
        assert adapter.deliver_binding is not None
        assert adapter.deliver_binding.team_id == TEAM_B

    def test_restoring_false_resumes_dispatch(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.set_restoring(TEAM_A, True)
        dispatcher.on_message(_make_sent_message(TEAM_A))
        assert not adapter.deliver_called

        dispatcher.set_restoring(TEAM_A, False)
        dispatcher.on_message(_make_sent_message(TEAM_A))
        assert adapter.deliver_called


# ---------------------------------------------------------------------------
# AC 3: non-SentMessage events are skipped
# ---------------------------------------------------------------------------


class TestSkipsNonSentMessage:
    """Dispatcher ignores non-SentMessage events."""

    def test_received_message_ignored(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(ReceivedMessage(message_id=uuid.uuid4()))

        assert not adapter.matches_called
        assert registry.sync_lookups == []

    def test_start_message_ignored(self) -> None:
        from akgentic.core.agent_config import BaseConfig

        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(StartMessage(config=BaseConfig()))

        assert not adapter.matches_called
        assert registry.sync_lookups == []


# ---------------------------------------------------------------------------
# AC 3: delivery is opt-in — no binding, no delivery; no user proxy, no lookup
# ---------------------------------------------------------------------------


class TestDeliveryIsOptIn:
    """AC 3: absence means skip, never guess."""

    def test_unregistered_agent_reaches_no_adapter(self) -> None:
        """A user proxy with no binding is the WebSocket's; nothing is posted."""
        adapter = _MatchingAdapter()
        registry = _registry_for()  # no bindings at all
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))

        assert registry.sync_lookups == [(TEAM_A, AGENT_A)]
        assert not adapter.matches_called
        assert not adapter.deliver_called

    def test_the_recipient_policy_is_left_to_the_adapters(self) -> None:
        """The dispatcher finds the binding and asks; it does not pre-filter.

        It used to drop anything whose recipient was not a user proxy, which
        decided for every channel at once and meant an adapter willing to
        carry such a message never saw it. Each adapter now says what it will
        take — ``TelegramChannelAdapter`` still refuses a non-user-proxy
        recipient, and its own specs pin that.
        """
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A, AGENT_B))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A, AGENT_B, is_user_proxy=False))

        assert adapter.matches_called
        assert adapter.deliver_called

    def test_an_unbound_agents_message_stops_at_the_lookup(self) -> None:
        """Intra-team traffic still stays in the team unless a binding names it."""
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A, AGENT_B))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A, "@Unbound_0", is_user_proxy=False))

        assert not adapter.matches_called
        assert not adapter.deliver_called

    def test_binding_is_looked_up_by_team_and_agent_name(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A, AGENT_B))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A, AGENT_B))

        assert registry.sync_lookups == [(TEAM_A, AGENT_B)]
        assert adapter.deliver_binding is not None
        assert adapter.deliver_binding.agent_name == AGENT_B


# ---------------------------------------------------------------------------
# AC 3: the adapters receive both the message and the binding
# ---------------------------------------------------------------------------


class TestDispatchToMatchingAdapter:
    """Dispatcher calls matches() then deliver(), handing both the binding."""

    def test_calls_matches_then_deliver(self) -> None:
        adapter = _MatchingAdapter()
        bound = _binding(TEAM_A)
        registry = _registry_for(bound)
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)
        sent = _make_sent_message(TEAM_A)

        dispatcher.on_message(sent)

        assert adapter.matches_called
        assert adapter.deliver_called
        assert adapter.matches_msg is sent
        assert adapter.deliver_msg is sent
        assert adapter.matches_binding is bound
        assert adapter.deliver_binding is bound


class TestNoAdapterMatch:
    """When no adapter matches, the message is silently skipped."""

    def test_no_match_no_exception(self) -> None:
        adapter = _NonMatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))

        assert adapter.matches_called
        assert not adapter.deliver_called


class TestMultiChannelDelivery:
    """With multiple adapters, ALL matching adapters receive deliver()."""

    def test_all_matching_adapters_deliver(self) -> None:
        first = _MatchingAdapter()
        second = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[first, second], registry=registry)
        sent = _make_sent_message(TEAM_A)

        dispatcher.on_message(sent)

        assert first.deliver_msg is sent
        assert second.deliver_msg is sent

    def test_skips_non_matching_then_delivers_to_match(self) -> None:
        non_match = _NonMatchingAdapter()
        match = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[non_match, match], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))

        assert non_match.matches_called
        assert not non_match.deliver_called
        assert match.deliver_called

    def test_delivers_to_multiple_with_non_matching_in_between(self) -> None:
        first = _MatchingAdapter()
        non_match = _NonMatchingAdapter()
        second = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(
            adapters=[first, non_match, second], registry=registry
        )
        sent = _make_sent_message(TEAM_A)

        dispatcher.on_message(sent)

        assert first.deliver_msg is sent
        assert not non_match.deliver_called
        assert second.deliver_msg is sent


class TestEmptyAdapterList:
    """Dispatcher with an empty adapter list handles messages without error."""

    def test_sent_message_with_no_adapters(self) -> None:
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))  # should not raise


# ---------------------------------------------------------------------------
# AC 4: on_message never blocks — the rule that replaced the narrow annotation
# ---------------------------------------------------------------------------


class TestOnMessageNeverAwaits:
    """AC 4: ``on_message`` touches only the synchronous face of the registry.

    The dispatcher now holds the full ``ChannelRegistry``, so nothing at the
    type level stops a developer awaiting on the delivery path. This is the
    guard that does: every async method raises the instant it is awaited, and
    ``on_message`` must reach none of them. It runs on a Pykka actor thread
    with no event loop — a missed delivery is recoverable, a wedged thread is
    not.
    """

    def test_on_message_never_awaits_the_registry(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A), trap_awaits=True)
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))

        assert registry.awaited == []
        assert registry.sync_lookups == [(TEAM_A, AGENT_A)]
        assert adapter.deliver_called

    def test_on_message_never_awaits_when_the_agent_is_unbound(self) -> None:
        registry = _registry_for(trap_awaits=True)
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.on_message(_make_sent_message(TEAM_A))

        assert registry.awaited == []


# ---------------------------------------------------------------------------
# on_stop keeps the team's bindings and notifies the adapters
# ---------------------------------------------------------------------------


class TestOnStop:
    """A stop is reversible, so the bindings stay: discard the entry, fan out."""

    def test_on_stop_makes_no_registry_call_at_all(self) -> None:
        """A stopped team keeps its chat, so there is nothing to release.

        Inverted from the pre-D6 spec, which pinned the release here. The
        release moved to the delete route, which is the one lifecycle change a
        conversation cannot survive; releasing on a stop turned every idle
        timeout into a lost conversation.
        """
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.on_stop(TEAM_A)

        assert registry.deregistered_teams == []
        assert registry.awaited == []
        assert registry.sync_lookups == []

    def test_on_stop_calls_all_adapters(self) -> None:
        """The fan-out stays; only the id's provenance changed."""
        a1 = _MatchingAdapter()
        a2 = _NonMatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[a1, a2], registry=registry)

        dispatcher.on_stop(TEAM_A)

        assert a1.stop_called
        assert a2.stop_called

    def test_adapters_receive_the_stopping_teams_id(self) -> None:
        """The id comes from the hook, not from anything the instance remembers.

        ``TEAM_A`` is the team the dispatcher has bindings for; stopping
        ``TEAM_B`` must reach the adapters as ``TEAM_B``.
        """
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_stop(TEAM_B)

        assert adapter.stop_team_id == TEAM_B

    def test_on_stop_discards_the_restoring_entry(self) -> None:
        """A team stopped mid-restore must not leak a UUID for the process's life."""
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.set_restoring(TEAM_A, True)
        dispatcher.on_stop(TEAM_A)

        assert TEAM_A not in dispatcher._restoring

    def test_a_registry_that_raises_on_every_method_is_never_touched(self) -> None:
        """With no registry call there is nothing left to swallow.

        Re-aimed from the pre-D6 spec, which pinned that a failing release did
        not cost the adapters their notice. ``trap_awaits`` raises the instant
        any async method is awaited, so a re-introduced release would surface
        as a raising ``on_stop`` rather than as a silent log line.
        """
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A), trap_awaits=True)
        registry.deregister_team_raises = True
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_stop(TEAM_A)  # must not raise

        assert registry.awaited == []
        assert adapter.stop_called
        assert adapter.stop_team_id == TEAM_A

    def test_on_stop_empty_adapter_list(self) -> None:
        registry = _registry_for()
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.on_stop(TEAM_A)  # should not raise


# ---------------------------------------------------------------------------
# The binding really does survive, on the file and on the sync index
# ---------------------------------------------------------------------------


class TestOnStopAgainstARealRegistry:
    """The stopped team keeps its chat, so the next message resumes it."""

    def test_the_binding_survives_the_stop(self, tmp_path: Path) -> None:
        """An hour of silence costs the user nothing.

        Both faces are read: the file is the authority for the inbound path,
        the in-process index is what the outbound dispatch consults, and a
        release that reached either would break a different half of the
        conversation.
        """
        registry = YamlChannelRegistry(registry_path=tmp_path / "channels.yaml")
        asyncio.run(registry.register(_binding(TEAM_A)))

        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)
        dispatcher.on_stop(TEAM_A)

        assert registry.find_binding_sync(TEAM_A, AGENT_A) is not None
        assert (
            asyncio.run(
                registry.find_binding(
                    ChannelAddress(channel="telegram", channel_user_id="987654321")
                )
            )
            is not None
        )


# ---------------------------------------------------------------------------
# on_stop_request stays a documented no-op (Protocol member, silently skipped
# if dropped from a structurally-typed subscriber)
# ---------------------------------------------------------------------------


class TestOnStopRequest:
    """The dispatcher has no work on the inactivity-timer signal."""

    def test_on_stop_request_returns_none(self) -> None:
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        assert dispatcher.on_stop_request(TEAM_A) is None

    def test_on_stop_request_does_not_call_adapter_methods(self) -> None:
        adapter = _MatchingAdapter()
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[adapter], registry=registry)

        dispatcher.on_stop_request(TEAM_A)

        assert not adapter.stop_called
        assert not adapter.matches_called
        assert not adapter.deliver_called

    def test_on_stop_request_does_not_mutate_restoring(self) -> None:
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.on_stop_request(TEAM_A)

        assert dispatcher._restoring == set()

    def test_on_stop_request_does_not_release_bindings(self) -> None:
        registry = _registry_for(_binding(TEAM_A))
        dispatcher = InteractionChannelDispatcher(adapters=[], registry=registry)

        dispatcher.on_stop_request(TEAM_A)

        assert registry.deregistered_teams == []


class TestProtocolCompliance:
    """InteractionChannelDispatcher structurally satisfies EventSubscriber."""

    def test_satisfies_event_subscriber_protocol(self) -> None:
        registry = _registry_for()
        dispatcher: EventSubscriber = InteractionChannelDispatcher(adapters=[], registry=registry)
        assert callable(dispatcher.set_restoring)
        assert callable(dispatcher.on_stop_request)
        assert callable(dispatcher.on_stop)
        assert callable(dispatcher.on_message)

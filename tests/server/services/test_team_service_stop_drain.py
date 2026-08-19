"""The drain-before-close guarantee, proven on the real team-stop path.

``tests/test_local_event_stream.py`` pins the ordering inside the reader, with
every removal performed by hand. These tests drive the path the defect lives on
instead: a wired ``TeamService``, a subscribed reader, and the removal that
``TeamService.stop_team``'s own teardown triggers.

Which removal is under test matters. ``stop_team`` leads to two:

1. ``EventStreamSubscriber.on_stop`` -> ``event_stream.remove(team_id)``, fired
   inside ``worker_handle.stop_team`` at the end of the orchestrator teardown —
   the canonical cleanup on every stop path, and the one these tests exercise.
2. ``TeamService.stop_team``'s own ``event_stream.remove``, a belt-and-suspenders
   that finds the entry already popped and returns without doing anything.

Neither is stubbed or disabled: by the time ``stop_team`` returns, the stream was
closed by (1), so any delivery observed afterwards crossed the canonical removal.

Nothing here asserts on the number, order or types of events a teardown emits.
The local ``akgentic-core`` checkout emits one event that the published release
CI installs does not, so any such assertion would be green here and red there.
Each test looks only for its own unique sentinel (see issue #412).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator

import pytest
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import WarningMessage
from akgentic.team.models import TeamStatus

from akgentic.infra.protocols.event_stream import EventStream, StreamClosed, StreamReader
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from tests.conftest import _team_payload, _write_team_entry
from tests.fixtures.events import build_warning_message


def _drain_available(reader: StreamReader, *, timeout: float = 0.1) -> list[Message]:
    """Consume what is currently there; returns on the first timeout.

    Makes the reader a live, caught-up subscriber. It is NOT exhaustive and
    no assertion may depend on it having consumed everything.
    """
    seen: list[Message] = []
    while True:
        event = reader.read_next(timeout=timeout)
        if event is None:
            return seen
        seen.append(event)


def _collect_until_closed(reader: StreamReader, *, deadline_s: float = 5.0) -> list[Message]:
    """Read until StreamClosed, returning everything handed over first."""
    events: list[Message] = []
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            event = reader.read_next(timeout=0.1)
        except StreamClosed:
            return events
        if event is not None:
            events.append(event)
    raise AssertionError("reader never raised StreamClosed")


def _wait_until_in_stream(
    stream: EventStream, team_id: uuid.UUID, marker: str, *, deadline_s: float = 5.0
) -> None:
    """Block until ``marker`` shows up in the stream's snapshot.

    ``emit_message`` crosses an actor boundary, so the append is asynchronous.
    ``read_from`` is a non-blocking snapshot that does not move any reader's
    cursor — which is the point: the sentinel must be confirmed present and
    still unconsumed when the stop runs.
    """
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if any(getattr(event, "content", None) == marker for event in stream.read_from(team_id)):
            return
        time.sleep(0.01)
    raise AssertionError(f"emitted marker '{marker}' never reached the stream")


def _caught_up_reader_with_pending_event(
    team_service: TeamService, team_id: uuid.UUID
) -> tuple[StreamReader, str]:
    """Subscribe, catch up, emit a unique marker, and confirm it is in the stream.

    Returns the reader — deliberately still short of the marker — and the marker.
    """
    stream = team_service.get_event_stream()
    reader = stream.subscribe(team_id, cursor=0)
    _drain_available(reader)
    marker = uuid.uuid4().hex
    team_service.emit_message(team_id, UserMessage(content=marker))
    _wait_until_in_stream(stream, team_id, marker)
    return reader, marker


def _user_message_contents(events: list[Message]) -> list[str]:
    """The ``content`` of every ``UserMessage`` among ``events``.

    Matching is on content, never on object identity: the message crosses an
    actor boundary and is re-stamped by ``Message.init``.
    """
    return [event.content for event in events if isinstance(event, UserMessage)]


@pytest.fixture()
def solo_services(seeded_settings: CommunitySettings) -> Generator[CommunityServices, None, None]:
    """Wired services carrying an extra namespace whose team is entry point only.

    The namespace is seeded BEFORE wiring: ``wire_community`` constructs the
    ``Catalog`` over ``settings.catalog_path``, so a namespace written afterwards
    is not reliably visible.
    """
    payload = _team_payload()
    payload["members"] = []
    _write_team_entry(seeded_settings.catalog_path, "solo-team", payload)
    services = wire_community(seeded_settings)
    yield services
    services.actor_system.shutdown()


def test_event_emitted_before_stop_is_delivered_after_stop_team(
    team_service: TeamService,
) -> None:
    """AC1: an event published before the stop still reaches a live reader after it.

    The reader is caught up and the sentinel is confirmed in the stream but
    deliberately unconsumed when ``stop_team`` runs, so the delivery below had to
    survive the removal the teardown performed.
    """
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_id = process.team_id
    reader, marker = _caught_up_reader_with_pending_event(team_service, team_id)

    try:
        team_service.stop_team(team_id)
        delivered = _collect_until_closed(reader)
    finally:
        reader.close()

    assert marker in _user_message_contents(delivered)


def test_event_appended_during_stream_removal_is_delivered(team_service: TeamService) -> None:
    """AC2: an event written inside the removal itself is delivered too.

    A one-shot wrapper reproduces the ``append()``-then-``remove()`` sequence
    ``EventStreamSubscriber.on_stop`` performs, deterministically rather than by
    racing a sleep. The wrapper only records: both ``remove`` call sites swallow
    every exception, so an assertion raised in there would be eaten silently and
    the test would pass while proving nothing.
    """
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_id = process.team_id
    stream = team_service.get_event_stream()
    reader = stream.subscribe(team_id, cursor=0)
    _drain_available(reader)

    marker = uuid.uuid4().hex
    sentinel = build_warning_message(content=marker)
    real_remove = stream.remove
    appended: list[int] = []

    def removing_after_a_final_write(tid: uuid.UUID) -> None:
        if not appended:  # one-shot: only the first removal, the canonical one
            appended.append(stream.append(tid, sentinel))
        real_remove(tid)

    stream.remove = removing_after_a_final_write  # type: ignore[method-assign]
    try:
        team_service.stop_team(team_id)
        delivered = _collect_until_closed(reader)
    finally:
        stream.remove = real_remove  # type: ignore[method-assign]
        reader.close()

    # A positive sequence number means the write was accepted — the stream was
    # still open when it landed, which is what makes the delivery meaningful.
    assert appended, "the wrapped remove never ran, so nothing was written in the window"
    assert appended[0] > 0
    assert marker in [event.content for event in delivered if isinstance(event, WarningMessage)]


def test_delivery_holds_for_an_entry_point_only_team(solo_services: CommunityServices) -> None:
    """AC3: the same guarantee for the shortest teardown the public path produces.

    ``TeamCard.entry_point`` is required and its ``headcount`` must be 1, so a
    literally zero-agent team is not constructible through ``create_team``; an
    entry-point-only roster is the tightest window there is. The claim is
    delivery, nothing about roster size or the orchestrator's internal
    finalisation shortcut.
    """
    team_service = solo_services.team_service
    assert team_service is not None
    process = team_service.create_team(catalog_namespace="solo-team", user_id="anonymous")
    team_id = process.team_id
    reader, marker = _caught_up_reader_with_pending_event(team_service, team_id)

    try:
        team_service.stop_team(team_id)
        delivered = _collect_until_closed(reader)
    finally:
        reader.close()

    assert marker in _user_message_contents(delivered)


def test_stop_team_still_stops_and_removes_the_stream(team_service: TeamService) -> None:
    """AC4: the drain does not weaken anything the stop already guaranteed.

    Stop returns, the team is persisted STOPPED, the stream entry is gone, and
    the drained reader is terminal rather than merely delayed.
    """
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_id = process.team_id
    stream = team_service.get_event_stream()
    reader, marker = _caught_up_reader_with_pending_event(team_service, team_id)

    try:
        team_service.stop_team(team_id)
        delivered = _collect_until_closed(reader)

        assert marker in _user_message_contents(delivered)
        stopped = team_service.get_team(team_id)
        assert stopped is not None
        assert stopped.status == TeamStatus.STOPPED
        assert stream.read_from(team_id) == []
        with pytest.raises(StreamClosed):
            reader.read_next(timeout=0.1)
    finally:
        reader.close()

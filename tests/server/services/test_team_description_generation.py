"""Story 80.3: the three doors — a first message through ``TeamService`` describes the team.

Real community wiring over a seeded catalog, with exactly two things faked on
the generator's module: ``create_model`` answers a recording ``FunctionModel``
and ``ThreadPoolExecutor`` is the inline executor, so the generation has
completed by the time the send returns. The write is observed on the wiring's
own event store and the notification on the persisted event log —
``PersistenceSubscriber`` writes ``emitMessage`` traffic, so the durable log is
the observable, reached after the orchestrator's actor thread has run.

Three doors trigger (``send_message``, ``send_message_to`` and the channel's
``initiate_team``, each with its HTTP route); ``send_message_from_to``,
``emit_message`` and ``process_human_input`` never do.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from akgentic.core.messages.message import UserMessage
from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.team.models import DescriptionOrigin, PersistedEvent
from fastapi.testclient import TestClient

from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.shared.channel_router import ChannelRouteContext
from akgentic.infra.protocols.channels import ChannelAddress
from akgentic.infra.server import description
from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.description import DESCRIPTION_CONTENT_TYPE
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from tests.conftest import _seed_catalog
from tests.fixtures.description import InlineExecutor, RecordingModel
from tests.fixtures.events import build_sent_message

LOGGER = "akgentic.infra.server.description"
MESSAGE = "Triage the inbound acme support cases"
EXPECTED = "Triage inbound acme support cases"


def _generating_settings(tmp_path: Path) -> CommunitySettings:
    """Community settings with the generator switched on and the catalog seeded."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
        description_provider="openai-chat",
        description_model="gpt-4o-mini",
    )
    _seed_catalog(settings.catalog_path)
    return settings


@pytest.fixture()
def settings(tmp_path: Path) -> CommunitySettings:
    return _generating_settings(tmp_path)


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> RecordingModel:
    """The recording model installed as the generator's ``create_model``, run inline."""
    recording = RecordingModel()
    monkeypatch.setattr(description, "create_model", lambda _cfg, _http=None: recording.build())
    monkeypatch.setattr(description, "ThreadPoolExecutor", InlineExecutor)
    return recording


@pytest.fixture()
def services(
    settings: CommunitySettings, model: RecordingModel
) -> Generator[CommunityServices, None, None]:
    """Community services wired with the generator switched on."""
    wired = wire_community(settings)
    yield wired
    wired.actor_system.shutdown(timeout=5)


@pytest.fixture()
def team_service(services: CommunityServices) -> TeamService:
    assert services.team_service is not None
    return services.team_service


def _description_notifications(
    services: CommunityServices, team_id: uuid.UUID
) -> list[NotificationMessage]:
    return [
        event.event
        for event in services.event_store.load_events(team_id)
        if isinstance(event.event, NotificationMessage)
        and event.event.content_type == DESCRIPTION_CONTENT_TYPE
    ]


def _wait_for_notification(
    services: CommunityServices, team_id: uuid.UUID
) -> list[NotificationMessage]:
    """The emit is a tell; the persistence subscriber writes it on the actor thread."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        found = _description_notifications(services, team_id)
        if found:
            return found
        time.sleep(0.02)
    return _description_notifications(services, team_id)


def _assert_described(services: CommunityServices, team_id: uuid.UUID) -> None:
    """The three effects every door ends in: the write, its origin, one notification."""
    persisted = services.event_store.load_team(team_id)
    assert persisted is not None
    assert persisted.team_description == EXPECTED
    assert persisted.description_origin is DescriptionOrigin.AUTO
    notifications = _wait_for_notification(services, team_id)
    assert len(notifications) == 1
    assert notifications[0].content == EXPECTED
    assert notifications[0].team_id == team_id  # stamped by the orchestrator
    assert notifications[0].timestamp is not None
    assert notifications[0].timestamp >= persisted.updated_at


# --- AC 11 (a): send_message and POST /teams/{id}/message ---


class TestSendMessageDoor:
    """AC11(a): ``TeamService.send_message`` and its route trigger the generator once."""

    def test_first_send_generates_writes_auto_and_persists_one_notification(
        self, services: CommunityServices, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")
        assert model.calls == 0

        team_service.send_message(process.team_id, MESSAGE)

        assert model.prompts == [MESSAGE]
        _assert_described(services, process.team_id)

    def test_the_http_door_answers_204_over_the_same_wiring(
        self, services: CommunityServices, settings: CommunitySettings, model: RecordingModel
    ) -> None:
        client = TestClient(create_app(services, settings))
        created = client.post("/teams", json={"catalog_namespace": "test-team"})
        assert created.status_code == 201
        team_id = uuid.UUID(created.json()["team_id"])

        response = client.post(f"/teams/{team_id}/message", json={"content": MESSAGE})

        assert response.status_code == 204
        assert model.prompts == [MESSAGE]
        _assert_described(services, team_id)


# --- AC 11 (b): send_message_to and POST /teams/{id}/message/{agent_name} ---


class TestSendMessageToDoor:
    """AC11(b): ``TeamService.send_message_to`` and its route trigger the generator once."""

    def test_send_to_generates_writes_auto_and_persists_one_notification(
        self, services: CommunityServices, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")

        team_service.send_message_to(process.team_id, "@Manager", MESSAGE)

        assert model.prompts == [MESSAGE]
        _assert_described(services, process.team_id)

    def test_the_agent_route_answers_204_over_the_same_wiring(
        self, services: CommunityServices, settings: CommunitySettings, model: RecordingModel
    ) -> None:
        client = TestClient(create_app(services, settings))
        created = client.post("/teams", json={"catalog_namespace": "test-team"})
        assert created.status_code == 201
        team_id = uuid.UUID(created.json()["team_id"])

        response = client.post(f"/teams/{team_id}/message/@Manager", json={"content": MESSAGE})

        assert response.status_code == 204
        assert model.prompts == [MESSAGE]
        _assert_described(services, team_id)


# --- AC 11 (c): the channel door ---


class TestChannelDoor:
    """AC11(c): ``ChannelRouteContext.initiate_team`` ends in ``send_message`` and triggers."""

    async def test_initiate_team_describes_and_a_bound_follow_up_does_not(
        self,
        tmp_path: Path,
        services: CommunityServices,
        team_service: TeamService,
        model: RecordingModel,
    ) -> None:
        ctx = ChannelRouteContext(
            address=ChannelAddress(channel="routed", channel_user_id="user-1"),
            registry=YamlChannelRegistry(registry_path=tmp_path / "channels.yaml"),
            team_service=team_service,
            adapters=[],
            default_catalog_entry="test-team",
        )

        process = await ctx.initiate_team(MESSAGE)

        assert model.prompts == [MESSAGE]
        _assert_described(services, process.team_id)

        # A bound chat sends as the bound agent — send_message_from_to — and
        # that door never triggers.
        sent = await ctx.send("and the contoso ones too")

        assert sent is True
        assert model.calls == 1


# --- AC 5, 11: what does not trigger ---


class TestNonDoors:
    """AC5/AC11: a settled record and the three other deliveries cost no model call."""

    def test_a_second_send_makes_no_further_model_call(
        self, services: CommunityServices, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")

        team_service.send_message(process.team_id, MESSAGE)
        team_service.send_message(process.team_id, "and the contoso ones too")

        assert model.calls == 1
        assert len(_wait_for_notification(services, process.team_id)) == 1

    def test_a_user_cleared_team_is_never_described(
        self, services: CommunityServices, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")
        team_service.update_team_description(process.team_id, None)

        team_service.send_message(process.team_id, MESSAGE)

        assert model.calls == 0
        persisted = services.event_store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description is None
        assert persisted.description_origin is DescriptionOrigin.USER
        assert _description_notifications(services, process.team_id) == []

    def test_send_message_from_to_does_not_trigger(
        self, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")

        team_service.send_message_from_to(process.team_id, "@Human", "@Manager", MESSAGE)

        assert model.calls == 0

    def test_emit_message_does_not_trigger(
        self, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")

        team_service.emit_message(process.team_id, UserMessage(content=MESSAGE))

        assert model.calls == 0

    def test_process_human_input_does_not_trigger(
        self,
        services: CommunityServices,
        team_service: TeamService,
        model: RecordingModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")
        sent = build_sent_message(content=MESSAGE)
        services.event_store.save_event(
            PersistedEvent(
                team_id=process.team_id, sequence=1, event=sent, timestamp=datetime.now(UTC)
            )
        )
        handle = services.runtime_cache.get(process.team_id)
        assert handle is not None
        monkeypatch.setattr(handle, "process_human_input", MagicMock())

        team_service.process_human_input(process.team_id, "yes", str(sent.message.id))

        assert model.calls == 0


# --- AC 4, 7: the seam, precisely ---


class TestSeam:
    """AC4/AC7: one record read per send, the endpoint's own store write, no worker verb."""

    def test_the_write_is_the_wired_stores_own_conditional_auto_write(
        self, services: CommunityServices, team_service: TeamService, model: RecordingModel
    ) -> None:
        generator = team_service._description_generator  # noqa: SLF001
        assert generator is not None
        assert generator._event_store is services.event_store  # noqa: SLF001
        real_update = services.event_store.update_team_description
        update = MagicMock(wraps=real_update)
        services.event_store.update_team_description = update  # type: ignore[method-assign]
        process = team_service.create_team("test-team", user_id="alice")

        team_service.send_message(process.team_id, MESSAGE)

        update.assert_called_once_with(process.team_id, EXPECTED, DescriptionOrigin.AUTO)
        _assert_described(services, process.team_id)

    def test_a_send_reads_the_record_once_whether_or_not_the_generator_fires(
        self, services: CommunityServices, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")
        real = team_service._services.worker_handle  # noqa: SLF001
        spy = MagicMock(wraps=real)
        team_service._services.worker_handle = spy  # noqa: SLF001

        # Fires: the generator's own emit reads once more, off the request
        # path in production; inline here, so it is the one extra call.
        team_service.send_message(process.team_id, MESSAGE)
        assert model.calls == 1
        assert spy.method_calls == [call.get_team(process.team_id), call.get_team(process.team_id)]

        # Does not fire: the record in hand already says described.
        spy.reset_mock()
        team_service.send_message(process.team_id, "and the contoso ones too")
        assert model.calls == 1
        assert spy.method_calls == [call.get_team(process.team_id)]

    def test_the_request_path_reads_the_record_once_while_the_model_is_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the real pool, the send itself makes exactly one ``get_team`` call."""
        gate = threading.Event()
        recording = RecordingModel(gate=gate)
        monkeypatch.setattr(description, "create_model", lambda _cfg, _http=None: recording.build())
        services = wire_community(_generating_settings(tmp_path))
        try:
            assert services.team_service is not None
            team_service = services.team_service
            process = team_service.create_team("test-team", user_id="alice")
            spy = MagicMock(wraps=team_service._services.worker_handle)  # noqa: SLF001
            team_service._services.worker_handle = spy  # noqa: SLF001

            team_service.send_message(process.team_id, MESSAGE)

            # Back on the caller's thread with the model still blocked.
            deadline = time.monotonic() + 5
            while not recording.prompts and time.monotonic() < deadline:
                time.sleep(0.005)
            assert recording.calls == 1
            assert spy.method_calls == [call.get_team(process.team_id)]
            before = services.event_store.load_team(process.team_id)
            assert before is not None
            assert before.team_description is None

            gate.set()
            # The write lands on the generator's thread; wait for it before reading.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                written = services.event_store.load_team(process.team_id)
                if written is not None and written.team_description is not None:
                    break
                time.sleep(0.005)
            _assert_described(services, process.team_id)
            # The emit's own read followed, on the generator's thread; no other verb.
            assert all(c == call.get_team(process.team_id) for c in spy.method_calls)
        finally:
            gate.set()
            generator = services.team_service._description_generator  # noqa: SLF001
            assert generator is not None
            generator._executor.shutdown(wait=False, cancel_futures=True)  # noqa: SLF001
            services.actor_system.shutdown(timeout=5)

    def test_an_inner_send_that_raises_schedules_nothing(
        self, team_service: TeamService, model: RecordingModel
    ) -> None:
        process = team_service.create_team("test-team", user_id="alice")

        with pytest.raises(ValueError, match="not found"):
            team_service.send_message_to(process.team_id, "@Nobody", MESSAGE)

        assert model.calls == 0


# --- AC 9: no cap at the service level ---


class TestNoCapThroughTheService:
    """AC9: a raising model is one WARNING per message, and the next message tries again."""

    def test_raising_model_answers_204_warns_once_and_the_next_send_tries_again(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        recording = RecordingModel(error=RuntimeError("provider down"))
        monkeypatch.setattr(description, "create_model", lambda _cfg, _http=None: recording.build())
        monkeypatch.setattr(description, "ThreadPoolExecutor", InlineExecutor)
        settings = _generating_settings(tmp_path)
        services = wire_community(settings)
        try:
            client = TestClient(create_app(services, settings))
            created = client.post("/teams", json={"catalog_namespace": "test-team"})
            team_id = uuid.UUID(created.json()["team_id"])

            with caplog.at_level(logging.DEBUG, logger=LOGGER):
                first = client.post(f"/teams/{team_id}/message", json={"content": MESSAGE})
                second = client.post(f"/teams/{team_id}/message", json={"content": MESSAGE})

            assert first.status_code == 204
            assert second.status_code == 204
            assert recording.calls == 2
            warnings = [
                r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING
            ]
            assert len(warnings) == 2
            assert all(str(team_id) in w.getMessage() for w in warnings)
            persisted = services.event_store.load_team(team_id)
            assert persisted is not None
            assert persisted.team_description is None
            assert _description_notifications(services, team_id) == []
        finally:
            services.actor_system.shutdown(timeout=5)


# --- AC 3: every tier inherits the generator through TeamService ---


class TestInheritance:
    """AC3: a ``TeamService`` built without settings reads the two variables itself."""

    def test_without_settings_and_with_the_variables_set_a_generator_is_built(
        self,
        tmp_path: Path,
        community_services: CommunityServices,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AKGENTIC_DESCRIPTION_PROVIDER", "openai-chat")
        monkeypatch.setenv("AKGENTIC_DESCRIPTION_MODEL", "gpt-4o-mini")

        service = TeamService(community_services, workspaces_root=tmp_path)

        assert service._description_generator is not None  # noqa: SLF001

    def test_without_settings_and_without_the_variables_there_is_none(
        self, tmp_path: Path, community_services: CommunityServices
    ) -> None:
        """The suite's autouse fixture keeps both variables unset."""
        service = TeamService(community_services, workspaces_root=tmp_path)

        assert service._description_generator is None  # noqa: SLF001

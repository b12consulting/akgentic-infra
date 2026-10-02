"""Story 80.2: the community door — a send through ``TeamService`` describes the team.

Real community wiring over a seeded catalog, with exactly two things faked on
the generator's module: ``create_model`` answers a recording ``FunctionModel``
and ``ThreadPoolExecutor`` is the inline executor, so the generation has
completed by the time ``send_message`` returns. The write is observed on the
wiring's own event store and the notification on the persisted event log —
``PersistenceSubscriber`` writes ``emitMessage`` traffic, so the durable log
is the observable, reached after the orchestrator's actor thread has run.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator

import pytest
from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.team.models import DescriptionOrigin
from fastapi.testclient import TestClient

from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from akgentic.infra.worker import description
from akgentic.infra.worker.description import DESCRIPTION_CONTENT_TYPE, DescribingTeamHandle
from akgentic.infra.worker.settings import WorkerSettings
from tests.fixtures.description import InlineExecutor, RecordingModel

MESSAGE = "Triage the inbound acme support cases"
EXPECTED = "Triage inbound acme support cases"


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> RecordingModel:
    """The recording model installed as the generator's ``create_model``."""
    recording = RecordingModel()
    monkeypatch.setattr(description, "create_model", lambda _cfg, _http=None: recording.build())
    monkeypatch.setattr(description, "ThreadPoolExecutor", InlineExecutor)
    return recording


@pytest.fixture()
def services(
    seeded_settings: CommunitySettings, model: RecordingModel
) -> Generator[CommunityServices, None, None]:
    """Community services wired with the generator switched on."""
    worker_settings = WorkerSettings(
        description_provider="openai-chat", description_model="gpt-4o-mini"
    )
    wired = wire_community(seeded_settings, worker_settings=worker_settings)
    yield wired
    wired.actor_system.shutdown(timeout=5)


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


class TestCommunityDoor:
    """AC4(a): ``TeamService.send_message`` on the real wiring triggers the generator."""

    def test_first_send_generates_writes_auto_and_persists_one_notification(
        self, services: CommunityServices, model: RecordingModel
    ) -> None:
        assert services.team_service is not None
        process = services.team_service.create_team("test-team", user_id="alice")
        assert isinstance(services.runtime_cache.get(process.team_id), DescribingTeamHandle)
        assert model.calls == 0

        services.team_service.send_message(process.team_id, MESSAGE)

        assert model.prompts == [MESSAGE]
        persisted = services.event_store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED
        assert persisted.description_origin is DescriptionOrigin.AUTO

        notifications = _wait_for_notification(services, process.team_id)
        assert len(notifications) == 1
        assert notifications[0].content == EXPECTED
        assert notifications[0].team_id == process.team_id  # stamped by the orchestrator
        assert notifications[0].timestamp is not None
        assert notifications[0].timestamp >= persisted.updated_at

    def test_a_second_send_makes_no_further_model_call(
        self, services: CommunityServices, model: RecordingModel
    ) -> None:
        assert services.team_service is not None
        process = services.team_service.create_team("test-team", user_id="alice")

        services.team_service.send_message(process.team_id, MESSAGE)
        services.team_service.send_message(process.team_id, "and the contoso ones too")

        assert model.calls == 1
        notifications = _wait_for_notification(services, process.team_id)
        assert len(notifications) == 1

    def test_the_http_door_answers_204_over_the_same_wiring(
        self,
        services: CommunityServices,
        seeded_settings: CommunitySettings,
        model: RecordingModel,
    ) -> None:
        """The community route ends in the same ``TeamService``; the hook fires once."""
        app = create_app(services, seeded_settings)
        client = TestClient(app)

        created = client.post("/teams", json={"catalog_namespace": "test-team"})
        assert created.status_code == 201
        team_id = uuid.UUID(created.json()["team_id"])

        response = client.post(f"/teams/{team_id}/message", json={"content": MESSAGE})

        assert response.status_code == 204
        assert model.prompts == [MESSAGE]
        persisted = services.event_store.load_team(team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED
        assert persisted.description_origin is DescriptionOrigin.AUTO
        assert len(_wait_for_notification(services, team_id)) == 1

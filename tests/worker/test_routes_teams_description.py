"""Story 80.2: the worker door — ``POST /teams/{id}/message`` describes the team.

In the style of ``test_routes_teams.py``: the handlers are called directly
with a ``WorkerServices``-shaped stub around a real ``LocalRuntimeCache``. The
cache carries a generator over a real ``TeamManager`` and the stateful
``FakeEventStore`` seeded with the team's ``Process``; the model is a
recording ``FunctionModel`` on an inline executor, so every effect is
observable the moment the route returns.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.llm import ModelConfig
from akgentic.team.manager import TeamManager
from akgentic.team.models import DescriptionOrigin, Process

from akgentic.infra.adapters.community.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.server.models import SendMessageRequest
from akgentic.infra.worker import description
from akgentic.infra.worker.description import (
    DESCRIPTION_CONTENT_TYPE,
    DescribingTeamHandle,
    TeamDescriptionGenerator,
)
from akgentic.infra.worker.routes.teams import create_team, send_message, send_message_to_agent
from tests.fixtures.description import InlineExecutor, RecordingModel
from tests.test_deps import FakeEventStore
from tests.worker.test_routes_teams import (
    _build_process,
    _build_services,
    _build_team_card,
    _FakeRuntime,
    _make_create_body,
)

MESSAGE = "Triage the inbound acme support cases"
EXPECTED = "Triage inbound acme support cases"


def _generator_over(process: Process) -> tuple[TeamDescriptionGenerator, FakeEventStore]:
    store = FakeEventStore()
    store.save_team(process)
    manager = TeamManager(actor_system=MagicMock(), event_store=store)
    generator = TeamDescriptionGenerator(
        ModelConfig(provider="openai-chat", model="gpt-4o-mini"),
        manager,
        executor=InlineExecutor(),
    )
    return generator, store


def _install(monkeypatch: pytest.MonkeyPatch, model: RecordingModel) -> None:
    monkeypatch.setattr(description, "create_model", lambda _cfg, _http=None: model.build())


def _description_notifications(runtime: _FakeRuntime) -> list[NotificationMessage]:
    return [
        emitted
        for emitted in runtime.emitted
        if isinstance(emitted, NotificationMessage)
        and emitted.content_type == DESCRIPTION_CONTENT_TYPE
    ]


class TestWorkerDoor:
    """AC4(b): the worker's message route triggers the generator through the cache."""

    def test_create_then_message_generates_writes_auto_and_emits_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = RecordingModel()
        _install(monkeypatch, model)
        team_id = uuid.uuid4()
        team_card = _build_team_card()
        runtime = _FakeRuntime(team_id)
        process = _build_process(team_id, team_card)
        generator, store = _generator_over(process)
        cache = LocalRuntimeCache(description_generator=generator)
        services = _build_services(runtime, process, cache)

        create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]
        assert isinstance(cache.get(team_id), DescribingTeamHandle)
        assert model.calls == 0

        result = send_message(team_id, SendMessageRequest(content=MESSAGE), services)  # type: ignore[arg-type]

        assert result is None  # the 204 path
        assert runtime.sent == [MESSAGE]
        assert model.prompts == [MESSAGE]
        persisted = store.load_team(team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED
        assert persisted.description_origin is DescriptionOrigin.AUTO
        notifications = _description_notifications(runtime)
        assert len(notifications) == 1
        assert notifications[0].content == EXPECTED
        assert notifications[0].timestamp is not None
        assert notifications[0].timestamp >= persisted.updated_at

    def test_the_agent_route_triggers_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = RecordingModel()
        _install(monkeypatch, model)
        team_id = uuid.uuid4()
        team_card = _build_team_card()
        runtime = _FakeRuntime(team_id)
        runtime.send_to = lambda _agent, content: runtime.sent.append(content)  # type: ignore[attr-defined]
        process = _build_process(team_id, team_card)
        generator, store = _generator_over(process)
        services = _build_services(
            runtime, process, LocalRuntimeCache(description_generator=generator)
        )
        create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

        result = send_message_to_agent(  # type: ignore[arg-type]
            team_id, "@Manager", SendMessageRequest(content=MESSAGE), services
        )

        assert result is None
        assert runtime.sent == [MESSAGE]
        assert model.prompts == [MESSAGE]
        persisted = store.load_team(team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED

    def test_a_second_message_makes_no_further_model_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = RecordingModel()
        _install(monkeypatch, model)
        team_id = uuid.uuid4()
        team_card = _build_team_card()
        runtime = _FakeRuntime(team_id)
        process = _build_process(team_id, team_card)
        generator, _store = _generator_over(process)
        services = _build_services(
            runtime, process, LocalRuntimeCache(description_generator=generator)
        )
        create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

        send_message(team_id, SendMessageRequest(content=MESSAGE), services)  # type: ignore[arg-type]
        send_message(team_id, SendMessageRequest(content="and the contoso ones"), services)  # type: ignore[arg-type]

        assert runtime.sent == [MESSAGE, "and the contoso ones"]
        assert model.calls == 1
        assert len(_description_notifications(runtime)) == 1

    def test_a_raising_model_still_answers_204_and_emits_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install(monkeypatch, model)
        team_id = uuid.uuid4()
        team_card = _build_team_card()
        runtime = _FakeRuntime(team_id)
        process = _build_process(team_id, team_card)
        generator, store = _generator_over(process)
        services = _build_services(
            runtime, process, LocalRuntimeCache(description_generator=generator)
        )
        create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

        result = send_message(team_id, SendMessageRequest(content=MESSAGE), services)  # type: ignore[arg-type]

        assert result is None
        assert runtime.sent == [MESSAGE]
        assert model.calls == 1
        assert runtime.emitted == []
        persisted = store.load_team(team_id)
        assert persisted is not None
        assert persisted.team_description is None

    def test_without_a_generator_the_route_behaves_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = RecordingModel()
        _install(monkeypatch, model)
        team_id = uuid.uuid4()
        team_card = _build_team_card()
        runtime = _FakeRuntime(team_id)
        process = _build_process(team_id, team_card)
        services = _build_services(runtime, process, LocalRuntimeCache())
        create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

        result = send_message(team_id, SendMessageRequest(content=MESSAGE), services)  # type: ignore[arg-type]

        assert result is None
        assert runtime.sent == [MESSAGE]
        assert model.calls == 0
        assert runtime.emitted == []

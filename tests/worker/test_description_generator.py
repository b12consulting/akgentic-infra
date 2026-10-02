"""Story 80.2: the worker-side team description generator and its wrapping handle.

The team side is a real ``TeamManager`` over the stateful ``FakeEventStore``
from ``tests/test_deps.py`` — the one with the port's AUTO-loses-to-USER
semantics — so the race spec can genuinely go red. Every model call is a
``FunctionModel`` installed by patching ``create_model`` on the module under
test; nothing here reaches a provider. Deterministic specs run the generator on
an inline executor; the asynchrony specs use the real pool with a model that
blocks on an ``Event``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Executor, Future
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.llm import ModelConfig
from akgentic.team.manager import TeamManager
from akgentic.team.models import (
    AgentCardRef,
    AgentRef,
    DescriptionOrigin,
    Process,
    TeamStatus,
)
from pydantic import ValidationError

from akgentic.infra.protocols.team_handle import TeamHandle
from akgentic.infra.server.models import MAX_TEAM_DESCRIPTION_LENGTH
from akgentic.infra.worker import description
from akgentic.infra.worker.description import (
    DEFAULT_MAX_ATTEMPTS,
    DESCRIPTION_CONTENT_TYPE,
    DescribingTeamHandle,
    TeamDescriptionGenerator,
    normalise_description,
)
from akgentic.infra.worker.settings import WorkerSettings
from tests.fixtures.description import InlineExecutor, RecordingModel
from tests.test_deps import FakeEventStore

LOGGER = "akgentic.infra.worker.description"
MESSAGE = "Triage the inbound acme support cases"
EXPECTED = "Triage inbound acme support cases"


# --- Fakes ---


def _process(
    team_id: uuid.UUID | None = None,
    *,
    team_description: str | None = None,
    description_origin: DescriptionOrigin = DescriptionOrigin.AUTO,
) -> Process:
    then = datetime.now(UTC) - timedelta(days=1)
    return Process(
        team_id=team_id or uuid.uuid4(),
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=then,
        updated_at=then,
        entry_point=AgentRef(name="@Manager", role="Manager"),
        agent_cards=[AgentCardRef(role="Manager", card_hash="0" * 64)],
        team_description=team_description,
        description_origin=description_origin,
    )


def _team_side(process: Process | None = None) -> tuple[TeamManager, FakeEventStore]:
    store = FakeEventStore()
    if process is not None:
        store.save_team(process)
    return TeamManager(actor_system=MagicMock(), event_store=store), store


def _model_cfg() -> ModelConfig:
    return ModelConfig(provider="openai-chat", model="gpt-4o-mini")


def _generator(
    manager: TeamManager,
    *,
    executor: Executor | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> TeamDescriptionGenerator:
    return TeamDescriptionGenerator(
        _model_cfg(),
        manager,
        max_attempts=max_attempts,
        executor=executor if executor is not None else InlineExecutor(),
    )


def _inner_handle(team_id: uuid.UUID) -> MagicMock:
    inner = MagicMock(spec=TeamHandle)
    inner.team_id = team_id
    return inner


def _install_model(monkeypatch: pytest.MonkeyPatch, model: RecordingModel) -> MagicMock:
    """Patch ``create_model`` by name on the module; the mock counts constructions."""
    factory = MagicMock(return_value=model.build())
    monkeypatch.setattr(description, "create_model", factory)
    return factory


def _notifications(inner: MagicMock) -> list[NotificationMessage]:
    return [call.args[0] for call in inner.emitMessage.call_args_list]


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> tuple[RecordingModel, MagicMock]:
    """A recording model installed as the module's ``create_model``."""
    model = RecordingModel()
    return model, _install_model(monkeypatch, model)


# --- AC 2: from_settings ---


class TestFromSettings:
    """AC2: off unless both settings are set, said once; nothing built at wiring."""

    def test_both_unset_returns_none_and_names_both_variables_once(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        _model, factory = wired
        manager, _store = _team_side()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = TeamDescriptionGenerator.from_settings(WorkerSettings(), manager)

        assert result is None
        infos = _records(caplog, logging.INFO)
        assert len(infos) == 1
        assert "AKGENTIC_WORKER_DESCRIPTION_PROVIDER" in infos[0].getMessage()
        assert "AKGENTIC_WORKER_DESCRIPTION_MODEL" in infos[0].getMessage()
        factory.assert_not_called()

    @pytest.mark.parametrize(
        "settings",
        [
            WorkerSettings(description_provider="openai-chat"),
            WorkerSettings(description_model="gpt-4o-mini"),
        ],
        ids=["provider-only", "model-only"],
    )
    def test_one_of_two_is_off(
        self, settings: WorkerSettings, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager, _store = _team_side()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = TeamDescriptionGenerator.from_settings(settings, manager)

        assert result is None
        assert len(_records(caplog, logging.INFO)) == 1

    def test_both_set_returns_generator_and_names_provider_and_model_once(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        _model, factory = wired
        manager, _store = _team_side()
        settings = WorkerSettings(
            description_provider="openai-chat", description_model="gpt-4o-mini"
        )
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = TeamDescriptionGenerator.from_settings(settings, manager)

        assert isinstance(result, TeamDescriptionGenerator)
        assert result.max_attempts == DEFAULT_MAX_ATTEMPTS
        infos = _records(caplog, logging.INFO)
        assert len(infos) == 1
        assert "openai-chat" in infos[0].getMessage()
        assert "gpt-4o-mini" in infos[0].getMessage()
        # Lazy: no Model and no Agent until the first generation.
        factory.assert_not_called()
        assert result._agent is None

    def test_unknown_provider_fails_at_wiring(self) -> None:
        manager, _store = _team_side()
        settings = WorkerSettings(description_provider="no-such-provider", description_model="m")
        with pytest.raises(ValidationError, match="provider"):
            TeamDescriptionGenerator.from_settings(settings, manager)


# --- AC 3 (wrap half): wrap ---


class TestWrap:
    """AC3: ``wrap`` yields a describing handle and never double-wraps."""

    def test_wrap_returns_describing_handle_with_the_inner_team_id(self) -> None:
        manager, _store = _team_side()
        team_id = uuid.uuid4()
        wrapped = _generator(manager).wrap(_inner_handle(team_id))

        assert isinstance(wrapped, DescribingTeamHandle)
        assert isinstance(wrapped, TeamHandle)
        assert wrapped.team_id == team_id

    def test_wrap_is_idempotent(self) -> None:
        manager, _store = _team_side()
        generator = _generator(manager)
        wrapped = generator.wrap(_inner_handle(uuid.uuid4()))

        assert generator.wrap(wrapped) is wrapped


# --- AC 9: normalise_description ---


class TestNormaliseDescription:
    """AC9: the output contract, one function."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Triage inbound acme support cases.", "Triage inbound acme support cases"),
            ('"Triage inbound acme support cases"', "Triage inbound acme support cases"),
            ("'Triage inbound acme support cases'", "Triage inbound acme support cases"),
            ("“Triage inbound acme support cases”", "Triage inbound acme support cases"),
            ('"Triage inbound acme support cases."', "Triage inbound acme support cases"),
            ("Triage inbound acme support cases...", "Triage inbound acme support cases"),
            ("  Triage inbound acme support cases  \n", "Triage inbound acme support cases"),
            ("\n\nFirst line\nSecond line", "First line"),
            ('""Nested""', '"Nested"'),
            ("", None),
            ("   \n  ", None),
            ('""', None),
            ("...", None),
        ],
    )
    def test_table(self, raw: str, expected: str | None) -> None:
        assert normalise_description(raw) == expected

    def test_over_length_output_is_truncated_not_rejected(self) -> None:
        raw = "x" * (MAX_TEAM_DESCRIPTION_LENGTH + 100)

        result = normalise_description(raw)

        assert result == raw[:MAX_TEAM_DESCRIPTION_LENGTH]
        assert result is not None
        assert len(result) == MAX_TEAM_DESCRIPTION_LENGTH


# --- Happy path, AC 4 effects at the unit level, AC 10's winning branch ---


class TestHappyPath:
    """One model call, one AUTO write, one notification through the inner handle."""

    def test_send_generates_writes_auto_and_notifies_once(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, factory = wired
        process = _process()
        manager, store = _team_side(process)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)

        inner.send.assert_called_once_with(MESSAGE)
        assert model.prompts == [MESSAGE]
        assert factory.call_count == 1
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED
        assert persisted.description_origin is DescriptionOrigin.AUTO
        notifications = _notifications(inner)
        assert len(notifications) == 1
        assert isinstance(notifications[0], NotificationMessage)
        assert notifications[0].content_type == DESCRIPTION_CONTENT_TYPE
        assert notifications[0].content == EXPECTED
        assert notifications[0].team_id is None  # the orchestrator stamps it
        infos = _records(caplog, logging.INFO)
        assert len(infos) == 1
        assert str(process.team_id) in infos[0].getMessage()
        assert EXPECTED not in infos[0].getMessage()
        assert not _records(caplog, logging.WARNING)

    def test_notification_is_not_older_than_the_write_it_reports(
        self, monkeypatch: pytest.MonkeyPatch, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        """The frontend drops a description frame older than the team's ``updated_at``.

        The write is slowed so a notification built before it is measurably
        older than the stamp the write sets; the order is what this pins.
        """
        process = _process()
        manager, store = _team_side(process)
        real_update = manager.update_description

        def slow_update(*args: Any, **kwargs: Any) -> Process | None:
            time.sleep(0.005)
            return real_update(*args, **kwargs)

        monkeypatch.setattr(manager, "update_description", slow_update)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        handle.send(MESSAGE)

        written = store.load_team(process.team_id)
        assert written is not None
        assert written.team_description == EXPECTED
        [notification] = _notifications(inner)
        assert notification.timestamp is not None
        assert notification.timestamp >= written.updated_at

    def test_send_to_triggers_too(self, wired: tuple[RecordingModel, MagicMock]) -> None:
        model, _factory = wired
        process = _process()
        manager, store = _team_side(process)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        handle.send_to("@Manager", MESSAGE)

        inner.send_to.assert_called_once_with("@Manager", MESSAGE)
        assert model.prompts == [MESSAGE]
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED

    def test_agent_is_built_once_and_reused(self, wired: tuple[RecordingModel, MagicMock]) -> None:
        model, factory = wired
        first, second = _process(), _process()
        manager, store = _team_side(first)
        store.save_team(second)
        generator = _generator(manager)

        generator.wrap(_inner_handle(first.team_id)).send(MESSAGE)
        generator.wrap(_inner_handle(second.team_id)).send(MESSAGE)

        assert model.calls == 2
        assert factory.call_count == 1

    def test_after_success_a_second_send_does_nothing_more(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        get_team = MagicMock(wraps=manager.get_team)
        manager.get_team = get_team  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        handle.send(MESSAGE)
        handle.send("a follow-up")

        assert inner.send.call_count == 2
        assert model.calls == 1
        assert get_team.call_count == 1
        assert len(_notifications(inner)) == 1


# --- AC 5: the common case makes no model call ---


class TestCommonCase:
    """AC5: a described or user-cleared team costs one store read, then nothing."""

    @pytest.mark.parametrize(
        "process",
        [
            _process(team_description="Already described"),
            _process(team_description=None, description_origin=DescriptionOrigin.USER),
            _process(team_description="Mine", description_origin=DescriptionOrigin.USER),
        ],
        ids=["auto-described", "user-cleared", "user-owned"],
    )
    def test_no_model_call_and_settled_after_one_read(
        self,
        process: Process,
        caplog: pytest.LogCaptureFixture,
        wired: tuple[RecordingModel, MagicMock],
    ) -> None:
        model, factory = wired
        manager, store = _team_side(process)
        get_team = MagicMock(wraps=manager.get_team)
        update = MagicMock(wraps=manager.update_description)
        manager.get_team = get_team  # type: ignore[method-assign]
        manager.update_description = update  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)
            handle.send("a second message")

        assert inner.send.call_count == 2
        assert model.calls == 0
        assert factory.call_count == 0
        update.assert_not_called()
        assert get_team.call_count == 1
        inner.emitMessage.assert_not_called()
        assert store.load_team(process.team_id) == process
        assert not _records(caplog, logging.WARNING)


# --- AC 6: never on the request path, never raising ---


class TestNeverOnRequestPath:
    """AC6: send returns before the model runs; any failure is one WARNING and silence."""

    def test_send_returns_while_the_model_call_is_still_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = threading.Event()
        model = RecordingModel(gate=gate)
        _install_model(monkeypatch, model)
        process = _process()
        manager, store = _team_side(process)
        threads: list[str] = []
        real_get_team = manager.get_team

        def get_team_recording_thread(team_id: uuid.UUID) -> Process | None:
            threads.append(threading.current_thread().name)
            return real_get_team(team_id)

        monkeypatch.setattr(manager, "get_team", get_team_recording_thread)
        generator = TeamDescriptionGenerator(_model_cfg(), manager)  # the real pool
        futures: list[Future[bool]] = []
        real_schedule = generator.schedule

        def capturing(*args: Any, **kwargs: Any) -> Future[bool]:
            future = real_schedule(*args, **kwargs)
            futures.append(future)
            return future

        monkeypatch.setattr(generator, "schedule", capturing)
        inner = _inner_handle(process.team_id)
        handle = generator.wrap(inner)
        try:
            handle.send(MESSAGE)

            # Back on the caller's thread with the model still blocked.
            inner.send.assert_called_once_with(MESSAGE)
            assert len(futures) == 1
            assert not futures[0].done()
            before = store.load_team(process.team_id)
            assert before is not None
            assert before.team_description is None
            inner.emitMessage.assert_not_called()

            gate.set()
            assert futures[0].result(timeout=5) is True
        finally:
            gate.set()
            generator.close()

        after = store.load_team(process.team_id)
        assert after is not None
        assert after.team_description == EXPECTED
        assert len(_notifications(inner)) == 1
        # The unit ran on the generator's own thread, not the caller's.
        assert len(threads) == 1
        assert threads[0].startswith("team-description")
        assert threads[0] != threading.current_thread().name

    def test_raising_model_leaves_none_and_logs_one_warning_naming_the_team(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install_model(monkeypatch, model)
        process = _process()
        manager, store = _team_side(process)
        update = MagicMock(wraps=manager.update_description)
        manager.update_description = update  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)  # must not raise

        assert model.calls == 1
        update.assert_not_called()
        inner.emitMessage.assert_not_called()
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description is None
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()
        assert MESSAGE not in warnings[0].getMessage()
        assert warnings[0].exc_info is not None

    def test_empty_model_output_is_a_counted_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = RecordingModel(reply='"."')
        _install_model(monkeypatch, model)
        process = _process()
        manager, store = _team_side(process)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager, max_attempts=1).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)
            handle.send(MESSAGE)

        assert model.calls == 1  # counted: the cap of one was consumed
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description is None
        inner.emitMessage.assert_not_called()
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()

    def test_raising_get_team_is_handled_the_same_way(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        manager.get_team = MagicMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)  # must not raise

        assert model.calls == 0
        inner.emitMessage.assert_not_called()
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()

    def test_raising_update_description_is_handled_the_same_way(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, store = _team_side(process)
        manager.update_description = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("store down")
        )
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)  # must not raise

        assert model.calls == 1
        inner.emitMessage.assert_not_called()
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description is None
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()

    def test_a_scheduling_failure_does_not_escape_send(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        process = _process()
        manager, _store = _team_side(process)
        generator = _generator(manager)
        generator.close()  # a shut executor refuses new work
        inner = _inner_handle(process.team_id)
        handle = generator.wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)  # must not raise

        inner.send.assert_called_once_with(MESSAGE)
        assert len(_records(caplog, logging.WARNING)) == 1


# --- AC 7: attempt cap ---


class TestAttemptCap:
    """AC7: per team, in memory, on the handle; a fresh handle starts over."""

    def test_three_failures_then_the_fourth_send_makes_no_call(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install_model(monkeypatch, model)
        process = _process()
        manager, _store = _team_side(process)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            for _ in range(3):
                handle.send(MESSAGE)
            assert model.calls == 3
            caplog.clear()
            handle.send(MESSAGE)

        assert model.calls == 3
        assert inner.send.call_count == 4
        debugs = [r for r in _records(caplog, logging.DEBUG) if "cap" in r.getMessage()]
        assert len(debugs) == 1
        assert str(process.team_id) in debugs[0].getMessage()
        assert not _records(caplog, logging.WARNING)

    def test_the_cap_is_a_constructor_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install_model(monkeypatch, model)
        process = _process()
        manager, _store = _team_side(process)
        handle = _generator(manager, max_attempts=1).wrap(_inner_handle(process.team_id))

        handle.send(MESSAGE)
        handle.send(MESSAGE)

        assert model.calls == 1

    def test_a_fresh_handle_for_the_same_team_restarts_the_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install_model(monkeypatch, model)
        process = _process()
        manager, _store = _team_side(process)
        generator = _generator(manager)
        exhausted = generator.wrap(_inner_handle(process.team_id))
        for _ in range(4):
            exhausted.send(MESSAGE)
        assert model.calls == 3

        # What resume does: a new handle stored for the same team_id.
        fresh = generator.wrap(_inner_handle(process.team_id))
        fresh.send(MESSAGE)

        assert model.calls == 4

    def test_a_send_during_an_in_flight_generation_is_dropped_and_not_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Five sends, one of them mid-flight: three model calls, then the cap.

        Had the in-flight send consumed an attempt, the fourth send would already
        have met the cap and the total would be two.
        """
        gate = threading.Event()
        model = RecordingModel(error=RuntimeError("provider down"), gate=gate)
        _install_model(monkeypatch, model)
        process = _process()
        manager, _store = _team_side(process)
        generator = TeamDescriptionGenerator(_model_cfg(), manager)  # the real pool
        futures: list[Future[bool]] = []
        real_schedule = generator.schedule

        def capturing(*args: Any, **kwargs: Any) -> Future[bool]:
            future = real_schedule(*args, **kwargs)
            futures.append(future)
            return future

        monkeypatch.setattr(generator, "schedule", capturing)
        inner = _inner_handle(process.team_id)
        handle = generator.wrap(inner)
        try:
            handle.send(MESSAGE)  # attempt 1, blocked on the gate
            deadline = time.monotonic() + 5
            while not model.prompts and time.monotonic() < deadline:
                time.sleep(0.005)
            assert model.calls == 1
            handle.send(MESSAGE)  # in flight: dropped, not counted
            assert len(futures) == 1
            gate.set()
            assert futures[0].result(timeout=5) is False

            handle.send(MESSAGE)  # attempt 2
            assert futures[1].result(timeout=5) is False
            handle.send(MESSAGE)  # attempt 3
            assert futures[2].result(timeout=5) is False
            handle.send(MESSAGE)  # cap reached: nothing scheduled
        finally:
            gate.set()
            generator.close()

        assert model.calls == 3
        assert len(futures) == 3
        assert inner.send.call_count == 5


# --- AC 8: the input is that message alone ---


class TestInput:
    """AC8: the prompt carries the triggering message's text and nothing else."""

    def test_message_payload_contributes_its_content(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, store = _team_side(process)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)
        payload = UserMessage(content=MESSAGE)

        handle.send(payload)

        inner.send.assert_called_once_with(payload)
        assert model.prompts == [MESSAGE]
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED

    @pytest.mark.parametrize(
        "payload",
        [UserMessage(content=""), UserMessage(content="   \n"), Message()],
        ids=["empty-content", "blank-content", "no-content"],
    )
    def test_message_without_usable_text_schedules_nothing_and_is_not_counted(
        self, payload: Message, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        get_team = MagicMock(wraps=manager.get_team)
        manager.get_team = get_team  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager, max_attempts=1).wrap(inner)

        handle.send(payload)
        assert model.calls == 0
        get_team.assert_not_called()

        # Not counted: the single attempt is still available to a real message.
        handle.send(MESSAGE)
        assert model.prompts == [MESSAGE]

    def test_a_blank_string_schedules_nothing(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        handle.send("   ")

        inner.send.assert_called_once_with("   ")
        assert model.calls == 0


# --- AC 10: the store decides the race ---


class TestRace:
    """AC10: the notification follows the store's answer, not the worker's intent."""

    def test_user_edit_between_read_and_write_is_reported_not_overwritten(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        owned = _process(team_description="Mine", description_origin=DescriptionOrigin.USER)
        manager, store = _team_side(owned)
        # The read saw a fresh AUTO record; by the time of the write the user has
        # taken the field. The fake store refuses the AUTO write and answers
        # what it holds.
        manager.get_team = MagicMock(  # type: ignore[method-assign]
            return_value=_process(owned.team_id)
        )
        inner = _inner_handle(owned.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)
            handle.send(MESSAGE)

        assert model.calls == 1  # settled after the loss: no second attempt
        inner.emitMessage.assert_not_called()
        assert store.load_team(owned.team_id) == owned
        assert not _records(caplog, logging.WARNING)
        assert not _records(caplog, logging.INFO)
        lost = [r for r in _records(caplog, logging.DEBUG) if "user edit" in r.getMessage()]
        assert len(lost) == 1
        assert str(owned.team_id) in lost[0].getMessage()

    def test_user_cleared_between_read_and_write_is_a_loss_too(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        cleared = _process(team_description=None, description_origin=DescriptionOrigin.USER)
        manager, store = _team_side(cleared)
        manager.get_team = MagicMock(  # type: ignore[method-assign]
            return_value=_process(cleared.team_id)
        )
        inner = _inner_handle(cleared.team_id)

        _generator(manager).wrap(inner).send(MESSAGE)

        inner.emitMessage.assert_not_called()
        assert store.load_team(cleared.team_id) == cleared

    def test_team_vanished_before_the_write_emits_nothing(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side()  # the store never held it
        manager.get_team = MagicMock(return_value=process)  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)

        assert model.calls == 1
        inner.emitMessage.assert_not_called()
        assert not _records(caplog, logging.WARNING)
        gone = [r for r in _records(caplog, logging.DEBUG) if "no team" in r.getMessage()]
        assert len(gone) == 1

    def test_team_vanished_before_the_read_makes_no_model_call(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        manager, _store = _team_side()
        inner = _inner_handle(uuid.uuid4())
        handle = _generator(manager).wrap(inner)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            handle.send(MESSAGE)
            handle.send(MESSAGE)

        assert model.calls == 0
        inner.emitMessage.assert_not_called()
        assert not _records(caplog, logging.WARNING)


# --- AC 11: which calls trigger, and pure delegation ---


class TestDelegation:
    """AC11: every call forwards unchanged; only send and send_to trigger."""

    def test_non_triggering_methods_forward_and_never_generate(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        get_team = MagicMock(wraps=manager.get_team)
        manager.get_team = get_team  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        handle = _generator(manager).wrap(inner)
        message = UserMessage(content=MESSAGE)
        subscriber = MagicMock()

        handle.send_from_to("@Manager", "@Human", MESSAGE)
        inner.send_from_to.assert_called_once_with("@Manager", "@Human", MESSAGE)
        assert model.calls == 0

        handle.emitMessage(message)
        inner.emitMessage.assert_called_once_with(message)
        assert model.calls == 0

        handle.process_human_input("yes", message)
        inner.process_human_input.assert_called_once_with("yes", message)
        assert model.calls == 0

        handle.subscribe(subscriber)
        inner.subscribe.assert_called_once_with(subscriber)
        assert model.calls == 0

        handle.unsubscribe(subscriber)
        inner.unsubscribe.assert_called_once_with(subscriber)
        assert model.calls == 0

        get_team.assert_not_called()

    def test_inner_send_raising_propagates_and_schedules_nothing(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        get_team = MagicMock(wraps=manager.get_team)
        manager.get_team = get_team  # type: ignore[method-assign]
        inner = _inner_handle(process.team_id)
        inner.send.side_effect = ValueError("team is not running")
        handle = _generator(manager).wrap(inner)

        with pytest.raises(ValueError, match="not running"):
            handle.send(MESSAGE)

        assert model.calls == 0
        get_team.assert_not_called()

    def test_inner_send_to_raising_propagates_and_schedules_nothing(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        manager, _store = _team_side(process)
        inner = _inner_handle(process.team_id)
        inner.send_to.side_effect = ValueError("Agent '@Nobody' not found")
        handle = _generator(manager).wrap(inner)

        with pytest.raises(ValueError, match="not found"):
            handle.send_to("@Nobody", MESSAGE)

        assert model.calls == 0

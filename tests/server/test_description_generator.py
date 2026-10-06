"""Story 80.3: the server-side team description generator, at the unit level.

The store is the stateful ``FakeEventStore`` from ``tests/test_deps.py`` — the
one with the port's AUTO-loses-to-USER semantics — so the race specs can
genuinely go red; the emit path is a ``MagicMock`` standing in for
``TeamService.emit_message``. Every model call is a ``FunctionModel`` installed
by patching ``create_model`` on the module under test; nothing here reaches a
provider. Deterministic specs run the generator on an inline executor; the
asynchrony specs use the real pool with a model that blocks on an ``Event``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Executor
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.llm import ModelConfig
from akgentic.team.models import (
    AgentCardRef,
    AgentRef,
    DescriptionOrigin,
    Process,
    TeamStatus,
)
from pydantic import ValidationError

from akgentic.infra.server import description
from akgentic.infra.server.description import (
    DESCRIPTION_CONTENT_TYPE,
    TeamDescriptionGenerator,
    normalise_description,
)
from akgentic.infra.server.models import MAX_TEAM_DESCRIPTION_LENGTH
from akgentic.infra.server.settings import ServerSettings
from tests.fixtures.description import InlineExecutor, RecordingModel
from tests.test_deps import FakeEventStore

LOGGER = "akgentic.infra.server.description"
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


def _store(*processes: Process) -> FakeEventStore:
    store = FakeEventStore()
    for process in processes:
        store.save_team(process)
    return store


def _model_cfg() -> ModelConfig:
    return ModelConfig(provider="openai-chat", model="gpt-4o-mini")


def _generator(
    store: FakeEventStore,
    emit: MagicMock,
    *,
    executor: Executor | None = None,
) -> TeamDescriptionGenerator:
    return TeamDescriptionGenerator(
        _model_cfg(),
        store,
        emit,
        executor=executor if executor is not None else InlineExecutor(),
    )


def _install_model(monkeypatch: pytest.MonkeyPatch, model: RecordingModel) -> MagicMock:
    """Patch ``create_model`` by name on the module; the mock counts constructions."""
    factory = MagicMock(return_value=model.build())
    monkeypatch.setattr(description, "create_model", factory)
    return factory


def _spy_update(store: FakeEventStore) -> MagicMock:
    """Wrap the fake store's conditional write in a spy that still performs it."""
    update = MagicMock(wraps=store.update_team_description)
    store.update_team_description = update  # type: ignore[method-assign]
    return update


def _notifications(emit: MagicMock) -> list[tuple[uuid.UUID, NotificationMessage]]:
    return [(call.args[0], call.args[1]) for call in emit.call_args_list]


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
        settings = ServerSettings(description_provider=None, description_model=None)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = TeamDescriptionGenerator.from_settings(settings, _store(), MagicMock())

        assert result is None
        infos = _records(caplog, logging.INFO)
        assert len(infos) == 1
        assert "AKGENTIC_DESCRIPTION_PROVIDER" in infos[0].getMessage()
        assert "AKGENTIC_DESCRIPTION_MODEL" in infos[0].getMessage()
        factory.assert_not_called()

    @pytest.mark.parametrize(
        "settings",
        [
            ServerSettings(description_provider="", description_model="gpt-4o-mini"),
            ServerSettings(description_provider="openai-chat", description_model=""),
        ],
        ids=["empty-provider", "empty-model"],
    )
    def test_an_empty_value_turns_it_off(self, settings: ServerSettings) -> None:
        assert TeamDescriptionGenerator.from_settings(settings, _store(), MagicMock()) is None

    @pytest.mark.parametrize(
        "settings",
        [
            ServerSettings(description_provider="openai-chat", description_model=None),
            ServerSettings(description_provider=None, description_model="gpt-4o-mini"),
        ],
        ids=["provider-only", "model-only"],
    )
    def test_one_of_two_is_off(
        self, settings: ServerSettings, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = TeamDescriptionGenerator.from_settings(settings, _store(), MagicMock())

        assert result is None
        assert len(_records(caplog, logging.INFO)) == 1

    def test_both_set_returns_generator_and_names_provider_and_model_once(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        _model, factory = wired
        settings = ServerSettings(
            description_provider="openai-chat", description_model="gpt-4o-mini"
        )
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = TeamDescriptionGenerator.from_settings(settings, _store(), MagicMock())

        assert isinstance(result, TeamDescriptionGenerator)
        infos = _records(caplog, logging.INFO)
        assert len(infos) == 1
        assert "openai-chat" in infos[0].getMessage()
        assert "gpt-4o-mini" in infos[0].getMessage()
        # Lazy: no Model and no Agent until the first generation.
        factory.assert_not_called()
        assert result._agent is None

    def test_unknown_provider_fails_at_construction(self) -> None:
        settings = ServerSettings(description_provider="no-such-provider", description_model="m")
        with pytest.raises(ValidationError, match="provider"):
            TeamDescriptionGenerator.from_settings(settings, _store(), MagicMock())


# --- AC 10: normalise_description ---


class TestNormaliseDescription:
    """AC10: the output contract, one function."""

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


# --- AC 5: the trigger condition ---


class TestTrigger:
    """AC5: one comparison on the record in hand; no model call, no store call otherwise."""

    @pytest.mark.parametrize(
        "process",
        [
            _process(team_description="Already described"),
            _process(team_description=None, description_origin=DescriptionOrigin.USER),
            _process(team_description="Mine", description_origin=DescriptionOrigin.USER),
        ],
        ids=["auto-described", "user-cleared", "user-owned"],
    )
    def test_settled_record_schedules_nothing(
        self,
        process: Process,
        caplog: pytest.LogCaptureFixture,
        wired: tuple[RecordingModel, MagicMock],
    ) -> None:
        model, factory = wired
        store = _store(process)
        update = _spy_update(store)
        emit = MagicMock()
        generator = _generator(store, emit)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            generator.maybe_generate(process, MESSAGE)

        assert model.calls == 0
        assert factory.call_count == 0
        update.assert_not_called()
        emit.assert_not_called()
        assert store.load_team(process.team_id) == process
        assert not _records(caplog, logging.WARNING)

    @pytest.mark.parametrize(
        "payload",
        ["   ", UserMessage(content=""), UserMessage(content="   \n"), Message()],
        ids=["blank-string", "empty-content", "blank-content", "no-content"],
    )
    def test_delivery_without_usable_text_schedules_nothing(
        self, payload: str | Message, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        store = _store(process)
        update = _spy_update(store)
        emit = MagicMock()

        _generator(store, emit).maybe_generate(process, payload)

        assert model.calls == 0
        update.assert_not_called()
        emit.assert_not_called()

    def test_message_payload_contributes_its_content(
        self, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        store = _store(process)

        _generator(store, MagicMock()).maybe_generate(process, UserMessage(content=MESSAGE))

        assert model.prompts == [MESSAGE]
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED


# --- AC 7, 8: the happy path and the write it makes ---


class TestHappyPath:
    """AC7/AC8: one model call, one AUTO write, one notification built after the write."""

    def test_generates_writes_auto_and_notifies_once(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, factory = wired
        process = _process()
        store = _store(process)
        update = _spy_update(store)
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(process, MESSAGE)

        assert model.prompts == [MESSAGE]
        assert factory.call_count == 1
        update.assert_called_once_with(process.team_id, EXPECTED, DescriptionOrigin.AUTO)
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED
        assert persisted.description_origin is DescriptionOrigin.AUTO
        [(emitted_id, notification)] = _notifications(emit)
        assert emitted_id == process.team_id
        assert isinstance(notification, NotificationMessage)
        assert notification.content_type == DESCRIPTION_CONTENT_TYPE
        assert notification.content == EXPECTED
        assert notification.team_id is None  # the orchestrator stamps it
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
        store = _store(process)
        real_update = store.update_team_description

        def slow_update(*args: Any, **kwargs: Any) -> Process | None:
            time.sleep(0.005)
            return real_update(*args, **kwargs)

        monkeypatch.setattr(store, "update_team_description", slow_update)
        emit = MagicMock()

        _generator(store, emit).maybe_generate(process, MESSAGE)

        written = store.load_team(process.team_id)
        assert written is not None
        assert written.team_description == EXPECTED
        [(_team_id, notification)] = _notifications(emit)
        assert notification.timestamp is not None
        assert notification.timestamp >= written.updated_at

    def test_agent_is_built_once_and_reused(self, wired: tuple[RecordingModel, MagicMock]) -> None:
        model, factory = wired
        first, second = _process(), _process()
        store = _store(first, second)
        generator = _generator(store, MagicMock())

        generator.maybe_generate(first, MESSAGE)
        generator.maybe_generate(second, MESSAGE)

        assert model.calls == 2
        assert factory.call_count == 1


# --- AC 6, 9: never raising, never capped ---


class TestNeverRaises:
    """AC6: any failure is one WARNING naming the team, never the content, and silence."""

    def test_raising_model_leaves_none_and_logs_one_warning_naming_the_team(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install_model(monkeypatch, model)
        process = _process()
        store = _store(process)
        update = _spy_update(store)
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(process, MESSAGE)  # must not raise

        assert model.calls == 1
        update.assert_not_called()
        emit.assert_not_called()
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description is None
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()
        assert MESSAGE not in warnings[0].getMessage()
        assert warnings[0].exc_info is not None

    def test_empty_model_output_is_one_warning_and_no_write(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = RecordingModel(reply='"."')
        _install_model(monkeypatch, model)
        process = _process()
        store = _store(process)
        update = _spy_update(store)
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(process, MESSAGE)

        assert model.calls == 1
        update.assert_not_called()
        emit.assert_not_called()
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()

    def test_raising_store_write_is_one_warning_and_no_emit(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        store = _store(process)
        store.update_team_description = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("store down")
        )
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(process, MESSAGE)  # must not raise

        assert model.calls == 1
        emit.assert_not_called()
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()

    def test_raising_emit_is_one_warning_and_the_text_is_persisted(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        process = _process()
        store = _store(process)
        emit = MagicMock(side_effect=RuntimeError("team vanished"))

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(process, MESSAGE)  # must not raise

        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description == EXPECTED
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(process.team_id) in warnings[0].getMessage()

    def test_a_scheduling_failure_does_not_escape(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        store = _store(process)
        generator = _generator(store, MagicMock())
        # A shut executor refuses new work; nothing on the generator shuts it
        # (no lifespan hook yet), so the spec reaches the pool directly.
        generator._executor.shutdown(wait=False, cancel_futures=True)  # noqa: SLF001

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            generator.maybe_generate(process, MESSAGE)  # must not raise

        assert model.calls == 0
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert "scheduling failed" in warnings[0].getMessage()
        assert str(process.team_id) in warnings[0].getMessage()


class TestNoCap:
    """AC9: no per-team counter, no latch — every eligible message costs one attempt."""

    def test_four_failures_are_four_model_calls_and_four_warnings(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = RecordingModel(error=RuntimeError("provider down"))
        _install_model(monkeypatch, model)
        process = _process()
        store = _store(process)
        emit = MagicMock()
        generator = _generator(store, emit)

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            for _ in range(4):
                generator.maybe_generate(process, MESSAGE)

        assert model.calls == 4
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 4
        assert all(str(process.team_id) in w.getMessage() for w in warnings)
        assert all(MESSAGE not in w.getMessage() for w in warnings)
        emit.assert_not_called()
        persisted = store.load_team(process.team_id)
        assert persisted is not None
        assert persisted.team_description is None


# --- AC 8: the store decides the race ---


class TestRace:
    """AC8: the notification follows the store's answer, not the request-path check."""

    def test_user_edit_between_check_and_write_is_reported_not_overwritten(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        owned = _process(team_description="Mine", description_origin=DescriptionOrigin.USER)
        store = _store(owned)
        # The request path saw a fresh AUTO record; by the time of the write the
        # user has taken the field. The fake store refuses the AUTO write and
        # answers what it holds.
        seen_on_request_path = _process(owned.team_id)
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(seen_on_request_path, MESSAGE)

        assert model.calls == 1
        emit.assert_not_called()
        assert store.load_team(owned.team_id) == owned
        assert not _records(caplog, logging.WARNING)
        assert not _records(caplog, logging.INFO)
        lost = [r for r in _records(caplog, logging.DEBUG) if "user edit" in r.getMessage()]
        assert len(lost) == 1
        assert str(owned.team_id) in lost[0].getMessage()

    def test_user_cleared_between_check_and_write_is_a_loss_too(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        cleared = _process(team_description=None, description_origin=DescriptionOrigin.USER)
        store = _store(cleared)
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(_process(cleared.team_id), MESSAGE)

        emit.assert_not_called()
        assert store.load_team(cleared.team_id) == cleared
        assert not _records(caplog, logging.WARNING)
        lost = [r for r in _records(caplog, logging.DEBUG) if "user edit" in r.getMessage()]
        assert len(lost) == 1

    def test_team_vanished_before_the_write_emits_nothing(
        self, caplog: pytest.LogCaptureFixture, wired: tuple[RecordingModel, MagicMock]
    ) -> None:
        model, _factory = wired
        process = _process()
        store = _store()  # the store never held it
        emit = MagicMock()

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _generator(store, emit).maybe_generate(process, MESSAGE)

        assert model.calls == 1
        emit.assert_not_called()
        assert not _records(caplog, logging.WARNING)
        gone = [r for r in _records(caplog, logging.DEBUG) if "no team" in r.getMessage()]
        assert len(gone) == 1


# --- AC 6: never on the request path ---


class TestNeverOnRequestPath:
    """AC6: ``maybe_generate`` returns while the model call is still pending."""

    def test_returns_while_the_model_call_is_still_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = threading.Event()
        model = RecordingModel(gate=gate)
        _install_model(monkeypatch, model)
        process = _process()
        store = _store(process)
        threads: list[str] = []
        real_update = store.update_team_description

        def update_recording_thread(*args: Any, **kwargs: Any) -> Process | None:
            # pydantic-ai runs the model function on an AnyIO worker thread, so
            # the generator's own thread is observed at the write, not the model.
            threads.append(threading.current_thread().name)
            return real_update(*args, **kwargs)

        monkeypatch.setattr(store, "update_team_description", update_recording_thread)
        emit = MagicMock()
        generator = TeamDescriptionGenerator(_model_cfg(), store, emit)  # the real pool
        try:
            generator.maybe_generate(process, MESSAGE)

            # Back on the caller's thread with the model still blocked.
            deadline = time.monotonic() + 5
            while not model.prompts and time.monotonic() < deadline:
                time.sleep(0.005)
            assert model.calls == 1
            before = store.load_team(process.team_id)
            assert before is not None
            assert before.team_description is None
            emit.assert_not_called()

            gate.set()
            deadline = time.monotonic() + 5
            while not emit.call_args_list and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            gate.set()
            generator._executor.shutdown(wait=False, cancel_futures=True)  # noqa: SLF001

        after = store.load_team(process.team_id)
        assert after is not None
        assert after.team_description == EXPECTED
        assert len(_notifications(emit)) == 1
        # The unit ran on the generator's own thread, not the caller's.
        assert len(threads) == 1
        assert threads[0].startswith("team-description")
        assert threads[0] != threading.current_thread().name

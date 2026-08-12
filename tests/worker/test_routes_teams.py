"""Shared worker team-route tests — create-team populates the runtime cache.

These tests pin ADR-001 Story 1: the shared ``create_team`` handler must wrap
the new runtime in a ``LocalTeamHandle`` and ``runtime_cache.store`` it before
returning, so that every other route on the same router — all of which resolve
the live handle via ``runtime_cache.get`` — can reach the freshly created team.
Without the store, a follow-up ``POST /teams/{id}/message`` 404s on a cache miss.

The handlers are exercised directly (the existing worker-route test style) with
a real ``LocalRuntimeCache`` and lightweight stubs for the team manager and
worker handle, so the assertion is on the genuine cache interaction.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.core.messages.message import UserMessage
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team import derive_metadata_indexes, make_index_entry
from akgentic.team.models import Process, TeamCard, TeamStatus
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from akgentic.infra.adapters.community.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.server.models import EmitMessageRequest, SendMessageRequest
from akgentic.infra.worker.routes.teams import (
    WorkerCreateTeamRequest,
    _process_to_response,
    create_team,
    delete_team,
    emit_notification,
    router,
    send_message,
    stop_team,
)
from akgentic.infra.worker.state_keys import SERVICES
from tests.fixtures.team_metadata import (
    ACME_METADATA_TYPE,
    AcmeCaseMetadata,
    make_metadata_body,
)

_TEAM_CARD_PAYLOAD = {
    "name": "Test Team",
    "description": "worker-route test team",
    "entry_point": {
        "card": {
            "role": "Human",
            "description": "Human user interface",
            "skills": [],
            "agent_class": "akgentic.core.agent.Akgent",
            "config": {"name": "@Human", "role": "Human"},
            "routes_to": ["@Manager"],
        },
        "headcount": 1,
        "members": [],
    },
    "members": [
        {
            "card": {
                "role": "Manager",
                "description": "Test manager agent",
                "skills": ["coordination"],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "@Manager", "role": "Manager"},
                "routes_to": [],
            },
            "headcount": 1,
            "members": [],
        },
    ],
    "message_types": [{"__type__": "akgentic.core.messages.UserMessage"}],
    "agent_profiles": [],
}


class _FakeRuntime:
    """Stand-in for ``TeamRuntime`` exposing only what ``LocalTeamHandle`` uses.

    ``LocalTeamHandle`` reads ``runtime.id`` and delegates ``send`` to it; a
    full actor-backed ``TeamRuntime`` is unnecessary for a route-level cache test.
    """

    def __init__(self, team_id: uuid.UUID) -> None:
        self.id = team_id
        self.sent: list[str] = []
        self.emitted: list[object] = []

    def send(self, content: str) -> None:
        self.sent.append(content)

    def emitMessage(self, message: object) -> None:  # noqa: N802
        self.emitted.append(message)


def _build_team_card(*, metadata_type: type[SerializableBaseModel] | None = None) -> TeamCard:
    """Build a validated minimal TeamCard for the worker create request.

    The worker receives the card *in the request body*, already resolved by the
    server, so declaring ``metadata_type`` here needs no catalog seeding — the
    class goes straight onto the card. Omitting it is the "team declares no
    metadata contract" card.
    """
    payload = dict(_TEAM_CARD_PAYLOAD)
    if metadata_type is not None:
        payload["metadata_type"] = metadata_type
    return TeamCard.model_validate(payload)


def _build_process(team_id: uuid.UUID, team_card: TeamCard) -> Process:
    """Build the persisted Process metadata the worker handle returns."""
    now = datetime.now(UTC)
    return Process(
        team_id=team_id,
        team_card=team_card,
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=now,
        updated_at=now,
    )


def _build_services(
    runtime: _FakeRuntime,
    process: Process,
    cache: LocalRuntimeCache,
) -> SimpleNamespace:
    """Assemble a WorkerServices-shaped stub around a real LocalRuntimeCache."""
    return SimpleNamespace(
        team_manager=SimpleNamespace(create_team=lambda **_kwargs: runtime),
        worker_handle=SimpleNamespace(get_team=lambda _tid: process),
        runtime_cache=cache,
    )


def _make_create_body(
    team_id: uuid.UUID,
    team_card: TeamCard,
    *,
    metadata: dict[str, Any] | None = None,
) -> WorkerCreateTeamRequest:
    """Build a valid WorkerCreateTeamRequest for the given team.

    ``metadata`` defaults to ``None`` so every pre-existing caller keeps
    producing exactly the body it produced before.
    """
    return WorkerCreateTeamRequest(
        team_card=team_card,
        user_id="user-1",
        user_email="user@example.com",
        team_id=team_id,
        metadata=metadata,
    )


def test_create_team_stores_handle_in_cache() -> None:
    """AC#1: POST /teams stores a LocalTeamHandle before returning (201 shape)."""
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    response = create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    cached = cache.get(team_id)
    assert isinstance(cached, LocalTeamHandle)
    assert cached.team_id == team_id
    # Response shape / id unchanged from today.
    assert response.team_id == team_id
    assert response.name == "Test Team"
    assert response.user_id == "user-1"
    assert response.status == TeamStatus.RUNNING.value
    # A team carrying no metadata reports None, never {} — asserted here since
    # the worker has no read route to assert it on.
    assert response.metadata is None


def test_create_then_message_hits_cache_and_returns_204() -> None:
    """AC#2: a message sent right after create resolves the cached handle."""
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    # send_message returns None (HTTP 204) only if the cache lookup hit.
    result = send_message(
        team_id,
        SendMessageRequest(content="hello team"),
        services,  # type: ignore[arg-type]
    )

    assert result is None
    assert runtime.sent == ["hello team"]


def test_send_message_typed_envelope_delivers_decoded_message() -> None:
    """The worker /message route accepts a serialized Message envelope (str|Message path).

    A ``{message: <model_dump(mode="json")>}`` body is resolved via the shared
    ``resolve_send_payload`` into the concrete typed ``Message`` and handed to
    the handle — the newly-available typed path, alongside the plain-str path.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    original = UserMessage(content="typed-hello")
    result = send_message(
        team_id,
        SendMessageRequest(message=original.model_dump(mode="json")),
        services,  # type: ignore[arg-type]
    )

    assert result is None
    assert len(runtime.sent) == 1
    delivered = runtime.sent[0]
    assert type(delivered) is UserMessage
    assert delivered == original


def test_send_message_bad_envelope_returns_400() -> None:
    """A message envelope with an unimportable __model__ tag surfaces a 400."""
    from fastapi import HTTPException

    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    body = SendMessageRequest(message={"__model__": "akgentic.core.messages.message.NoSuchMessage"})
    with pytest.raises(HTTPException) as exc_info:
        send_message(team_id, body, services)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert runtime.sent == []


def test_send_message_non_message_envelope_returns_400() -> None:
    """A message envelope that decodes to a non-Message surfaces a 400."""
    from fastapi import HTTPException

    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    with pytest.raises(HTTPException) as exc_info:
        send_message(
            team_id,
            SendMessageRequest(message={"hello": "world"}),
            services,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 400
    assert runtime.sent == []


def test_create_team_store_is_idempotent_overwrite() -> None:
    """AC#3: re-creating the same team id overwrites the cache entry harmlessly."""
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    cache = LocalRuntimeCache()

    first_runtime = _FakeRuntime(team_id)
    services_a = _build_services(first_runtime, _build_process(team_id, team_card), cache)
    create_team(_make_create_body(team_id, team_card), services_a)  # type: ignore[arg-type]
    first_handle = cache.get(team_id)

    second_runtime = _FakeRuntime(team_id)
    services_b = _build_services(second_runtime, _build_process(team_id, team_card), cache)
    create_team(_make_create_body(team_id, team_card), services_b)  # type: ignore[arg-type]
    second_handle = cache.get(team_id)

    assert isinstance(second_handle, LocalTeamHandle)
    assert second_handle is not first_handle
    assert second_handle.team_id == team_id


def _build_lifecycle_services(
    runtime: _FakeRuntime,
    process: Process,
    cache: LocalRuntimeCache,
) -> SimpleNamespace:
    """WorkerServices stub whose worker_handle also supports stop/delete (no-ops)."""
    return SimpleNamespace(
        team_manager=SimpleNamespace(create_team=lambda **_kwargs: runtime),
        worker_handle=SimpleNamespace(
            get_team=lambda _tid: process,
            stop_team=lambda _tid: None,
            delete_team=lambda _tid: None,
        ),
        runtime_cache=cache,
    )


def test_stop_team_evicts_handle_from_cache() -> None:
    """Regression: stopping a team removes its handle so the cache cannot pin it.

    Without the eviction the worker-lifetime LocalRuntimeCache retains the whole
    TeamRuntime graph (proxies, ActorRefs, pykka AttrInfo) of every stopped team.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    cache = LocalRuntimeCache()
    services = _build_lifecycle_services(
        _FakeRuntime(team_id), _build_process(team_id, team_card), cache
    )

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]
    assert cache.get(team_id) is not None

    stop_team(team_id, services)  # type: ignore[arg-type]

    assert cache.get(team_id) is None, "stop_team must evict the handle from runtime_cache"


def test_delete_team_evicts_handle_from_cache() -> None:
    """Regression: deleting a team removes its handle from the cache."""
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    cache = LocalRuntimeCache()
    services = _build_lifecycle_services(
        _FakeRuntime(team_id), _build_process(team_id, team_card), cache
    )

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]
    assert cache.get(team_id) is not None

    delete_team(team_id, services)  # type: ignore[arg-type]

    assert cache.get(team_id) is None, "delete_team must evict the handle from runtime_cache"


def test_emit_notification_delivers_decoded_message_and_returns_204() -> None:
    """The worker /notification route decodes the envelope and calls emitMessage.

    A ``{message: <model_dump(mode="json")>}`` body is decoded via the shared
    ``decode_message`` into the concrete typed ``Message`` and published through
    ``handle.emitMessage`` (durable store + live stream, no agent processing).
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    original = UserMessage(content="notify-hello")
    result = emit_notification(
        team_id,
        EmitMessageRequest(message=original.model_dump(mode="json")),
        services,  # type: ignore[arg-type]
    )

    assert result is None
    assert runtime.sent == []
    assert len(runtime.emitted) == 1
    delivered = runtime.emitted[0]
    assert type(delivered) is UserMessage
    assert delivered == original


def test_emit_notification_without_create_returns_404() -> None:
    """A valid envelope for an un-stored team 404s (cache gates the route)."""
    from fastapi import HTTPException

    team_id = uuid.uuid4()
    cache = LocalRuntimeCache()
    services = SimpleNamespace(runtime_cache=cache)

    body = EmitMessageRequest(message=UserMessage(content="nobody home").model_dump(mode="json"))
    with pytest.raises(HTTPException) as exc_info:
        emit_notification(team_id, body, services)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 404


def test_emit_notification_bad_envelope_returns_400() -> None:
    """A notification envelope with an unimportable __model__ tag surfaces a 400."""
    from fastapi import HTTPException

    team_id = uuid.uuid4()
    team_card = _build_team_card()
    runtime = _FakeRuntime(team_id)
    process = _build_process(team_id, team_card)
    cache = LocalRuntimeCache()
    services = _build_services(runtime, process, cache)

    create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    body = EmitMessageRequest(message={"__model__": "akgentic.core.messages.message.NoSuchMessage"})
    with pytest.raises(HTTPException) as exc_info:
        emit_notification(team_id, body, services)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert runtime.emitted == []


def test_send_message_without_create_returns_404() -> None:
    """Control: an un-stored team still 404s, proving the cache gates the route."""
    from fastapi import HTTPException

    team_id = uuid.uuid4()
    cache = LocalRuntimeCache()
    services = SimpleNamespace(runtime_cache=cache)

    with pytest.raises(HTTPException) as exc_info:
        send_message(
            team_id,
            SendMessageRequest(content="nobody home"),
            services,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Story 54.1 — the worker's INBOUND metadata half.
#
# The worker revalidates rather than trusting the server: it holds the resolved
# TeamCard, so it knows metadata_type and runs the same check the server ran. A
# worker is reachable by anything holding its address, so "the server already
# checked" is a deployment assumption, not a security property.
#
# ``ACME_METADATA_TYPE`` is the ``__model__`` value throughout: a real,
# importable, harmless class. A nonexistent path would let these tests pass on an
# ImportError even if the worker had honoured the tag — the false green that
# hides exactly the vulnerability the rule exists for.
# ---------------------------------------------------------------------------


class _PersistingTeamManager:
    """TeamManager stub that stores what it is handed, as the real one does.

    ``create_team`` is the call that writes the team into the event store, so
    recording it here is what makes "validation ran before anything was created"
    observable: a route that created first and validated afterwards would leave a
    team behind even while answering 422. Counting the store is the assertion;
    the 422 alone would pass either way.
    """

    def __init__(self, runtime: _FakeRuntime, team_card: TeamCard) -> None:
        self._runtime = runtime
        self._team_card = team_card
        self.teams: dict[uuid.UUID, Process] = {}
        self.calls: list[dict[str, Any]] = []

    def create_team(self, **kwargs: Any) -> _FakeRuntime:
        """Persist a Process carrying the forwarded metadata; return the runtime."""
        self.calls.append(kwargs)
        team_id = kwargs.get("team_id") or self._runtime.id
        base = _build_process(team_id, self._team_card)
        self.teams[team_id] = base.model_copy(update={"metadata": kwargs.get("metadata")})
        return self._runtime


def _build_metadata_services(
    runtime: _FakeRuntime,
    team_card: TeamCard,
    cache: LocalRuntimeCache,
) -> SimpleNamespace:
    """WorkerServices stub whose team manager genuinely persists the create.

    Unlike ``_build_services``, nothing is pre-seeded: the store starts empty, so
    the team count before and after a request is a real measurement of what the
    route did rather than of what the fixture arranged.
    """
    manager = _PersistingTeamManager(runtime, team_card)
    return SimpleNamespace(
        team_manager=manager,
        worker_handle=SimpleNamespace(get_team=lambda tid: manager.teams.get(tid)),
        runtime_cache=cache,
    )


def _has_model_tag(value: Any) -> bool:
    """Report whether ``__model__`` appears anywhere in *value*.

    Deliberately a local walk rather than the shipped scanner: checking the strip
    with the code it ships beside would let a shared blind spot — a container
    neither of them recurses into — pass as a green test.
    """
    if isinstance(value, dict):
        return "__model__" in value or any(_has_model_tag(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_model_tag(item) for item in value)
    return False


def test_worker_create_with_metadata_persists_it_and_reads_back_as_plain_json() -> None:
    """AC #2/#8: a worker-routed create carries metadata through to the store.

    The nested ``owner`` and the ``watchers`` list are populated so the outbound
    recursion runs on real data — the two shapes a top-level-only strip misses.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, cache)
    body = make_metadata_body(
        note="escalated",
        owner={"email": "ops@contoso.example", "squad": "support"},
        watchers=[{"email": "watcher@contoso.example"}],
    )

    created = create_team(_make_create_body(team_id, team_card, metadata=body), services)  # type: ignore[arg-type]

    assert created.team_id == team_id
    # The VALIDATED MODEL reached the team manager, not the raw dict: the derived
    # index is computed once, inside akgentic-team, from a real instance.
    forwarded = services.team_manager.calls[-1]["metadata"]
    assert isinstance(forwarded, AcmeCaseMetadata)
    assert forwarded.tenant == "acme"
    assert forwarded.owner is not None
    assert forwarded.owner.email == "ops@contoso.example"

    # Read the PERSISTED Process back out of the store and convert it exactly as
    # every outbound worker response does. Deliberately not `created`: asserting
    # on the create response would turn this test from "it was persisted" into
    # "it was echoed", which is the one thing the readback exists to rule out.
    fetched = _process_to_response(services.team_manager.teams[team_id])
    assert fetched.metadata is not None
    assert fetched.metadata["tenant"] == "acme"
    assert fetched.metadata["case"] == "C-1234"
    assert fetched.metadata["note"] == "escalated"
    owner = fetched.metadata["owner"]
    assert isinstance(owner, dict)
    assert owner["email"] == "ops@contoso.example"
    watchers = fetched.metadata["watchers"]
    assert isinstance(watchers, list)
    assert watchers[0]["email"] == "watcher@contoso.example"
    assert not _has_model_tag(fetched.metadata)


def test_worker_create_response_carries_the_metadata() -> None:
    """The 201 body itself carries the metadata, not just a later GET."""
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, LocalRuntimeCache())

    response = create_team(  # type: ignore[arg-type]
        _make_create_body(team_id, team_card, metadata=make_metadata_body(note="from-create")),
        services,
    )

    assert response.metadata is not None
    assert response.metadata["note"] == "from-create"
    assert not _has_model_tag(response.metadata)


def test_worker_create_with_schema_failure_is_422_and_creates_nothing() -> None:
    """AC #3/#6: a body failing the card's schema is refused AT THE WORKER."""
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, cache)
    before = len(services.team_manager.teams)

    with pytest.raises(HTTPException) as exc_info:
        create_team(  # type: ignore[arg-type]
            _make_create_body(team_id, team_card, metadata={"tenant": "acme"}),
            services,
        )

    assert exc_info.value.status_code == 422
    assert "case" in str(exc_info.value.detail)
    assert len(services.team_manager.teams) == before
    assert cache.get(team_id) is None


def test_worker_create_metadata_for_an_untyped_card_is_422_and_creates_nothing() -> None:
    """AC #4/#6: metadata for a card declaring ``metadata_type=None`` is refused."""
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, cache)
    before = len(services.team_manager.teams)

    with pytest.raises(HTTPException) as exc_info:
        create_team(  # type: ignore[arg-type]
            _make_create_body(team_id, team_card, metadata={"tenant": "acme"}),
            services,
        )

    assert exc_info.value.status_code == 422
    assert "no metadata contract" in str(exc_info.value.detail)
    assert len(services.team_manager.teams) == before
    assert cache.get(team_id) is None


def test_worker_create_with_a_model_tag_is_422_and_creates_nothing() -> None:
    """AC #5/#6: a ``__model__`` key is refused, and the class named is importable.

    Pydantic would not catch this: unknown keys are ignored and the serializer
    strips the tag for its own declared class, so a tagged body validates
    *cleanly* and the key vanishes with no signal. The explicit recursive scan is
    the only thing that turns it into a 422 — which is why the worker reuses the
    shared helper rather than calling ``model_validate`` itself.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, cache)
    before = len(services.team_manager.teams)
    tagged = {"__model__": ACME_METADATA_TYPE, "tenant": "acme", "case": "C-1234"}

    with pytest.raises(HTTPException) as exc_info:
        create_team(_make_create_body(team_id, team_card, metadata=tagged), services)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 422
    assert "__model__" in str(exc_info.value.detail)
    assert len(services.team_manager.teams) == before
    assert cache.get(team_id) is None


def test_worker_create_with_a_nested_model_tag_is_422() -> None:
    """AC #5: a ``__model__`` one level down is refused too — the scan recurses."""
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, LocalRuntimeCache())
    nested = {
        "tenant": "acme",
        "case": "C-1234",
        "owner": {"__model__": ACME_METADATA_TYPE, "email": "ops@contoso.example"},
    }

    with pytest.raises(HTTPException) as exc_info:
        create_team(_make_create_body(team_id, team_card, metadata=nested), services)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 422
    assert "__model__" in str(exc_info.value.detail)
    assert services.team_manager.teams == {}


def test_worker_create_model_tag_beats_the_no_contract_message() -> None:
    """The scan runs first and unconditionally, exactly as it does server-side.

    Against a card declaring no contract the caller must still be told about
    ``__model__``: the type-naming attempt is the security-relevant condition,
    and "this team takes no metadata" would hide that it was noticed at all.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, LocalRuntimeCache())

    with pytest.raises(HTTPException) as exc_info:
        create_team(  # type: ignore[arg-type]
            _make_create_body(team_id, team_card, metadata={"__model__": ACME_METADATA_TYPE}),
            services,
        )

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 422
    assert "__model__" in detail
    assert "no metadata contract" not in detail


def test_worker_create_rejection_is_422_not_the_action_error_mapping() -> None:
    """AC #9: the rejection never travels through the ValueError → 404/409 mapper.

    That helper maps any message without "not found" / "deleted" to 409, so a
    regression routing this through it would report a validation failure as a
    conflict.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, LocalRuntimeCache())

    with pytest.raises(HTTPException) as exc_info:
        create_team(  # type: ignore[arg-type]
            _make_create_body(team_id, team_card, metadata={"tenant": "acme"}),
            services,
        )

    assert exc_info.value.status_code == 422
    # The mapper would have answered 409 carrying this very message, so the
    # status code is the only thing separating the two paths — asserting the
    # code is not 404/409 after pinning it to 422 would assert nothing at all.
    assert "metadata" in str(exc_info.value.detail)


def test_worker_create_without_metadata_is_unchanged() -> None:
    """AC #7: omitting metadata behaves exactly as it did before this story.

    Asserted field by field rather than against a frozen response dict: the
    response has carried a ``metadata`` key since the outbound half shipped, so a
    whole-dict comparison would pin the wrong thing.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, cache)

    response = create_team(_make_create_body(team_id, team_card), services)  # type: ignore[arg-type]

    assert response.team_id == team_id
    assert response.name == "Test Team"
    assert response.user_id == "user-1"
    assert response.status == TeamStatus.RUNNING.value
    assert response.metadata is None
    # ...and nothing was persisted for it either.
    assert services.team_manager.teams[team_id].metadata is None
    assert isinstance(cache.get(team_id), LocalTeamHandle)


def test_worker_create_request_survives_the_json_round_trip() -> None:
    """The seam the tier adapters consume: ``metadata_type`` must survive JSON.

    Every other test here builds the request in-process, with the metadata_type
    class already sitting on the card. Production never does that: the tier's
    placement adapter serializes this body, HTTP carries it as text, and FastAPI
    validates it back. If the card's ``__type__`` tag did not survive that trip
    the worker would see ``metadata_type=None`` and answer every metadata create
    with "this team declares no metadata contract" — while every in-process test
    here stayed green, and the failure surfaced only in the deployment repos
    blocked on this story.

    ``json.loads(model_dump_json())`` rather than ``model_dump(mode="json")``: a
    real text round-trip cannot let a live Python object through by accident.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    body = make_metadata_body(note="over-http")
    original = _make_create_body(team_id, team_card, metadata=body)

    parsed = WorkerCreateTeamRequest.model_validate(json.loads(original.model_dump_json()))

    assert parsed.team_card.metadata_type is AcmeCaseMetadata
    assert parsed.metadata == original.metadata

    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), parsed.team_card, cache)
    response = create_team(parsed, services)  # type: ignore[arg-type]

    # Validated against the reconstructed type, not waved through as a raw dict.
    assert isinstance(services.team_manager.calls[-1]["metadata"], AcmeCaseMetadata)
    assert response.metadata is not None
    assert response.metadata["note"] == "over-http"


def test_worker_create_over_the_json_round_trip_still_rejects_a_model_tag() -> None:
    """The rejection half survives the wire too, not just the accept half.

    A body that reached the route through JSON is the only form an attacker ever
    sends. Pinning the 422 on the in-process object alone would leave the actual
    attack path unasserted.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    tagged = {"__model__": ACME_METADATA_TYPE, "tenant": "acme", "case": "C-1234"}
    original = _make_create_body(team_id, team_card, metadata=tagged)

    parsed = WorkerCreateTeamRequest.model_validate(json.loads(original.model_dump_json()))
    cache = LocalRuntimeCache()
    services = _build_metadata_services(_FakeRuntime(team_id), parsed.team_card, cache)

    with pytest.raises(HTTPException) as exc_info:
        create_team(parsed, services)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 422
    assert "__model__" in str(exc_info.value.detail)
    assert services.team_manager.teams == {}


def test_worker_create_with_empty_metadata_creates_the_team() -> None:
    """An empty body is not an error, even for a card that declares a type.

    A declared type constrains metadata's *shape*, not its *presence* — the same
    rule the server applies, inherited by calling the same helper.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card(metadata_type=AcmeCaseMetadata)
    services = _build_metadata_services(_FakeRuntime(team_id), team_card, LocalRuntimeCache())

    response = create_team(_make_create_body(team_id, team_card, metadata={}), services)  # type: ignore[arg-type]

    assert response.metadata is None
    assert services.team_manager.calls[-1]["metadata"] is None


# ---------------------------------------------------------------------------
# Story 54.2 — PATCH /teams/{team_id}/metadata, the worker's metadata WRITE.
#
# Driven through a real ``TestClient`` rather than by calling the handler: the
# response is the persisted ``Process`` with its ``__model__`` tag INTACT — the
# module's one exception to the outbound strip — because the caller is a tier
# adapter reconstructing a typed ``Process``, not a client. That reconstruction
# only exists on the far side of JSON, so an in-process assertion would pin
# nothing and the failure would surface first in the two downstream repos.
# ---------------------------------------------------------------------------


class _MetadataWorkerHandle:
    """WorkerHandle stub running the ordered write path ``TeamManager`` owns.

    The index is re-derived through the shipped ``derive_metadata_indexes`` —
    hand-assembling ``"key|value"`` strings here would test the test. Persisted
    with ``model_copy(update=...)`` so a field added to ``Process`` later rides
    along (Golden Rule #12).
    """

    def __init__(self, process: Process, *, push_fails: bool = False) -> None:
        self.teams: dict[uuid.UUID, Process] = {process.team_id: process}
        self.get_team_calls = 0
        self.push_failures = 0
        self._push_fails = push_fails

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        """Resolve the persisted team; counted, so a re-read is observable."""
        self.get_team_calls += 1
        return self.teams.get(team_id)

    def update_team_metadata(
        self,
        team_id: uuid.UUID,
        metadata: SerializableBaseModel | None,
    ) -> Process:
        """Single DB write of value + re-derived index, then a best-effort push."""
        process = self.teams.get(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status is TeamStatus.DELETED:
            msg = f"Cannot update metadata for team {team_id}: team has been deleted"
            raise ValueError(msg)
        updated = process.model_copy(
            update={
                "updated_at": datetime.now(UTC),
                "metadata": metadata,
                "metadata_indexes": derive_metadata_indexes(metadata),
            }
        )
        self.teams[team_id] = updated
        self._push()
        return updated

    def _push(self) -> None:
        """Push the new value to the live orchestrator, swallowing a failure.

        The swallow belongs to ``akgentic-team`` and is tested there; mirrored
        here so the route sees exactly what production hands it — a normal
        return after a push that *genuinely* raised. The database write above is
        the system of record and the actor re-reads from ``Process`` on its next
        resume, so ``push_failures`` counts real swallowed failures rather than
        the fixture's intention to have one.
        """
        try:
            self._push_to_orchestrator()
        except RuntimeError:
            self.push_failures += 1

    def _push_to_orchestrator(self) -> None:
        """The orchestrator hop itself; raises when the test asks it to fail."""
        if self._push_fails:
            msg = "orchestrator unreachable"
            raise RuntimeError(msg)


def _build_metadata_process(
    team_id: uuid.UUID,
    *,
    metadata_type: type[SerializableBaseModel] | None = AcmeCaseMetadata,
    metadata: SerializableBaseModel | None = None,
    status: TeamStatus = TeamStatus.RUNNING,
) -> Process:
    """Build the persisted Process the update route resolves and rewrites."""
    process = _build_process(team_id, _build_team_card(metadata_type=metadata_type))
    return process.model_copy(
        update={
            "status": status,
            "metadata": metadata,
            "metadata_indexes": derive_metadata_indexes(metadata),
        }
    )


def _build_update_client(
    handle: _MetadataWorkerHandle,
    cache: LocalRuntimeCache | None = None,
) -> TestClient:
    """Mount the worker router on a bare app carrying only what the route uses."""
    app = FastAPI()
    app.include_router(router)
    services = SimpleNamespace(worker_handle=handle, runtime_cache=cache or LocalRuntimeCache())
    SERVICES.set(app, services)  # type: ignore[arg-type]
    return TestClient(app)


def test_worker_metadata_update_round_trips_the_typed_value_through_json() -> None:
    """AC #4: the 200 body is the persisted Process, reconstructible by the caller.

    The tag survives here and nowhere else in this module. A tier adapter must
    rebuild a ``Process`` — metadata included, as the team's concrete declared
    class — to satisfy ``WorkerHandle.update_team_metadata() -> Process``, and
    the ``__model__`` tag is what makes that rebuild possible. Reconstructed from
    ``response.json()``, so the serialization seam is genuinely crossed.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    client = _build_update_client(handle)
    body = make_metadata_body(
        note="escalated",
        owner={"email": "ops@contoso.example", "squad": "support"},
        watchers=[{"email": "watcher@contoso.example"}],
    )

    response = client.patch(f"/teams/{team_id}/metadata", json={"metadata": body})

    assert response.status_code == 200
    payload = response.json()
    # Not a TeamResponse and not dump_metadata's output: both would have dropped
    # the tag, and TeamResponse carries no team_card and no metadata_indexes.
    assert payload["metadata"]["__model__"] == ACME_METADATA_TYPE
    restored = Process.model_validate(payload)
    assert restored.team_id == team_id
    assert restored.team_card.metadata_type is AcmeCaseMetadata
    assert isinstance(restored.metadata, AcmeCaseMetadata)
    assert restored.metadata.tenant == "acme"
    assert restored.metadata.case == "C-1234"
    assert restored.metadata.note == "escalated"
    assert restored.metadata.owner is not None
    assert restored.metadata.owner.email == "ops@contoso.example"
    assert restored.metadata.watchers[0].email == "watcher@contoso.example"
    assert restored.metadata_indexes
    assert restored.metadata_indexes == derive_metadata_indexes(restored.metadata)


def test_worker_metadata_update_returns_the_rederived_index_not_an_echo() -> None:
    """AC #5: the response carries the index the NEW document derives.

    A route that echoed the request body would carry no ``metadata_indexes`` at
    all, and one that returned the pre-update Process would carry the old
    entries — so the assertion is on the index, not on the value.
    """
    team_id = uuid.uuid4()
    seeded = AcmeCaseMetadata(tenant="acme", case="C-1234")
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id, metadata=seeded))
    client = _build_update_client(handle)

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": make_metadata_body(case="C-9999")},
    )

    assert response.status_code == 200
    after = response.json()["metadata_indexes"]
    assert make_index_entry("case", "C-9999") in after
    assert make_index_entry("case", "C-1234") not in after
    assert after == derive_metadata_indexes(handle.teams[team_id].metadata)


def test_worker_metadata_update_shrinks_the_index_when_an_indexed_field_is_omitted() -> None:
    """AC #6: replace, never merge — an omitted indexed field loses its entry.

    ``channel`` is indexed *and* optional, so its disappearance is observable in
    the index rather than merely overwritten. "The value changed" would not
    distinguish a replace from a merge; a strictly SMALLER index does.
    """
    team_id = uuid.uuid4()
    seeded = AcmeCaseMetadata(tenant="acme", case="C-1234", channel="email")
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id, metadata=seeded))
    client = _build_update_client(handle)
    before = list(handle.teams[team_id].metadata_indexes)
    assert make_index_entry("channel", "email") in before

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": make_metadata_body(tenant="acme", case="C-1234")},
    )

    assert response.status_code == 200
    restored = Process.model_validate(response.json())
    assert isinstance(restored.metadata, AcmeCaseMetadata)
    assert restored.metadata.channel is None
    after = restored.metadata_indexes
    assert len(after) < len(before)
    assert make_index_entry("channel", "email") not in after
    # No channel entry under any value: derivation of a channel=None model emits
    # none, so whole-list equality is the strict form of "the entry is gone".
    assert after == derive_metadata_indexes(restored.metadata)


def test_worker_metadata_update_succeeds_when_the_orchestrator_push_fails() -> None:
    """AC #7: a failed best-effort push is still a 200 carrying the written value.

    The write path is database-first by decision: the index stays truthful and
    the actor self-heals from ``Process`` on its next resume, so turning a failed
    push into an error would invert that ordering and misdescribe a write that
    stands. ``get_team`` is counted because the route's other obligation is to
    add no re-read "confirming" the update.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id), push_fails=True)
    client = _build_update_client(handle)

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": make_metadata_body(case="C-9999", channel="phone")},
    )

    assert response.status_code == 200
    assert handle.push_failures == 1
    assert handle.get_team_calls == 1, "the route must not re-read after the update"
    restored = Process.model_validate(response.json())
    persisted = handle.teams[team_id]
    assert isinstance(restored.metadata, AcmeCaseMetadata)
    assert restored.metadata.case == "C-9999"
    assert restored.metadata.channel == "phone"
    assert restored.metadata_indexes == persisted.metadata_indexes
    assert make_index_entry("channel", "phone") in restored.metadata_indexes


def test_worker_metadata_update_clears_the_value_on_an_empty_document() -> None:
    """An empty document clears; it is not an error, even for a typed card.

    A declared type constrains metadata's *shape*, not its *presence* — the same
    rule the server applies, inherited by calling the same helper.
    """
    team_id = uuid.uuid4()
    seeded = AcmeCaseMetadata(tenant="acme", case="C-1234", channel="email")
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id, metadata=seeded))
    client = _build_update_client(handle)

    response = client.patch(f"/teams/{team_id}/metadata", json={"metadata": {}})

    assert response.status_code == 200
    restored = Process.model_validate(response.json())
    assert restored.metadata is None
    assert restored.metadata_indexes == []


def test_worker_metadata_update_leaves_the_runtime_cache_alone() -> None:
    """The cache maps team ids to live handles and holds no metadata.

    The module's other mutating routes all touch it — create stores, stop and
    delete evict — so pattern-matching leads straight into it. An eviction here
    would 404 the five cache-reading routes for a team that is plainly running.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    cache = LocalRuntimeCache()
    team_handle = LocalTeamHandle(_FakeRuntime(team_id))
    cache.store(team_id, team_handle)
    client = _build_update_client(handle, cache)

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": make_metadata_body()},
    )

    assert response.status_code == 200
    assert cache.get(team_id) is team_handle


def test_worker_metadata_update_for_an_untyped_card_is_422_and_writes_nothing() -> None:
    """AC #8: a non-empty document against a card declaring no metadata_type."""
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id, metadata_type=None))
    client = _build_update_client(handle)
    before = handle.teams[team_id]

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": {"tenant": "acme"}},
    )

    assert response.status_code == 422
    assert "no metadata contract" in response.json()["detail"]
    assert handle.teams[team_id] is before


def test_worker_metadata_update_with_schema_failure_is_422_and_writes_nothing() -> None:
    """AC #9: a document failing the card's declared schema names the field."""
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    client = _build_update_client(handle)
    before = handle.teams[team_id]

    response = client.patch(f"/teams/{team_id}/metadata", json={"metadata": {"tenant": "acme"}})

    assert response.status_code == 422
    assert "case" in response.json()["detail"]
    assert handle.teams[team_id] is before


def test_worker_metadata_update_with_a_model_tag_is_422_and_writes_nothing() -> None:
    """AC #9: a ``__model__`` key is refused, and the class named is importable.

    A fake dotted path would let this pass on an ImportError even if the route
    had honoured the tag — the false green that hides the very gadget the rule
    exists for. Pydantic catches none of this: unknown keys are ignored and the
    serializer strips the tag for its own declared class, so a tagged body
    validates *cleanly* and the key vanishes with no signal.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    client = _build_update_client(handle)
    before = handle.teams[team_id]
    tagged = {"__model__": ACME_METADATA_TYPE, "tenant": "acme", "case": "C-1234"}

    response = client.patch(f"/teams/{team_id}/metadata", json={"metadata": tagged})

    assert response.status_code == 422
    assert "__model__" in response.json()["detail"]
    assert handle.teams[team_id] is before


def test_worker_metadata_update_with_a_nested_model_tag_is_422() -> None:
    """AC #9: a ``__model__`` one level down is refused too — the scan recurses."""
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    client = _build_update_client(handle)
    before = handle.teams[team_id]
    nested = {
        "tenant": "acme",
        "case": "C-1234",
        "owner": {"__model__": ACME_METADATA_TYPE, "email": "ops@contoso.example"},
    }

    response = client.patch(f"/teams/{team_id}/metadata", json={"metadata": nested})

    assert response.status_code == 422
    assert "__model__" in response.json()["detail"]
    assert handle.teams[team_id] is before


def test_worker_metadata_update_model_tag_beats_the_no_contract_message() -> None:
    """AC #9: the scan runs first and unconditionally, as it does server-side.

    Against a card declaring no contract the caller must still be told about
    ``__model__``: the type-naming attempt is the security-relevant condition,
    and "this team takes no metadata" would hide that it was noticed at all.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id, metadata_type=None))
    client = _build_update_client(handle)

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": {"__model__": ACME_METADATA_TYPE}},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "__model__" in detail
    assert "no metadata contract" not in detail


def test_worker_metadata_update_rejection_is_422_not_the_action_error_mapping() -> None:
    """AC #11: the rejection never travels through the ValueError → 404/409 mapper.

    That helper maps any message carrying neither "not found" nor "deleted" to
    409, so a regression routing this through it would report a validation
    failure as a conflict. The status code is the only thing separating the two
    paths — asserting it is not 404/409 after pinning it to 422 would assert
    nothing at all, so the detail text is asserted instead.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    client = _build_update_client(handle)

    response = client.patch(f"/teams/{team_id}/metadata", json={"metadata": {"tenant": "acme"}})

    assert response.status_code == 422
    assert "metadata field 'case' is invalid" in response.json()["detail"]


def test_worker_metadata_update_for_an_unknown_team_is_404() -> None:
    """AC #10: an id the worker cannot resolve is a 404 before any validation."""
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(_build_metadata_process(team_id))
    client = _build_update_client(handle)

    response = client.patch(
        f"/teams/{uuid.uuid4()}/metadata",
        json={"metadata": make_metadata_body()},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Team not found"


def test_worker_metadata_update_for_a_deleted_team_is_404() -> None:
    """AC #10: a deleted team 404s through the lifecycle mapper, not 409.

    A deleted team still resolves, so the refusal comes from the update call as
    a ``ValueError`` — which is exactly what ``_raise_action_error`` is for.
    """
    team_id = uuid.uuid4()
    handle = _MetadataWorkerHandle(
        _build_metadata_process(team_id, status=TeamStatus.DELETED),
    )
    client = _build_update_client(handle)
    before = handle.teams[team_id]

    response = client.patch(
        f"/teams/{team_id}/metadata",
        json={"metadata": make_metadata_body()},
    )

    assert response.status_code == 404
    assert "deleted" in response.json()["detail"]
    assert handle.teams[team_id] is before

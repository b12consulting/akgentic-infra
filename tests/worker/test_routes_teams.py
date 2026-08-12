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

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.core.messages.message import UserMessage
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.models import Process, TeamCard, TeamStatus
from fastapi import HTTPException

from akgentic.infra.adapters.community.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.server.models import EmitMessageRequest, SendMessageRequest
from akgentic.infra.worker.routes.teams import (
    WorkerCreateTeamRequest,
    create_team,
    delete_team,
    emit_notification,
    get_team,
    send_message,
    stop_team,
)

from tests.fixtures.team_metadata import (
    ACME_METADATA_TYPE,
    AcmeCaseMetadata,
    AcmeOwner,
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


def test_get_team_returns_metadata_with_the_tag_stripped() -> None:
    """The worker fills ``TeamResponse.metadata`` through the shared helper.

    ``TeamResponse`` is one model with two producers. The worker's converter left
    the field at its default, so a team carrying metadata was reported as
    ``metadata: null`` here while the server router reported the real value — and
    a bare ``model_dump`` would have leaked the ``__model__`` tag the server-side
    API refuses back in. Nested sub-model included, since that is the depth a
    top-level-only strip would miss.
    """
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    process = _build_process(team_id, team_card)
    process.metadata = AcmeCaseMetadata(
        tenant="acme",
        case="C-1234",
        owner=AcmeOwner(email="ops@contoso.example"),
    )
    services = _build_services(_FakeRuntime(team_id), process, LocalRuntimeCache())

    response = get_team(team_id, services)  # type: ignore[arg-type]

    assert response.metadata is not None
    assert response.metadata["tenant"] == "acme"
    assert "__model__" not in response.metadata
    owner = response.metadata["owner"]
    assert isinstance(owner, dict)
    assert "__model__" not in owner
    assert owner["email"] == "ops@contoso.example"


def test_get_team_without_metadata_returns_none() -> None:
    """A team carrying no metadata still reports ``None``, not an empty dict."""
    team_id = uuid.uuid4()
    team_card = _build_team_card()
    process = _build_process(team_id, team_card)
    services = _build_services(_FakeRuntime(team_id), process, LocalRuntimeCache())

    assert get_team(team_id, services).metadata is None  # type: ignore[arg-type]


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

    body = EmitMessageRequest(
        message={"__model__": "akgentic.core.messages.message.NoSuchMessage"}
    )
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

    fetched = get_team(team_id, services)  # type: ignore[arg-type]
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
    assert exc_info.value.status_code not in (404, 409)


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

"""Tests for channel routing — ChannelRouteContext, DefaultChannelRouter and router resolution."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from akgentic.core.messages.message import Message
from akgentic.team.models import AgentCardRef, AgentRef, Process, TeamStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.adapters.channels.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
)
from akgentic.infra.adapters.channels.channel_router import (
    ChannelRouteContext,
    DefaultChannelRouter,
)
from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.errors import TeamNotFoundError, TeamStateConflictError
from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelBinding,
    ChannelCommand,
    ChannelMessage,
    InteractionChannelRouter,
    JsonValue,
)
from akgentic.infra.server.routes.webhook import router as webhook_router

_THIS_MODULE = "tests.adapters.channels.test_channel_router"


# --- Stubs, resolvable by FQCN from this module ---


class StubParser:
    """ChannelParser for the ``routed`` channel; returns whatever it was told to."""

    next_message: ChannelMessage | None = None

    def __init__(self, **config: str) -> None:
        pass

    @property
    def channel_name(self) -> str:
        return "routed"

    @property
    def default_catalog_entry(self) -> str:
        return "routed-default"

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        if StubParser.next_message is not None:
            return StubParser.next_message
        return ChannelMessage(content=str(payload.get("text", "")), channel_user_id="user-1")


class OtherStubParser(StubParser):
    """A second channel, so per-channel wiring can be told from shared wiring."""

    @property
    def channel_name(self) -> str:
        return "other"


class StubAdapter:
    """InteractionChannelAdapter recording its notices."""

    def __init__(self, **config: str) -> None:
        self.notices: list[tuple[ChannelAddress, str]] = []

    def matches(self, msg: object, binding: object) -> bool:
        return False

    def deliver(self, msg: object, binding: object) -> None:
        pass

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        self.notices.append((address, text))

    def on_stop(self, team_id: uuid.UUID) -> None:
        pass


class RecordingRouter:
    """A custom router that records what it was handed and does nothing else."""

    instances: list[RecordingRouter] = []

    def __init__(self, **config: str) -> None:
        self.config = config
        self.calls: list[tuple[ChannelMessage, ChannelRouteContext]] = []
        RecordingRouter.instances.append(self)

    async def route(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        self.calls.append((message, ctx))


class NotARouter:
    """Has no ``route`` — must be refused at wiring time."""

    def __init__(self, **config: str) -> None:
        pass


class ReplyOnlyRouter(DefaultChannelRouter):
    """Keeps the default commands, but never starts a team for a stranger."""

    async def on_unbound(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        ctx.notify("Say /new to start.")


class StubPlacement:
    """What a resume that bypassed ``restore_team`` would reach.

    Mirrors ``PlacementStrategy.resume_team``: it brings the runtime back up,
    and the persisted status with it, and it warms **no cache** — the store is
    ``TeamService.restore_team``'s own next line. So a bypass leaves the team
    running with a cold handle, which is exactly how the real one fails.

    Held privately, as the real service holds it
    (``TeamService._services.placement``), so the bypass has to be written the
    same way against this stub as against production.
    """

    def __init__(self, service: StubTeamService) -> None:
        self._service = service
        self.resume_team_calls: list[uuid.UUID] = []

    def resume_team(self, team_id: uuid.UUID) -> uuid.UUID:
        self.resume_team_calls.append(team_id)
        process = self._service.teams[team_id]
        self._service.teams[team_id] = process.model_copy(update={"status": TeamStatus.RUNNING})
        self._service.cold_handles.add(team_id)
        return team_id


class StubTeamService:
    """The TeamService the router context calls, recording every call.

    Synchronous like the real one — ``ChannelRouteContext`` calls it through
    ``asyncio.to_thread``. Given a registry, it also records whether the chat was
    already bound when each message was sent: the ordering the create / bind /
    send sequence exists to guarantee.
    """

    def __init__(self, registry: YamlChannelRegistry | None = None) -> None:
        self.send_message_calls: list[tuple[uuid.UUID, str | Message]] = []
        self.send_from_to_calls: list[tuple[uuid.UUID, str, str, str | Message]] = []
        self.create_team_calls: list[tuple[str, str, dict[str, JsonValue] | None]] = []
        self.create_team_keys: list[uuid.UUID | None] = []
        self.bound_at_send: list[bool] = []
        self.next_team_id = uuid.uuid4()
        self.teams: dict[uuid.UUID, Process] = {}
        self.restore_team_calls: list[uuid.UUID] = []
        # Teams whose runtime is up but whose handle the service never cached.
        # Only a resume that bypasses ``restore_team`` can put one here, which
        # is the one cold-handle transition this stub can honestly know about:
        # a team a spec seeds directly is seeded warm, as the real cache would
        # hold a team that has been running all along.
        self.cold_handles: set[uuid.UUID] = set()
        self._placement = StubPlacement(self)
        self._registry = registry

    def _require_running(self, team_id: uuid.UUID) -> None:
        """Refuse a send the real ``_get_running_handle`` would refuse.

        Mirrors it in both checks and in their order: a team that exists but is
        not running is a ``TeamStateConflictError``, and a running team with no
        cached handle is the bare ``ValueError`` the webhook surfaces as a 500.
        A team the stub does not hold is left alone — the real service would
        raise ``TeamNotFoundError`` there, and the context's own
        default-recipient lookup already does.
        """
        process = self.teams.get(team_id)
        if process is None:
            return
        if process.status != TeamStatus.RUNNING:
            msg = f"Team {team_id} is not running"
            raise TeamStateConflictError(msg)
        if team_id in self.cold_handles:
            msg = f"Team {team_id} handle not cached"
            raise ValueError(msg)

    def restore_team(self, team_id: uuid.UUID) -> Process:
        """Mirror ``TeamService.restore_team``: guard, place, then warm the cache.

        The guards and their error types are the real ones, because the router
        branches on them: already-running is the race AC 6 resolves by sending
        anyway, and not-found / deleted are what fall to the third row.
        """
        process = self.teams.get(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
        if process.status == TeamStatus.RUNNING:
            msg = f"Team {team_id} is already running"
            raise TeamStateConflictError(msg)
        if process.status == TeamStatus.DELETED:
            msg = f"Team {team_id} has been deleted"
            raise ValueError(msg)
        self.restore_team_calls.append(team_id)
        self._placement.resume_team(team_id)
        self.cold_handles.discard(team_id)
        return self.teams[team_id]

    def send_message(self, team_id: uuid.UUID, content: str | Message) -> None:
        self._require_running(team_id)
        self.send_message_calls.append((team_id, content))
        if self._registry is not None:
            binding = self._registry.find_binding_sync(team_id, "@HumanProxy_0")
            self.bound_at_send.append(binding is not None)

    def send_message_from_to(
        self, team_id: uuid.UUID, sender_name: str, recipient_name: str, content: str | Message
    ) -> None:
        self._require_running(team_id)
        self.send_from_to_calls.append((team_id, sender_name, recipient_name, content))
        self.send_message_calls.append((team_id, content))
        if self._registry is not None:
            binding = self._registry.find_binding_sync(team_id, sender_name)
            self.bound_at_send.append(binding is not None)

    def create_team(
        self,
        catalog_namespace: str,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Process:
        self.create_team_calls.append((user_id, catalog_namespace, metadata))
        self.create_team_keys.append(team_id)
        now = datetime.now(UTC)
        process = Process(
            team_id=self.next_team_id,
            status=TeamStatus.RUNNING,
            user_id=user_id,
            created_at=now,
            updated_at=now,
            entry_point=AgentRef(name="@HumanProxy_0", role="human_support"),
            supervisors=[AgentRef(name="@Manager_0", role="manager")],
            agent_cards=[
                AgentCardRef(role="human_support", card_hash="stub-hash"),
                AgentCardRef(role="manager", card_hash="stub-hash-manager"),
            ],
        )
        self.teams[process.team_id] = process
        return process

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        return self.teams.get(team_id)


def _config(router_fqcn: str | None = None) -> dict[str, ChannelConfig]:
    return {
        "routed": ChannelConfig(
            parser_fqcn=f"{_THIS_MODULE}.StubParser",
            router_fqcn=router_fqcn,
            adapter_fqcn=f"{_THIS_MODULE}.StubAdapter",
            config={"token": "t0k3n"},
        )
    }


def _ctx(
    registry: YamlChannelRegistry,
    team_service: StubTeamService,
    adapter: StubAdapter | None = None,
) -> ChannelRouteContext:
    return ChannelRouteContext(
        address=ChannelAddress(channel="routed", channel_user_id="user-1"),
        registry=registry,
        team_service=team_service,  # type: ignore[arg-type]  # structural stand-in
        adapters=[adapter] if adapter is not None else [],
        default_catalog_entry="routed-default",
    )


@pytest.fixture(autouse=True)
def _reset_stub_state() -> None:
    StubParser.next_message = None
    RecordingRouter.instances.clear()


# --- Resolution ---


def test_a_channel_naming_no_router_gets_the_default() -> None:
    registry = ChannelParserRegistry(_config())
    assert isinstance(registry.get_router("routed"), DefaultChannelRouter)


def test_a_named_router_is_built_with_the_channel_config() -> None:
    """The shared ``config`` kwargs reach the router exactly as they reach the parser."""
    registry = ChannelParserRegistry(_config(f"{_THIS_MODULE}.RecordingRouter"))

    router = registry.get_router("routed")

    assert isinstance(router, RecordingRouter)
    assert router.config == {"token": "t0k3n"}


def test_a_class_without_route_is_refused_at_wiring() -> None:
    with pytest.raises(TypeError, match="InteractionChannelRouter"):
        ChannelParserRegistry(_config(f"{_THIS_MODULE}.NotARouter"))


def test_the_default_router_accepts_config_meant_for_the_other_two() -> None:
    """Configured explicitly, the default router must survive the shared kwargs."""
    registry = ChannelParserRegistry(
        _config("akgentic.infra.adapters.channels.channel_router.DefaultChannelRouter")
    )
    assert isinstance(registry.get_router("routed"), DefaultChannelRouter)


def test_the_default_router_satisfies_the_protocol() -> None:
    assert isinstance(DefaultChannelRouter(), InteractionChannelRouter)


# --- The route delegates ---


def _build_app(parser_registry: ChannelParserRegistry, tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.state.channel_parser_registry = parser_registry
    app.state.channel_registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    app.state.team_service = StubTeamService()
    app.include_router(webhook_router)
    return app


def test_the_route_hands_the_message_to_the_channels_router(tmp_path: Path) -> None:
    """A router that does nothing is a no-op: nothing is created, nothing replied."""
    parser_registry = ChannelParserRegistry(_config(f"{_THIS_MODULE}.RecordingRouter"))
    app = _build_app(parser_registry, tmp_path)
    client = TestClient(app)

    resp = client.post("/webhook/routed", json={"text": "hello"})

    assert resp.status_code == 204
    (router,) = RecordingRouter.instances
    ((message, ctx),) = router.calls
    assert message.content == "hello"
    assert ctx.address == ChannelAddress(channel="routed", channel_user_id="user-1")
    assert ctx.default_catalog_entry == "routed-default"
    assert app.state.team_service.create_team_calls == []
    assert app.state.team_service.send_message_calls == []


# --- ChannelRouteContext ---


async def test_initiate_team_always_binds_the_conversation(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    ctx = _ctx(registry, team_service)

    process = await ctx.initiate_team(None)

    binding = await registry.find_binding(
        ChannelAddress(channel="routed", channel_user_id="user-1")
    )
    assert binding is not None
    assert binding.team_id == process.team_id
    assert binding.agent_name == "@HumanProxy_0"
    # Outbound delivery reads the same record, so it must be answerable there too.
    assert registry.find_binding_sync(process.team_id, "@HumanProxy_0") == binding
    assert team_service.create_team_calls == [("user-1", "routed-default", None)]
    # None is "create silently": no first message at all, not an empty one.
    assert team_service.send_message_calls == []


async def test_initiate_team_separates_team_and_binding_metadata(tmp_path: Path) -> None:
    """Team metadata goes to creation; binding metadata goes to the stored record."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    ctx = _ctx(registry, team_service)

    await ctx.initiate_team(
        "hi",
        catalog_entry="chosen",
        team_metadata={"case": "42"},
        binding_metadata={"thread": "t-9"},
    )

    assert team_service.create_team_calls == [("user-1", "chosen", {"case": "42"})]
    # A fresh registry reads the file, so this proves the field is persisted and
    # not just held in the writer's memory.
    reread = await YamlChannelRegistry(tmp_path / "registry.yaml").find_binding(
        ChannelAddress(channel="routed", channel_user_id="user-1")
    )
    assert reread is not None
    assert reread.metadata == {"thread": "t-9"}


async def test_the_first_message_is_sent_only_once_the_chat_is_bound(tmp_path: Path) -> None:
    """Create, bind, THEN send: a team answering at once must find the chat.

    Sent before the binding exists, the team's first reply reaches an outbound
    path where ``find_binding_sync`` answers None, and is silently lost.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService(registry)

    await _ctx(registry, team_service).initiate_team("first")

    assert team_service.send_message_calls == [(team_service.next_team_id, "first")]
    assert team_service.bound_at_send == [True]


async def test_release_returns_what_it_released(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_id = uuid.uuid4()
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-1", team_id=team_id, agent_name="@H_0"
        )
    )
    ctx = _ctx(registry, StubTeamService())

    released = await ctx.release()

    assert released is not None
    assert released.team_id == team_id
    assert (
        await registry.find_binding(ChannelAddress(channel="routed", channel_user_id="user-1"))
        is None
    )
    assert await ctx.release() is None


# --- Subclassing the default ---


async def test_a_subclass_replacing_initiation_keeps_the_commands(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    ctx = _ctx(registry, team_service, adapter)
    router = ReplyOnlyRouter()

    await router.route(ChannelMessage(content="hello", channel_user_id="user-1"), ctx)
    assert team_service.create_team_calls == []
    assert adapter.notices[-1][1] == "Say /new to start."

    new = ChannelMessage(
        content="/new go", channel_user_id="user-1", command=ChannelCommand(name="new", rest="go")
    )
    await router.route(new, ctx)
    assert team_service.create_team_calls == [("user-1", "routed-default", None)]
    assert team_service.send_message_calls == [(team_service.next_team_id, "go")]
    assert (
        await registry.find_binding(ChannelAddress(channel="routed", channel_user_id="user-1"))
        is not None
    )


async def test_a_bound_message_replies_to_the_bound_team(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_id = uuid.uuid4()
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-1", team_id=team_id, agent_name="@H_0"
        )
    )
    team_service = StubTeamService()
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        ChannelMessage(content="more", channel_user_id="user-1"), _ctx(registry, team_service)
    )

    assert team_service.send_message_calls == [(team_id, "more")]
    assert team_service.create_team_calls == []


# --- The default router forwards what the parser lifted ---

_ADDRESS = ChannelAddress(channel="routed", channel_user_id="user-1")


async def test_an_unbound_message_carries_its_binding_metadata_onto_the_binding(
    tmp_path: Path,
) -> None:
    """``on_unbound`` must pass ``message.binding_metadata`` to the new binding.

    ``ctx.initiate_team`` storing it is proven above; this proves the default
    router hands it over. Without it the parser's field reaches nothing.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    message = ChannelMessage(
        content="hello", channel_user_id="user-1", binding_metadata={"thread": "t-1"}
    )

    await DefaultChannelRouter().route(message, _ctx(registry, StubTeamService()))

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.metadata == {"thread": "t-1"}


async def test_new_carries_its_binding_metadata_onto_the_fresh_binding(tmp_path: Path) -> None:
    """``/new`` starts a team exactly as an unbound message does — metadata included."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    await registry.register(
        ChannelBinding(
            channel="routed",
            channel_user_id="user-1",
            team_id=uuid.uuid4(),
            agent_name="@H_0",
            metadata={"thread": "stale"},
        )
    )
    message = ChannelMessage(
        content="/new go",
        channel_user_id="user-1",
        command=ChannelCommand(name="new", rest="go"),
        binding_metadata={"thread": "t-2"},
    )

    await DefaultChannelRouter().route(message, _ctx(registry, StubTeamService()))

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.metadata == {"thread": "t-2"}


# --- The context cannot address another conversation's team ---


async def test_send_reaches_the_bound_team_and_no_other(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    bound = uuid.uuid4()
    await registry.register(
        ChannelBinding(channel="routed", channel_user_id="user-1", team_id=bound, agent_name="@H_0")
    )
    # Another conversation's team exists in the same registry.
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-2", team_id=uuid.uuid4(), agent_name="@H_0"
        )
    )
    team_service = StubTeamService()
    team_service.teams[bound] = _team_with_supervisor(bound)

    sent = await _ctx(registry, team_service).send("more")

    assert sent is True
    assert team_service.send_message_calls == [(bound, "more")]


async def test_send_on_an_unbound_conversation_sends_nothing(tmp_path: Path) -> None:
    team_service = StubTeamService()

    sent = await _ctx(YamlChannelRegistry(tmp_path / "registry.yaml"), team_service).send("hi")

    assert sent is False
    assert team_service.send_message_calls == []


_CONTEXT_SURFACE = {
    "address",
    "default_catalog_entry",
    "find_binding",
    "release",
    "initiate_team",
    "send",
    "send_to",
    "bound_process",
    "resume_bound_team",
    "bind_team",
    "notify",
}


def test_the_context_exposes_no_service_that_takes_a_team_id(tmp_path: Path) -> None:
    """The registry and the team service each have a method accepting an arbitrary team id.

    Re-exposing one — under any name — makes "address only the bound team" a
    rule a router has to remember instead of one it cannot break: the payload
    is unauthenticated, so that is the difference between a guarantee and a
    hope. The public surface is pinned whole, so a new method must be added
    here deliberately, after checking it resolves its team from the binding.

    ``bind_team`` is the one listed method that does not, and it is listed
    knowingly: it takes a team id so a user can re-attach their chat to a team
    they name, which is why the default router keeps it behind
    ``allow_register``. Adding a second such method is a decision, not a
    detail — that is what pinning the surface is for.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    ctx = _ctx(registry, team_service)

    public = {name for name in dir(ctx) if not name.startswith("_")}

    assert public == _CONTEXT_SURFACE
    exposed = [getattr(ctx, name) for name in public]
    assert not any(value is registry or value is team_service for value in exposed)


# --- The creation key travels parser → router → team_service ---


async def test_an_unbound_message_carries_its_creation_key_to_team_creation(
    tmp_path: Path,
) -> None:
    team_service = StubTeamService()
    key = uuid.uuid4()
    message = ChannelMessage(content="hello", channel_user_id="user-1", team_id=key)

    await DefaultChannelRouter().route(
        message, _ctx(YamlChannelRegistry(tmp_path / "registry.yaml"), team_service)
    )

    assert team_service.create_team_keys == [key]


async def test_new_carries_its_creation_key_to_team_creation(tmp_path: Path) -> None:
    team_service = StubTeamService()
    key = uuid.uuid4()
    message = ChannelMessage(
        content="/new go",
        channel_user_id="user-1",
        command=ChannelCommand(name="new", rest="go"),
        team_id=key,
    )

    await DefaultChannelRouter().route(
        message, _ctx(YamlChannelRegistry(tmp_path / "registry.yaml"), team_service)
    )

    assert team_service.create_team_keys == [key]


async def test_a_collapsed_initiation_still_delivers_its_own_message(tmp_path: Path) -> None:
    """The loser of a collapse binds to the winner's team and sends to it — nothing is lost.

    The team_service hands both initiations the same team, as a collapsed creation
    does. Both messages must reach it: a loser that skipped its send because the
    chat looked bound already would silently drop the second message.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    key = uuid.uuid4()
    ctx = _ctx(registry, team_service)

    first = await ctx.initiate_team("one", team_id=key)
    second = await ctx.initiate_team("two", team_id=key)

    assert first.team_id == second.team_id
    assert team_service.send_message_calls == [(first.team_id, "one"), (first.team_id, "two")]
    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == first.team_id


# --- The context calls TeamService directly, off the event loop ---


class _TextMessage(Message):
    """A typed message carrying text, to exercise the blank check on a Message."""

    content: str = ""


class _ThreadRecordingTeamService(StubTeamService):
    """Records which thread each TeamService call ran on."""

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[str] = []

    def create_team(self, *args: object, **kwargs: object) -> Process:  # type: ignore[override]
        self.threads.append(threading.current_thread().name)
        return super().create_team(*args, **kwargs)  # type: ignore[arg-type]

    def send_message(self, team_id: uuid.UUID, content: str | Message) -> None:
        self.threads.append(threading.current_thread().name)
        super().send_message(team_id, content)

    def send_message_from_to(
        self, team_id: uuid.UUID, sender_name: str, recipient_name: str, content: str | Message
    ) -> None:
        self.threads.append(threading.current_thread().name)
        super().send_message_from_to(team_id, sender_name, recipient_name, content)

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        self.threads.append(threading.current_thread().name)
        return super().get_team(team_id)


async def test_every_team_service_call_runs_off_the_event_loop(tmp_path: Path) -> None:
    """TeamService is synchronous and creation spawns actors for about a second.

    Called on the loop, it would stall every other request, WebSocket and
    background task in the process for that long.
    """
    team_service = _ThreadRecordingTeamService()
    ctx = _ctx(YamlChannelRegistry(tmp_path / "registry.yaml"), team_service)
    loop_thread = threading.current_thread().name

    await ctx.initiate_team("first")  # create_team + send_message
    await ctx.send("second")  # get_team (the default recipient) + send_message_from_to
    await ctx.send_to("@Manager_0", "third")  # send_message_from_to
    await ctx.bound_process()  # get_team

    assert len(team_service.threads) == 6
    assert loop_thread not in team_service.threads


async def test_a_blank_message_to_a_bound_team_is_not_sent(tmp_path: Path) -> None:
    """An empty prompt costs an LLM call answering nothing, and a reply guessing at it."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-1", team_id=uuid.uuid4(), agent_name="@H_0"
        )
    )
    team_service = StubTeamService()

    sent = await _ctx(registry, team_service).send("  \n\t ")

    assert sent is False
    assert team_service.send_message_calls == []


async def test_a_blank_first_message_still_creates_and_binds_the_team(tmp_path: Path) -> None:
    """Only the prompt is withheld: the binding is what lets the next message continue."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()

    process = await _ctx(registry, team_service).initiate_team("   ")

    assert team_service.create_team_calls == [("user-1", "routed-default", None)]
    assert team_service.send_message_calls == []
    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == process.team_id


async def test_a_typed_message_is_judged_by_its_content(tmp_path: Path) -> None:
    """Blank ``content`` is dropped; a Message with no content field is kept whole."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_id = uuid.uuid4()
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-1", team_id=team_id, agent_name="@H_0"
        )
    )
    team_service = StubTeamService()
    team_service.teams[team_id] = _team_with_supervisor(team_id)
    ctx = _ctx(registry, team_service)
    opaque = Message()

    assert await ctx.send(_TextMessage(content=" ")) is False
    assert await ctx.send(opaque) is True
    assert team_service.send_message_calls == [(team_id, opaque)]


async def test_a_preformed_message_reaches_the_team_as_the_same_instance(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_id = uuid.uuid4()
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-1", team_id=team_id, agent_name="@H_0"
        )
    )
    team_service = StubTeamService()
    team_service.teams[team_id] = _team_with_supervisor(team_id)
    message = _TextMessage(content="hello")

    await _ctx(registry, team_service).send(message)

    assert team_service.send_message_calls[0][1] is message


async def test_a_refused_creation_propagates_and_binds_nothing(tmp_path: Path) -> None:
    """The team service's error reaches the route unchanged, through the thread hop.

    And no binding is left behind for a team that was never created.
    """
    from akgentic.infra.errors import MetadataValidationError

    class Refusing(StubTeamService):
        def create_team(self, *args: object, **kwargs: object) -> Process:  # type: ignore[override]
            raise MetadataValidationError("case.id must be an integer")

    registry = YamlChannelRegistry(tmp_path / "registry.yaml")

    with pytest.raises(MetadataValidationError, match="case.id"):
        await _ctx(registry, Refusing()).initiate_team("hi")

    assert await registry.find_binding(_ADDRESS) is None


async def test_status_reads_the_bound_teams_state(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    ctx = _ctx(registry, team_service)
    created = await ctx.initiate_team(None)

    process = await ctx.bound_process()

    assert process is not None
    assert process.team_id == created.team_id


# --- /register: binding a chat to a team the message names ---


_ANOTHER_TEAM = uuid.UUID("11111111-2222-3333-4444-555555555555")
_ANOTHER_AGENT = "@HumanProxy_0"


def _enabled_router() -> DefaultChannelRouter:
    """The default router with ``/register`` turned on, as a channel config would."""
    return DefaultChannelRouter(allow_register="true")


def _register(rest: str = "", quoted: str | None = None) -> ChannelMessage:
    return ChannelMessage(
        content=f"/register {rest}".strip(),
        channel_user_id="user-1",
        command=ChannelCommand(name="register", rest=rest),
        quoted_text=quoted,
    )


async def test_register_is_refused_unless_the_channel_enables_it(tmp_path: Path) -> None:
    """The gate is the whole security story: off, the command binds nothing.

    An unauthenticated payload naming any team id must not move a chat onto
    that team, so the default router ships the command disabled and says so.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    adapter = StubAdapter()
    ctx = _ctx(registry, StubTeamService(), adapter)

    await DefaultChannelRouter().route(_register(f"{_ANOTHER_TEAM} {_ANOTHER_AGENT}"), ctx)

    assert await registry.find_binding(_ADDRESS) is None
    assert "not enabled" in adapter.notices[0][1]


def test_a_channel_naming_no_router_still_gets_its_own_config() -> None:
    """The default router is built per channel, with that channel's config.

    One shared config-less instance would silently drop every key a channel
    set for the router — ``allow_register`` among them, so the command answers
    "not enabled" on a channel whose config enables it, with nothing in the
    logs to say why.
    """
    config = _config()
    config["routed"].config["allow_register"] = "true"
    registry = ChannelParserRegistry(config)

    router = registry.get_router("routed")

    assert isinstance(router, DefaultChannelRouter)
    assert router._allow_register is True


def test_two_channels_do_not_share_one_default_router() -> None:
    """Enabling the command on one channel must not enable it on another."""
    config = _config()
    config["routed"].config["allow_register"] = "true"
    config["other"] = ChannelConfig(
        parser_fqcn=f"{_THIS_MODULE}.OtherStubParser",
        adapter_fqcn=f"{_THIS_MODULE}.StubAdapter",
        config={},
    )
    registry = ChannelParserRegistry(config)

    assert registry.get_router("routed")._allow_register is True  # type: ignore[union-attr]
    assert registry.get_router("other")._allow_register is False  # type: ignore[union-attr]


async def test_register_is_enabled_through_the_channel_config_end_to_end(
    tmp_path: Path,
) -> None:
    """The whole path: settings config -> registry -> router -> binding.

    The unit specs build the router directly, so every one of them passed while
    the wiring dropped the flag.
    """
    config = _config()
    config["routed"].config["allow_register"] = "true"
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    parser_registry = ChannelParserRegistry(config)
    StubParser.next_message = _register(f"{_ANOTHER_TEAM} {_ANOTHER_AGENT}")

    await parser_registry.get_router("routed").route(
        _register(f"{_ANOTHER_TEAM} {_ANOTHER_AGENT}"), _ctx(registry, team_service, adapter)
    )

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM


async def test_register_binds_the_team_and_agent_named_in_the_command(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    adapter = StubAdapter()
    ctx = _ctx(registry, StubTeamService(), adapter)

    await _enabled_router().route(_register(f"{_ANOTHER_TEAM} {_ANOTHER_AGENT}"), ctx)

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM
    assert binding.agent_name == _ANOTHER_AGENT
    assert str(_ANOTHER_TEAM) in adapter.notices[0][1]


async def test_register_binds_without_consulting_the_team_service(tmp_path: Path) -> None:
    """Neither name is verified, on purpose.

    A lookup would turn the command into an existence oracle — a chat could
    ask "is this id live?" and read the answer off the reply — while checking
    nothing that matters, since the payload carries no identity to compare
    against the team's owner. A binding naming nothing is inert: the next
    message finds no team, and the outbound path never matches it.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = _ThreadRecordingTeamService()
    ctx = _ctx(registry, team_service, StubAdapter())

    await _enabled_router().route(_register(f"{uuid.uuid4()} @NoSuchAgent_9"), ctx)

    assert team_service.threads == []
    assert await registry.find_binding(_ADDRESS) is not None


async def test_register_takes_both_names_from_the_replied_to_message(tmp_path: Path) -> None:
    """A bare ``/register`` reads the message it answers.

    This is what makes the command usable: the bot's own messages carry the
    team id and the agent name back into the chat.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ctx = _ctx(registry, StubTeamService(), StubAdapter())

    await _enabled_router().route(
        _register(quoted=f"Bound to team {_ANOTHER_TEAM} as {_ANOTHER_AGENT}."), ctx
    )

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM
    assert binding.agent_name == _ANOTHER_AGENT


async def test_register_prefers_the_command_text_over_the_quotation(tmp_path: Path) -> None:
    """What the user typed now beats what they replied to — for each name."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ctx = _ctx(registry, StubTeamService(), StubAdapter())
    quoted_team = uuid.uuid4()

    await _enabled_router().route(
        _register(f"{_ANOTHER_TEAM} {_ANOTHER_AGENT}", quoted=f"team {quoted_team} @Other_0"), ctx
    )

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM
    assert binding.agent_name == _ANOTHER_AGENT


async def test_register_takes_the_team_from_the_reply_and_the_agent_from_the_message(
    tmp_path: Path,
) -> None:
    """Each name falls back on its own, because they arrive from different places.

    A notice names the team and no agent. Replying to one and typing the agent
    supplies the missing half by hand — the case the command exists for.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ctx = _ctx(registry, StubTeamService(), StubAdapter())

    await _enabled_router().route(
        _register("@Expert_1", quoted=f"Started a new session — team {_ANOTHER_TEAM}."), ctx
    )

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM
    assert binding.agent_name == "@Expert_1"


async def test_register_takes_the_agent_from_the_reply_and_the_team_from_the_message(
    tmp_path: Path,
) -> None:
    """The mirror case: the quotation names the agent, the user types the team."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ctx = _ctx(registry, StubTeamService(), StubAdapter())

    await _enabled_router().route(
        _register(str(_ANOTHER_TEAM), quoted="@Expert_1 here, what would you like?"), ctx
    )

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM
    assert binding.agent_name == "@Expert_1"


async def test_register_binds_any_agent_name_the_message_carries(tmp_path: Path) -> None:
    """An agent below the first layer is a legitimate recipient.

    ``Process`` names the entry point and the first-layer supervisors only, so
    a deeper member like ``@Expert_1`` could never be confirmed — refusing what
    cannot be confirmed would reject valid agents.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ctx = _ctx(registry, StubTeamService(), StubAdapter())

    await _enabled_router().route(_register(f"{_ANOTHER_TEAM} @Expert_1"), ctx)

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.agent_name == "@Expert_1"


@pytest.mark.parametrize(
    "rest",
    ["11111111-2222-3333-4444-555555555555", "@HumanProxy_0", "nothing useful here"],
    ids=["team-only", "agent-only", "neither"],
)
async def test_register_needs_both_names_and_says_so(tmp_path: Path, rest: str) -> None:
    """Half a pair binds nothing: the binding is meaningless without both.

    There is no entry-point default, because learning the entry point's name
    would mean looking the team up — the one thing this command does not do.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    adapter = StubAdapter()
    ctx = _ctx(registry, StubTeamService(), adapter)

    await _enabled_router().route(_register(rest), ctx)

    assert await registry.find_binding(_ADDRESS) is None
    assert "/register" in adapter.notices[0][1]


async def test_register_replaces_the_previous_binding(tmp_path: Path) -> None:
    """A chat moves between teams and never holds two."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    ctx = _ctx(registry, team_service, StubAdapter())
    first = await ctx.initiate_team(None)

    await _enabled_router().route(_register(f"{_ANOTHER_TEAM} {_ANOTHER_AGENT}"), ctx)

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == _ANOTHER_TEAM
    assert registry.find_binding_sync(first.team_id, "@HumanProxy_0") is None


async def test_an_unknown_command_is_still_ordinary_text(tmp_path: Path) -> None:
    """Enabling register changes nothing for the fall-through branch."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    ctx = _ctx(registry, team_service, StubAdapter())
    message = ChannelMessage(
        content="/whatever please",
        channel_user_id="user-1",
        command=ChannelCommand(name="whatever", rest="please"),
    )

    await _enabled_router().route(message, ctx)

    assert team_service.create_team_calls


# --- A bound message is the bound agent speaking ---


async def _bind(registry: YamlChannelRegistry, agent_name: str = "@HumanProxy_0") -> uuid.UUID:
    """Bind the conversation to a team, as initiation would."""
    team_id = uuid.uuid4()
    await registry.register(
        ChannelBinding(
            channel="routed",
            channel_user_id="user-1",
            team_id=team_id,
            agent_name=agent_name,
        )
    )
    return team_id


def _inbound(content: str, quoted: str | None = None) -> ChannelMessage:
    return ChannelMessage(content=content, channel_user_id="user-1", quoted_text=quoted)


async def test_a_bound_message_is_sent_as_the_bound_agent(tmp_path: Path) -> None:
    """The chat IS the bound agent, so its messages carry that agent as sender.

    ``send_message_from_to`` takes a proxy for the sender and calls ``send()``
    on it, so the team sees one of its own members speaking rather than an
    anonymous injection.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry, "@Expert_1")
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(_inbound("a question"), _ctx(registry, team_service))

    assert team_service.send_from_to_calls == [(team_id, "@Expert_1", "@Manager_0", "a question")]


async def test_a_bound_message_goes_to_the_supervisor_by_default(tmp_path: Path) -> None:
    """No name in the message: the first supervisor answers."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service))

    assert team_service.send_from_to_calls[0][2] == "@Manager_0"


async def test_a_named_agent_in_the_message_is_the_recipient(tmp_path: Path) -> None:
    """Addressing an agent by name sends to it instead of the supervisor."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Expert_1 tell me a joke"), _ctx(registry, team_service)
    )

    assert team_service.send_from_to_calls[0][2] == "@Expert_1"


async def test_the_recipient_can_come_from_the_replied_to_message(tmp_path: Path) -> None:
    """Answering an agent's message in the chat addresses that agent.

    The bot's outbound messages name their sender, so replying to one is how a
    user carries on a conversation with that member without retyping its name.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("and another one?", quoted="@Expert_1: why did the programmer quit?"),
        _ctx(registry, team_service),
    )

    assert team_service.send_from_to_calls[0][2] == "@Expert_1"


async def test_the_typed_name_beats_the_quoted_one(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Manager_0 take this back", quoted="@Expert_1: here is the joke"),
        _ctx(registry, team_service),
    )

    assert team_service.send_from_to_calls[0][2] == "@Manager_0"


async def test_the_content_keeps_the_name_the_user_typed(tmp_path: Path) -> None:
    """An @Name is the user's sentence, not markup to strip."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Expert_1 tell me a joke"), _ctx(registry, team_service)
    )

    assert team_service.send_from_to_calls[0][3] == "@Expert_1 tell me a joke"


async def test_a_chat_bound_to_the_only_supervisor_addresses_the_entry_point(
    tmp_path: Path,
) -> None:
    """The sender is the bound agent on every path, fallback included.

    Sending through the team's default entry would have stamped the entry point
    as the sender, so a chat bound to ``@Manager_0`` would have spoken with
    someone else's voice — and no reader of the transcript could untangle it.
    The entry point is the seat left to address, and nobody talks to themselves.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry, "@Manager_0")
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service))

    assert team_service.send_from_to_calls == [(team_id, "@Manager_0", "@HumanProxy_0", "hello")]


def _lone_team(team_id: uuid.UUID) -> Process:
    """A team that is its own entry point and declares no supervisors."""
    now = datetime.now(UTC)
    return Process(
        team_id=team_id,
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=now,
        updated_at=now,
        entry_point=AgentRef(name="@HumanProxy_0", role="human_support"),
        agent_cards=[AgentCardRef(role="human_support", card_hash="stub-hash")],
    )


async def test_a_team_with_nobody_else_to_address_asks_the_user_to_name_one(
    tmp_path: Path,
) -> None:
    """No fallback sends the message anyway, and the user is told what to do.

    Every fallback available here changes who is speaking: the team's default
    entry stamps the entry point as the sender, and the bound agent itself
    would answer its own message. Delivering under the wrong name is worse
    than not delivering, and the user can fix it in their next message.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry, "@HumanProxy_0")
    team_service.teams[team_id] = _lone_team(team_id)

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service, adapter))

    assert team_service.send_from_to_calls == []
    assert team_service.send_message_calls == []
    assert "@Agent" in adapter.notices[0][1]


async def test_naming_the_agent_works_where_the_default_failed(tmp_path: Path) -> None:
    """The remedy the notice names actually works."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry, "@HumanProxy_0")
    team_service.teams[team_id] = _lone_team(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Expert_1 hello"), _ctx(registry, team_service, StubAdapter())
    )

    assert team_service.send_from_to_calls == [
        (team_id, "@HumanProxy_0", "@Expert_1", "@Expert_1 hello")
    ]


async def test_a_binding_naming_an_unknown_team_is_reported_to_the_chat(tmp_path: Path) -> None:
    """A binding that outlived its team is answered, not a silent redirect.

    This used to raise ``TeamNotFoundError`` out to the webhook, which answered
    non-2xx and the channel redelivered for ever. It is the third row of
    ADR-045 §D7: nothing is sent, nothing is created, the binding stays, and
    the user is told — with a remedy they can act on now. Healing it by
    starting a fresh team and rebinding is issue #487's.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry, "@HumanProxy_0")

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service, adapter))

    assert team_service.send_from_to_calls == []
    assert team_service.send_message_calls == []
    assert team_service.create_team_calls == []
    assert team_service.restore_team_calls == []
    ((_, notice),) = adapter.notices
    assert str(team_id) in notice
    assert "/new" in notice
    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == team_id


async def test_a_blank_bound_message_is_still_not_sent(tmp_path: Path) -> None:
    """The blank rule survives the sender change, on both paths."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)
    ctx = _ctx(registry, team_service)

    assert await ctx.send("   ") is False
    assert await ctx.send_to("@Manager_0", "\n") is False
    assert team_service.send_from_to_calls == []
    assert team_service.send_message_calls == []


def _team_with_supervisor(team_id: uuid.UUID) -> Process:
    """A persisted team whose first supervisor is @Manager_0."""
    now = datetime.now(UTC)
    return Process(
        team_id=team_id,
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=now,
        updated_at=now,
        entry_point=AgentRef(name="@HumanProxy_0", role="human_support"),
        supervisors=[AgentRef(name="@Manager_0", role="manager")],
        agent_cards=[
            AgentCardRef(role="human_support", card_hash="stub-hash"),
            AgentCardRef(role="manager", card_hash="stub-hash-manager"),
        ],
    )


def _stopped_team(team_id: uuid.UUID) -> Process:
    """The same team after an idle timeout stopped it."""
    return _team_with_supervisor(team_id).model_copy(update={"status": TeamStatus.STOPPED})


class _RacingTeamService(StubTeamService):
    """A team that comes up between the router's state read and its resume.

    ``get_team`` answers STOPPED — the read the router branches on — while the
    team itself is already RUNNING, so the resume meets
    ``TeamStateConflictError`` exactly as the real service raises it when
    something else got there first.
    """

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        process = super().get_team(team_id)
        if process is None:
            return None
        return process.model_copy(update={"status": TeamStatus.STOPPED})


# --- Row two: a stopped bound team is resumed, silently, and the message lands ---


async def test_a_message_to_a_stopped_bound_team_resumes_it_and_is_delivered(
    tmp_path: Path,
) -> None:
    """The inversion of the regression §D6 shipped alone — ADR-045 §D7 row two.

    The binding survives a stop (§D6), so this hook meets teams that are not
    running. It used to send to them unconditionally: the send failed out of
    ``TeamService._get_running_handle`` with a ``TeamStateConflictError``, the
    webhook answered non-2xx, and the channel redelivered for ever. Now the
    state is resolved first and a stopped team is brought back.

    **Silently.** A notice on every post-idle message would make the idle
    timeout a user-visible rule again, which is what §D6 removes. And exactly
    one resume, no second team: the conversation continued, it did not restart.

    The resume goes through ``restore_team``, which is what warms the handle
    cache — hence ``cold_handles``: a resume through the placement alone leaves
    the team running with a cold handle, and the send that follows dies on
    ``handle not cached``.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _stopped_team(team_id)

    await DefaultChannelRouter().route(
        _inbound("still there?"), _ctx(registry, team_service, adapter)
    )

    assert team_service.restore_team_calls == [team_id]
    assert team_service.cold_handles == set()
    assert team_service.send_from_to_calls == [
        (team_id, "@HumanProxy_0", "@Manager_0", "still there?")
    ]
    assert team_service.create_team_calls == []
    assert adapter.notices == []
    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == team_id


async def test_a_named_recipient_message_also_resumes_the_stopped_bound_team(
    tmp_path: Path,
) -> None:
    """Row two on the other dispatch path.

    ``on_bound`` resolves the state once, above the recipient branch. Put the
    check inside the default-recipient branch only and this spec goes red
    while its sibling above stays green — which is half the regression left
    behind.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _stopped_team(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Expert_1 still there?"), _ctx(registry, team_service, adapter)
    )

    assert team_service.restore_team_calls == [team_id]
    assert team_service.send_from_to_calls == [
        (team_id, "@HumanProxy_0", "@Expert_1", "@Expert_1 still there?")
    ]
    assert adapter.notices == []


async def test_a_deleted_bound_team_is_reported_rather_than_resumed(tmp_path: Path) -> None:
    """A team the service still knows, but has deleted, is the third row too.

    ``restore_team`` refuses a deleted team with a bare ``ValueError``, and
    that falls through to the notice with the not-found case — there is
    nothing to resume and nothing to send.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id).model_copy(
        update={"status": TeamStatus.DELETED}
    )

    await DefaultChannelRouter().route(_inbound("hello?"), _ctx(registry, team_service, adapter))

    assert team_service.send_from_to_calls == []
    assert team_service.create_team_calls == []
    ((_, notice),) = adapter.notices
    assert "/new" in notice


async def test_a_team_brought_up_before_the_resume_lands_still_gets_the_message(
    tmp_path: Path,
) -> None:
    """The race between the state read and the resume is resolved, not propagated.

    The read said stopped; by the time the resume ran, a concurrent message or
    an operator had already brought the team up. ``TeamStateConflictError``
    means the team is running, which is what the message needed — so it goes,
    and the user never learns there was a race.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = _RacingTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service, adapter))

    assert team_service.restore_team_calls == []
    assert team_service.send_from_to_calls == [(team_id, "@HumanProxy_0", "@Manager_0", "hello")]
    assert adapter.notices == []


# --- Row one: a running team behaves exactly as it did, on both paths ---


async def test_a_running_bound_team_is_not_resumed_on_the_default_path(tmp_path: Path) -> None:
    """Row one is byte-for-byte what it was: no resume, same recipient, same content."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service, adapter))

    assert team_service.restore_team_calls == []
    assert team_service.send_from_to_calls == [(team_id, "@HumanProxy_0", "@Manager_0", "hello")]
    assert adapter.notices == []


async def test_a_running_bound_team_is_not_resumed_on_the_named_path(tmp_path: Path) -> None:
    """Row one on the named-recipient path — the name and the content survive it."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Expert_1 a joke please"), _ctx(registry, team_service, adapter)
    )

    assert team_service.restore_team_calls == []
    assert team_service.send_from_to_calls == [
        (team_id, "@HumanProxy_0", "@Expert_1", "@Expert_1 a joke please")
    ]
    assert adapter.notices == []


# --- A blank body costs nothing, whatever state the bound team is in ---


async def test_a_blank_message_does_not_resume_a_stopped_bound_team(tmp_path: Path) -> None:
    """The blank rule runs before the state is resolved, not after it.

    ``ctx.send`` and ``ctx.send_to`` hold the rule, and both run *below* the
    state resolution. Resolve first and a caption-less photo or a stray
    newline places and starts a whole team runtime, and then the message it
    was resumed for is thrown away — a larger version of the LLM call
    ``_is_blank`` was written to avoid.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _stopped_team(team_id)

    await DefaultChannelRouter().route(_inbound("  \n\t "), _ctx(registry, team_service, adapter))

    assert team_service.restore_team_calls == []
    assert team_service.send_from_to_calls == []
    assert team_service.send_message_calls == []
    assert team_service.create_team_calls == []
    assert adapter.notices == []
    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == team_id


async def test_a_blank_message_to_a_lost_bound_team_is_not_reported(tmp_path: Path) -> None:
    """Row three is owed to a user who typed something, and a blank is not that.

    The notice exists so that nobody is left with silence after saying
    something. Announcing a lost team because a sticker arrived is noise the
    user cannot act on and did not ask for.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    await _bind(registry)

    await DefaultChannelRouter().route(_inbound("   "), _ctx(registry, team_service, adapter))

    assert adapter.notices == []
    assert team_service.send_from_to_calls == []
    assert team_service.create_team_calls == []


# --- Every creation announces its team and agent ---


async def test_a_first_message_announces_the_team_it_started(tmp_path: Path) -> None:
    """An unbound chat learns its team id the moment one is created for it.

    The notice is the only place a chat ever sees that id, and ``register``
    reads it back out of a replied-to message — so without this, re-attaching
    a chat means fetching the id from the web UI.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()

    await DefaultChannelRouter().route(
        _inbound("hello there"), _ctx(registry, team_service, adapter)
    )

    assert len(adapter.notices) == 1
    assert str(team_service.next_team_id) in adapter.notices[0][1]


async def test_the_announcement_names_the_bound_agent_too(tmp_path: Path) -> None:
    """``register`` needs both names, so the notice a user replies to carries both."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()

    await DefaultChannelRouter().route(_inbound("hello"), _ctx(registry, team_service, adapter))

    assert "@HumanProxy_0" in adapter.notices[0][1]


async def test_new_and_a_first_message_announce_the_same_way(tmp_path: Path) -> None:
    """One creation path, one notice: the two must not drift apart."""
    first_adapter = StubAdapter()
    first_service = StubTeamService()
    shared_team_id = first_service.next_team_id
    await DefaultChannelRouter().route(
        _inbound("hello"),
        _ctx(YamlChannelRegistry(tmp_path / "a.yaml"), first_service, first_adapter),
    )

    new_adapter = StubAdapter()
    new_service = StubTeamService()
    new_service.next_team_id = shared_team_id
    await DefaultChannelRouter().route(
        ChannelMessage(
            content="/new hello",
            channel_user_id="user-1",
            command=ChannelCommand(name="new", rest="hello"),
        ),
        _ctx(YamlChannelRegistry(tmp_path / "b.yaml"), new_service, new_adapter),
    )

    assert first_adapter.notices[0][1] == new_adapter.notices[0][1]


async def test_a_registered_chat_can_be_rebound_from_the_announcement(tmp_path: Path) -> None:
    """End to end: the notice a chat received is enough to bind another chat.

    This is the round trip the two features exist for — the announcement
    carries a team id and an agent name, and ``register`` reads both back out
    of a reply.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    adapter = StubAdapter()
    ctx = _ctx(registry, team_service, adapter)
    await DefaultChannelRouter().route(_inbound("hello"), ctx)
    announcement = adapter.notices[0][1]
    await ctx.release()

    await _enabled_router().route(_register(quoted=announcement), ctx)

    binding = await registry.find_binding(_ADDRESS)
    assert binding is not None
    assert binding.team_id == team_service.next_team_id
    assert binding.agent_name == "@HumanProxy_0"


async def test_a_leading_name_addresses_and_the_rest_is_the_message(tmp_path: Path) -> None:
    """ "@Expert, ask a joke to @Support" is for the Expert.

    The second name is what the Expert is being asked to do. Routing on any
    name in the sentence would send the user's instruction to the agent it
    names as its subject.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("@Expert_1, ask a joke to @Support_0"), _ctx(registry, team_service)
    )

    assert team_service.send_from_to_calls == [
        (team_id, "@HumanProxy_0", "@Expert_1", "@Expert_1, ask a joke to @Support_0")
    ]


@pytest.mark.parametrize(
    "content",
    [
        "ask @Expert_1 for a joke",
        "the answer came from @Expert_1",
        "tell @Expert_1 I said hello",
    ],
    ids=["mid-sentence", "trailing", "instruction"],
)
async def test_a_name_that_is_not_first_does_not_address(tmp_path: Path, content: str) -> None:
    """A name inside the sentence belongs to the sentence."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(_inbound(content), _ctx(registry, team_service))

    assert team_service.send_from_to_calls[0][2] == "@Manager_0"


async def test_leading_whitespace_does_not_hide_the_address(tmp_path: Path) -> None:
    """A copy-pasted message often carries a leading newline."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound("  \n@Expert_1 hello"), _ctx(registry, team_service)
    )

    assert team_service.send_from_to_calls[0][2] == "@Expert_1"


async def test_a_quoted_name_is_found_anywhere_in_the_bots_own_text(tmp_path: Path) -> None:
    """The quotation is the adapter's wording, not the user's sentence.

    A delivered message reads "You received a message from @Expert_1: …", so
    the name never comes first there. Applying the leading-name rule to it
    would make replying useless.
    """
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_service = StubTeamService()
    team_id = await _bind(registry)
    team_service.teams[team_id] = _team_with_supervisor(team_id)

    await DefaultChannelRouter().route(
        _inbound(
            "and another one?", quoted="You received a message from @Expert_1: \n\nhere it is"
        ),
        _ctx(registry, team_service),
    )

    assert team_service.send_from_to_calls[0][2] == "@Expert_1"

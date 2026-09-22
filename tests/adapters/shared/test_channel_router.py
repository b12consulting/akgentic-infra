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

from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.shared.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
)
from akgentic.infra.adapters.shared.channel_router import (
    ChannelRouteContext,
    DefaultChannelRouter,
)
from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelBinding,
    ChannelCommand,
    ChannelMessage,
    InteractionChannelRouter,
    JsonValue,
)
from akgentic.infra.server.routes.webhook import router as webhook_router

_THIS_MODULE = "tests.adapters.shared.test_channel_router"


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


class StubTeamService:
    """The TeamService the router context calls, recording every call.

    Synchronous like the real one — ``ChannelRouteContext`` calls it through
    ``asyncio.to_thread``. Given a registry, it also records whether the chat was
    already bound when each message was sent: the ordering the create / bind /
    send sequence exists to guarantee.
    """

    def __init__(self, registry: YamlChannelRegistry | None = None) -> None:
        self.send_message_calls: list[tuple[uuid.UUID, str | Message]] = []
        self.create_team_calls: list[tuple[str, str, dict[str, JsonValue] | None]] = []
        self.create_team_keys: list[uuid.UUID | None] = []
        self.bound_at_send: list[bool] = []
        self.next_team_id = uuid.uuid4()
        self.teams: dict[uuid.UUID, Process] = {}
        self._registry = registry

    def send_message(self, team_id: uuid.UUID, content: str | Message) -> None:
        self.send_message_calls.append((team_id, content))
        if self._registry is not None:
            binding = self._registry.find_binding_sync(team_id, "@HumanProxy_0")
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
            agent_cards=[AgentCardRef(role="human_support", card_hash="stub-hash")],
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
        _config("akgentic.infra.adapters.shared.channel_router.DefaultChannelRouter")
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
    "bound_process",
    "notify",
}


def test_the_context_exposes_no_service_that_takes_a_team_id(tmp_path: Path) -> None:
    """The registry and the team service each have a method accepting an arbitrary team id.

    Re-exposing one — under any name — makes "address only the bound team" a
    rule a router has to remember instead of one it cannot break: the payload
    is unauthenticated, so that is the difference between a guarantee and a
    hope. The public surface is pinned whole, so a new method must be added
    here deliberately, after checking it resolves its team from the binding.
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
    await ctx.send("second")  # send_message
    await ctx.bound_process()  # get_team

    assert len(team_service.threads) == 4
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

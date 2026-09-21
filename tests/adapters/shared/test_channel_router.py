"""Tests for channel routing — ChannelRouteContext, DefaultChannelRouter and router resolution."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
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
    InitiatedTeam,
    InteractionChannelRouter,
    JsonValue,
)
from akgentic.infra.server.routes.webhook import router as webhook_router

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message


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


class StubIngestion:
    """InteractionChannelIngestion recording every call.

    Given a registry, it also records whether the chat was already bound when
    each message was sent — the ordering the split exists to guarantee.
    """

    def __init__(self, registry: YamlChannelRegistry | None = None) -> None:
        self.send_message_calls: list[tuple[uuid.UUID, str | Message]] = []
        self.create_team_calls: list[tuple[str, str, dict[str, JsonValue] | None]] = []
        self.bound_at_send: list[bool] = []
        self.next_team_id = uuid.uuid4()
        self._registry = registry

    async def send_message(
        self,
        team_id: uuid.UUID,
        content: str | Message,
        original_message_id: str | None = None,
    ) -> None:
        self.send_message_calls.append((team_id, content))
        if self._registry is not None:
            binding = self._registry.find_binding_sync(team_id, "@HumanProxy_0")
            self.bound_at_send.append(binding is not None)

    async def create_team(
        self,
        channel_user_id: str,
        catalog_entry_id: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> InitiatedTeam:
        self.create_team_calls.append((channel_user_id, catalog_entry_id, metadata))
        return InitiatedTeam(team_id=self.next_team_id, entry_point_name="@HumanProxy_0")


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
    ingestion: StubIngestion,
    adapter: StubAdapter | None = None,
) -> ChannelRouteContext:
    return ChannelRouteContext(
        address=ChannelAddress(channel="routed", channel_user_id="user-1"),
        registry=registry,
        ingestion=ingestion,
        team_service=None,  # type: ignore[arg-type]  # no spec here reaches `status`
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
    app.state.ingestion = StubIngestion()
    app.state.team_service = object()
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
    assert app.state.ingestion.create_team_calls == []
    assert app.state.ingestion.send_message_calls == []


# --- ChannelRouteContext ---


async def test_initiate_team_always_binds_the_conversation(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ingestion = StubIngestion()
    ctx = _ctx(registry, ingestion)

    initiated = await ctx.initiate_team(None)

    binding = await registry.find_binding(
        ChannelAddress(channel="routed", channel_user_id="user-1")
    )
    assert binding is not None
    assert binding.team_id == initiated.team_id
    assert binding.agent_name == "@HumanProxy_0"
    # Outbound delivery reads the same record, so it must be answerable there too.
    assert registry.find_binding_sync(initiated.team_id, "@HumanProxy_0") == binding
    assert ingestion.create_team_calls == [("user-1", "routed-default", None)]
    # None is "create silently": no first message at all, not an empty one.
    assert ingestion.send_message_calls == []


async def test_initiate_team_separates_team_and_binding_metadata(tmp_path: Path) -> None:
    """Team metadata goes to creation; binding metadata goes to the stored record."""
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ingestion = StubIngestion()
    ctx = _ctx(registry, ingestion)

    await ctx.initiate_team(
        "hi",
        catalog_entry="chosen",
        team_metadata={"case": "42"},
        binding_metadata={"thread": "t-9"},
    )

    assert ingestion.create_team_calls == [("user-1", "chosen", {"case": "42"})]
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
    ingestion = StubIngestion(registry)

    await _ctx(registry, ingestion).initiate_team("first")

    assert ingestion.send_message_calls == [(ingestion.next_team_id, "first")]
    assert ingestion.bound_at_send == [True]


async def test_release_returns_what_it_released(tmp_path: Path) -> None:
    registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    team_id = uuid.uuid4()
    await registry.register(
        ChannelBinding(
            channel="routed", channel_user_id="user-1", team_id=team_id, agent_name="@H_0"
        )
    )
    ctx = _ctx(registry, StubIngestion())

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
    ingestion = StubIngestion()
    adapter = StubAdapter()
    ctx = _ctx(registry, ingestion, adapter)
    router = ReplyOnlyRouter()

    await router.route(ChannelMessage(content="hello", channel_user_id="user-1"), ctx)
    assert ingestion.create_team_calls == []
    assert adapter.notices[-1][1] == "Say /new to start."

    new = ChannelMessage(
        content="/new go", channel_user_id="user-1", command=ChannelCommand(name="new", rest="go")
    )
    await router.route(new, ctx)
    assert ingestion.create_team_calls == [("user-1", "routed-default", None)]
    assert ingestion.send_message_calls == [(ingestion.next_team_id, "go")]
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
    ingestion = StubIngestion()

    await DefaultChannelRouter().route(
        ChannelMessage(content="more", channel_user_id="user-1"), _ctx(registry, ingestion)
    )

    assert ingestion.send_message_calls == [(team_id, "more")]
    assert ingestion.create_team_calls == []

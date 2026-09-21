"""Tests for the webhook route — POST /webhook/{channel} through the default channel router."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from akgentic.team.models import AgentCardRef, AgentRef, Process, TeamStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.shared.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelBinding,
    ChannelCommand,
    ChannelMessage,
    InitiatedTeam,
    JsonValue,
)
from akgentic.infra.server.errors import add_server_exception_handlers
from akgentic.infra.server.routes.webhook import router as webhook_router

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message

# ---------------------------------------------------------------------------
# Stub classes satisfying protocols via structural subtyping
# ---------------------------------------------------------------------------


class StubParser:
    """Stub ChannelParser that returns a configurable ChannelMessage."""

    def __init__(
        self,
        channel: str = "test-channel",
        default_entry: str = "default-catalog",
    ) -> None:
        self._channel = channel
        self._default_entry = default_entry
        self._next_message: ChannelMessage | None = None

    @property
    def channel_name(self) -> str:
        return self._channel

    @property
    def default_catalog_entry(self) -> str:
        return self._default_entry

    def set_next_message(self, msg: ChannelMessage) -> None:
        """Configure the message that parse() will return."""
        self._next_message = msg

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        if self._next_message is not None:
            return self._next_message
        return ChannelMessage(
            content=str(payload.get("text", "")),
            channel_user_id=str(payload.get("user", "unknown")),
        )


class StubIngestion:
    """Stub InteractionChannelIngestion that tracks calls."""

    def __init__(self) -> None:
        self.send_message_calls: list[tuple[uuid.UUID, str | Message, str | None]] = []
        # Anything the route passes to send_message beyond the three declared
        # parameters lands here, so "the reply path forwards no metadata" is an
        # assertion about recorded evidence rather than about a TypeError.
        self.send_message_extra_kwargs: list[dict[str, object]] = []
        self.create_team_calls: list[tuple[str, str, dict[str, JsonValue] | None]] = []
        self._next_team_id: uuid.UUID = uuid.uuid4()
        self._next_entry_point_name: str = "@HumanProxy_0"

    def set_next_team_id(self, team_id: uuid.UUID) -> None:
        self._next_team_id = team_id

    def set_next_entry_point_name(self, entry_point_name: str) -> None:
        """Configure the entry-point name the next initiation reports.

        The route copies this into the binding it writes, so a spec that wants
        to assert *which* value reached ``agent_name`` needs a seam to set it.
        """
        self._next_entry_point_name = entry_point_name

    async def send_message(
        self,
        team_id: uuid.UUID,
        content: str | Message,
        original_message_id: str | None = None,
        **extra: object,
    ) -> None:
        self.send_message_calls.append((team_id, content, original_message_id))
        self.send_message_extra_kwargs.append(extra)

    async def create_team(
        self,
        channel_user_id: str,
        catalog_entry_id: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> InitiatedTeam:
        self.create_team_calls.append((channel_user_id, catalog_entry_id, metadata))
        return InitiatedTeam(
            team_id=self._next_team_id,
            entry_point_name=self._next_entry_point_name,
        )


class StubNoticeAdapter:
    """Records the acknowledgements the route fans out, and to which address."""

    def __init__(self, channel: str = "test-channel") -> None:
        self._channel = channel
        self.notices: list[tuple[ChannelAddress, str]] = []

    def matches(self, msg: object, binding: object) -> bool:
        return False

    def deliver(self, msg: object, binding: object) -> None:
        pass

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        # Mirrors the real adapter: a notice for another channel is not ours.
        if address.channel != self._channel:
            return
        self.notices.append((address, text))

    def on_stop(self, team_id: uuid.UUID) -> None:
        pass


class StubTeamService:
    """Returns a **real** ``Process``, never a look-alike.

    A fake carrying only a ``.status`` attribute cannot notice the field being
    renamed, and ``status`` reports that field's value verbatim.
    """

    def __init__(self) -> None:
        self.teams: dict[uuid.UUID, Process] = {}
        self.get_team_calls: list[uuid.UUID] = []

    def add_running_team(self, team_id: uuid.UUID, status: TeamStatus = TeamStatus.RUNNING) -> None:
        now = datetime.now(UTC)
        self.teams[team_id] = Process(
            team_id=team_id,
            status=status,
            created_at=now,
            updated_at=now,
            entry_point=AgentRef(name="@HumanProxy_0", role="human_support"),
            agent_cards=[AgentCardRef(role="human_support", card_hash="stub-hash")],
        )

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        self.get_team_calls.append(team_id)
        return self.teams.get(team_id)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _build_parser_registry(
    parser: StubParser,
    monkeypatch: pytest.MonkeyPatch | None = None,
    adapter: StubNoticeAdapter | None = None,
) -> ChannelParserRegistry:
    """Build a ChannelParserRegistry with a pre-registered stub parser.

    Constructs via the public API with an empty config, then monkeypatches
    get_parser to return the stub. This avoids __new__ hacks and private
    attribute access. ``get_adapters`` is patched the same way, rather than
    reaching into ``registry._adapters``.
    """
    registry = ChannelParserRegistry(channels_config={})

    original_get_parser = registry.get_parser

    def _patched_get_parser(channel_name: str) -> StubParser | None:
        if channel_name == parser.channel_name:
            return parser  # type: ignore[return-value]
        return original_get_parser(channel_name)

    adapters = [adapter] if adapter is not None else []

    def _patched_get_adapters() -> list[StubNoticeAdapter]:
        return list(adapters)

    if monkeypatch is not None:
        monkeypatch.setattr(registry, "get_parser", _patched_get_parser)
        monkeypatch.setattr(registry, "get_adapters", _patched_get_adapters)
    else:
        registry.get_parser = _patched_get_parser  # type: ignore[assignment]
        registry.get_adapters = _patched_get_adapters  # type: ignore[assignment]

    return registry


def _build_app(
    parser: StubParser,
    ingestion: StubIngestion,
    channel_registry: YamlChannelRegistry,
    adapter: StubNoticeAdapter | None = None,
    team_service: StubTeamService | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with the webhook router wired.

    ``add_server_exception_handlers`` is the same registration the real assembly
    installs, so a ``ServerError`` raised by the ingestion layer is mapped here
    exactly as it is in production — a status assertion against a bare
    ``FastAPI()`` would only prove the TestClient re-raises.
    """
    app = FastAPI()
    parser_registry = _build_parser_registry(parser, adapter=adapter)
    app.state.channel_parser_registry = parser_registry
    app.state.channel_registry = channel_registry
    app.state.ingestion = ingestion
    app.state.team_service = team_service or StubTeamService()
    app.include_router(webhook_router)
    add_server_exception_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebhookReplyFlow:
    """A bound conversation's message → send_message to the bound team, verbatim."""

    async def test_reply_flow_calls_send_message(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="reply msg",
                channel_user_id="user-1",
                channel_message_id="msg-abc",
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-1",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(ingestion.send_message_calls) == 1
        call = ingestion.send_message_calls[0]
        assert call[0] == team_id
        assert call[1] == "reply msg"
        assert call[2] == "msg-abc"


class TestWebhookContinuationFlow:
    """AC #4: no team_id but registered team → send_message (forwarding message_id)."""

    async def test_continuation_flow_calls_send_message(self, tmp_path: Path) -> None:
        existing_team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="continuation msg",
                channel_user_id="user-2",
                channel_message_id="msg-cont",
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        # Pre-register a team for this user
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-2",
                team_id=existing_team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(ingestion.send_message_calls) == 1
        call = ingestion.send_message_calls[0]
        assert call[0] == existing_team_id
        assert call[1] == "continuation msg"
        # Story 30.3: the continuation flow forwards the parsed message_id as the
        # third positional arg, in lockstep with the reply flow (TestWebhookReplyFlow).
        assert call[2] == "msg-cont"


class TestWebhookInitiationFlow:
    """No binding for the conversation → create_team, bind, then send."""

    def test_initiation_flow_creates_the_team_then_sends(self, tmp_path: Path) -> None:
        parser = StubParser(default_entry="my-catalog-entry")
        parser.set_next_message(
            ChannelMessage(
                content="new convo",
                channel_user_id="user-3",
            )
        )
        new_team_id = uuid.uuid4()
        ingestion = StubIngestion()
        ingestion.set_next_team_id(new_team_id)
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.create_team_calls == [("user-3", "my-catalog-entry", None)]
        assert ingestion.send_message_calls == [(new_team_id, "new convo", None)]

    async def test_initiation_registers_in_channel_registry(self, tmp_path: Path) -> None:
        parser = StubParser()
        parser.set_next_message(ChannelMessage(content="hello", channel_user_id="user-4"))
        new_team_id = uuid.uuid4()
        ingestion = StubIngestion()
        ingestion.set_next_team_id(new_team_id)
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        client.post("/webhook/test-channel", json={"text": "hi"})

        # Verify registration happened
        found = await registry.find_binding(
            ChannelAddress(channel="test-channel", channel_user_id="user-4")
        )
        assert found is not None
        assert found.team_id == new_team_id

    async def test_initiation_binds_the_channel_to_the_entry_point_agent(
        self, tmp_path: Path
    ) -> None:
        """AC 12: the persisted record carries the ingestion's entry-point name.

        ``agent_name`` must be what ``create_team`` reported, not the channel
        user or anything else the route has to hand — the outbound lookup starts
        from an agent, and a wrong name there is a silent delivery failure.
        ``channel`` must be the path segment: the route is the only component
        that knows which channel the message arrived on.
        """
        parser = StubParser()
        parser.set_next_message(ChannelMessage(content="hello", channel_user_id="user-6"))
        new_team_id = uuid.uuid4()
        ingestion = StubIngestion()
        ingestion.set_next_team_id(new_team_id)
        ingestion.set_next_entry_point_name("@HumanProxy_0")
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        client.post("/webhook/test-channel", json={"text": "hi"})

        binding = await registry.find_binding(
            ChannelAddress(channel="test-channel", channel_user_id="user-6")
        )
        assert binding == ChannelBinding(
            channel="test-channel",
            channel_user_id="user-6",
            team_id=new_team_id,
            agent_name="@HumanProxy_0",
        )

    async def test_a_corrupt_channel_section_leaves_no_orphan_team(self, tmp_path: Path) -> None:
        """Every team this branch creates is bound, corrupt store included.

        Guarding ``find_binding`` alone moved the failure *past the point of no
        return*: pre-epic a scalar section raised in ``find_team``, before any
        team existed; guarded, the read answers None, the branch creates a team,
        and an unguarded ``register`` raised only then — leaking one orphan team
        per delivery retry, and a channel retries with backoff.

        The invariant is stated as this branch never creating a team it cannot
        bind. Reordering is not available — the binding carries the team id, so
        it cannot be written first — so the guarantee is bought by making the
        registry write total for the input class that breaks it.
        """
        registry_path = tmp_path / "registry.yaml"
        registry_path.write_text(yaml.safe_dump({"test-channel": "oops"}), encoding="utf-8")
        parser = StubParser()
        parser.set_next_message(ChannelMessage(content="hello", channel_user_id="user-corrupt"))
        new_team_id = uuid.uuid4()
        ingestion = StubIngestion()
        ingestion.set_next_team_id(new_team_id)
        registry = YamlChannelRegistry(registry_path)
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(ingestion.create_team_calls) == 1
        binding = await registry.find_binding(
            ChannelAddress(channel="test-channel", channel_user_id="user-corrupt")
        )
        assert binding is not None
        assert binding.team_id == new_team_id


class TestWebhookUnknownChannel:
    """AC #2: unknown channel → 404."""

    def test_unknown_channel_returns_404(self, tmp_path: Path) -> None:
        parser = StubParser(channel="known-channel")
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/unknown-channel", json={"text": "hi"})

        assert resp.status_code == 404
        assert "Unknown channel" in resp.json()["detail"]


class TestWebhookStatusCode:
    """AC: all successful flows return 204 No Content."""

    async def test_reply_returns_204(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="msg",
                channel_user_id="u",
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="u",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})
        assert resp.status_code == 204

    def test_initiation_returns_204(self, tmp_path: Path) -> None:
        parser = StubParser()
        parser.set_next_message(ChannelMessage(content="msg", channel_user_id="u"))
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# AC #4: Form-data and unsupported content-type handling
# ---------------------------------------------------------------------------


class TestWebhookFormData:
    """AC #4: webhook handles application/x-www-form-urlencoded."""

    def test_form_data_payload_parsed(self, tmp_path: Path) -> None:
        parser = StubParser()
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post(
            "/webhook/test-channel",
            data={"text": "form hello", "user": "form-user"},
        )

        assert resp.status_code == 204
        assert len(ingestion.create_team_calls) == 1
        assert ingestion.create_team_calls[0][0] == "form-user"
        assert ingestion.send_message_calls[0][1] == "form hello"


class TestWebhookContentTypeEdgeCases:
    """AC #4: content-type edge cases."""

    def test_missing_content_type_returns_415(self, tmp_path: Path) -> None:
        """Request with no content-type header returns 415."""
        parser = StubParser()
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post(
            "/webhook/test-channel",
            content=b"some data",
            headers={"content-type": ""},
        )

        assert resp.status_code == 415

    async def test_json_with_charset_param(self, tmp_path: Path) -> None:
        """application/json; charset=utf-8 is handled as JSON."""
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="charset msg",
                channel_user_id="u-charset",
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="u-charset",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post(
            "/webhook/test-channel",
            json={"text": "hi"},
            headers={"content-type": "application/json; charset=utf-8"},
        )

        assert resp.status_code == 204
        assert len(ingestion.send_message_calls) == 1


class TestWebhookUnsupportedContentType:
    """AC #4: unsupported content-type returns 415."""

    def test_unsupported_content_type_returns_415(self, tmp_path: Path) -> None:
        parser = StubParser()
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post(
            "/webhook/test-channel",
            content=b"<xml>data</xml>",
            headers={"content-type": "application/xml"},
        )

        assert resp.status_code == 415
        assert "Unsupported content type" in resp.json()["detail"]


class TestWebhookMalformedPayload:
    """A parser-raised ValueError is acknowledged and dropped, never refused.

    A channel treats any non-2xx as a failed delivery and redelivers with
    backoff, so answering 4xx to an update the parser can never accept queues it
    forever. Dropping is the only terminal answer available.
    """

    def test_parser_value_error_is_acknowledged_not_refused(self, tmp_path: Path) -> None:
        class RaisingParser(StubParser):
            async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
                del payload
                raise ValueError("payload missing required field 'message'")

        parser = RaisingParser()
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"update_id": 1})

        assert resp.status_code == 204
        assert resp.content == b""
        assert ingestion.send_message_calls == []
        assert ingestion.create_team_calls == []


# ---------------------------------------------------------------------------
# Business metadata on the initiation branch
# ---------------------------------------------------------------------------


class TestWebhookMetadataForwarding:
    """The metadata the parser lifted reaches team creation, and only there."""

    def test_initiation_forwards_parsed_metadata(self, tmp_path: Path) -> None:
        metadata: dict[str, JsonValue] = {"tenant": "acme", "case": {"id": 7, "tags": ["a"]}}
        parser = StubParser(default_entry="my-catalog-entry")
        parser.set_next_message(
            ChannelMessage(
                content="new convo",
                channel_user_id="user-m1",
                metadata=metadata,
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(ingestion.create_team_calls) == 1
        assert ingestion.create_team_calls[0][2] == metadata

    def test_initiation_without_metadata_forwards_none(self, tmp_path: Path) -> None:
        parser = StubParser(default_entry="my-catalog-entry")
        parser.set_next_message(ChannelMessage(content="new convo", channel_user_id="user-m2"))
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.create_team_calls[0][2] is None

    async def test_reply_flow_forwards_no_metadata(self, tmp_path: Path) -> None:
        """A reply addresses a team whose metadata was fixed at creation."""
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="reply msg",
                channel_user_id="user-m3",
                metadata={"tenant": "acme"},
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-m3",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.create_team_calls == []
        assert ingestion.send_message_extra_kwargs == [{}]

    async def test_continuation_flow_forwards_no_metadata(self, tmp_path: Path) -> None:
        """A continuation addresses an existing team — same reasoning as a reply."""
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="continuation msg",
                channel_user_id="user-m4",
                metadata={"tenant": "acme"},
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-m4",
                team_id=uuid.uuid4(),
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.create_team_calls == []
        assert ingestion.send_message_extra_kwargs == [{}]


class TestWebhookMetadataRejection:
    """A metadata body the card refuses answers 422, with the validator's words."""

    def test_invalid_metadata_returns_422_with_validator_message(self, tmp_path: Path) -> None:
        detail = "metadata field 'case.id' must be an integer"

        class RefusingIngestion(StubIngestion):
            async def create_team(
                self,
                channel_user_id: str,
                catalog_entry_id: str,
                metadata: dict[str, JsonValue] | None = None,
            ) -> InitiatedTeam:
                raise MetadataValidationError(detail)

        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="new convo",
                channel_user_id="user-m5",
                metadata={"case": {"id": "seven"}},
            )
        )
        ingestion = RefusingIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 422
        body = resp.json()
        assert body["detail"] == detail
        assert body["code"] == "invalid_metadata"


# ---------------------------------------------------------------------------
# Session commands — new, unregister, status
# ---------------------------------------------------------------------------


def _command_message(
    name: str,
    rest: str = "",
    channel_user_id: str = "user-cmd",
) -> ChannelMessage:
    """A parsed message carrying a command, with ``content`` left whole.

    ``content`` keeps the command word exactly as the parser leaves it: the
    fall-through path depends on the team seeing what the user typed.
    """
    text = f"/{name} {rest}".rstrip()
    return ChannelMessage(
        content=text,
        channel_user_id=channel_user_id,
        command=ChannelCommand(name=name, rest=rest),
    )


class TestUnrecognisedCommandFallsThrough:
    """G1: only three names are consumed; every other one is content."""

    async def test_roster_reaches_the_team_as_text(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(_command_message("roster", "please", channel_user_id="user-r"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-r",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        # The whole text, command word included — not the rest, and not nothing.
        assert len(ingestion.send_message_calls) == 1
        assert ingestion.send_message_calls[0][1] == "/roster please"
        assert adapter.notices == []


class TestCommandUnregister:
    """AC 12: the binding is released, and the caller is told either way."""

    async def test_unregister_removes_the_binding_and_acknowledges(self, tmp_path: Path) -> None:
        parser = StubParser()
        parser.set_next_message(_command_message("unregister", channel_user_id="user-u1"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-u1",
                team_id=uuid.uuid4(),
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert (
            await registry.find_binding(
                ChannelAddress(channel="test-channel", channel_user_id="user-u1")
            )
            is None
        )
        assert len(adapter.notices) == 1
        address, _text = adapter.notices[0]
        assert address.channel == "test-channel"
        assert address.channel_user_id == "user-u1"
        assert ingestion.send_message_calls == []

    async def test_unregister_with_no_binding_still_acknowledges(self, tmp_path: Path) -> None:
        """G3c: silence is indistinguishable from a command that did nothing."""
        parser = StubParser()
        parser.set_next_message(_command_message("unregister", channel_user_id="user-u2"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(adapter.notices) == 1
        assert ingestion.send_message_calls == []
        assert ingestion.create_team_calls == []

    async def test_unregister_releases_only_the_callers_own_binding(self, tmp_path: Path) -> None:
        """G6: a command acts on the caller's own conversation and nothing else.

        A status code alone cannot tell "acted on my own session" from "acted on
        another" — so this asserts on the *other* conversation's record, which
        must survive untouched.
        """
        other_team = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(_command_message("unregister", channel_user_id="user-attacker"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-victim",
                team_id=other_team,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        victim = await registry.find_binding(
            ChannelAddress(channel="test-channel", channel_user_id="user-victim")
        )
        assert victim is not None
        assert victim.team_id == other_team
        # Consumed as a command, so the reply branch was never reached.
        assert ingestion.send_message_calls == []


class TestCommandNew:
    """AC 9-11: release, replace, and answer — in that order."""

    async def test_new_replaces_the_binding_with_the_new_team(self, tmp_path: Path) -> None:
        """G4: a deregister running after the register leaves no binding at all."""
        old_team = uuid.uuid4()
        new_team = uuid.uuid4()
        parser = StubParser(default_entry="my-catalog-entry")
        parser.set_next_message(_command_message("new", "hello", channel_user_id="user-n1"))
        ingestion = StubIngestion()
        ingestion.set_next_team_id(new_team)
        ingestion.set_next_entry_point_name("@HumanProxy_0")
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-n1",
                team_id=old_team,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        binding = await registry.find_binding(
            ChannelAddress(channel="test-channel", channel_user_id="user-n1")
        )
        assert binding is not None
        assert binding.team_id == new_team
        # G5: the first message is the command's rest, not the whole text.
        assert ingestion.create_team_calls == [("user-n1", "my-catalog-entry", None)]
        assert ingestion.send_message_calls == [(new_team, "hello", None)]
        assert len(adapter.notices) == 1
        assert str(new_team) in adapter.notices[0][1]

    async def test_new_does_not_stop_the_old_team(self, tmp_path: Path) -> None:
        """Abandoning is the affordance; stopping is a lifecycle change."""
        old_team = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(_command_message("new", channel_user_id="user-n2"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-n2",
                team_id=old_team,
                agent_name="@HumanProxy_0",
            )
        )
        team_service = StubTeamService()
        team_service.add_running_team(old_team)
        client = TestClient(
            _build_app(parser, ingestion, registry, adapter=adapter, team_service=team_service)
        )

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert team_service.teams[old_team].status is TeamStatus.RUNNING

    async def test_new_with_no_prior_binding_still_initiates_and_binds(
        self, tmp_path: Path
    ) -> None:
        new_team = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(_command_message("new", "start here", channel_user_id="user-n3"))
        ingestion = StubIngestion()
        ingestion.set_next_team_id(new_team)
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        binding = await registry.find_binding(
            ChannelAddress(channel="test-channel", channel_user_id="user-n3")
        )
        assert binding is not None
        assert binding.team_id == new_team
        assert ingestion.send_message_calls == [(new_team, "start here", None)]

    async def test_new_alone_creates_the_team_and_sends_nothing(self, tmp_path: Path) -> None:
        """``/new`` alone sends no first message, which is why it acknowledges."""
        parser = StubParser()
        parser.set_next_message(_command_message("new", channel_user_id="user-n4"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(ingestion.create_team_calls) == 1
        assert ingestion.send_message_calls == []
        assert len(adapter.notices) == 1

    async def test_new_honours_a_catalog_entry_the_parser_set(self, tmp_path: Path) -> None:
        """Identical to the initiation branch's expression, not a second rule."""
        parser = StubParser(default_entry="the-default")
        message = _command_message("new", "hello", channel_user_id="user-n5")
        parser.set_next_message(message.model_copy(update={"catalog_entry": "chosen-entry"}))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.create_team_calls[0][1] == "chosen-entry"


class TestCommandStatus:
    """AC 13: the bound team *and its lifecycle state*."""

    async def test_status_reports_the_team_and_its_state(self, tmp_path: Path) -> None:
        """G8: the id alone drops the dead-binding diagnosis entirely."""
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(_command_message("status", channel_user_id="user-s1"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-s1",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        team_service = StubTeamService()
        team_service.add_running_team(team_id, TeamStatus.STOPPED)
        client = TestClient(
            _build_app(parser, ingestion, registry, adapter=adapter, team_service=team_service)
        )

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(adapter.notices) == 1
        text = adapter.notices[0][1]
        assert str(team_id) in text
        assert TeamStatus.STOPPED.value in text
        assert team_service.get_team_calls == [team_id]

    async def test_status_reports_a_binding_whose_team_is_gone(self, tmp_path: Path) -> None:
        """The dead-binding case: the record outlived the team."""
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(_command_message("status", channel_user_id="user-s2"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        await registry.register(
            ChannelBinding(
                channel="test-channel",
                channel_user_id="user-s2",
                team_id=team_id,
                agent_name="@HumanProxy_0",
            )
        )
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        text = adapter.notices[0][1]
        assert str(team_id) in text
        assert "no longer known" in text

    async def test_status_with_no_binding_reports_no_session(self, tmp_path: Path) -> None:
        """G8b: the unbound path must answer, not raise on a None binding."""
        parser = StubParser()
        parser.set_next_message(_command_message("status", channel_user_id="user-s3"))
        ingestion = StubIngestion()
        adapter = StubNoticeAdapter()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry, adapter=adapter))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(adapter.notices) == 1
        address, text = adapter.notices[0]
        assert address == ChannelAddress(channel="test-channel", channel_user_id="user-s3")
        assert "No active session" in text

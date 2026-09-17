"""Tests for the webhook route — POST /webhook/{channel} with 3 routing flows."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.shared.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.protocols.channels import (
    ChannelBinding,
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
        self.route_reply_calls: list[tuple[uuid.UUID, str | Message, str | None]] = []
        # Anything the route passes to route_reply beyond the three declared
        # parameters lands here, so "the reply path forwards no metadata" is an
        # assertion about recorded evidence rather than about a TypeError.
        self.route_reply_extra_kwargs: list[dict[str, object]] = []
        self.initiate_team_calls: list[tuple[str, str, str, dict[str, JsonValue] | None]] = []
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

    async def route_reply(
        self,
        team_id: uuid.UUID,
        content: str | Message,
        original_message_id: str | None = None,
        **extra: object,
    ) -> None:
        self.route_reply_calls.append((team_id, content, original_message_id))
        self.route_reply_extra_kwargs.append(extra)

    async def initiate_team(
        self,
        content: str,
        channel_user_id: str,
        catalog_entry_id: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> InitiatedTeam:
        self.initiate_team_calls.append((content, channel_user_id, catalog_entry_id, metadata))
        return InitiatedTeam(
            team_id=self._next_team_id,
            entry_point_name=self._next_entry_point_name,
        )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _build_parser_registry(
    parser: StubParser,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> ChannelParserRegistry:
    """Build a ChannelParserRegistry with a pre-registered stub parser.

    Constructs via the public API with an empty config, then monkeypatches
    get_parser to return the stub. This avoids __new__ hacks and private
    attribute access.
    """
    registry = ChannelParserRegistry(channels_config={})

    original_get_parser = registry.get_parser

    def _patched_get_parser(channel_name: str) -> StubParser | None:
        if channel_name == parser.channel_name:
            return parser  # type: ignore[return-value]
        return original_get_parser(channel_name)

    if monkeypatch is not None:
        monkeypatch.setattr(registry, "get_parser", _patched_get_parser)
    else:
        registry.get_parser = _patched_get_parser  # type: ignore[assignment]

    return registry


def _build_app(
    parser: StubParser,
    ingestion: StubIngestion,
    channel_registry: YamlChannelRegistry,
) -> FastAPI:
    """Build a minimal FastAPI app with the webhook router wired.

    ``add_server_exception_handlers`` is the same registration the real assembly
    installs, so a ``ServerError`` raised by the ingestion layer is mapped here
    exactly as it is in production — a status assertion against a bare
    ``FastAPI()`` would only prove the TestClient re-raises.
    """
    app = FastAPI()
    parser_registry = _build_parser_registry(parser)
    app.state.channel_parser_registry = parser_registry
    app.state.channel_registry = channel_registry
    app.state.ingestion = ingestion
    app.include_router(webhook_router)
    add_server_exception_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebhookReplyFlow:
    """AC #3: team_id in parsed message → route_reply."""

    def test_reply_flow_calls_route_reply(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="reply msg",
                channel_user_id="user-1",
                team_id=team_id,
                message_id="msg-abc",
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert len(ingestion.route_reply_calls) == 1
        call = ingestion.route_reply_calls[0]
        assert call[0] == team_id
        assert call[1] == "reply msg"
        assert call[2] == "msg-abc"


class TestWebhookContinuationFlow:
    """AC #4: no team_id but registered team → route_reply (forwarding message_id)."""

    async def test_continuation_flow_calls_route_reply(self, tmp_path: Path) -> None:
        existing_team_id = uuid.uuid4()
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="continuation msg",
                channel_user_id="user-2",
                message_id="msg-cont",
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
        assert len(ingestion.route_reply_calls) == 1
        call = ingestion.route_reply_calls[0]
        assert call[0] == existing_team_id
        assert call[1] == "continuation msg"
        # Story 30.3: the continuation flow forwards the parsed message_id as the
        # third positional arg, in lockstep with the reply flow (TestWebhookReplyFlow).
        assert call[2] == "msg-cont"


class TestWebhookInitiationFlow:
    """AC #5: no team_id and no existing team → initiate_team + register."""

    def test_initiation_flow_calls_initiate_team(self, tmp_path: Path) -> None:
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
        assert len(ingestion.initiate_team_calls) == 1
        call = ingestion.initiate_team_calls[0]
        assert call[0] == "new convo"
        assert call[1] == "user-3"
        assert call[2] == "my-catalog-entry"

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
        found = await registry.find_team("test-channel", "user-4")
        assert found == new_team_id

    async def test_initiation_binds_the_channel_to_the_entry_point_agent(
        self, tmp_path: Path
    ) -> None:
        """AC 12: the persisted record carries the ingestion's entry-point name.

        ``agent_name`` must be what ``initiate_team`` reported, not the channel
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

        binding = await registry.find_binding("test-channel", "user-6")
        assert binding == ChannelBinding(
            channel="test-channel",
            channel_user_id="user-6",
            team_id=new_team_id,
            agent_name="@HumanProxy_0",
        )


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

    def test_reply_returns_204(self, tmp_path: Path) -> None:
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="msg",
                channel_user_id="u",
                team_id=uuid.uuid4(),
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
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
        assert len(ingestion.initiate_team_calls) == 1
        call = ingestion.initiate_team_calls[0]
        assert call[0] == "form hello"
        assert call[1] == "form-user"


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

    def test_json_with_charset_param(self, tmp_path: Path) -> None:
        """application/json; charset=utf-8 is handled as JSON."""
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="charset msg",
                channel_user_id="u-charset",
                team_id=uuid.uuid4(),
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post(
            "/webhook/test-channel",
            json={"text": "hi"},
            headers={"content-type": "application/json; charset=utf-8"},
        )

        assert resp.status_code == 204
        assert len(ingestion.route_reply_calls) == 1


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
    """Parser-raised ValueError should surface as 400, not 500."""

    def test_parser_value_error_returns_400(self, tmp_path: Path) -> None:
        class RaisingParser(StubParser):
            async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
                del payload
                raise ValueError("payload missing required field 'message'")

        parser = RaisingParser()
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"update_id": 1})

        assert resp.status_code == 400
        assert "payload missing required field 'message'" in resp.json()["detail"]


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
        assert len(ingestion.initiate_team_calls) == 1
        assert ingestion.initiate_team_calls[0][3] == metadata

    def test_initiation_without_metadata_forwards_none(self, tmp_path: Path) -> None:
        parser = StubParser(default_entry="my-catalog-entry")
        parser.set_next_message(ChannelMessage(content="new convo", channel_user_id="user-m2"))
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.initiate_team_calls[0][3] is None

    def test_reply_flow_forwards_no_metadata(self, tmp_path: Path) -> None:
        """A reply addresses a team whose metadata was fixed at creation."""
        parser = StubParser()
        parser.set_next_message(
            ChannelMessage(
                content="reply msg",
                channel_user_id="user-m3",
                team_id=uuid.uuid4(),
                metadata={"tenant": "acme"},
            )
        )
        ingestion = StubIngestion()
        registry = YamlChannelRegistry(tmp_path / "registry.yaml")
        client = TestClient(_build_app(parser, ingestion, registry))

        resp = client.post("/webhook/test-channel", json={"text": "hi"})

        assert resp.status_code == 204
        assert ingestion.initiate_team_calls == []
        assert ingestion.route_reply_extra_kwargs == [{}]

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
        assert ingestion.initiate_team_calls == []
        assert ingestion.route_reply_extra_kwargs == [{}]


class TestWebhookMetadataRejection:
    """A metadata body the card refuses answers 422, with the validator's words."""

    def test_invalid_metadata_returns_422_with_validator_message(self, tmp_path: Path) -> None:
        detail = "metadata field 'case.id' must be an integer"

        class RefusingIngestion(StubIngestion):
            async def initiate_team(
                self,
                content: str,
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

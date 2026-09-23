"""Validate protocol definitions are structurally correct."""

from __future__ import annotations

import inspect
import uuid
from typing import Protocol, get_type_hints


def test_placement_strategy_is_protocol() -> None:
    """PlacementStrategy uses typing.Protocol base."""
    from akgentic.infra.protocols import PlacementStrategy

    assert Protocol in inspect.getmro(PlacementStrategy)


def test_placement_strategy_has_create_team() -> None:
    """PlacementStrategy defines create_team with team_card and user_id parameters."""
    from akgentic.infra.protocols import PlacementStrategy

    assert hasattr(PlacementStrategy, "create_team")
    sig = inspect.signature(PlacementStrategy.create_team)
    assert "team_card" in sig.parameters
    assert "user_id" in sig.parameters


def test_placement_strategy_has_resume_team() -> None:
    """PlacementStrategy defines resume_team with team_id parameter."""
    from akgentic.infra.protocols import PlacementStrategy

    assert hasattr(PlacementStrategy, "resume_team")
    sig = inspect.signature(PlacementStrategy.resume_team)
    assert "team_id" in sig.parameters


def test_resume_team_lives_on_placement_not_on_the_worker_handle() -> None:
    """Where a returning team runs is a placement decision, not a handle's.

    Stated negatively as well as positively: a handle already names a worker,
    so a ``resume_team`` left on ``WorkerHandle`` would let a tier grow a second
    worker selector inside it and bring a stopped team back wherever its last
    handle happens to point.
    """
    from akgentic.infra.protocols import PlacementStrategy, WorkerHandle

    assert not hasattr(WorkerHandle, "resume_team")
    assert hasattr(PlacementStrategy, "resume_team")


def test_auth_strategy_is_protocol() -> None:
    """AuthStrategy uses typing.Protocol base."""
    from akgentic.infra.protocols import AuthStrategy

    assert Protocol in inspect.getmro(AuthStrategy)


def test_auth_strategy_declares_async_contract() -> None:
    """AuthStrategy declares the async resolver contract; the sync member stays gone.

    The contract is ``async resolve_request_user`` + ``get_auth_routes``; there
    is no synchronous ``authenticate`` entry point (removed in Story 40.1).
    """
    import inspect

    from akgentic.infra.protocols import AuthStrategy

    assert not hasattr(AuthStrategy, "authenticate")
    assert hasattr(AuthStrategy, "resolve_request_user")
    assert inspect.iscoroutinefunction(AuthStrategy.resolve_request_user)
    assert "connection" in inspect.signature(AuthStrategy.resolve_request_user).parameters
    assert hasattr(AuthStrategy, "get_auth_routes")


def test_recovery_policy_is_protocol() -> None:
    """RecoveryPolicy uses typing.Protocol base."""
    from akgentic.infra.protocols import RecoveryPolicy

    assert Protocol in inspect.getmro(RecoveryPolicy)


def test_recovery_policy_has_recover() -> None:
    """RecoveryPolicy defines recover with instance_id and team_ids parameters."""
    from akgentic.infra.protocols import RecoveryPolicy

    assert hasattr(RecoveryPolicy, "recover")
    sig = inspect.signature(RecoveryPolicy.recover)
    assert "instance_id" in sig.parameters
    assert "team_ids" in sig.parameters


def test_health_monitor_is_protocol() -> None:
    """HealthMonitor uses typing.Protocol base."""
    from akgentic.infra.protocols import HealthMonitor

    assert Protocol in inspect.getmro(HealthMonitor)


def test_health_monitor_has_check_health() -> None:
    """HealthMonitor defines check_health method."""
    from akgentic.infra.protocols import HealthMonitor

    assert hasattr(HealthMonitor, "check_health")
    sig = inspect.signature(HealthMonitor.check_health)
    # Only self parameter
    assert len(sig.parameters) == 1


# --- InteractionChannelAdapter ---


def test_interaction_channel_adapter_is_protocol() -> None:
    """InteractionChannelAdapter uses typing.Protocol base."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert Protocol in inspect.getmro(InteractionChannelAdapter)


def test_interaction_channel_adapter_has_matches() -> None:
    """InteractionChannelAdapter defines matches taking exactly msg and binding.

    The binding names the destination chat, which the address cannot. An
    ``isinstance`` check compares method *names* only, so this signature
    assertion is the only thing that notices a one-argument adapter — and a
    membership check would survive an extra parameter being appended, so the
    whole list is compared.
    """
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "matches")
    sig = inspect.signature(InteractionChannelAdapter.matches)
    assert list(sig.parameters) == ["self", "msg", "binding"]


def test_interaction_channel_adapter_has_deliver() -> None:
    """InteractionChannelAdapter defines deliver taking exactly msg and binding."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "deliver")
    sig = inspect.signature(InteractionChannelAdapter.deliver)
    assert list(sig.parameters) == ["self", "msg", "binding"]


def test_interaction_channel_adapter_has_deliver_notice() -> None:
    """deliver_notice takes an address, not a binding.

    A binding requires a ``team_id`` and an ``agent_name``; ``status`` may find
    no binding and ``unregister`` has just destroyed one, so a binding-typed
    parameter could only be satisfied by fabricating a team id. The type is the
    guard: an unbound path literally cannot name a team.
    """
    from akgentic.infra.protocols import ChannelAddress, InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "deliver_notice")
    sig = inspect.signature(InteractionChannelAdapter.deliver_notice)
    assert list(sig.parameters) == ["self", "address", "text"]

    hints = get_type_hints(InteractionChannelAdapter.deliver_notice)
    assert hints["address"] is ChannelAddress
    assert hints["text"] is str
    assert hints["return"] is type(None)


def test_interaction_channel_adapter_has_on_stop() -> None:
    """InteractionChannelAdapter defines on_stop with team_id parameter."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "on_stop")
    sig = inspect.signature(InteractionChannelAdapter.on_stop)
    assert "team_id" in sig.parameters


def test_interaction_channel_adapter_matches_returns_bool() -> None:
    """InteractionChannelAdapter.matches has bool return annotation."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    sig = inspect.signature(InteractionChannelAdapter.matches)
    assert sig.return_annotation is bool or sig.return_annotation == "bool"


def test_interaction_channel_adapter_deliver_returns_none() -> None:
    """InteractionChannelAdapter.deliver has None return annotation."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    sig = inspect.signature(InteractionChannelAdapter.deliver)
    assert sig.return_annotation is None or sig.return_annotation == "None"


def test_interaction_channel_adapter_on_stop_returns_none() -> None:
    """InteractionChannelAdapter.on_stop returns None."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    hints = get_type_hints(InteractionChannelAdapter.on_stop)
    assert hints["return"] is type(None)


def test_interaction_channel_adapter_structural_subtyping() -> None:
    """A concrete class satisfying InteractionChannelAdapter is recognized."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    class FakeAdapter:
        def matches(self, msg: object, binding: object) -> bool:
            return True

        def deliver(self, msg: object, binding: object) -> None:
            pass

        def deliver_notice(self, address: object, text: str) -> None:
            pass

        def on_stop(self, team_id: uuid.UUID) -> None:
            pass

    assert isinstance(FakeAdapter(), InteractionChannelAdapter)


def test_an_adapter_without_deliver_notice_is_not_an_adapter() -> None:
    """The negative case: dropping ``deliver_notice`` must fail the check.

    The positive assertion above cannot notice the member being removed from the
    Protocol — every fake would simply carry one method more than required. This
    is what pins ``deliver_notice`` as a *requirement* rather than a convention.
    """
    from akgentic.infra.protocols import InteractionChannelAdapter

    class AdapterMissingDeliverNotice:
        def matches(self, msg: object, binding: object) -> bool:
            return True

        def deliver(self, msg: object, binding: object) -> None:
            pass

        def on_stop(self, team_id: uuid.UUID) -> None:
            pass

    assert not isinstance(AdapterMissingDeliverNotice(), InteractionChannelAdapter)


# --- ChannelParser ---


def test_channel_parser_is_protocol() -> None:
    """ChannelParser uses typing.Protocol base."""
    from akgentic.infra.protocols import ChannelParser

    assert Protocol in inspect.getmro(ChannelParser)


def test_channel_parser_has_parse() -> None:
    """ChannelParser defines async parse with payload parameter."""
    from akgentic.infra.protocols import ChannelParser

    assert hasattr(ChannelParser, "parse")
    sig = inspect.signature(ChannelParser.parse)
    assert "payload" in sig.parameters
    assert inspect.iscoroutinefunction(ChannelParser.parse)


def test_channel_parser_has_channel_name_property() -> None:
    """ChannelParser defines channel_name property."""
    from akgentic.infra.protocols import ChannelParser

    assert hasattr(ChannelParser, "channel_name")


def test_channel_parser_has_default_catalog_entry_property() -> None:
    """ChannelParser defines default_catalog_entry property."""
    from akgentic.infra.protocols import ChannelParser

    assert hasattr(ChannelParser, "default_catalog_entry")


def test_channel_parser_return_type() -> None:
    """ChannelParser.parse returns ChannelMessage."""
    from akgentic.infra.protocols import ChannelMessage, ChannelParser

    hints = get_type_hints(ChannelParser.parse)
    assert hints["return"] is ChannelMessage


def test_channel_parser_structural_subtyping() -> None:
    """A concrete class satisfying ChannelParser is recognized."""
    from akgentic.infra.protocols import ChannelMessage, ChannelParser

    class FakeParser:
        @property
        def channel_name(self) -> str:
            return "fake"

        @property
        def default_catalog_entry(self) -> str:
            return "default-fake"

        async def parse(self, payload: dict[str, object]) -> ChannelMessage:
            return ChannelMessage(content="", channel_user_id="")

    assert isinstance(FakeParser(), ChannelParser)


# --- ChannelRegistry ---


def test_channel_registry_is_protocol() -> None:
    """ChannelRegistry uses typing.Protocol base."""
    from akgentic.infra.protocols import ChannelRegistry

    assert Protocol in inspect.getmro(ChannelRegistry)


def test_channel_registry_has_register() -> None:
    """ChannelRegistry defines async register taking the whole binding.

    The three positional values are gone, not joined by a fourth: passing the
    record whole is what lets a field added to ``ChannelBinding`` later reach
    storage without every call site being revisited.
    """
    from akgentic.infra.protocols import ChannelBinding, ChannelRegistry

    assert hasattr(ChannelRegistry, "register")
    sig = inspect.signature(ChannelRegistry.register)
    assert "binding" in sig.parameters
    assert set(sig.parameters) == {"self", "binding"}
    assert inspect.iscoroutinefunction(ChannelRegistry.register)

    hints = get_type_hints(ChannelRegistry.register)
    assert hints["binding"] is ChannelBinding


def test_channel_registry_has_no_find_team() -> None:
    """The team is read off ``find_binding``; a second lookup is not owed by any tier."""
    from akgentic.infra.protocols import ChannelRegistry

    assert not hasattr(ChannelRegistry, "find_team")


def test_channel_registry_has_deregister() -> None:
    """ChannelRegistry defines async deregister(address)."""
    from akgentic.infra.protocols import ChannelAddress, ChannelRegistry

    sig = inspect.signature(ChannelRegistry.deregister)
    assert list(sig.parameters) == ["self", "address"]
    assert get_type_hints(ChannelRegistry.deregister)["address"] is ChannelAddress
    assert inspect.iscoroutinefunction(ChannelRegistry.deregister)


def test_channel_registry_register_returns_none() -> None:
    """ChannelRegistry.register returns None."""
    from akgentic.infra.protocols import ChannelRegistry

    hints = get_type_hints(ChannelRegistry.register)
    assert hints["return"] is type(None)


def test_channel_registry_deregister_returns_none() -> None:
    """ChannelRegistry.deregister returns None."""
    from akgentic.infra.protocols import ChannelRegistry

    hints = get_type_hints(ChannelRegistry.deregister)
    assert hints["return"] is type(None)


def test_channel_registry_has_find_binding() -> None:
    """ChannelRegistry defines async find_binding(address) returning ChannelBinding | None."""
    from akgentic.infra.protocols import ChannelAddress, ChannelBinding, ChannelRegistry

    assert hasattr(ChannelRegistry, "find_binding")
    sig = inspect.signature(ChannelRegistry.find_binding)
    assert list(sig.parameters) == ["self", "address"]
    assert inspect.iscoroutinefunction(ChannelRegistry.find_binding)

    hints = get_type_hints(ChannelRegistry.find_binding)
    assert hints["address"] is ChannelAddress
    assert hints["return"] == ChannelBinding | None


def test_channel_registry_has_deregister_team() -> None:
    """ChannelRegistry defines async deregister_team taking a team_id."""
    from akgentic.infra.protocols import ChannelRegistry

    assert hasattr(ChannelRegistry, "deregister_team")
    sig = inspect.signature(ChannelRegistry.deregister_team)
    assert set(sig.parameters) == {"self", "team_id"}
    assert inspect.iscoroutinefunction(ChannelRegistry.deregister_team)

    hints = get_type_hints(ChannelRegistry.deregister_team)
    assert hints["team_id"] is uuid.UUID
    assert hints["return"] is type(None)


def test_channel_registry_structural_subtyping() -> None:
    """A concrete class satisfying ChannelRegistry is recognized.

    ``@runtime_checkable`` checks method *presence*, so this fake must carry
    every method the Protocol declares — including ``find_binding_sync``,
    inherited from ``ChannelRegistryReadSync``.
    """
    from akgentic.infra.protocols import ChannelAddress, ChannelBinding, ChannelRegistry

    class FakeRegistry:
        async def register(self, binding: ChannelBinding) -> None:
            pass

        async def find_binding(self, address: ChannelAddress) -> ChannelBinding | None:
            return None

        async def deregister(self, address: ChannelAddress) -> None:
            pass

        async def deregister_team(self, team_id: uuid.UUID) -> None:
            pass

        def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
            return None

    assert isinstance(FakeRegistry(), ChannelRegistry)


# --- ChannelRegistryReadSync ---


def test_channel_registry_read_sync_is_protocol() -> None:
    """ChannelRegistryReadSync uses typing.Protocol base."""
    from akgentic.infra.protocols import ChannelRegistryReadSync

    assert Protocol in inspect.getmro(ChannelRegistryReadSync)


def test_channel_registry_read_sync_find_binding_sync_is_not_a_coroutine() -> None:
    """find_binding_sync is declared ``def``, not ``async def``.

    It is called from ``EventSubscriber.on_message``, which runs in a Pykka
    actor thread with no event loop: a coroutine here could not be awaited and
    would have to be bridged, which is the blocking this Protocol exists to
    avoid.
    """
    from akgentic.infra.protocols import ChannelBinding, ChannelRegistryReadSync

    assert hasattr(ChannelRegistryReadSync, "find_binding_sync")
    assert not inspect.iscoroutinefunction(ChannelRegistryReadSync.find_binding_sync)

    sig = inspect.signature(ChannelRegistryReadSync.find_binding_sync)
    assert set(sig.parameters) == {"self", "team_id", "agent_name"}

    hints = get_type_hints(ChannelRegistryReadSync.find_binding_sync)
    assert hints["team_id"] is uuid.UUID
    assert hints["agent_name"] is str
    assert hints["return"] == ChannelBinding | None


def test_channel_registry_inherits_the_sync_read() -> None:
    """ChannelRegistry is a ChannelRegistryReadSync, so no cast is needed anywhere.

    The narrow reading — the two Protocols merely declared side by side — leaves
    a dispatcher typed on the sync face unable to accept
    ``TierServices.channel_registry`` without a cast or a third union Protocol.
    """
    from akgentic.infra.protocols import ChannelRegistry, ChannelRegistryReadSync

    assert ChannelRegistryReadSync in inspect.getmro(ChannelRegistry)
    assert hasattr(ChannelRegistry, "find_binding_sync")


def test_channel_registry_read_sync_structural_subtyping() -> None:
    """A class with only the sync read satisfies ChannelRegistryReadSync."""
    from akgentic.infra.protocols import ChannelBinding, ChannelRegistryReadSync

    class FakeSyncReader:
        def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
            return None

    assert isinstance(FakeSyncReader(), ChannelRegistryReadSync)


# --- ChannelMessage ---


def test_channel_message_is_pydantic_model() -> None:
    """ChannelMessage is a Pydantic BaseModel with correct fields."""
    from pydantic import BaseModel

    from akgentic.infra.protocols import ChannelMessage

    assert issubclass(ChannelMessage, BaseModel)
    fields = ChannelMessage.model_fields
    assert "content" in fields
    assert "channel_user_id" in fields
    assert "channel_message_id" in fields
    # ``team_id`` is back with a different meaning: a CREATION KEY for a team the
    # message may start, never the address of an existing one. The property that
    # makes it safe — a key naming an existing team, or one being created for
    # another user, is refused — is guarded where it is enforced, in the
    # LocalPlacement creation-key specs, not by this field's absence.
    assert "team_id" in fields
    # The bare name is Telegram's own key for a different thing, and
    # ``HumanInputRequest.message_id`` is a third. Reintroducing it here puts
    # three unrelated ids behind one name again.
    assert "message_id" not in fields


def test_channel_message_field_descriptions() -> None:
    """ChannelMessage fields have descriptions."""
    from akgentic.infra.protocols import ChannelMessage

    for name, field_info in ChannelMessage.model_fields.items():
        assert field_info.description is not None, f"Field {name} missing description"


def test_channel_message_optional_defaults() -> None:
    """ChannelMessage channel_message_id defaults to None."""
    from akgentic.infra.protocols import ChannelMessage

    msg = ChannelMessage(content="hello", channel_user_id="u1")
    assert msg.channel_message_id is None


def test_channel_message_with_all_fields() -> None:
    """ChannelMessage can be created with all fields."""
    from akgentic.infra.protocols import ChannelMessage

    msg = ChannelMessage(
        content="hello",
        channel_user_id="u1",
        channel_message_id="msg-123",
    )
    assert msg.content == "hello"
    assert msg.channel_user_id == "u1"
    assert msg.channel_message_id == "msg-123"


def test_channel_message_metadata_defaults_to_none() -> None:
    """Omitting metadata leaves it None — no parser is obliged to supply it."""
    from akgentic.infra.protocols import ChannelMessage

    msg = ChannelMessage(content="hello", channel_user_id="u1")
    assert msg.team_metadata is None


def test_channel_message_nested_metadata_survives_round_trip() -> None:
    """Nested metadata validates and round-trips unchanged.

    The field is annotated with the recursive ``JsonValue`` alias, so it has to
    build a real Pydantic schema — an annotation that never validates would pass
    construction and lose the nesting on the way back in.
    """
    from akgentic.infra.protocols import ChannelMessage

    metadata = {"case": {"id": 7, "tags": ["a"]}, "tenant": "acme"}
    msg = ChannelMessage(content="hello", channel_user_id="u1", team_metadata=metadata)

    assert msg.team_metadata == metadata

    restored = ChannelMessage.model_validate(msg.model_dump())
    assert restored.team_metadata == metadata
    assert restored.team_metadata is not None
    assert restored.team_metadata["case"] == {"id": 7, "tags": ["a"]}


# --- ChannelCommand ---


def test_channel_command_rest_defaults_to_empty() -> None:
    """``/new`` with nothing after it is a command with an empty ``rest``."""
    from akgentic.infra.protocols import ChannelCommand

    command = ChannelCommand(name="new")

    assert command.name == "new"
    assert command.rest == ""


def test_channel_command_name_is_a_plain_string() -> None:
    """An unrecognised command must be a *value*, not a validation error.

    The channel layer consumes three names and passes every other one through
    as text. An enum here would turn ``/shrug`` into a 400 at parse time, which
    is precisely the fall-through the design forbids breaking.
    """
    from akgentic.infra.protocols import ChannelCommand

    command = ChannelCommand(name="shrug", rest="whatever")

    assert command.name == "shrug"
    assert ChannelCommand.model_fields["name"].annotation is str


def test_channel_message_command_defaults_to_none() -> None:
    """A message built without a command carries none — no parser is obliged to."""
    from akgentic.infra.protocols import ChannelMessage

    msg = ChannelMessage(content="hello", channel_user_id="u1")

    assert msg.command is None


def test_channel_message_command_round_trips() -> None:
    """The command survives a dump/validate cycle as a model, not a dict."""
    from akgentic.infra.protocols import ChannelCommand, ChannelMessage

    msg = ChannelMessage(
        content="/new Fix the invoice",
        channel_user_id="u1",
        command=ChannelCommand(name="new", rest="Fix the invoice"),
    )

    restored = ChannelMessage.model_validate(msg.model_dump(mode="json"))

    assert restored.command == ChannelCommand(name="new", rest="Fix the invoice")


# --- ChannelBinding ---


def test_channel_binding_is_pydantic_model_with_its_fields() -> None:
    """ChannelBinding carries the four bound values plus router-owned metadata."""
    from pydantic import BaseModel

    from akgentic.infra.protocols import ChannelBinding

    assert issubclass(ChannelBinding, BaseModel)
    assert set(ChannelBinding.model_fields) == {
        "channel",
        "channel_user_id",
        "team_id",
        "agent_name",
        "metadata",
    }


def test_channel_binding_is_a_channel_address_losing_no_field() -> None:
    """The split names a concept the design already had, and loses no field.

    ``metadata`` is declared on ``ChannelAddress``, not on the binding. The
    notice path (``deliver_notice``) is handed a bare address and never a
    binding, so an adapter routing on metadata — Signal choosing which of
    several bot accounts answers — would otherwise have it for agent messages
    and not for acknowledgements. Pydantic orders base-class fields first, so
    it sits between the address pair and the binding pair.

    **Field ORDER is deliberately not asserted, because it does not ship.** An
    earlier version of this test pinned the dumped key order on the stated
    grounds that it "is what the YAML registry writes". That premise was false:
    ``YamlChannelRegistry`` persists with ``yaml.safe_dump``, whose
    ``sort_keys`` defaults to True, so every record is written alphabetically
    and Pydantic's declaration order never reaches the file. The assertion
    could only ever fail for a reordering no reader could observe — and it did,
    when ``metadata`` moved to the base class, against a change that altered
    nothing on disk.

    What must hold is what a reader actually depends on: every field is still
    present, and a record persisted before the move still loads.
    """
    from akgentic.infra.protocols import ChannelAddress, ChannelBinding

    assert issubclass(ChannelBinding, ChannelAddress)
    assert set(ChannelAddress.model_fields) == {"channel", "channel_user_id", "metadata"}
    assert set(ChannelBinding.model_fields) == {
        "channel",
        "channel_user_id",
        "metadata",
        "team_id",
        "agent_name",
    }

    binding = ChannelBinding(
        channel="telegram",
        channel_user_id="987654321",
        team_id=uuid.uuid4(),
        agent_name="@HumanProxy_0",
    )
    dumped = {key for key in binding.model_dump(mode="json") if not key.startswith("__")}
    assert dumped == {"channel", "channel_user_id", "team_id", "agent_name", "metadata"}

    # A record written before ``metadata`` existed — the shape on disk in any
    # registry that has not been rewritten since — must still load.
    team_id = uuid.uuid4()
    legacy = ChannelBinding.model_validate(
        {
            "agent_name": "@Human",
            "channel": "signal",
            "channel_user_id": "+32470000000",
            "team_id": str(team_id),
        }
    )
    assert legacy.metadata == {}
    assert legacy.team_id == team_id


def test_channel_address_carries_no_team() -> None:
    """A bare address is constructible without a team — the whole point of it."""
    from akgentic.infra.protocols import ChannelAddress

    address = ChannelAddress(channel="telegram", channel_user_id="987654321")

    assert address.channel == "telegram"
    assert address.channel_user_id == "987654321"
    assert not hasattr(address, "team_id")


def test_channel_binding_field_descriptions() -> None:
    """Every ChannelBinding field has a description."""
    from akgentic.infra.protocols import ChannelBinding

    for name, field_info in ChannelBinding.model_fields.items():
        assert field_info.description is not None, f"Field {name} missing description"


def test_channel_binding_adds_no_config_and_no_private_state() -> None:
    """ChannelBinding stays serializable by construction (Golden Rule #1b).

    All four types are serializable, so this model declares no ``ConfigDict`` of
    its own and holds no ``PrivateAttr``. It is persisted by dumping the model
    whole, so anything unserializable — or any runtime state smuggled into a
    field — would break the write path silently.

    The comparison is against the base's config rather than against a literal:
    ``SerializableBaseModel`` sets ``arbitrary_types_allowed`` for the whole
    framework, so an absolute assertion would either be false today or would
    pin a decision that is not this model's to make. What is this model's to
    make is whether it *adds* anything — and it must not.
    """
    from akgentic.core.utils.serializer import SerializableBaseModel

    from akgentic.infra.protocols import ChannelBinding

    assert ChannelBinding.model_config == SerializableBaseModel.model_config
    assert ChannelBinding.__private_attributes__ == {}


def test_channel_binding_round_trips_through_json_mode() -> None:
    """A dumped binding validates back to an equal model, team_id still a UUID."""
    from akgentic.infra.protocols import ChannelBinding

    team_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
    binding = ChannelBinding(
        channel="telegram",
        channel_user_id="987654321",
        team_id=team_id,
        agent_name="@HumanProxy_0",
    )

    dumped = binding.model_dump(mode="json")
    assert dumped["team_id"] == str(team_id)

    restored = ChannelBinding.model_validate(dumped)
    assert restored == binding
    assert isinstance(restored.team_id, uuid.UUID)


# --- TeamHandle ---


def test_team_handle_is_protocol() -> None:
    """TeamHandle uses typing.Protocol base."""
    from akgentic.infra.protocols import TeamHandle

    assert Protocol in inspect.getmro(TeamHandle)


def test_team_handle_has_team_id_property() -> None:
    """TeamHandle defines team_id property returning uuid.UUID."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "team_id")
    hints = get_type_hints(TeamHandle.team_id.fget)  # type: ignore[union-attr]
    assert hints["return"] is uuid.UUID


def test_team_handle_is_runtime_checkable() -> None:
    """TeamHandle has @runtime_checkable decorator."""
    from akgentic.infra.protocols import TeamHandle

    assert getattr(TeamHandle, "__protocol_attrs__", None) is not None or hasattr(
        TeamHandle, "_is_runtime_protocol"
    )

    # Verify isinstance works (runtime_checkable requirement)
    class FakeHandle:
        @property
        def team_id(self) -> object:
            import uuid

            return uuid.uuid4()

        def send(self, content: str) -> None:
            pass

        def send_to(self, agent_name: str, content: str) -> None:
            pass

        def send_from_to(self, sender_name: str, recipient_name: str, content: str) -> None:
            pass

        def emitMessage(self, message: object) -> None:  # noqa: N802
            pass

        def process_human_input(self, content: str, message: object) -> None:
            pass

        def subscribe(self, subscriber: object) -> None:
            pass

        def unsubscribe(self, subscriber: object) -> None:
            pass

    assert isinstance(FakeHandle(), TeamHandle)


def test_team_handle_has_send() -> None:
    """TeamHandle defines send with content parameter."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "send")
    sig = inspect.signature(TeamHandle.send)
    assert "content" in sig.parameters


def test_team_handle_has_send_to() -> None:
    """TeamHandle defines send_to with agent_name and content parameters."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "send_to")
    sig = inspect.signature(TeamHandle.send_to)
    assert "agent_name" in sig.parameters
    assert "content" in sig.parameters


def test_team_handle_has_send_from_to() -> None:
    """TeamHandle defines send_from_to with sender_name, recipient_name, and content parameters."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "send_from_to")
    sig = inspect.signature(TeamHandle.send_from_to)
    assert "sender_name" in sig.parameters
    assert "recipient_name" in sig.parameters
    assert "content" in sig.parameters


def test_team_handle_send_from_to_returns_none() -> None:
    """TeamHandle.send_from_to returns None."""
    from akgentic.core.messages.message import Message

    from akgentic.infra.protocols import TeamHandle

    hints = get_type_hints(TeamHandle.send_from_to, localns={"Message": Message})
    assert hints["return"] is type(None)


def test_team_handle_has_emit_message() -> None:
    """TeamHandle defines emitMessage with a message parameter, returning None."""
    from akgentic.core.messages.message import Message

    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "emitMessage")
    sig = inspect.signature(TeamHandle.emitMessage)
    assert "message" in sig.parameters
    hints = get_type_hints(TeamHandle.emitMessage, localns={"Message": Message})
    assert hints["return"] is type(None)


def test_team_handle_has_process_human_input() -> None:
    """TeamHandle defines process_human_input with content and message parameters."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "process_human_input")
    sig = inspect.signature(TeamHandle.process_human_input)
    assert "content" in sig.parameters
    assert "message" in sig.parameters


def test_team_handle_has_subscribe() -> None:
    """TeamHandle defines subscribe with subscriber parameter."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "subscribe")
    sig = inspect.signature(TeamHandle.subscribe)
    assert "subscriber" in sig.parameters


def test_team_handle_has_unsubscribe() -> None:
    """TeamHandle defines unsubscribe with subscriber parameter."""
    from akgentic.infra.protocols import TeamHandle

    assert hasattr(TeamHandle, "unsubscribe")
    sig = inspect.signature(TeamHandle.unsubscribe)
    assert "subscriber" in sig.parameters


def test_team_handle_method_count() -> None:
    """TeamHandle has exactly 7 public methods (emitMessage added per ADR-22)."""
    from akgentic.infra.protocols import TeamHandle

    public_methods = [
        m for m in dir(TeamHandle) if not m.startswith("_") and callable(getattr(TeamHandle, m))
    ]
    assert len(public_methods) == 7


def test_team_handle_send_returns_none() -> None:
    """TeamHandle.send returns None."""
    from akgentic.core.messages.message import Message

    from akgentic.infra.protocols import TeamHandle

    hints = get_type_hints(TeamHandle.send, localns={"Message": Message})
    assert hints["return"] is type(None)


def test_team_handle_send_to_returns_none() -> None:
    """TeamHandle.send_to returns None."""
    from akgentic.core.messages.message import Message

    from akgentic.infra.protocols import TeamHandle

    hints = get_type_hints(TeamHandle.send_to, localns={"Message": Message})
    assert hints["return"] is type(None)


# --- RuntimeCache ---


def test_runtime_cache_is_protocol() -> None:
    """RuntimeCache uses typing.Protocol base."""
    from akgentic.infra.protocols import RuntimeCache

    assert Protocol in inspect.getmro(RuntimeCache)


def test_runtime_cache_is_runtime_checkable() -> None:
    """RuntimeCache has @runtime_checkable decorator and isinstance works."""
    from akgentic.infra.protocols import RuntimeCache

    class FakeCache:
        def store(self, team_id: uuid.UUID, handle: object) -> None:
            pass

        def get(self, team_id: uuid.UUID) -> object:
            return None

        def remove(self, team_id: uuid.UUID) -> None:
            pass

    assert isinstance(FakeCache(), RuntimeCache)


def test_runtime_cache_has_store() -> None:
    """RuntimeCache defines store with team_id and handle parameters."""
    from akgentic.infra.protocols import RuntimeCache

    assert hasattr(RuntimeCache, "store")
    sig = inspect.signature(RuntimeCache.store)
    assert "team_id" in sig.parameters
    assert "handle" in sig.parameters


def test_runtime_cache_has_get() -> None:
    """RuntimeCache defines get with team_id parameter."""
    from akgentic.infra.protocols import RuntimeCache

    assert hasattr(RuntimeCache, "get")
    sig = inspect.signature(RuntimeCache.get)
    assert "team_id" in sig.parameters


def test_runtime_cache_has_remove() -> None:
    """RuntimeCache defines remove with team_id parameter."""
    from akgentic.infra.protocols import RuntimeCache

    assert hasattr(RuntimeCache, "remove")
    sig = inspect.signature(RuntimeCache.remove)
    assert "team_id" in sig.parameters


def test_runtime_cache_store_returns_none() -> None:
    """RuntimeCache.store returns None."""
    from akgentic.infra.protocols import RuntimeCache

    hints = get_type_hints(RuntimeCache.store)
    assert hints["return"] is type(None)


def test_runtime_cache_remove_returns_none() -> None:
    """RuntimeCache.remove returns None."""
    from akgentic.infra.protocols import RuntimeCache

    hints = get_type_hints(RuntimeCache.remove)
    assert hints["return"] is type(None)


def test_runtime_cache_method_count() -> None:
    """RuntimeCache has exactly 3 public methods."""
    from akgentic.infra.protocols import RuntimeCache

    public_methods = [
        m for m in dir(RuntimeCache) if not m.startswith("_") and callable(getattr(RuntimeCache, m))
    ]
    assert len(public_methods) == 3


# --- Non-channel protocols (unchanged) ---


def test_placement_strategy_return_type() -> None:
    """PlacementStrategy.create_team returns TeamHandle."""
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.team.models import TeamCard

    from akgentic.infra.protocols import PlacementStrategy, TeamHandle

    hints = get_type_hints(
        PlacementStrategy.create_team,
        localns={
            "TeamCard": TeamCard,
            "TeamHandle": TeamHandle,
            "SerializableBaseModel": SerializableBaseModel,
        },
    )
    assert hints["return"] is TeamHandle
    # Optional metadata pass-through: the seam that carries a validated model
    # from the service down to the persisted Process.
    assert hints["metadata"] == SerializableBaseModel | None


def test_placement_strategy_resume_team_returns_team_handle() -> None:
    """PlacementStrategy.resume_team returns TeamHandle."""
    from akgentic.infra.protocols import PlacementStrategy, TeamHandle

    hints = get_type_hints(
        PlacementStrategy.resume_team,
        localns={"TeamHandle": TeamHandle},
    )
    assert hints["return"] is TeamHandle


def test_recovery_policy_return_type() -> None:
    """RecoveryPolicy.recover returns None."""
    from akgentic.infra.protocols import RecoveryPolicy

    hints = get_type_hints(RecoveryPolicy.recover)
    assert hints["return"] is type(None)


def test_health_monitor_return_type() -> None:
    """HealthMonitor.check_health returns list[uuid.UUID]."""
    from akgentic.infra.protocols import HealthMonitor

    hints = get_type_hints(HealthMonitor.check_health)
    assert hints["return"] == list[uuid.UUID]


# --- WorkerHandle ---


def test_worker_handle_is_protocol() -> None:
    """WorkerHandle uses typing.Protocol base."""
    from akgentic.infra.protocols import WorkerHandle

    assert Protocol in inspect.getmro(WorkerHandle)


def test_worker_handle_is_runtime_checkable() -> None:
    """WorkerHandle has @runtime_checkable decorator and isinstance works."""
    from akgentic.infra.protocols import WorkerHandle

    class FakeWorkerHandle:
        def stop_team(self, team_id: uuid.UUID) -> None:
            pass

        def delete_team(self, team_id: uuid.UUID) -> None:
            pass

        def get_team(self, team_id: uuid.UUID) -> object:
            return None

        def update_team_metadata(self, team_id: uuid.UUID, metadata: object) -> object:
            return None

        def stop_all(self) -> None:
            pass

    assert isinstance(FakeWorkerHandle(), WorkerHandle)


def test_worker_handle_has_stop_team() -> None:
    """WorkerHandle defines stop_team with team_id parameter."""
    from akgentic.infra.protocols import WorkerHandle

    assert hasattr(WorkerHandle, "stop_team")
    sig = inspect.signature(WorkerHandle.stop_team)
    assert "team_id" in sig.parameters


def test_worker_handle_has_delete_team() -> None:
    """WorkerHandle defines delete_team with team_id parameter."""
    from akgentic.infra.protocols import WorkerHandle

    assert hasattr(WorkerHandle, "delete_team")
    sig = inspect.signature(WorkerHandle.delete_team)
    assert "team_id" in sig.parameters


def test_worker_handle_has_get_team() -> None:
    """WorkerHandle defines get_team with team_id parameter."""
    from akgentic.infra.protocols import WorkerHandle

    assert hasattr(WorkerHandle, "get_team")
    sig = inspect.signature(WorkerHandle.get_team)
    assert "team_id" in sig.parameters


def test_worker_handle_stop_team_returns_none() -> None:
    """WorkerHandle.stop_team returns None."""
    from akgentic.infra.protocols import WorkerHandle

    hints = get_type_hints(WorkerHandle.stop_team)
    assert hints["return"] is type(None)


def test_worker_handle_delete_team_returns_none() -> None:
    """WorkerHandle.delete_team returns None."""
    from akgentic.infra.protocols import WorkerHandle

    hints = get_type_hints(WorkerHandle.delete_team)
    assert hints["return"] is type(None)


def test_worker_handle_get_team_returns_process_or_none() -> None:
    """WorkerHandle.get_team returns Process | None."""
    from akgentic.team.models import Process

    from akgentic.infra.protocols import WorkerHandle

    hints = get_type_hints(
        WorkerHandle.get_team,
        localns={"Process": Process},
    )
    assert hints["return"] == Process | None


def test_worker_handle_has_update_team_metadata() -> None:
    """WorkerHandle defines update_team_metadata with team_id and metadata parameters."""
    from akgentic.infra.protocols import WorkerHandle

    assert hasattr(WorkerHandle, "update_team_metadata")
    sig = inspect.signature(WorkerHandle.update_team_metadata)
    assert "team_id" in sig.parameters
    assert "metadata" in sig.parameters


def test_worker_handle_update_team_metadata_types() -> None:
    """update_team_metadata takes a validated model (or None) and returns a Process.

    The metadata parameter is the *model*, not a raw dict: validation belongs
    above this seam, and the type below it is what the store persists.
    """
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.team.models import Process

    from akgentic.infra.protocols import WorkerHandle

    hints = get_type_hints(
        WorkerHandle.update_team_metadata,
        localns={"Process": Process, "SerializableBaseModel": SerializableBaseModel},
    )
    assert hints["metadata"] == SerializableBaseModel | None
    assert hints["return"] is Process


def test_worker_handle_has_stop_all() -> None:
    """WorkerHandle defines stop_all with no parameters (besides self)."""
    from akgentic.infra.protocols import WorkerHandle

    assert hasattr(WorkerHandle, "stop_all")
    sig = inspect.signature(WorkerHandle.stop_all)
    params = [p for p in sig.parameters if p != "self"]
    assert params == []


def test_worker_handle_stop_all_returns_none() -> None:
    """WorkerHandle.stop_all returns None."""
    from akgentic.infra.protocols import WorkerHandle

    hints = get_type_hints(WorkerHandle.stop_all)
    assert hints["return"] is type(None)


def test_worker_handle_method_count() -> None:
    """WorkerHandle has exactly 5 public methods.

    A count rather than a set, so widening the Protocol is a deliberate act:
    every tier implementation and every test fake has to grow the method too,
    and this failing is the reminder to sweep them. It was 6 until ``resume_team``
    moved to ``PlacementStrategy``.
    """
    from akgentic.infra.protocols import WorkerHandle

    public_methods = [
        m for m in dir(WorkerHandle) if not m.startswith("_") and callable(getattr(WorkerHandle, m))
    ]
    assert len(public_methods) == 5

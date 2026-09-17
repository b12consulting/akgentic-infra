"""Tests for YamlChannelRegistry — one binding record serving two indexes."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pytest
import yaml

from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.protocols import ChannelBinding, ChannelRegistry, ChannelRegistryReadSync


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    """Return a temporary path for the registry YAML file."""
    return tmp_path / "channel_registry.yaml"


@pytest.fixture()
def registry(registry_path: Path) -> YamlChannelRegistry:
    """Return a fresh YamlChannelRegistry instance."""
    return YamlChannelRegistry(registry_path)


def _binding(
    channel: str = "telegram",
    channel_user_id: str = "987654321",
    team_id: uuid.UUID | None = None,
    agent_name: str = "@HumanProxy_0",
) -> ChannelBinding:
    """Build a ChannelBinding with sensible defaults for the values under test."""
    return ChannelBinding(
        channel=channel,
        channel_user_id=channel_user_id,
        team_id=team_id if team_id is not None else uuid.uuid4(),
        agent_name=agent_name,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


async def test_satisfies_channel_registry_protocol(registry_path: Path) -> None:
    """YamlChannelRegistry satisfies the ChannelRegistry protocol.

    Built over ``tmp_path`` rather than a fixed ``/tmp`` name: construction
    reads and parses the file as of this story, so a shared path would make
    these two specs depend on whatever else happens to be on the machine.
    """
    assert isinstance(YamlChannelRegistry(registry_path), ChannelRegistry)


async def test_satisfies_channel_registry_read_sync_protocol(registry_path: Path) -> None:
    """YamlChannelRegistry also satisfies the synchronous read face."""
    assert isinstance(YamlChannelRegistry(registry_path), ChannelRegistryReadSync)


# ---------------------------------------------------------------------------
# AC 8 — the stored record is the dumped model, in full
# ---------------------------------------------------------------------------


async def test_register_persists_the_dumped_binding_in_full(
    registry: YamlChannelRegistry, registry_path: Path
) -> None:
    """The persisted mapping equals ``binding.model_dump(mode="json")`` exactly.

    Not "contains the fields we listed": a hand-listed subset is correct on the
    day it is written and silently drops whatever field ``ChannelBinding`` gains
    next.
    """
    binding = _binding(channel="telegram", channel_user_id="987654321")
    await registry.register(binding)

    data = yaml.safe_load(registry_path.read_text())  # noqa: ASYNC240
    assert data["telegram"]["987654321"] == binding.model_dump(mode="json")


async def test_register_creates_mapping(registry: YamlChannelRegistry, registry_path: Path) -> None:
    """register() persists a binding under channel → channel_user_id."""
    binding = _binding(channel="whatsapp", channel_user_id="+1234567890")
    await registry.register(binding)

    data = yaml.safe_load(registry_path.read_text())  # noqa: ASYNC240
    assert data["whatsapp"]["+1234567890"]["team_id"] == str(binding.team_id)
    assert data["whatsapp"]["+1234567890"]["agent_name"] == binding.agent_name


async def test_find_binding_returns_the_whole_record(registry: YamlChannelRegistry) -> None:
    """find_binding() returns an equal ChannelBinding, agent name included."""
    binding = _binding(channel="slack", channel_user_id="U12345", agent_name="@Assistant_0")
    await registry.register(binding)

    assert await registry.find_binding("slack", "U12345") == binding


async def test_find_binding_returns_none_for_unknown_user(registry: YamlChannelRegistry) -> None:
    """find_binding() returns None when nothing is stored for that user."""
    assert await registry.find_binding("slack", "U99999") is None


# ---------------------------------------------------------------------------
# find_team — unchanged signature, derived from the binding
# ---------------------------------------------------------------------------


async def test_find_team_returns_uuid(registry: YamlChannelRegistry) -> None:
    """find_team() returns the UUID for a registered channel user."""
    binding = _binding(channel="slack", channel_user_id="U12345")
    await registry.register(binding)

    result = await registry.find_team("slack", "U12345")
    assert result == binding.team_id


async def test_find_team_returns_none_for_unknown_channel(registry: YamlChannelRegistry) -> None:
    """find_team() returns None for an unregistered channel."""
    assert await registry.find_team("whatsapp", "+9999999999") is None


async def test_find_team_returns_none_for_unknown_user(registry: YamlChannelRegistry) -> None:
    """find_team() returns None when channel exists but user doesn't."""
    await registry.register(_binding(channel="slack", channel_user_id="U11111"))

    assert await registry.find_team("slack", "U99999") is None


async def test_uuid_serialization_roundtrip(registry: YamlChannelRegistry) -> None:
    """UUID is serialized to string and deserialized back correctly."""
    team_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
    await registry.register(_binding(channel="whatsapp", channel_user_id="+1", team_id=team_id))

    result = await registry.find_team("whatsapp", "+1")
    assert result == team_id
    assert isinstance(result, uuid.UUID)


async def test_multiple_channels(registry: YamlChannelRegistry) -> None:
    """Multiple channels with different users are tracked independently."""
    b1 = _binding(channel="whatsapp", channel_user_id="+111")
    b2 = _binding(channel="slack", channel_user_id="U222")
    await registry.register(b1)
    await registry.register(b2)

    assert await registry.find_team("whatsapp", "+111") == b1.team_id
    assert await registry.find_team("slack", "U222") == b2.team_id


async def test_multiple_users_same_channel(registry: YamlChannelRegistry) -> None:
    """Multiple users in the same channel each get their own binding."""
    b1 = _binding(channel="slack", channel_user_id="U111")
    b2 = _binding(channel="slack", channel_user_id="U222")
    await registry.register(b1)
    await registry.register(b2)

    assert await registry.find_team("slack", "U111") == b1.team_id
    assert await registry.find_team("slack", "U222") == b2.team_id


async def test_register_overwrites_existing(registry: YamlChannelRegistry) -> None:
    """Re-registering the same user replaces the stored binding."""
    first = _binding(channel="slack", channel_user_id="U111", agent_name="@HumanProxy_0")
    second = _binding(channel="slack", channel_user_id="U111", agent_name="@HumanProxy_1")
    await registry.register(first)
    await registry.register(second)

    assert await registry.find_binding("slack", "U111") == second


async def test_register_overwrite_releases_the_stale_sync_entry(
    registry: YamlChannelRegistry,
) -> None:
    """The replaced binding stops answering the sync read.

    The memory index is rebuilt from what was written, so a superseded
    ``(team_id, agent_name)`` cannot linger and route a later message to a chat
    that has moved on.
    """
    first = _binding(channel="slack", channel_user_id="U111")
    second = _binding(channel="slack", channel_user_id="U111")
    await registry.register(first)
    await registry.register(second)

    assert registry.find_binding_sync(first.team_id, first.agent_name) is None
    assert registry.find_binding_sync(second.team_id, second.agent_name) == second


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------


async def test_deregister_removes_mapping(registry: YamlChannelRegistry) -> None:
    """deregister() removes the binding for a channel user."""
    await registry.register(_binding(channel="whatsapp", channel_user_id="+1234567890"))
    await registry.deregister("whatsapp", "+1234567890")

    assert await registry.find_team("whatsapp", "+1234567890") is None


async def test_deregister_clears_the_sync_index_too(registry: YamlChannelRegistry) -> None:
    """A deregistered binding stops answering the sync read as well."""
    binding = _binding(channel="whatsapp", channel_user_id="+1234567890")
    await registry.register(binding)
    await registry.deregister("whatsapp", "+1234567890")

    assert registry.find_binding_sync(binding.team_id, binding.agent_name) is None


async def test_deregister_unknown_channel_is_noop(registry: YamlChannelRegistry) -> None:
    """deregister() on an unknown channel does not raise."""
    await registry.deregister("nonexistent", "nobody")


async def test_deregister_unknown_user_is_noop(registry: YamlChannelRegistry) -> None:
    """deregister() for unknown user in existing channel does not raise."""
    await registry.register(_binding(channel="slack", channel_user_id="U11111"))
    await registry.deregister("slack", "U99999")


async def test_deregister_removes_empty_channel_section(
    registry: YamlChannelRegistry, registry_path: Path
) -> None:
    """Deregistering the last user in a channel removes the channel section."""
    await registry.register(_binding(channel="whatsapp", channel_user_id="+111"))
    await registry.deregister("whatsapp", "+111")

    data = yaml.safe_load(registry_path.read_text())  # noqa: ASYNC240
    assert data is None or "whatsapp" not in (data or {})


# ---------------------------------------------------------------------------
# AC 11 — deregister_team clears file *and* memory
# ---------------------------------------------------------------------------


async def test_deregister_team_clears_both_indexes(registry: YamlChannelRegistry) -> None:
    """deregister_team() leaves neither the disk read nor the memory read answering.

    Updating one and not the other is the defect this spec exists to catch: the
    inbound path would start a new team while the outbound path kept delivering
    into the old chat.
    """
    binding = _binding(channel="telegram", channel_user_id="987654321")
    await registry.register(binding)

    await registry.deregister_team(binding.team_id)

    assert await registry.find_team("telegram", "987654321") is None
    assert registry.find_binding_sync(binding.team_id, binding.agent_name) is None


async def test_deregister_team_removes_every_channel(registry: YamlChannelRegistry) -> None:
    """Every record for the team goes, across all channels; others are untouched."""
    team_id = uuid.uuid4()
    await registry.register(_binding(channel="telegram", channel_user_id="111", team_id=team_id))
    await registry.register(_binding(channel="slack", channel_user_id="U222", team_id=team_id))
    survivor = _binding(channel="slack", channel_user_id="U333")
    await registry.register(survivor)

    await registry.deregister_team(team_id)

    assert await registry.find_team("telegram", "111") is None
    assert await registry.find_team("slack", "U222") is None
    assert await registry.find_binding("slack", "U333") == survivor


async def test_deregister_team_drops_emptied_channel_sections(
    registry: YamlChannelRegistry, registry_path: Path
) -> None:
    """A channel section emptied by deregister_team is removed, as deregister does."""
    binding = _binding(channel="telegram", channel_user_id="111")
    await registry.register(binding)

    await registry.deregister_team(binding.team_id)

    data = yaml.safe_load(registry_path.read_text())  # noqa: ASYNC240
    assert data is None or "telegram" not in (data or {})


async def test_deregister_team_unknown_team_is_noop(registry: YamlChannelRegistry) -> None:
    """deregister_team() for a team with no bindings does not raise or clear others."""
    survivor = _binding(channel="slack", channel_user_id="U111")
    await registry.register(survivor)

    await registry.deregister_team(uuid.uuid4())

    assert await registry.find_binding("slack", "U111") == survivor


# ---------------------------------------------------------------------------
# AC 10 — the sync read answers from memory
# ---------------------------------------------------------------------------


async def test_find_binding_sync_returns_the_registered_binding(
    registry: YamlChannelRegistry,
) -> None:
    """The sync read is keyed by (team_id, agent_name)."""
    binding = _binding(channel="telegram", channel_user_id="987654321")
    await registry.register(binding)

    assert registry.find_binding_sync(binding.team_id, binding.agent_name) == binding


async def test_find_binding_sync_answers_after_the_file_is_deleted(
    registry: YamlChannelRegistry, registry_path: Path
) -> None:
    """AC 10(a): the sync read never touches the disk.

    Deleting the file is the only way to prove it: an implementation that calls
    ``_load()`` returns None here, and would block a Pykka actor thread on file
    I/O in production.
    """
    binding = _binding(channel="telegram", channel_user_id="987654321")
    await registry.register(binding)
    registry_path.unlink()  # noqa: ASYNC240

    assert registry.find_binding_sync(binding.team_id, binding.agent_name) == binding


async def test_find_binding_sync_answers_from_a_second_instance(
    registry: YamlChannelRegistry, registry_path: Path
) -> None:
    """AC 10(b): the index is primed at construction, so a restart still answers.

    An index that is only filled on write answers None here — and would leave a
    restarted server unable to deliver to any conversation it did not itself
    create.
    """
    binding = _binding(channel="telegram", channel_user_id="987654321")
    await registry.register(binding)

    reopened = YamlChannelRegistry(registry_path)

    assert reopened.find_binding_sync(binding.team_id, binding.agent_name) == binding


async def test_find_binding_sync_returns_none_for_unknown_agent(
    registry: YamlChannelRegistry,
) -> None:
    """An agent the index has never seen answers None, not an exception."""
    binding = _binding()
    await registry.register(binding)

    assert registry.find_binding_sync(binding.team_id, "@Nobody_9") is None
    assert registry.find_binding_sync(uuid.uuid4(), binding.agent_name) is None


# ---------------------------------------------------------------------------
# AC 9 — a legacy record reads as absent
# ---------------------------------------------------------------------------


async def test_legacy_string_record_reads_as_absent(
    registry_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A pre-binding ``channel_user_id: "<team-uuid>"`` record is ignored, not fatal.

    No migration: the record carries no agent name, so a binding synthesised
    from it could never satisfy an outbound lookup and would present as a silent
    delivery failure. Reading it as absent sends the next inbound message down
    the initiation branch, which writes a proper binding.
    """
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": {"987654321": str(uuid.uuid4())}}),
        encoding="utf-8",
    )
    reg = YamlChannelRegistry(registry_path)

    with caplog.at_level(logging.WARNING):
        assert await reg.find_team("telegram", "987654321") is None
        assert await reg.find_binding("telegram", "987654321") is None

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert all("telegram" in message and "987654321" in message for message in warnings)


async def test_legacy_record_leaves_the_sync_index_empty(registry_path: Path) -> None:
    """Priming skips a legacy record rather than failing construction."""
    team_id = uuid.uuid4()
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": {"987654321": str(team_id)}}),
        encoding="utf-8",
    )

    reg = YamlChannelRegistry(registry_path)

    assert reg.find_binding_sync(team_id, "@HumanProxy_0") is None


async def test_a_legacy_record_does_not_block_a_fresh_binding(registry_path: Path) -> None:
    """The next inbound message self-heals the record in place."""
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": {"987654321": str(uuid.uuid4())}}),
        encoding="utf-8",
    )
    reg = YamlChannelRegistry(registry_path)

    binding = _binding(channel="telegram", channel_user_id="987654321")
    await reg.register(binding)

    assert await reg.find_binding("telegram", "987654321") == binding
    assert reg.find_binding_sync(binding.team_id, binding.agent_name) == binding


# ---------------------------------------------------------------------------
# An unreadable record is absent, whatever shape it is unreadable in
# ---------------------------------------------------------------------------


async def test_a_mapping_record_that_does_not_validate_reads_as_absent(
    registry_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mapping that is not a valid binding is absent too, not a 500.

    AC 9 names the pre-binding *string* record, but the binding format's own
    failure mode is a mapping with a field missing or misspelled — a hand-edit,
    or a file half-written when the process died. The async surface is
    deliberately disk-backed so a registry edited out of band is still
    honoured, which makes a malformed edit a reachable input rather than a
    corruption that cannot happen. The inbound path can act on neither shape,
    so both take the same self-healing initiation branch.
    """
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": {"987654321": {"team_id": str(uuid.uuid4())}}}),
        encoding="utf-8",
    )
    reg = YamlChannelRegistry(registry_path)

    with caplog.at_level(logging.WARNING):
        assert await reg.find_binding("telegram", "987654321") is None
        assert await reg.find_team("telegram", "987654321") is None

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert all("telegram" in message and "987654321" in message for message in warnings)


async def test_a_malformed_record_does_not_break_construction(registry_path: Path) -> None:
    """Priming walks the whole file, so one bad record must not fail the constructor.

    ``__init__`` reads the file as of this story; before it, construction did
    no I/O and a bad record cost exactly one lookup. Priming puts every record
    on the path ``wire_community`` runs at startup, so raising here takes the
    server down instead of degrading one conversation.
    """
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": {"987654321": {"team_id": str(uuid.uuid4())}}}),
        encoding="utf-8",
    )

    reg = YamlChannelRegistry(registry_path)

    assert reg.find_binding_sync(uuid.uuid4(), "@HumanProxy_0") is None


async def test_a_corrupt_channel_section_does_not_break_construction(
    registry_path: Path,
) -> None:
    """A channel section that is not a mapping of users is skipped, not fatal.

    Same reasoning one level up: priming iterates the sections too, so a
    hand-edit that leaves a scalar where a mapping belongs would otherwise
    raise out of the constructor.
    """
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": "not-a-mapping"}),
        encoding="utf-8",
    )

    reg = YamlChannelRegistry(registry_path)

    assert reg.find_binding_sync(uuid.uuid4(), "@HumanProxy_0") is None


async def test_a_malformed_record_does_not_block_a_fresh_binding(registry_path: Path) -> None:
    """The next inbound message overwrites it, exactly as for a legacy record."""
    registry_path.write_text(  # noqa: ASYNC240
        yaml.safe_dump({"telegram": {"987654321": {"team_id": str(uuid.uuid4())}}}),
        encoding="utf-8",
    )
    reg = YamlChannelRegistry(registry_path)

    binding = _binding(channel="telegram", channel_user_id="987654321")
    await reg.register(binding)

    assert await reg.find_binding("telegram", "987654321") == binding
    assert reg.find_binding_sync(binding.team_id, binding.agent_name) == binding


# ---------------------------------------------------------------------------
# Empty / missing files
# ---------------------------------------------------------------------------


async def test_missing_file_treated_as_empty(registry_path: Path) -> None:
    """A non-existent YAML file is treated as an empty registry."""
    reg = YamlChannelRegistry(registry_path)
    assert await reg.find_team("whatsapp", "+1234567890") is None


async def test_empty_file_treated_as_empty(registry_path: Path) -> None:
    """An empty YAML file is treated as an empty registry."""
    registry_path.write_text("")  # noqa: ASYNC240
    reg = YamlChannelRegistry(registry_path)
    assert await reg.find_team("whatsapp", "+1234567890") is None


# ---------------------------------------------------------------------------
# Disabled registry
# ---------------------------------------------------------------------------


async def test_disabled_registry_find_team_returns_none() -> None:
    """With no path configured the registry is disabled: find_team returns None."""
    reg = YamlChannelRegistry()
    assert await reg.find_team("whatsapp", "+1234567890") is None


async def test_disabled_registry_find_binding_returns_none() -> None:
    """find_binding() returns None when the registry is disabled."""
    reg = YamlChannelRegistry()
    assert await reg.find_binding("whatsapp", "+1234567890") is None


async def test_disabled_registry_register_is_noop() -> None:
    """register() is a no-op when the registry is disabled (no file I/O, no error)."""
    reg = YamlChannelRegistry()
    await reg.register(_binding(channel="slack", channel_user_id="U111"))
    assert await reg.find_team("slack", "U111") is None


async def test_disabled_registry_find_binding_sync_returns_none() -> None:
    """A disabled registry stays fully disabled — no in-memory consolation.

    Populating the index while the file stays empty would leave half a registry:
    outbound delivery working for this process only, and silently stopping at
    the next restart.
    """
    reg = YamlChannelRegistry()
    binding = _binding(channel="slack", channel_user_id="U111")

    await reg.register(binding)

    assert reg.find_binding_sync(binding.team_id, binding.agent_name) is None


async def test_disabled_registry_deregister_is_noop() -> None:
    """deregister() is a no-op when the registry is disabled."""
    reg = YamlChannelRegistry()
    await reg.deregister("slack", "U111")


async def test_disabled_registry_deregister_team_is_noop() -> None:
    """deregister_team() is a no-op when the registry is disabled."""
    reg = YamlChannelRegistry()
    await reg.deregister_team(uuid.uuid4())

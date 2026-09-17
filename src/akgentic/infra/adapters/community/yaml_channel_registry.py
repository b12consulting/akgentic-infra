"""YAML-backed channel registry — stores one binding record per channel user."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from akgentic.infra.protocols.channels import ChannelBinding

if TYPE_CHECKING:
    from pathlib import Path

    from akgentic.infra.protocols.channels import JsonValue

logger = logging.getLogger(__name__)


class YamlChannelRegistry:
    """Persists ``ChannelBinding`` records in a YAML file.

    File format — one record per ``(channel, channel_user_id)``, written as the
    dumped model in full::

        telegram:
          "987654321":
            __model__: akgentic.infra.protocols.channels.ChannelBinding
            channel: telegram
            channel_user_id: "987654321"
            team_id: 550e8400-e29b-41d4-a716-446655440000
            agent_name: "@HumanProxy_0"

    ``__model__`` is written by ``SerializableBaseModel``'s serializer, not by
    this class; it is listed here because it is in the file an operator opens.

    Satisfies the ``ChannelRegistry`` protocol, including its inherited
    synchronous read.

    Two indexes, one record:

    - The **file** is the authority for the async surface. It is re-read on
      every call, so a registry edited out of band is still honoured on the
      inbound path.
    - An **in-process index** keyed by ``(team_id, agent_name)`` answers
      ``find_binding_sync`` alone. It is primed at construction and rebuilt from
      the data every mutation writes, so the two cannot drift apart within a
      process. The sync read performs no I/O: it is called from a Pykka actor
      thread that must not block (ADR-043 §D5).

    When ``registry_path`` is ``None`` the registry is disabled: reads return
    ``None``, the mutations are no-ops (no file I/O), and the index stays empty —
    ``register`` does not populate it as an in-memory consolation, because half
    a registry is harder to reason about than none.

    Records written before the binding format — ``channel_user_id:
    "<team-uuid>"`` — read as **absent**, with one warning. The next inbound
    message from that conversation takes the initiation branch and writes a
    proper binding.
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = registry_path
        self._sync_index: dict[tuple[uuid.UUID, str], ChannelBinding] = self._index_from_data(
            self._load()
        )

    # -- storage -----------------------------------------------------------

    def _load(self) -> dict[str, dict[str, JsonValue]]:
        """Load registry data from YAML, returning empty dict if disabled or missing."""
        if self._path is None or not self._path.exists():
            return {}
        text = self._path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is None:
            return {}
        return dict(data)

    def _save(self, data: dict[str, dict[str, JsonValue]]) -> None:
        """Write registry data to YAML (callers guard the disabled, path-less case)."""
        assert self._path is not None  # mutations return early when disabled
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")

    def _commit(self, data: dict[str, dict[str, JsonValue]]) -> None:
        """Persist ``data`` and rebuild the sync index from the same snapshot.

        Both indexes move together by construction: there is no code path that
        writes the file and forgets the index, or the reverse.
        """
        self._save(data)
        self._sync_index = self._index_from_data(data)

    # -- record parsing ----------------------------------------------------

    @staticmethod
    def _binding_from_record(record: JsonValue) -> ChannelBinding | None:
        """Parse one stored record, or return None if it is not a readable binding.

        Two shapes read as absent rather than raising: the pre-binding string
        form, and a mapping that does not validate — a field missing or
        misspelled by a hand-edit, or a file half-written when the process
        died. The async surface is deliberately disk-backed so a registry
        edited out of band is still honoured, which makes a malformed edit a
        reachable input rather than a corruption that cannot happen. The
        inbound path can act on neither shape, so both take the same
        self-healing initiation branch.
        """
        if not isinstance(record, dict):
            return None
        try:
            return ChannelBinding.model_validate(record)
        except ValidationError:
            return None

    @classmethod
    def _index_from_data(
        cls, data: dict[str, dict[str, JsonValue]]
    ) -> dict[tuple[uuid.UUID, str], ChannelBinding]:
        """Build the ``(team_id, agent_name)`` index, skipping unreadable records."""
        index: dict[tuple[uuid.UUID, str], ChannelBinding] = {}
        for channel_data in data.values():
            # Priming runs at construction, on the path the server starts from,
            # so it must be total: a section left as a scalar by a hand-edit is
            # skipped rather than raising out of __init__.
            if not isinstance(channel_data, dict):
                continue
            for record in channel_data.values():
                binding = cls._binding_from_record(record)
                if binding is not None:
                    index[(binding.team_id, binding.agent_name)] = binding
        return index

    # -- async surface (inbound path) --------------------------------------

    async def register(self, binding: ChannelBinding) -> None:
        """Store a binding, replacing any existing one for the same channel user.

        The record is written as ``binding.model_dump(mode="json")`` in full, so
        a field added to ``ChannelBinding`` later is persisted without anybody
        remembering to add it here.
        """
        if self._path is None:
            return
        data = self._load()
        if binding.channel not in data:
            data[binding.channel] = {}
        data[binding.channel][binding.channel_user_id] = binding.model_dump(mode="json")
        self._commit(data)
        logger.debug(
            "Channel registry: registered %s/%s → team %s, agent %s",
            binding.channel,
            binding.channel_user_id,
            binding.team_id,
            binding.agent_name,
        )

    async def find_binding(self, channel: str, channel_user_id: str) -> ChannelBinding | None:
        """Look up the whole binding for a channel user, or return None."""
        channel_data = self._load().get(channel, {})
        if not isinstance(channel_data, dict):
            # A hand-edit leaving a scalar where the per-user mapping belongs.
            # Construction already skips it; this read must too, because the
            # reply branch performs its security check through here and a check
            # that raises cannot be performed at all. The section is left on
            # disk: filtering it out at load time would erase the operator's
            # edit on the next mutation, which writes back what it read.
            logger.warning(
                "Channel registry: ignoring section for %s — not a mapping of channel users; "
                "the next inbound message will start a new team",
                channel,
            )
            return None
        record = channel_data.get(channel_user_id)
        if record is None:
            logger.debug("Channel registry: lookup %s/%s → None", channel, channel_user_id)
            return None
        binding = self._binding_from_record(record)
        if binding is None:
            logger.warning(
                "Channel registry: ignoring record for %s/%s — not a binding mapping; "
                "the next inbound message will start a new team",
                channel,
                channel_user_id,
            )
            return None
        logger.debug(
            "Channel registry: lookup %s/%s → team %s",
            channel,
            channel_user_id,
            binding.team_id,
        )
        return binding

    async def find_team(self, channel: str, channel_user_id: str) -> uuid.UUID | None:
        """Look up the team for a channel user, or return None.

        Derived from the binding rather than parsed here, so one record shape is
        read in exactly one place.
        """
        binding = await self.find_binding(channel, channel_user_id)
        return None if binding is None else binding.team_id

    async def deregister(self, channel: str, channel_user_id: str) -> None:
        """Remove a channel user's binding if it exists (no-op when disabled)."""
        if self._path is None:
            return
        data = self._load()
        channel_data = data.get(channel)
        if channel_data is None:
            return
        if not isinstance(channel_data, dict):
            # A hand-edit leaving a scalar where the per-user mapping belongs.
            # Popping straight off it raises ``AttributeError``; this read is
            # skipped instead, and the section is left on disk — filtering it
            # out at load time would erase the operator's edit on the next
            # ``register``, which writes back what it read.
            logger.warning(
                "Channel registry: ignoring section for %s on deregister — "
                "not a mapping of channel users",
                channel,
            )
            return
        channel_data.pop(channel_user_id, None)
        if not channel_data:
            del data[channel]
        self._commit(data)
        logger.debug("Channel registry: deregistered %s/%s", channel, channel_user_id)

    async def deregister_team(self, team_id: uuid.UUID) -> None:
        """Remove every binding for a team, across all channels (no-op when disabled)."""
        if self._path is None:
            return
        data = self._load()
        if not self._prune_team(data, team_id):
            return
        self._commit(data)
        logger.debug("Channel registry: deregistered every binding for team %s", team_id)

    @classmethod
    def _prune_team(cls, data: dict[str, dict[str, JsonValue]], team_id: uuid.UUID) -> bool:
        """Drop every record for ``team_id`` in place; return whether anything went."""
        removed = False
        for channel in list(data):
            channel_data = data[channel]
            if not isinstance(channel_data, dict):
                # Iterating a scalar yields its *characters*, and indexing a str
                # with one raises ``TypeError``. This runs on team teardown, via
                # ``asyncio.run`` on the orchestrator thread, where a raise lands
                # far from its cause — so the section is skipped and left alone.
                logger.warning(
                    "Channel registry: ignoring section for %s while pruning a team — "
                    "not a mapping of channel users",
                    channel,
                )
                continue
            emptied = False
            for channel_user_id in list(channel_data):
                binding = cls._binding_from_record(channel_data[channel_user_id])
                if binding is not None and binding.team_id == team_id:
                    del channel_data[channel_user_id]
                    emptied = True
            if emptied and not channel_data:
                del data[channel]
            removed = removed or emptied
        return removed

    # -- sync surface (outbound path) --------------------------------------

    def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
        """Return the binding for one agent of one team, from memory only.

        No file read, no ``await``: this runs in a Pykka actor thread with no
        event loop. An answer this index has not seen is ``None``.
        """
        return self._sync_index.get((team_id, agent_name))

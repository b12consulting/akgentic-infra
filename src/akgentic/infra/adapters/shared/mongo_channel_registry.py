"""MongoDB-backed ``ChannelRegistry`` — the channel bindings of any tier on Mongo.

Implements the :class:`~akgentic.infra.protocols.channels.ChannelRegistry`
Protocol on one MongoDB collection (``channel_users``) in the database the
caller hands in, so the bindings live beside the teams they point at. Each
document is a whole :class:`~akgentic.infra.protocols.channels.ChannelBinding`
as ``model_dump(mode="json")`` writes it.

Two indexes, created together and lazily on the first async operation:

- a **unique** compound index on ``(channel, channel_user_id)``, the inbound
  path's key;
- a **non-unique** compound index on ``(team_id, agent_name)``, which makes the
  outbound ``find_binding_sync`` one indexed point read rather than a scan, and
  serves ``deregister_team``.

Threading: the async methods run their synchronous pymongo work — the index
creation included — through ``asyncio.to_thread``, so no driver call runs on the
event loop. ``find_binding_sync`` stays a direct call on the caller's thread (a
Pykka actor thread in production), which the protocol permits for one indexed
point read.

A document without ``agent_name`` — written before bindings named an agent —
reads as absent: the next message starts a team and ``register`` overwrites it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from akgentic.infra.adapters.shared._stored_channel_binding import read_stored_binding
from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding, ChannelRegistry

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "channel_users"


class MongoChannelRegistry(ChannelRegistry):
    """Mongo-backed :class:`ChannelRegistry` on ``db["channel_users"]``.

    Stores one ``ChannelBinding`` per ``(channel, channel_user_id)``. ``team_id``
    is stored as a string, which is what ``model_dump(mode="json")`` produces.

    Construction does no I/O. The first async operation creates the unique
    ``(channel, channel_user_id)`` index and the ``(team_id, agent_name)`` index
    that ``find_binding_sync`` reads through. Every async method offloads its
    driver calls with ``asyncio.to_thread``; ``find_binding_sync`` does not.

    Args:
        db: The database holding the ``channel_users`` collection — the
            caller's, so the bindings share it with the teams they name. The
            registry creates, configures and closes no client.
    """

    # ``Any``: a BSON document has no static shape at the driver boundary.
    def __init__(self, db: Database[Any]) -> None:
        # ``Any``: a BSON document has no static shape at the driver boundary.
        self._collection: Collection[Any] = db[_COLLECTION_NAME]
        # A plain bool that worker threads may race on: two concurrent first
        # calls both run ``create_index``, which is idempotent, so no lock.
        self._index_ensured: bool = False

    def _ensure_index(self) -> None:
        """Create both indexes once, on the first async operation.

        Lazy and idempotent — gated on ``_index_ensured`` so nothing touches the
        collection at construction time. Always called from a worker thread.
        """
        if self._index_ensured:
            return
        self._collection.create_index([("channel", 1), ("channel_user_id", 1)], unique=True)
        self._collection.create_index([("team_id", 1), ("agent_name", 1)])
        self._index_ensured = True

    @staticmethod
    def _address_filter(address: ChannelAddress) -> dict[str, str]:
        return {"channel": address.channel, "channel_user_id": address.channel_user_id}

    # -- async surface (inbound path), offloaded ----------------------------

    async def register(self, binding: ChannelBinding) -> None:
        """Store the whole binding, replacing any existing one for its address."""
        await asyncio.to_thread(self._register_blocking, binding)
        logger.debug(
            "Registered channel binding: channel=%s, channel_user_id=%s, team_id=%s, agent=%s",
            binding.channel,
            binding.channel_user_id,
            binding.team_id,
            binding.agent_name,
        )

    def _register_blocking(self, binding: ChannelBinding) -> None:
        self._ensure_index()
        self._collection.update_one(
            self._address_filter(binding),
            {"$set": binding.model_dump(mode="json")},
            upsert=True,
        )

    async def find_binding(self, address: ChannelAddress) -> ChannelBinding | None:
        """Return the binding for ``address``, or ``None``."""
        return await asyncio.to_thread(self._find_binding_blocking, address)

    def _find_binding_blocking(self, address: ChannelAddress) -> ChannelBinding | None:
        self._ensure_index()
        doc = self._collection.find_one(self._address_filter(address), {"_id": 0})
        return read_stored_binding(doc)

    async def deregister(self, address: ChannelAddress) -> None:
        """Remove the binding for ``address``; a no-op when absent."""
        await asyncio.to_thread(self._deregister_blocking, address)
        logger.debug(
            "Deregistered channel binding: channel=%s, channel_user_id=%s",
            address.channel,
            address.channel_user_id,
        )

    def _deregister_blocking(self, address: ChannelAddress) -> None:
        self._ensure_index()
        self._collection.delete_one(self._address_filter(address))

    async def deregister_team(self, team_id: uuid.UUID) -> None:
        """Remove every binding for ``team_id``, across channels; a no-op when none."""
        await asyncio.to_thread(self._deregister_team_blocking, team_id)
        logger.debug("Deregistered channel bindings of team_id=%s", team_id)

    def _deregister_team_blocking(self, team_id: uuid.UUID) -> None:
        self._ensure_index()
        self._collection.delete_many({"team_id": str(team_id)})

    # -- sync surface (outbound path), on the caller's thread ---------------

    def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
        """Return the binding for one agent of one team, or ``None``.

        One point read through the ``(team_id, agent_name)`` index, on the
        caller's thread — never a scan, never async I/O. A direct query rather
        than an in-process index: another process may have written the binding
        (department's server writes it, its worker reads it), and the server
        keeps no cross-request state, so any replica must answer from the
        collection.

        It does not create the indexes; they live in the database, created by
        the first async operation of *any* instance on it. A collection written
        before this class existed (department's earlier registry built only the
        unique index) has no ``(team_id, agent_name)`` index until such an
        operation runs, and until then this read scans. A deployment adopting
        this class on existing bindings creates that index at rollout.
        """
        doc = self._collection.find_one(
            {"team_id": str(team_id), "agent_name": agent_name}, {"_id": 0}
        )
        return read_stored_binding(doc)

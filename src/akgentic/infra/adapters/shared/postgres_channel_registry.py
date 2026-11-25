"""PostgreSQL-backed ``ChannelRegistry`` — the channel bindings of any tier on Postgres.

Implements the :class:`~akgentic.infra.protocols.channels.ChannelRegistry`
Protocol on one table, ``channel_users`` (the Mongo collection's name), in the
database the caller's DSN names, so the bindings live beside the teams they
point at. Each row's ``data`` column holds a whole
:class:`~akgentic.infra.protocols.channels.ChannelBinding` as
``model_dump(mode="json")`` writes it; the ``channel``, ``channel_user_id``,
``team_id`` and ``agent_name`` columns exist only to be keyed and indexed.

Two indexes, both created by :func:`init_db` — a deployment step
(``python -m akgentic.infra.scripts.init_db``), **never** the server:

- ``channel_users_idx``, the **unique** index Nagra builds from the natural key
  ``(channel, channel_user_id)``, the inbound path's key;
- ``channel_users_team_agent_idx`` on ``(team_id, agent_name)``, which makes the
  outbound ``find_binding_sync`` one indexed point read rather than a scan, and
  serves ``deregister_team`` through its leading column.

A database the script has not run against has no table: every channel
operation fails with ``relation "channel_users" does not exist``.

Threading: each async method runs its synchronous Nagra work through **one**
``asyncio.to_thread`` call, so no driver call runs on the event loop.
``find_binding_sync`` stays one direct point read on the caller's thread (a
Pykka actor thread in production), which the protocol permits.

Nagra is imported only inside the functions that use it, so importing this
module needs no driver. Every method opens its own ``nagra.Transaction(dsn)``;
Nagra keeps one connection pool per DSN, shared with the team event store and
the catalog repository on the same DSN.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from akgentic.infra.adapters.shared._stored_channel_binding import read_stored_binding
from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding, ChannelRegistry

logger = logging.getLogger(__name__)

__all__ = ["PostgresChannelRegistry", "init_db"]

_SCHEMA_PATH = Path(__file__).with_name("postgres_channel_registry_schema.toml")

_TEAM_AGENT_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS channel_users_team_agent_idx ON channel_users (team_id, agent_name)"
)

_REGISTER_SQL = (
    "INSERT INTO channel_users (channel, channel_user_id, team_id, agent_name, data) "
    "VALUES (%s, %s, %s, %s, %s) "
    "ON CONFLICT (channel, channel_user_id) DO UPDATE SET "
    "team_id = EXCLUDED.team_id, agent_name = EXCLUDED.agent_name, data = EXCLUDED.data"
)
_FIND_BY_ADDRESS_SQL = "SELECT data FROM channel_users WHERE channel = %s AND channel_user_id = %s"
_FIND_BY_TEAM_AGENT_SQL = (
    "SELECT data FROM channel_users WHERE team_id = %s AND agent_name = %s LIMIT 1"
)
_DELETE_BY_ADDRESS_SQL = "DELETE FROM channel_users WHERE channel = %s AND channel_user_id = %s"
_DELETE_BY_TEAM_SQL = "DELETE FROM channel_users WHERE team_id = %s"


def init_db(dsn: str) -> None:
    """Create the ``channel_users`` table and both its indexes; idempotent.

    Loads the schema into a private ``nagra.Schema()`` — never
    ``Schema.default``, which the team and catalog packages fill in the same
    process — then, in one transaction, creates what is missing (table,
    columns, the unique natural-key index) and the ``(team_id, agent_name)``
    index. The index DDL runs after the table exists.

    A deployment step, run by ``python -m akgentic.infra.scripts.init_db``;
    the server never calls it.

    Args:
        dsn: A ``postgresql://`` connection string.
    """
    from nagra import Schema, Transaction

    schema = Schema()
    schema.load_toml(_SCHEMA_PATH)
    with Transaction(dsn) as trn:
        schema.create_tables(trn=trn)
        trn.execute(_TEAM_AGENT_INDEX_SQL)


# ``Any``: a JSON document has no static shape at the driver boundary.
def _stored_document(row: tuple[object, ...] | None) -> Mapping[str, Any] | None:
    """The ``data`` of a ``SELECT data`` row, decoded; ``None`` when there is no row.

    psycopg 3 decodes a ``json`` column to Python objects; a string is decoded
    here. A value that is not a JSON object reads as an empty document, which
    ``read_stored_binding`` reports and ignores like any other unreadable one.
    """
    if row is None:
        return None
    data = row[0]
    if isinstance(data, str | bytes):
        data = json.loads(data)
    return data if isinstance(data, Mapping) else {}


class PostgresChannelRegistry(ChannelRegistry):
    """Postgres-backed :class:`ChannelRegistry` on the ``channel_users`` table.

    Stores one ``ChannelBinding`` per ``(channel, channel_user_id)``, whole, in
    the ``data`` column. The table and its two indexes — the unique
    ``(channel, channel_user_id)`` one and ``(team_id, agent_name)`` — are
    created by :func:`init_db`, never by this class or the server.

    Construction does no I/O and imports no driver. Every async method runs its
    Nagra work through one ``asyncio.to_thread`` call; ``find_binding_sync`` is
    one point read through the ``(team_id, agent_name)`` index on the caller's
    thread, never offloaded.

    Args:
        dsn: A ``postgresql://`` connection string — the caller's, so the
            bindings share the database with the teams they name.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _execute(self, sql: str, params: tuple[str, ...]) -> None:
        from nagra import Transaction

        with Transaction(self._dsn) as trn:
            trn.execute(sql, params)

    def _fetch_binding(self, sql: str, params: tuple[str, ...]) -> ChannelBinding | None:
        from nagra import Transaction

        with Transaction(self._dsn) as trn:
            row = trn.execute(sql, params).fetchone()
        return read_stored_binding(_stored_document(row))

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
        params = (
            binding.channel,
            binding.channel_user_id,
            str(binding.team_id),
            binding.agent_name,
            json.dumps(binding.model_dump(mode="json")),
        )
        self._execute(_REGISTER_SQL, params)

    async def find_binding(self, address: ChannelAddress) -> ChannelBinding | None:
        """Return the binding for ``address``, or ``None``."""
        return await asyncio.to_thread(self._find_binding_blocking, address)

    def _find_binding_blocking(self, address: ChannelAddress) -> ChannelBinding | None:
        return self._fetch_binding(_FIND_BY_ADDRESS_SQL, (address.channel, address.channel_user_id))

    async def deregister(self, address: ChannelAddress) -> None:
        """Remove the binding for ``address``; a no-op when absent."""
        await asyncio.to_thread(self._deregister_blocking, address)
        logger.debug(
            "Deregistered channel binding: channel=%s, channel_user_id=%s",
            address.channel,
            address.channel_user_id,
        )

    def _deregister_blocking(self, address: ChannelAddress) -> None:
        self._execute(_DELETE_BY_ADDRESS_SQL, (address.channel, address.channel_user_id))

    async def deregister_team(self, team_id: uuid.UUID) -> None:
        """Remove every binding for ``team_id``, across channels; a no-op when none."""
        await asyncio.to_thread(self._deregister_team_blocking, team_id)
        logger.debug("Deregistered channel bindings of team_id=%s", team_id)

    def _deregister_team_blocking(self, team_id: uuid.UUID) -> None:
        self._execute(_DELETE_BY_TEAM_SQL, (str(team_id),))

    # -- sync surface (outbound path), on the caller's thread ---------------

    def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
        """Return the binding for one agent of one team, or ``None``.

        One point read through ``channel_users_team_agent_idx`` on the caller's
        thread — never a scan, never async I/O. A direct query rather than an
        in-process index: another process may have written the binding, and the
        server keeps no cross-request state, so any replica must answer from the
        table.
        """
        return self._fetch_binding(_FIND_BY_TEAM_AGENT_SQL, (str(team_id), agent_name))

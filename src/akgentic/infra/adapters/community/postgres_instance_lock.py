"""One community server per PostgreSQL database, enforced by an advisory lock.

The community tier runs every team in-process and, at boot, restores each team
the store lists as ``RUNNING`` (``LocalRuntimeCache.warm``). Two servers on one
database would both restore the same teams and then write the same event
sequence numbers. With YAML that took two servers sharing one directory; with
Postgres it takes only the same connection string, so the server now refuses
to be the second.

The lock is a session-level ``pg_try_advisory_lock`` held on a dedicated
connection for the life of the server. Postgres releases it when that
connection closes — on a clean shutdown (:meth:`release`), and equally when the
process is killed — so a crash never leaves the database locked.

The connection carries ``application_name = akgentic-community-server``, so an
operator who meets :class:`InstanceLockedError` can find the holder::

    SELECT pid, client_addr, backend_start FROM pg_stat_activity
    WHERE application_name = 'akgentic-community-server';
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg

__all__ = ["LOCK_NAME", "InstanceLockedError", "PostgresInstanceLock"]

logger = logging.getLogger(__name__)

#: The advisory-lock key, hashed in Postgres with ``hashtext``. Database-scoped:
#: two databases on one server do not contend.
LOCK_NAME = "akgentic.infra.community.server"

APPLICATION_NAME = "akgentic-community-server"


class InstanceLockedError(RuntimeError):
    """Another community server already holds this database."""


class PostgresInstanceLock:
    """Hold the community-server advisory lock on a dedicated connection.

    Args:
        conn_string: ``postgresql://`` DSN of the database to lock. Never logged.
    """

    def __init__(self, conn_string: str) -> None:
        self._conn_string = conn_string
        self._conn: psycopg.Connection[tuple[object, ...]] | None = None

    @property
    def held(self) -> bool:
        """Whether this instance currently holds the lock."""
        return self._conn is not None

    def acquire(self) -> None:
        """Take the lock, or raise without waiting if another server holds it.

        Idempotent: acquiring a lock this instance already holds is a no-op.

        Raises:
            InstanceLockedError: Another session holds the lock.
            psycopg.OperationalError: The database cannot be reached.
        """
        if self._conn is not None:
            return
        import psycopg

        conn = psycopg.connect(
            self._conn_string, autocommit=True, application_name=APPLICATION_NAME
        )
        try:
            row = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_NAME,)).fetchone()
        except BaseException:
            conn.close()
            raise
        if row is None or row[0] is not True:
            conn.close()
            raise InstanceLockedError(
                "Another akgentic community server is already using this PostgreSQL "
                "database (advisory lock "
                f"'{LOCK_NAME}'). Run one server per database: each server restores "
                "the teams it finds RUNNING at boot, so a second one would run the "
                "same teams twice. Find the holder with: SELECT pid, client_addr, "
                "backend_start FROM pg_stat_activity WHERE application_name = "
                f"'{APPLICATION_NAME}';"
            )
        self._conn = conn
        logger.info("Acquired the community-server lock on the PostgreSQL database")

    def release(self) -> None:
        """Release the lock and close its connection. Idempotent, never raises.

        Closing the connection is what releases a session-level advisory lock;
        the explicit unlock only makes the release visible in the server log
        before the connection goes.
        """
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_NAME,))
        except Exception:  # noqa: BLE001 — closing below releases it anyway
            logger.warning("Explicit unlock failed; closing the connection releases the lock")
        finally:
            conn.close()
        logger.info("Released the community-server lock on the PostgreSQL database")

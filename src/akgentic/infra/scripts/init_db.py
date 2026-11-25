"""Init-container entry point that creates the channel bindings table.

Runs :func:`akgentic.infra.adapters.shared.postgres_channel_registry.init_db`
against the Postgres instance named by ``DB_CONN_STRING_PERSISTENCE`` — the
same variable as ``akgentic.team.scripts.init_db`` and
``akgentic.catalog.scripts.init_db``, so one ``.env`` drives all three:

    command: ["python", "-m", "akgentic.infra.scripts.init_db"]

The server never creates the ``channel_users`` table; this script is the
deployment step that does. It is idempotent, so re-running it against an
initialized database is safe.

The DSN must start with ``postgresql://``. The scheme is checked here, before
Nagra sees the value, because Nagra's error for any other scheme echoes the
whole DSN — password included. The log line names the variable and the
required scheme, never the value.

Exit codes:
    0 — the table and its indexes were created or already present
    2 — ``DB_CONN_STRING_PERSISTENCE`` is unset or not a ``postgresql://`` DSN
    1 — any other failure (Nagra not installed, connection refused, SQL error)
"""

from __future__ import annotations

import logging
import os
import sys

_ENV_VAR = "DB_CONN_STRING_PERSISTENCE"
_SCHEME = "postgresql://"

logger = logging.getLogger(__name__)


def main() -> int:
    """Create the ``channel_users`` table against the configured Postgres instance.

    Returns:
        Process exit code — 0 on success, 2 on a missing or non-Postgres DSN,
        1 on any other error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    dsn = os.environ.get(_ENV_VAR)
    if not dsn:
        logger.error("%s is not set; cannot initialize the channel bindings table", _ENV_VAR)
        return 2
    if not dsn.startswith(_SCHEME):
        logger.error("%s must start with '%s'", _ENV_VAR, _SCHEME)
        return 2

    try:
        from akgentic.infra.adapters.shared.postgres_channel_registry import init_db

        init_db(dsn)
    except ImportError as exc:
        logger.error(
            "Postgres backend unavailable: %s. "
            'Install with: pip install "akgentic-infra[postgres]"',
            exc,
        )
        return 1
    except Exception:
        logger.exception("init_db failed")
        return 1

    logger.info("Channel bindings table initialized successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

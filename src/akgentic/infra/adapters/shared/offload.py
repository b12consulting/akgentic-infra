"""``Offload`` — how a database registry moves its blocking driver work off the event loop.

The database channel registries (``MongoChannelRegistry``,
``PostgresChannelRegistry``) run each async method's synchronous driver work
through one call of an ``Offload``. The default is ``asyncio.to_thread``, the
shared default executor. A tier that forbids store I/O on that shared pool
injects its own — a dedicated bounded pool with a per-call timeout and a
circuit breaker, for instance — at construction or through ``mongo_stores`` /
``postgres_stores``.

Anything callable as ``await offload(fn, *args)`` that returns ``fn(*args)``
satisfies it: ``asyncio.to_thread`` itself, or a bound method such as
``async def offload(self, fn, *args, timeout_s=...)``. Extra keyword
parameters with defaults are allowed; the registries never pass any.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol


class Offload(Protocol):
    """Run ``fn(*args)`` somewhere that is not the event loop and await its result.

    One call per async registry method. The result, and any exception ``fn``
    raises, must come back unchanged: a caller distinguishes driver errors by
    type.
    """

    def __call__[*Ts, R](self, fn: Callable[[*Ts], R], /, *args: *Ts) -> Awaitable[R]:
        """Run ``fn(*args)`` off the event loop; await to get its return value."""
        ...

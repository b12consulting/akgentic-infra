"""The one reader of a stored channel binding, shared by the database registries.

``MongoChannelRegistry`` and ``PostgresChannelRegistry`` store the same document
— a whole ``ChannelBinding`` as ``model_dump(mode="json")`` writes it — so they
read it back by the same rule: a document that validates is the binding, one
that does not (written before bindings named an agent, or malformed) reads as
absent, with one warning.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from akgentic.infra.protocols.channels import ChannelBinding

logger = logging.getLogger(__name__)


# ``Any``: a BSON document has no static shape at the driver boundary.
def read_stored_binding(doc: Mapping[str, Any] | None) -> ChannelBinding | None:
    """Validate a stored document, reading a legacy or malformed one as absent."""
    if doc is None:
        return None
    try:
        return ChannelBinding.model_validate(doc)
    except ValidationError:
        logger.warning(
            "Ignoring unreadable channel binding for channel=%s, channel_user_id=%s",
            doc.get("channel"),
            doc.get("channel_user_id"),
        )
        return None

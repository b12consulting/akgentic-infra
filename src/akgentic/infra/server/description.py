"""Team description generator — one line from a team's first message, written once.

A team's description is a generated default the user can take ownership of
(ADR-041). The generator here produces it, on the server, inside ``TeamService``:
once a delivery to a team has been handed over, the service shows the generator
the ``Process`` it already resolved for that delivery. If the record has no
description and is still generator-owned, a small model reduces that one
message to a line off the request thread, the line is written through the
store's conditional AUTO write, and a ``NotificationMessage`` tells the open
page through the service's own ``emit_message``.

Three things shape it:

* The trigger is the ``Process`` the service already read, so the common
  case — a team that has its description — costs one comparison and no I/O.
* The store's filter decides the race: a user edit that lands while the model
  is thinking wins, the AUTO write is refused, and the generator only reports
  the loss. There is no read-then-check here and no per-team state anywhere.
* The notification is constructed only after the write returned and stood, so
  its timestamp is never older than the ``updated_at`` the write set.

``create_model`` and ``ThreadPoolExecutor`` are imported by name on purpose:
tests patch them on this module to keep every model call fake and every
generation synchronous.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.infra.server.models import MAX_TEAM_DESCRIPTION_LENGTH
from akgentic.infra.server.settings import ServerSettings
from akgentic.llm import ModelConfig, create_model
from akgentic.team.models import DescriptionOrigin, Process
from akgentic.team.ports import EventStore

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message

logger = logging.getLogger(__name__)

DESCRIPTION_CONTENT_TYPE = "team_description"
"""``NotificationMessage.content_type`` carrying a freshly generated description."""

_DESCRIPTION_INSTRUCTIONS = (
    "You are given the first message a user sent to a team of AI agents. "
    "Answer with one plain-text line of about ten words naming what the team "
    "was asked to do. No quotes, no trailing period, no preamble, no explanation."
)

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"))


def normalise_description(raw: str) -> str | None:
    """Reduce a model's answer to the one line the store accepts.

    First non-blank line, stripped; one layer of matching surrounding quotes
    removed; trailing periods dropped; hard-truncated to the same cap the
    endpoint enforces. A longer answer is cut, never rejected.

    Args:
        raw: The model output, verbatim.

    Returns:
        The persisted form, or ``None`` when nothing usable is left.
    """
    line = next((candidate.strip() for candidate in raw.splitlines() if candidate.strip()), "")
    for opening, closing in _QUOTE_PAIRS:
        if len(line) >= 2 and line.startswith(opening) and line.endswith(closing):
            line = line[1:-1].strip()
            break
    line = line.rstrip(".").strip()
    line = line[:MAX_TEAM_DESCRIPTION_LENGTH].strip()
    return line or None


def _content_of(content: str | Message) -> str | None:
    """The text a delivery carries, or ``None`` when it carries none."""
    text = content if isinstance(content, str) else getattr(content, "content", None)
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    return stripped or None


class TeamDescriptionGenerator:
    """Generates a team's first description from its first message, off the request path.

    The model is built lazily, the way ``SummarizingCompaction`` builds its
    summarizer, so constructing the generator needs no provider environment.
    :meth:`maybe_describe` is the request-path half — one comparison, one
    executor submit, never raising. :meth:`_generate` is the background unit —
    the model call, the conditional write and the notification — and never
    raises either.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        event_store: EventStore,
        emit: Callable[[uuid.UUID, Message], None],
        *,
        executor: Executor | None = None,
    ) -> None:
        """Hold the model configuration, the store and the emit path; build nothing yet.

        Args:
            model_cfg: The small model that writes the line.
            event_store: Where the conditional AUTO write goes — the server's
                own store, the same one the description endpoint writes.
            emit: How a notification reaches a team's event stream; the
                service's ``emit_message``.
            executor: Where generation runs. Defaults to one dedicated thread.
        """
        self._model_cfg = model_cfg
        self._event_store = event_store
        self._emit = emit
        self._executor: Executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="team-description"
        )
        self._agent: Agent[None, str] | None = None

    @classmethod
    def from_settings(
        cls,
        settings: ServerSettings,
        event_store: EventStore,
        emit: Callable[[uuid.UUID, Message], None],
    ) -> TeamDescriptionGenerator | None:
        """Build the generator from server settings, or ``None`` when it is off.

        The only reader of the two description settings and the only place the
        on/off decision is logged. An unknown provider fails here, at wiring,
        through ``ModelConfig``'s own validation rather than at the first message.

        Args:
            settings: The tier's server settings carrying the provider and model ids.
            event_store: The store the generator writes.
            emit: The service's notification path.

        Returns:
            A generator when both settings are set, else ``None``.

        Raises:
            pydantic.ValidationError: If the provider is not one ``ModelConfig`` knows.
        """
        if settings.description_provider is None or settings.description_model is None:
            logger.info(
                "Team description generator disabled: set "
                "AKGENTIC_DESCRIPTION_PROVIDER and AKGENTIC_DESCRIPTION_MODEL to enable it"
            )
            return None
        model_cfg = ModelConfig.model_validate(
            {"provider": settings.description_provider, "model": settings.description_model}
        )
        logger.info(
            "Team description generator enabled: provider=%s model=%s",
            model_cfg.provider,
            model_cfg.model,
        )
        return cls(model_cfg, event_store, emit)

    def maybe_describe(self, process: Process, content: str | Message) -> bool:
        """Schedule one generation attempt if this delivery should describe the team.

        Called by the service after the handle accepted the delivery, with the
        ``Process`` it resolved for it. Fires only while the record says no
        description and generator-owned, and only for a delivery that carries
        text. Never raises into the send path.

        Args:
            process: The record the delivery was resolved against.
            content: What was delivered; a ``Message`` contributes its ``content``.

        Returns:
            ``True`` when an attempt was scheduled.
        """
        if (
            process.team_description is not None
            or process.description_origin is not DescriptionOrigin.AUTO
        ):
            return False
        text = _content_of(content)
        if text is None:
            return False
        try:
            self._executor.submit(self._generate, process.team_id, text)
        except Exception:
            logger.warning(
                "Team description scheduling failed: team_id=%s", process.team_id, exc_info=True
            )
            return False
        return True

    def close(self) -> None:
        """Shut the executor down without waiting on an in-flight provider call."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --- Background unit ---

    def _build_agent(self) -> Agent[None, str]:
        """Build and cache the one-shot agent on first use."""
        if self._agent is None:
            self._agent = Agent(
                model=create_model(self._model_cfg),
                instructions=_DESCRIPTION_INSTRUCTIONS,
                output_type=str,
            )
        return self._agent

    def _generate(self, team_id: uuid.UUID, content: str) -> None:
        """The whole attempt: generate, write, notify. Never raises."""
        try:
            text = normalise_description(self._build_agent().run_sync(content).output)
            if text is None:
                logger.warning(
                    "Team description generation produced no usable text: team_id=%s", team_id
                )
                return
            self._commit(team_id, text)
        except Exception:
            logger.warning("Team description generation failed: team_id=%s", team_id, exc_info=True)

    def _commit(self, team_id: uuid.UUID, text: str) -> None:
        """Write through the conditional AUTO path and notify only if the write stood.

        The notification is constructed after the write returns, so its
        timestamp is never older than the ``updated_at`` the write set — the
        frontend drops a description frame older than the team's last write.
        """
        updated = self._event_store.update_team_description(team_id, text, DescriptionOrigin.AUTO)
        if updated is None:
            logger.debug("Team description write found no team: team_id=%s", team_id)
            return
        if (
            updated.description_origin is DescriptionOrigin.AUTO
            and updated.team_description == text
        ):
            notification = NotificationMessage(content_type=DESCRIPTION_CONTENT_TYPE, content=text)
            self._emit(team_id, notification)
            logger.info("Team description generated: team_id=%s length=%d", team_id, len(text))
            return
        logger.debug("Team description generation lost to a user edit: team_id=%s", team_id)

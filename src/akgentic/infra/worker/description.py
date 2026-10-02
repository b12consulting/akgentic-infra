"""Team description generator — one line from a team's first message, written once.

A team's description is a generated default the user can take ownership of
(ADR-041). The generator here produces it: on the first delivery to a team
whose ``Process`` has no description and is still generator-owned, a small
model reduces that one message to a line, the line is written through the
store's conditional AUTO write, and a ``NotificationMessage`` tells the open
page. A user edit that lands while the model is thinking wins, because the
store's filter refuses the AUTO write and the worker only reports the loss.

Two pieces, both installed by ``LocalRuntimeCache.store()``:

* :class:`TeamDescriptionGenerator` holds the model, the ``TeamManager`` and a
  single-thread executor; everything it does runs off the request thread.
* :class:`DescribingTeamHandle` wraps a live ``TeamHandle``, hands every call
  through unchanged, and after ``send`` / ``send_to`` asks the generator for
  one attempt — latched once the description is seen to be settled, capped per
  handle so a broken provider cannot cost a model call per message forever.

``create_model`` and ``ThreadPoolExecutor`` are imported by name on purpose:
tests patch them on this module to keep every model call fake and every
generation synchronous.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from akgentic.core.messages.orchestrator import NotificationMessage
from akgentic.infra.protocols.team_handle import TeamHandle
from akgentic.infra.server.models import MAX_TEAM_DESCRIPTION_LENGTH
from akgentic.infra.worker.settings import WorkerSettings
from akgentic.llm import ModelConfig, create_model
from akgentic.team.manager import TeamManager
from akgentic.team.models import DescriptionOrigin

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message
    from akgentic.core.orchestrator import EventSubscriber

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
"""Generation attempts per handle lifetime before the generator gives up on a team."""

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
    One background unit — :meth:`_generate` — does the store read, the model
    call, the conditional write and the notification, and never raises.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        team_manager: TeamManager,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        executor: Executor | None = None,
    ) -> None:
        """Hold the model configuration and the team side; build nothing yet.

        Args:
            model_cfg: The small model that writes the line.
            team_manager: Where the ``Process`` is read and the AUTO write goes.
            max_attempts: Generation attempts per handle lifetime.
            executor: Where generation runs. Defaults to one dedicated thread.
        """
        self._model_cfg = model_cfg
        self._team_manager = team_manager
        self._max_attempts = max_attempts
        self._executor: Executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="team-description"
        )
        self._agent: Agent[None, str] | None = None

    @property
    def max_attempts(self) -> int:
        """Generation attempts a wrapping handle may make before it stops asking."""
        return self._max_attempts

    @classmethod
    def from_settings(
        cls, settings: WorkerSettings, team_manager: TeamManager
    ) -> TeamDescriptionGenerator | None:
        """Build the generator from worker settings, or ``None`` when it is off.

        The only reader of the two description settings and the only place the
        on/off decision is logged. An unknown provider fails here, at wiring,
        through ``ModelConfig``'s own validation rather than at the first message.

        Args:
            settings: The worker settings carrying the provider and model ids.
            team_manager: The team side the generator writes through.

        Returns:
            A generator when both settings are set, else ``None``.

        Raises:
            pydantic.ValidationError: If the provider is not one ``ModelConfig`` knows.
        """
        if settings.description_provider is None or settings.description_model is None:
            logger.info(
                "Team description generator disabled: set "
                "AKGENTIC_WORKER_DESCRIPTION_PROVIDER and AKGENTIC_WORKER_DESCRIPTION_MODEL "
                "to enable it"
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
        return cls(model_cfg, team_manager)

    def wrap(self, handle: TeamHandle) -> TeamHandle:
        """Return a describing handle around ``handle``; an already-wrapped one as is."""
        if isinstance(handle, DescribingTeamHandle):
            return handle
        return DescribingTeamHandle(handle, self)

    def schedule(
        self, team_id: uuid.UUID, content: str, emit: Callable[[Message], None]
    ) -> Future[bool]:
        """Run one generation attempt on the executor.

        Args:
            team_id: The team to describe.
            content: The triggering message's text, and nothing else.
            emit: How the notification reaches the team's event stream.

        Returns:
            A future resolving ``True`` when the handle should stop asking
            (described, user-owned, or gone) and ``False`` on a counted failure.
        """
        return self._executor.submit(self._generate, team_id, content, emit)

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

    def _generate(self, team_id: uuid.UUID, content: str, emit: Callable[[Message], None]) -> bool:
        """The whole attempt: read, generate, write, notify. Never raises."""
        try:
            process = self._team_manager.get_team(team_id)
            if process is None:
                logger.debug("Team description skipped, team not found: team_id=%s", team_id)
                return True
            if (
                process.team_description is not None
                or process.description_origin is DescriptionOrigin.USER
            ):
                logger.debug("Team description already settled: team_id=%s", team_id)
                return True
            text = normalise_description(self._build_agent().run_sync(content).output)
            if text is None:
                logger.warning(
                    "Team description generation produced no usable text: team_id=%s", team_id
                )
                return False
            self._commit(team_id, text, emit)
            return True
        except Exception:
            logger.warning("Team description generation failed: team_id=%s", team_id, exc_info=True)
            return False

    def _commit(self, team_id: uuid.UUID, text: str, emit: Callable[[Message], None]) -> None:
        """Write through the conditional AUTO path and notify only if the write stood.

        The notification is constructed after the write returns, so its
        timestamp is never older than the ``updated_at`` the write set — the
        frontend drops a description frame older than the team's last write.
        """
        updated = self._team_manager.update_description(team_id, text, DescriptionOrigin.AUTO)
        if updated is None:
            logger.debug("Team description write found no team: team_id=%s", team_id)
            return
        if (
            updated.description_origin is DescriptionOrigin.AUTO
            and updated.team_description == text
        ):
            emit(NotificationMessage(content_type=DESCRIPTION_CONTENT_TYPE, content=text))
            logger.info("Team description generated: team_id=%s length=%d", team_id, len(text))
            return
        logger.debug("Team description generation lost to a user edit: team_id=%s", team_id)


class DescribingTeamHandle(TeamHandle):
    """A ``TeamHandle`` that asks the generator for a description after a delivery.

    Every method forwards to the inner handle unchanged. ``send`` and
    ``send_to`` additionally schedule one generation attempt once the message
    has been handed to the team; the other deliveries never trigger. The three
    flags below are the handle's whole state: attempts made, one in flight, and
    settled — after which nothing is scheduled again for this handle's lifetime.
    """

    def __init__(self, inner: TeamHandle, generator: TeamDescriptionGenerator) -> None:
        self._inner = inner
        self._generator = generator
        self._attempts = 0
        self._in_flight = False
        self._settled = False
        self._lock = threading.Lock()

    @property
    def team_id(self) -> uuid.UUID:
        """The unique identifier of the team this handle points to."""
        return self._inner.team_id

    def send(self, content: str | Message) -> None:
        """Deliver to the team's entry point, then consider describing the team."""
        self._inner.send(content)
        self._maybe_describe(content)

    def send_to(self, agent_name: str, content: str | Message) -> None:
        """Deliver to one agent, then consider describing the team."""
        self._inner.send_to(agent_name, content)
        self._maybe_describe(content)

    def send_from_to(self, sender_name: str, recipient_name: str, content: str | Message) -> None:
        """Agent-to-agent delivery; never triggers generation."""
        self._inner.send_from_to(sender_name, recipient_name, content)

    def emitMessage(self, message: Message) -> None:  # noqa: N802
        """Publish a pre-formed message; never triggers generation."""
        self._inner.emitMessage(message)

    def process_human_input(self, content: str, message: Message) -> None:
        """Route a human reply; never triggers generation."""
        self._inner.process_human_input(content, message)

    def subscribe(self, subscriber: EventSubscriber) -> None:
        """Register an event subscriber with the team's orchestrator."""
        self._inner.subscribe(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Remove an event subscriber from the team's orchestrator."""
        self._inner.unsubscribe(subscriber)

    # --- Trigger ---

    def _maybe_describe(self, content: str | Message) -> None:
        """Schedule one attempt if this delivery carries text and the handle allows it."""
        text = _content_of(content)
        if text is None or not self._claim_attempt():
            return
        try:
            future = self._generator.schedule(self.team_id, text, self._inner.emitMessage)
        except Exception:
            logger.warning(
                "Team description scheduling failed: team_id=%s", self.team_id, exc_info=True
            )
            with self._lock:
                self._in_flight = False
            return
        future.add_done_callback(self._on_done)

    def _claim_attempt(self) -> bool:
        """Take the next attempt slot, or report why none is available."""
        with self._lock:
            if self._settled or self._in_flight:
                return False
            if self._attempts >= self._generator.max_attempts:
                logger.debug("Team description attempt cap reached: team_id=%s", self.team_id)
                return False
            self._attempts += 1
            self._in_flight = True
            return True

    def _on_done(self, future: Future[bool]) -> None:
        """Release the in-flight slot; latch when the attempt settled the team."""
        try:
            settled = future.result() is True
        except Exception:
            # ``_generate`` never raises; a cancelled future on shutdown does.
            settled = False
        with self._lock:
            self._in_flight = False
            if settled:
                self._settled = True

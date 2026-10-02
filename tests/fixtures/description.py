"""Test doubles for the team description generator, shared by its unit and door specs.

``RecordingModel`` keeps every model call fake and observable; ``InlineExecutor``
makes a generation complete before ``send`` returns, so a spec needs no polling.
Both are installed by patching the names the generator imports on
``akgentic.infra.server.description``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

DEFAULT_REPLY = "Triage inbound acme support cases."
"""What ``RecordingModel`` answers unless told otherwise; normalises by losing its period."""


class InlineExecutor(Executor):
    """Runs the submitted callable synchronously and answers a settled future.

    Refuses work after ``shutdown`` exactly as ``ThreadPoolExecutor`` does, so
    a scheduling-failure spec exercises the real refusal rather than a no-op.
    Constructed with the same keywords the generator passes its real pool, so
    the class can stand in for ``ThreadPoolExecutor`` on the module.
    """

    def __init__(self, max_workers: int | None = None, thread_name_prefix: str = "") -> None:
        self._closed = False

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        if self._closed:
            msg = "cannot schedule new futures after shutdown"
            raise RuntimeError(msg)
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 — mirror the executor contract
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._closed = True


def prompt_text(messages: list[ModelMessage]) -> str:
    """Everything the model was asked, joined — the whole prompt, not a part of it."""
    chunks: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    chunks.append(part.content)
    return "\n".join(chunks)


class RecordingModel:
    """A ``FunctionModel`` factory that records each call's user prompt.

    ``error`` makes every call raise; ``gate`` makes every call block until the
    test releases it, which is how an asynchrony spec holds a generation
    mid-flight.
    """

    def __init__(
        self,
        reply: str = DEFAULT_REPLY,
        *,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.prompts: list[str] = []
        self._reply = reply
        self._error = error
        self._gate = gate

    def build(self) -> FunctionModel:
        """The model to hand back from a patched ``create_model``."""

        def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            self.prompts.append(prompt_text(messages))
            if self._gate is not None:
                self._gate.wait()
            if self._error is not None:
                raise self._error
            return ModelResponse(parts=[TextPart(self._reply)])

        return FunctionModel(respond)

    @property
    def calls(self) -> int:
        """How many times the model was asked."""
        return len(self.prompts)

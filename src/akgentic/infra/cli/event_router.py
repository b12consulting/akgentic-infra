"""Event routing and dispatch for chat events."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from akgentic.infra.cli.renderers import RichRenderer

if TYPE_CHECKING:
    from textual.widget import Widget

    from akgentic.infra.cli.tui.colors import AgentColorRegistry

# Event types to display vs skip
_DISPLAY_EVENTS = {"SentMessage", "ErrorMessage", "EventMessage"}


class EventRouter:
    """Parse raw event dicts and dispatch to the renderer."""

    def __init__(
        self,
        renderer: RichRenderer,
        on_human_input: Callable[[str, str], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._renderer = renderer
        self._on_human_input = on_human_input
        self._log = logger or logging.getLogger(__name__)

    def route(self, data: dict[str, Any]) -> bool:
        """Parse, classify, and render an event. Returns True if rendered.

        Malformed events are logged at DEBUG and skipped -- never raised.
        """
        try:
            return self._route_inner(data)
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            self._log.debug("Malformed event skipped: %s", exc)
            return False

    def _route_inner(self, data: dict[str, Any]) -> bool:
        """Core dispatch logic -- classify by __model__ suffix and delegate."""
        event = data.get("event", data)
        if isinstance(event, str):
            event = json.loads(event)

        model = event.get("__model__", "")
        short_model = model.rsplit(".", 1)[-1] if model else ""
        if short_model not in _DISPLAY_EVENTS:
            return False

        if short_model == "ErrorMessage":
            return self._handle_error_message(event)

        if short_model == "EventMessage":
            return self._handle_event_message(event, data)

        # SentMessage
        return self._handle_sent_message(event)

    def _handle_error_message(self, event: dict[str, Any]) -> bool:
        """Render an ErrorMessage event."""
        content = event.get("exception_value", event.get("content", event.get("error", "")))
        self._renderer.render_error(str(content))
        return True

    def _handle_sent_message(self, event: dict[str, Any]) -> bool:
        """Render a SentMessage -- agent response with sender and content."""
        raw_sender = event.get("sender", "agent")
        sender = (
            raw_sender.get("name", "agent") if isinstance(raw_sender, dict) else str(raw_sender)
        )
        message = event.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if content:
            self._renderer.render_agent_message(sender, str(content))
            return True
        return False

    def _handle_event_message(self, event: dict[str, Any], outer_data: dict[str, Any]) -> bool:
        """Handle EventMessage -- inspect nested event for tool calls / human input."""
        nested = event.get("event", {})
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except (json.JSONDecodeError, TypeError):
                return False

        if not isinstance(nested, dict):
            return False

        nested_model = nested.get("__model__", "")

        if "tool_name" in nested:
            self._handle_tool_call(nested)
            return True

        if "HumanInput" in nested_model or "RequestInput" in nested_model:
            prompt_text = str(nested.get("prompt", nested.get("content", "Input requested")))
            self._renderer.render_human_input_request(prompt_text)
            if self._on_human_input is not None:
                self._notify_human_input(outer_data)
            return True

        return False

    def _handle_tool_call(self, nested: dict[str, Any]) -> None:
        """Extract and render a tool call event."""
        tool_name = str(nested["tool_name"])
        raw_input = nested.get("arguments", nested.get("args", nested.get("input", "")))
        if isinstance(raw_input, dict | list):
            tool_input = json.dumps(raw_input, indent=2)
        else:
            tool_input = str(raw_input)
        raw_output = nested.get("result")
        if raw_output is None:
            tool_output = None
        elif isinstance(raw_output, dict | list):
            tool_output = json.dumps(raw_output, indent=2)
        else:
            tool_output = str(raw_output)
        self._renderer.render_tool_call(tool_name, tool_input, tool_output)

    def _notify_human_input(self, outer_data: dict[str, Any]) -> None:
        """Extract message_id and agent_name from outer event data and invoke callback."""
        raw_id = outer_data.get("id")
        if not raw_id:
            return
        message_id = str(raw_id)
        raw_sender = outer_data.get("sender", "Agent")
        if isinstance(raw_sender, dict):
            agent_name = str(raw_sender.get("name", "Agent"))
        else:
            agent_name = str(raw_sender)
        if self._on_human_input is not None:
            self._on_human_input(message_id, agent_name)

    def to_widget(
        self,
        data: dict[str, Any],
        color_registry: AgentColorRegistry,
    ) -> Widget | None:
        """Parse a raw event dict and return the appropriate Textual widget.

        Returns ``None`` for unrecognized or malformed events.
        This method does NOT invoke callbacks or use the renderer -- it is
        a pure event-to-widget translation for TUI usage.
        """
        try:
            event = data.get("event", data)
            if isinstance(event, str):
                event = json.loads(event)

            model = event.get("__model__", "")
            short_model = model.rsplit(".", 1)[-1] if model else ""
            if short_model not in _DISPLAY_EVENTS:
                return None

            if short_model == "ErrorMessage":
                return self._error_to_widget(event)
            if short_model == "SentMessage":
                return self._sent_to_widget(event, color_registry)
            if short_model == "EventMessage":
                return self._event_to_widget(event)
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            self._log.debug("to_widget: malformed event skipped: %s", exc)
        return None

    def _error_to_widget(self, event: dict[str, Any]) -> Widget:
        """Convert an ErrorMessage event to an ErrorWidget."""
        from akgentic.infra.cli.tui.widgets.error import ErrorWidget

        content = event.get("exception_value", event.get("content", event.get("error", "")))
        return ErrorWidget(content=str(content))

    def _sent_to_widget(
        self, event: dict[str, Any], color_registry: AgentColorRegistry
    ) -> Widget | None:
        """Convert a SentMessage event to an AgentMessage widget.

        Skips messages from @Human — those are echoes of the user's own
        message, already displayed as a UserMessage widget.
        """
        from akgentic.infra.cli.tui.widgets.agent_message import AgentMessage

        raw_sender = event.get("sender", "agent")
        sender = (
            raw_sender.get("name", "agent") if isinstance(raw_sender, dict) else str(raw_sender)
        )
        if sender == "@Human":
            return None
        message = event.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not content:
            return None
        color = color_registry.get(sender)
        return AgentMessage(sender=sender, content=str(content), color=color)

    def _event_to_widget(self, event: dict[str, Any]) -> Widget | None:
        """Convert an EventMessage event to a ToolCallWidget or HumanInputPrompt."""
        nested = event.get("event", {})
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(nested, dict):
            return None

        if "tool_name" in nested:
            return self._tool_call_to_widget(nested)

        nested_model = nested.get("__model__", "")
        if "HumanInput" in nested_model or "RequestInput" in nested_model:
            from akgentic.infra.cli.tui.widgets.human_input import HumanInputPrompt

            prompt_text = str(nested.get("prompt", nested.get("content", "Input requested")))
            return HumanInputPrompt(prompt_text=prompt_text)
        return None

    def _tool_call_to_widget(self, nested: dict[str, Any]) -> Widget:
        """Convert a tool call nested event to a ToolCallWidget."""
        from akgentic.infra.cli.tui.widgets.tool_call import ToolCallWidget

        tool_name = str(nested["tool_name"])
        raw_input = nested.get("arguments", nested.get("args", nested.get("input", "")))
        if isinstance(raw_input, dict | list):
            tool_input = json.dumps(raw_input, indent=2)
        else:
            tool_input = str(raw_input)
        raw_output = nested.get("result")
        if raw_output is None:
            tool_output = None
        elif isinstance(raw_output, dict | list):
            tool_output = json.dumps(raw_output, indent=2)
        else:
            tool_output = str(raw_output)
        return ToolCallWidget(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)

"""Tests for EventRouter -- event routing and dispatch."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from akgentic.infra.cli.event_router import EventRouter
from akgentic.infra.cli.tui.colors import AgentColorRegistry
from akgentic.infra.cli.tui.widgets.agent_message import AgentMessage
from akgentic.infra.cli.tui.widgets.error import ErrorWidget
from akgentic.infra.cli.tui.widgets.human_input import HumanInputPrompt
from akgentic.infra.cli.tui.widgets.tool_call import ToolCallWidget
from tests.fixtures.events import (
    make_error_message,
    make_event_message,
    make_sent_message,
    make_start_message,
    make_tool_call_event,
)

from .conftest import captured_renderer as _captured_renderer


def _make_router(
    renderer: Any = None,
    on_human_input: Any = None,
) -> tuple[EventRouter, Any]:
    """Build an EventRouter with a captured renderer."""
    if renderer is None:
        renderer, buf = _captured_renderer()
    else:
        buf = None
    router = EventRouter(renderer, on_human_input=on_human_input)
    return router, buf


class TestRouteSentMessage:
    def test_valid_sent_message_renders(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route({"event": make_sent_message(content="hello")})
        assert result is True
        out = buf.getvalue()
        assert "sender" in out
        assert "hello" in out

    def test_sent_message_empty_content_returns_false(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route({"event": make_sent_message(content="")})
        assert result is False

    def test_sent_message_dict_sender(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route({"event": make_sent_message(content="from dict sender")})
        assert result is True
        out = buf.getvalue()
        assert "sender" in out
        assert "from dict sender" in out


class TestRouteErrorMessage:
    def test_valid_error_message_renders(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route({"event": make_error_message(exception_value="something broke")})
        assert result is True
        out = buf.getvalue()
        assert "error" in out
        assert "something broke" in out


class TestRouteToolCall:
    def test_tool_call_renders(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route(
            {"event": make_event_message(event=make_tool_call_event(tool_name="search"))}
        )
        assert result is True
        out = buf.getvalue()
        assert "search" in out


class TestRouteHumanInput:
    def test_human_input_invokes_callback(self) -> None:
        renderer, buf = _captured_renderer()
        callback = MagicMock()
        router = EventRouter(renderer, on_human_input=callback)
        data = {
            "id": "msg-123",
            "sender": {"name": "Agent"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "Enter your name",
                },
            },
        }
        result = router.route(data)
        assert result is True
        out = buf.getvalue()
        assert "Human Input Required" in out
        assert "Enter your name" in out
        callback.assert_called_once_with("msg-123", "Agent")

    def test_human_input_renders_without_callback(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer, on_human_input=None)
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "Please provide input",
                },
            },
        }
        result = router.route(data)
        assert result is True
        out = buf.getvalue()
        assert "Human Input Required" in out


class TestRouteUnknownModel:
    def test_unknown_model_returns_false(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route({"event": make_start_message()})
        assert result is False


class TestRouteMalformedEvents:
    def test_missing_model_returns_false(self) -> None:
        renderer, buf = _captured_renderer()
        logger = logging.getLogger("test.event_router")
        router = EventRouter(renderer, logger=logger)
        result = router.route({"event": {"data": "no model"}})
        assert result is False

    def test_empty_dict_returns_false(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        result = router.route({})
        assert result is False


class TestRouteJsonStringEvent:
    def test_json_string_event_payload(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        import json

        event_payload = json.dumps(
            {
                "__model__": "some.module.SentMessage",
                "sender": "Agent",
                "message": {"content": "hi"},
            }
        )
        result = router.route({"event": event_payload})
        assert result is True
        out = buf.getvalue()
        assert "hi" in out

    def test_invalid_json_string_returns_false(self) -> None:
        renderer, buf = _captured_renderer()
        logger = logging.getLogger("test.event_router")
        router = EventRouter(renderer, logger=logger)
        result = router.route({"event": "not valid json {{"})
        assert result is False


class TestNotifyHumanInput:
    def test_extracts_id_and_sender(self) -> None:
        renderer, buf = _captured_renderer()
        callback = MagicMock()
        router = EventRouter(renderer, on_human_input=callback)
        data = {
            "id": "msg-456",
            "sender": {"name": "BotX"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "question?",
                },
            },
        }
        router.route(data)
        callback.assert_called_once_with("msg-456", "BotX")

    def test_missing_id_callback_not_invoked(self) -> None:
        renderer, buf = _captured_renderer()
        callback = MagicMock()
        router = EventRouter(renderer, on_human_input=callback)
        data = {
            "sender": {"name": "BotX"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "question?",
                },
            },
        }
        router.route(data)
        callback.assert_not_called()

    def test_none_id_callback_not_invoked(self) -> None:
        renderer, buf = _captured_renderer()
        callback = MagicMock()
        router = EventRouter(renderer, on_human_input=callback)
        data = {
            "id": None,
            "sender": {"name": "BotX"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "question?",
                },
            },
        }
        router.route(data)
        callback.assert_not_called()

    def test_string_sender_extracted(self) -> None:
        renderer, buf = _captured_renderer()
        callback = MagicMock()
        router = EventRouter(renderer, on_human_input=callback)
        data = {
            "id": "msg-789",
            "sender": "SimpleAgent",
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "q?",
                },
            },
        }
        router.route(data)
        callback.assert_called_once_with("msg-789", "SimpleAgent")


class TestToolCallArgFormats:
    def test_dict_arguments_rendered_as_json(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "tool_name": "search",
                    "arguments": {"query": "test", "limit": 10},
                    "result": {"items": ["a", "b"]},
                },
            },
        }
        result = router.route(data)
        assert result is True
        out = buf.getvalue()
        assert "search" in out

    def test_list_arguments_rendered_as_json(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "tool_name": "multi_search",
                    "arguments": ["query1", "query2"],
                    "result": None,
                },
            },
        }
        result = router.route(data)
        assert result is True
        out = buf.getvalue()
        assert "multi_search" in out


class TestNestedEventJsonString:
    def test_nested_event_json_string_parsed(self) -> None:
        """EventMessage with a JSON-string nested event should be parsed."""
        import json

        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        nested = json.dumps({"tool_name": "calc", "arguments": "2+2", "result": "4"})
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": nested,
            },
        }
        result = router.route(data)
        assert result is True
        out = buf.getvalue()
        assert "calc" in out

    def test_nested_event_invalid_json_string_returns_false(self) -> None:
        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": "not valid json {{",
            },
        }
        result = router.route(data)
        assert result is False

    def test_nested_event_non_dict_returns_false(self) -> None:
        """If nested event parses to a non-dict (e.g., list), return False."""
        import json

        renderer, buf = _captured_renderer()
        router = EventRouter(renderer)
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": json.dumps([1, 2, 3]),
            },
        }
        result = router.route(data)
        assert result is False


# ---------------------------------------------------------------------------
# to_widget() tests
# ---------------------------------------------------------------------------


def _make_widget_router() -> tuple[EventRouter, AgentColorRegistry]:
    """Build an EventRouter and AgentColorRegistry for to_widget tests."""
    renderer, _ = _captured_renderer()
    router = EventRouter(renderer)
    registry = AgentColorRegistry()
    return router, registry


class TestToWidgetSentMessage:
    def test_returns_agent_message(self) -> None:
        router, registry = _make_widget_router()
        data = {"event": make_sent_message(content="hello")}
        widget = router.to_widget(data, registry)
        assert isinstance(widget, AgentMessage)

    def test_empty_content_returns_none(self) -> None:
        router, registry = _make_widget_router()
        data = {"event": make_sent_message(content="")}
        widget = router.to_widget(data, registry)
        assert widget is None

    def test_uses_color_registry(self) -> None:
        router, registry = _make_widget_router()
        data = {"event": make_sent_message(content="hi")}
        router.to_widget(data, registry)
        # sender from make_sent_message defaults to "sender"
        assert "sender" in registry._map


class TestToWidgetErrorMessage:
    def test_returns_error_widget(self) -> None:
        router, registry = _make_widget_router()
        data = {"event": make_error_message(exception_value="boom")}
        widget = router.to_widget(data, registry)
        assert isinstance(widget, ErrorWidget)


class TestToWidgetToolCall:
    def test_returns_tool_call_widget(self) -> None:
        router, registry = _make_widget_router()
        data = {"event": make_event_message(event=make_tool_call_event(tool_name="search"))}
        widget = router.to_widget(data, registry)
        assert isinstance(widget, ToolCallWidget)

    def test_tool_call_with_dict_args(self) -> None:
        router, registry = _make_widget_router()
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "tool_name": "calc",
                    "arguments": {"x": 1},
                    "result": {"answer": 42},
                },
            },
        }
        widget = router.to_widget(data, registry)
        assert isinstance(widget, ToolCallWidget)


class TestToWidgetHumanInput:
    def test_returns_human_input_prompt(self) -> None:
        router, registry = _make_widget_router()
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "Enter name",
                },
            },
        }
        widget = router.to_widget(data, registry)
        assert isinstance(widget, HumanInputPrompt)

    def test_request_input_model(self) -> None:
        router, registry = _make_widget_router()
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "RequestInput",
                    "prompt": "provide data",
                },
            },
        }
        widget = router.to_widget(data, registry)
        assert isinstance(widget, HumanInputPrompt)


class TestToWidgetUnknown:
    def test_unknown_model_returns_none(self) -> None:
        router, registry = _make_widget_router()
        data = {"event": make_start_message()}
        widget = router.to_widget(data, registry)
        assert widget is None


class TestToWidgetJsonStringEvent:
    def test_json_string_outer_event(self) -> None:
        """to_widget handles a JSON-string 'event' field (same as route)."""
        import json

        router, registry = _make_widget_router()
        event_payload = json.dumps(
            {
                "__model__": "some.module.SentMessage",
                "sender": "Agent",
                "message": {"content": "via json string"},
            }
        )
        widget = router.to_widget({"event": event_payload}, registry)
        assert isinstance(widget, AgentMessage)

    def test_nested_event_json_string_returns_tool_widget(self) -> None:
        """to_widget handles a JSON-string nested event inside EventMessage."""
        import json

        router, registry = _make_widget_router()
        nested = json.dumps({"tool_name": "calc", "arguments": "2+2", "result": "4"})
        data = {
            "event": {
                "__model__": "EventMessage",
                "event": nested,
            },
        }
        widget = router.to_widget(data, registry)
        assert isinstance(widget, ToolCallWidget)


class TestToWidgetMalformed:
    def test_empty_dict_returns_none(self) -> None:
        router, registry = _make_widget_router()
        widget = router.to_widget({}, registry)
        assert widget is None

    def test_missing_model_returns_none(self) -> None:
        router, registry = _make_widget_router()
        widget = router.to_widget({"event": {"data": "no model"}}, registry)
        assert widget is None

    def test_invalid_json_event_returns_none(self) -> None:
        router, registry = _make_widget_router()
        widget = router.to_widget({"event": "not json {{"}, registry)
        assert widget is None

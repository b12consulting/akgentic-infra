"""Tests for LocalTeamHandle — community-tier TeamHandle adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akgentic.infra.adapters.local_team_handle import LocalTeamHandle
from akgentic.infra.protocols.team_handle import TeamHandle


class TestLocalTeamHandleProtocolConformance:
    """AC1: LocalTeamHandle satisfies the TeamHandle protocol."""

    def test_isinstance_check(self) -> None:
        """isinstance(LocalTeamHandle(...), TeamHandle) returns True."""
        runtime = MagicMock()
        handle = LocalTeamHandle(runtime)
        assert isinstance(handle, TeamHandle)


class TestLocalTeamHandleSend:
    """AC1: send() delegates to TeamRuntime.send()."""

    def test_send_delegates_to_runtime(self) -> None:
        """send(content) calls self._runtime.send(content)."""
        runtime = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.send("hello")
        runtime.send.assert_called_once_with("hello")

    def test_send_passes_content_exactly(self) -> None:
        """Content string is forwarded unchanged."""
        runtime = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.send("multi\nline\ncontent")
        runtime.send.assert_called_once_with("multi\nline\ncontent")


class TestLocalTeamHandleSendTo:
    """AC1: send_to() delegates to TeamRuntime.send_to()."""

    def test_send_to_delegates_to_runtime(self) -> None:
        """send_to(agent_name, content) calls runtime.send_to(agent_name, content)."""
        runtime = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.send_to("analyst", "do analysis")
        runtime.send_to.assert_called_once_with("analyst", "do analysis")


class TestLocalTeamHandleProcessHumanInput:
    """AC1: process_human_input() encapsulates HumanProxy lookup + delegation."""

    def test_process_human_input_delegates(self) -> None:
        """Finds HumanProxy in agent_cards, resolves addr, calls proxy."""
        from akgentic.agent import HumanProxy

        runtime = MagicMock()
        mock_card = MagicMock()
        mock_card.get_agent_class.return_value = HumanProxy
        runtime.team.agent_cards = {"human": mock_card}

        mock_addr = MagicMock()
        runtime.addrs = {"human": mock_addr}

        mock_proxy = MagicMock()
        runtime.actor_system.proxy_ask.return_value = mock_proxy

        message = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.process_human_input("user input", message)

        runtime.actor_system.proxy_ask.assert_called_once_with(mock_addr, HumanProxy)
        mock_proxy.process_human_input.assert_called_once_with("user input", message)

    def test_process_human_input_raises_if_no_human_proxy(self) -> None:
        """Raises ValueError when no HumanProxy agent exists in team."""
        runtime = MagicMock()
        mock_card = MagicMock()
        mock_card.get_agent_class.return_value = type("OtherAgent", (), {})
        runtime.team.agent_cards = {"other": mock_card}

        handle = LocalTeamHandle(runtime)
        message = MagicMock()
        with pytest.raises(ValueError, match="No HumanProxy found in team"):
            handle.process_human_input("input", message)

    def test_process_human_input_raises_if_addr_missing(self) -> None:
        """Raises ValueError when HumanProxy exists but has no resolved address."""
        from akgentic.agent import HumanProxy

        runtime = MagicMock()
        mock_card = MagicMock()
        mock_card.get_agent_class.return_value = HumanProxy
        runtime.team.agent_cards = {"human": mock_card}
        runtime.addrs = {}  # no addr for "human"

        handle = LocalTeamHandle(runtime)
        message = MagicMock()
        with pytest.raises(ValueError, match="HumanProxy 'human' found but has no resolved address"):
            handle.process_human_input("input", message)

    def test_process_human_input_raises_for_empty_agent_cards(self) -> None:
        """Raises ValueError when agent_cards is empty."""
        runtime = MagicMock()
        runtime.team.agent_cards = {}

        handle = LocalTeamHandle(runtime)
        message = MagicMock()
        with pytest.raises(ValueError, match="No HumanProxy found in team"):
            handle.process_human_input("input", message)

    def test_process_human_input_finds_human_proxy_among_multiple_cards(self) -> None:
        """Finds HumanProxy when it is not the first card in the dict."""
        from akgentic.agent import HumanProxy

        runtime = MagicMock()
        other_card = MagicMock()
        other_card.get_agent_class.return_value = type("OtherAgent", (), {})
        human_card = MagicMock()
        human_card.get_agent_class.return_value = HumanProxy
        runtime.team.agent_cards = {"analyst": other_card, "human": human_card}

        mock_addr = MagicMock()
        runtime.addrs = {"analyst": MagicMock(), "human": mock_addr}

        mock_proxy = MagicMock()
        runtime.actor_system.proxy_ask.return_value = mock_proxy

        message = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.process_human_input("user input", message)

        runtime.actor_system.proxy_ask.assert_called_once_with(mock_addr, HumanProxy)
        mock_proxy.process_human_input.assert_called_once_with("user input", message)


class TestLocalTeamHandleSubscribe:
    """AC1: subscribe() delegates via orchestrator proxy."""

    def test_subscribe_delegates_to_orchestrator(self) -> None:
        """Gets orchestrator proxy and calls subscribe(subscriber)."""
        from akgentic.core.orchestrator import Orchestrator

        runtime = MagicMock()
        mock_orch_proxy = MagicMock()
        runtime.actor_system.proxy_ask.return_value = mock_orch_proxy

        subscriber = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.subscribe(subscriber)

        runtime.actor_system.proxy_ask.assert_called_once_with(
            runtime.orchestrator_addr, Orchestrator,
        )
        mock_orch_proxy.subscribe.assert_called_once_with(subscriber)


class TestLocalTeamHandleUnsubscribe:
    """AC1: unsubscribe() delegates via orchestrator proxy."""

    def test_unsubscribe_delegates_to_orchestrator(self) -> None:
        """Gets orchestrator proxy and calls unsubscribe(subscriber)."""
        from akgentic.core.orchestrator import Orchestrator

        runtime = MagicMock()
        mock_orch_proxy = MagicMock()
        runtime.actor_system.proxy_ask.return_value = mock_orch_proxy

        subscriber = MagicMock()
        handle = LocalTeamHandle(runtime)
        handle.unsubscribe(subscriber)

        runtime.actor_system.proxy_ask.assert_called_once_with(
            runtime.orchestrator_addr, Orchestrator,
        )
        mock_orch_proxy.unsubscribe.assert_called_once_with(subscriber)

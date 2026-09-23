"""TeamsChannelAdapter — delivers outbound messages via the Bot Framework Connector."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage
    from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

logger = logging.getLogger(__name__)

# Mirrors ``TeamsChannelParser.channel_name``. A literal here rather than a
# Protocol member: an adapter does not advertise its channel the way a parser
# does, so the comparison belongs to the one implementation that knows the
# answer (ADR-043 §D6).
TEAMS_CHANNEL = "teams"

# The Bot Framework Connector resource every bot token is issued for,
# regardless of tenancy.
CONNECTOR_SCOPE = "https://api.botframework.com/.default"

# A **multi-tenant** bot authenticates against this shared authority rather
# than against its own tenant. A single-tenant bot uses its tenant id instead,
# and getting this wrong is the classic Teams failure: inbound works, every
# reply 401s, and the error names none of it.
MULTI_TENANT_AUTHORITY = "botframework.com"
LOGIN_HOST = "https://login.microsoftonline.com"

SINGLE_TENANT = "singletenant"

# Refresh this long before the token actually expires, so a reply is never sent
# with a credential that dies in flight.
TOKEN_EXPIRY_MARGIN_S = 300.0

_UNKNOWN_SENDER = "an unknown sender"


class TeamsChannelAdapter:
    """Delivers outbound agent messages to Teams conversations.

    Satisfies the ``InteractionChannelAdapter`` protocol via structural
    subtyping.

    Transport:
        A reply is a POST to ``{serviceUrl}/v3/conversations/{id}/activities``
        carrying a bearer token from Microsoft Entra's client-credentials flow.

        **There is no single endpoint to configure.** Bot Framework hosts each
        conversation on a regional service and names it in the inbound
        activity's ``serviceUrl``; the parser stores that on the binding and
        this adapter reads it back. ``teams_service_url`` is only a fallback
        for a binding written before that existed.

    Tenancy:
        ``teams_app_type`` decides the token authority — ``SingleTenant`` uses
        the bot's own tenant, anything else uses Microsoft's shared
        ``botframework.com`` authority. A mismatch produces a 401 on every
        outbound message while inbound continues to work perfectly, which is a
        genuinely confusing failure, so the resolved authority is logged once
        at construction.

    Threading:
        ``deliver`` runs in a Pykka actor thread and ``deliver_notice`` in the
        FastAPI one, so the cached token is guarded by a lock. Without it two
        threads racing an expiry would each fetch, and one would overwrite the
        other's token with an older one.

    Args:
        teams_app_id: The bot's Entra application (client) id.
        teams_app_password: Its client secret.
        teams_app_type: ``SingleTenant`` or ``MultiTenant``, matching the Azure
            Bot resource's *Microsoft App Type*.
        teams_tenant_id: Directory (tenant) id. Required for ``SingleTenant``.
        teams_service_url: Fallback connector base URL, used only when a
            binding carries none.
    """

    def __init__(
        self,
        teams_app_id: str = "",
        teams_app_password: str = "",
        teams_app_type: str = "MultiTenant",
        teams_tenant_id: str = "",
        teams_service_url: str = "",
        **_kwargs: str,
    ) -> None:
        self._app_id = teams_app_id
        self._password = teams_app_password
        self._tenant_id = teams_tenant_id
        self._fallback_service_url = teams_service_url
        self._authority = _resolve_authority(teams_app_type, teams_tenant_id)
        self._client = httpx.Client(timeout=15.0)
        self._lock = threading.Lock()
        self._token: str | None = None
        self._token_expires_at = 0.0
        logger.info("Teams adapter using token authority %s", self._token_url())

    def matches(self, msg: SentMessage, binding: ChannelBinding) -> bool:
        """Check if this adapter should deliver the message.

        Returns True when the recipient actor is a ``UserProxy``, or a subclass
        such as ``HumanProxy``, and the binding belongs to Teams. The recipient
        check is structural rather than a comparison against the recipient's
        ``role`` string, so a team is free to name its human-in-the-loop member
        anything.

        Args:
            msg: The outbound message to check.
            binding: The recipient agent's channel binding.

        Returns:
            True if the recipient is a UserProxy agent bound to Teams.
        """
        if binding.channel != TEAMS_CHANNEL:
            return False
        try:
            return msg.recipient.is_user_proxy
        except Exception:  # noqa: BLE001
            return False

    # -- authentication ----------------------------------------------------

    def _token_url(self) -> str:
        """The Entra token endpoint this bot's tenancy requires."""
        return f"{LOGIN_HOST}/{self._authority}/oauth2/v2.0/token"

    def _bearer_token(self) -> str | None:
        """Return a valid access token, fetching one when the cache is cold or stale.

        Returns None on failure rather than raising: this is called from an
        actor thread, where an escape would take the dispatch with it.
        """
        with self._lock:
            if self._token is not None and time.monotonic() < self._token_expires_at:
                return self._token
            token = self._fetch_token()
            if token is not None:
                self._token = token
            return self._token if token is not None else None

    def _fetch_token(self) -> str | None:
        """Request a client-credentials token, updating the cached expiry."""
        try:
            response = self._client.post(
                self._token_url(),
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._app_id,
                    "client_secret": self._password,
                    "scope": CONNECTOR_SCOPE,
                },
            )
        except (httpx.HTTPError, RuntimeError):
            logger.exception("Teams token request failed")
            return None
        if not response.is_success:
            logger.error(
                "Teams token request rejected (%d): %s — check the app id, the secret, "
                "and that teams_app_type matches the Azure Bot's Microsoft App Type",
                response.status_code,
                response.text,
            )
            return None
        try:
            body = response.json()
        except ValueError:
            logger.error("Teams token endpoint returned non-JSON: %s", response.text[:200])  # noqa: TRY400
            return None
        token = body.get("access_token")
        if not isinstance(token, str) or not token:
            logger.error("Teams token response carried no access_token: %s", body)
            return None
        expires_in = body.get("expires_in")
        lifetime = float(expires_in) if isinstance(expires_in, (int, float)) else 0.0
        self._token_expires_at = time.monotonic() + max(lifetime - TOKEN_EXPIRY_MARGIN_S, 0.0)
        return token

    # -- delivery ----------------------------------------------------------

    def _service_url(self, address: ChannelAddress) -> str:
        """The connector base URL for this conversation.

        Per-conversation rather than global: Bot Framework hosts conversations
        regionally and names the host only on the inbound activity.
        """
        service_url = address.metadata.get("service_url")
        if isinstance(service_url, str) and service_url:
            return service_url
        return self._fallback_service_url

    def _post(self, address: ChannelAddress, text: str) -> None:
        """POST one activity, logging every failure rather than raising.

        The single place this adapter talks to the connector, so a message and
        a notice cannot drift apart in how they are sent or how failures are
        handled.

        ``RuntimeError`` is caught alongside ``httpx.HTTPError`` because a
        closed client raises it and it is **not** an ``httpx.HTTPError``. On
        the ``deliver`` path that escape crashes an actor thread; on the
        ``deliver_notice`` path it turns a command that already took effect
        into a 500.

        Args:
            address: The conversation to answer. A ``ChannelBinding``
                satisfies this.
            text: The message body. Blank means nothing to say.
        """
        if not text.strip():
            logger.debug(
                "Nothing to deliver to Teams conversation %s — blank body", address.channel_user_id
            )
            return
        service_url = self._service_url(address)
        if not service_url:
            # Refusing beats guessing: there is no default region, and posting
            # to the wrong connector cannot reach the conversation anyway.
            logger.error(
                "No serviceUrl for Teams conversation %s — the binding predates it being "
                "stored, or the activity carried none. The next inbound message repairs it.",
                address.channel_user_id,
            )
            return
        token = self._bearer_token()
        if token is None:
            logger.error("No Teams access token; dropping message to %s", address.channel_user_id)
            return
        url = f"{service_url.rstrip('/')}/v3/conversations/{address.channel_user_id}/activities"
        try:
            response = self._client.post(
                url,
                json={"type": "message", "text": text},
                headers={"Authorization": f"Bearer {token}"},
            )
            if not response.is_success:
                logger.error("Teams API error %d: %s", response.status_code, response.text)
        except (httpx.HTTPError, RuntimeError):
            logger.exception("Failed to post to Teams conversation %s", address.channel_user_id)

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        """Deliver an outbound message to a Teams conversation.

        Logs transport errors without raising — delivery failures must not
        crash the caller, which here is a Pykka actor thread.

        Args:
            msg: The message to deliver.
            binding: The recipient agent's binding, naming the conversation.
        """
        # No ``or str(msg.message)`` fallback: an empty ``content`` is falsy, so
        # that idiom quietly posts the message model's repr into a human's chat.
        text = getattr(msg.message, "content", "") or ""
        # An agent with nothing to say produces nothing. The attribution below
        # would make every message non-blank, so this must precede it.
        if not text.strip():
            logger.debug("Agent produced no text; nothing delivered to %s", binding.channel_user_id)
            return
        sender_name = msg.sender.name if msg.sender else _UNKNOWN_SENDER
        self._post(binding, f"**{sender_name}**\n\n{text}")

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        """Deliver a channel-layer acknowledgement to a Teams conversation.

        Notices are fanned out to every configured adapter, so the channel
        comparison is what stops a Telegram chat id being posted here — the
        same check ``matches()`` performs on a binding.

        Args:
            address: The conversation to answer.
            text: The acknowledgement text.
        """
        if address.channel != TEAMS_CHANNEL:
            return
        self._post(address, text)

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Note that a team stopped; release nothing.

        This adapter holds no per-team state. Its httpx client and cached token
        are **process-scoped** and shared by every team, so releasing them here
        would mute every other conversation — silently, because ``deliver()``
        logs and swallows its errors.

        Args:
            team_id: The team being stopped.
        """
        logger.debug("TeamsAdapter stopped: team_id=%s", team_id)


def _resolve_authority(app_type: str, tenant_id: str) -> str:
    """Return the Entra authority segment for this bot's tenancy.

    A single-tenant bot without a tenant id cannot work at all, so it fails
    loudly at construction rather than 401-ing on every reply later.

    Args:
        app_type: The Azure Bot's *Microsoft App Type*, matched case-insensitively.
        tenant_id: Directory (tenant) id.

    Returns:
        The tenant id for a single-tenant bot, ``botframework.com`` otherwise.

    Raises:
        ValueError: If single-tenant is declared with no tenant id.
    """
    if app_type.strip().lower() != SINGLE_TENANT:
        return MULTI_TENANT_AUTHORITY
    if not tenant_id:
        msg = (
            "teams_app_type is 'SingleTenant' but no teams_tenant_id was given. "
            "A single-tenant bot authenticates against its own tenant, so the "
            "directory (tenant) id is required."
        )
        raise ValueError(msg)
    return tenant_id

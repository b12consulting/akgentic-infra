"""Reading a leading ``/command`` out of plain text, for channels with no markup.

Telegram marks a command for you: it reports the leading token as a
``bot_command`` MessageEntity, so a ``/slash`` mid-sentence is unambiguously
text. Signal and Teams send plain strings and mark nothing, so the token has to
be recognised from the text alone — the same rule, needed by both, and worth
having in one place so a fix reaches both rather than one.
"""

from __future__ import annotations

import re

from akgentic.infra.protocols.channels import ChannelCommand

# The trailing lookahead is what keeps ``/path/to/file`` text: a command word
# must END the message or be followed by whitespace, so a slash-separated path
# never parses as a command. The leading ``[A-Za-z]`` does the same for
# ``/2fa`` and for a bare ``/`` (ADR-043 §D10, applied to a channel that
# carries no entities).
_COMMAND_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9_]*)(?=\s|$)")


def parse_leading_command(text: str) -> ChannelCommand | None:
    """Lift the leading ``/command`` token out of ``text``, or return None.

    Args:
        text: The message's full text, returned to the caller untouched. The
            caller keeps it whole: a command the channel layer does not consume
            must reach the team looking exactly as the user typed it.

    Returns:
        The parsed command, or None when the text does not open with a command
        word — no leading slash, a slash followed by a digit or punctuation, or
        a token running straight into more non-space characters.
    """
    match = _COMMAND_RE.match(text)
    if match is None:
        return None
    # ``lstrip`` drops the separating whitespace; the remainder is verbatim —
    # not lowercased, internal whitespace preserved.
    return ChannelCommand(name=match.group(1).lower(), rest=text[match.end() :].lstrip())

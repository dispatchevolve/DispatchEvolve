"""Shared pre-provider input limits for DispatchEvolve LLM calls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


# UTF-8 bytes are a conservative upper bound on BPE token count.  The envelope
# leaves ample room below the supported model contexts and for response tokens.
MAX_PROMPT_UTF8_BYTES = 500_000


class PromptSizeError(ValueError):
    """An LLM input cannot be sent within the context envelope."""


def prompt_utf8_bytes(system: str, messages: Iterable[Mapping[str, Any]]) -> int:
    return len(system.encode("utf-8")) + sum(
        len(str(message.get("content", "")).encode("utf-8")) for message in messages
    )


def require_prompt_within_limit(
    system: str,
    messages: Iterable[Mapping[str, Any]],
    *,
    role: str,
) -> int:
    size = prompt_utf8_bytes(system, messages)
    if size > MAX_PROMPT_UTF8_BYTES:
        raise PromptSizeError(
            f"{role} provider input is {size} UTF-8 bytes, exceeding the "
            f"{MAX_PROMPT_UTF8_BYTES}-byte pre-send envelope"
        )
    return size

"""One-shot, format-only repair for an existing semantic LLM action."""

from __future__ import annotations

import hashlib
import json

from .prompt_store import RenderedPrompt


FORMAT_REPAIR_VERSION = "format-only-repair-v1"


class ResponseFormatError(ValueError):
    """The response cannot be parsed under its frozen output contract."""


def format_repair_prompt(
    original: RenderedPrompt, previous_response: str, validation_error: Exception,
) -> RenderedPrompt:
    user = f"""This is a format-only correction of the same semantic task.
Do not reconsider or change the substantive decision unless required to make the response internally valid.
Return the complete corrected response only. Do not add commentary or Markdown fences.

<ORIGINAL_USER_INSTRUCTION>
{original.user}
</ORIGINAL_USER_INSTRUCTION>

<PREVIOUS_RESPONSE>
{previous_response}
</PREVIOUS_RESPONSE>

<FORMAT_VALIDATION_ERROR>
{type(validation_error).__name__}: {validation_error}
</FORMAT_VALIDATION_ERROR>
"""
    digest = hashlib.sha256(json.dumps({
        "version": FORMAT_REPAIR_VERSION,
        "role": original.role,
        "system": original.system,
        "user": user,
    }, sort_keys=True).encode()).hexdigest()
    return RenderedPrompt(original.role, original.system, user, digest)

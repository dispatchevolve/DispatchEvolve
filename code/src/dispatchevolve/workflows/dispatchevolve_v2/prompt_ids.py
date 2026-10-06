"""Persistent experiment-local identifiers used only in LLM prompts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .protocol import atomic_json


_NAMESPACE_FORMATS = {
    "candidate": ("", 5),
    "reference": ("R", 5),
    "attempt": ("A", 5),
}


class PromptIdRegistry:
    """Map internal content identities to short, stable, experiment-local IDs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_file():
            document = self._read()
            if document.get("schema_version") != 1:
                raise ValueError("unsupported Prompt ID registry schema")
        else:
            atomic_json(self.path, {"schema_version": 1, "namespaces": {}})

    def _read(self) -> dict[str, Any]:
        import json

        return json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _format(namespace: str, number: int) -> str:
        try:
            prefix, width = _NAMESPACE_FORMATS[namespace]
        except KeyError as exc:
            raise ValueError(f"unsupported Prompt ID namespace: {namespace}") from exc
        if number <= 0 or number > 99_999:
            raise ValueError(f"{namespace} Prompt ID exhausted its five-digit range")
        return f"{prefix}{number:0{width}d}"

    def get(self, namespace: str, internal_id: str) -> str:
        internal = str(internal_id)
        document = self._read()
        namespaces = document.setdefault("namespaces", {})
        state = namespaces.setdefault(namespace, {
            "next_number": 1,
            "internal_to_prompt": {},
        })
        mapping = state["internal_to_prompt"]
        if internal in mapping:
            return str(mapping[internal])
        number = int(state["next_number"])
        prompt_id = self._format(namespace, number)
        mapping[internal] = prompt_id
        state["next_number"] = number + 1
        atomic_json(self.path, document)
        return prompt_id

    def get_many(self, namespace: str, internal_ids: list[str] | tuple[str, ...]) -> dict[str, str]:
        return {str(identifier): self.get(namespace, str(identifier)) for identifier in internal_ids}

    def resolve(self, namespace: str, prompt_id: str) -> str:
        value = str(prompt_id)
        document = self._read()
        state = document.get("namespaces", {}).get(namespace, {})
        matches = [
            internal
            for internal, displayed in state.get("internal_to_prompt", {}).items()
            if displayed == value
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous {namespace} Prompt ID: {value}")
        return str(matches[0])

    def validate(self, namespace: str, value: str) -> None:
        prefix, width = _NAMESPACE_FORMATS[namespace]
        if re.fullmatch(rf"{re.escape(prefix)}[0-9]{{{width}}}", str(value)) is None:
            raise ValueError(f"invalid {namespace} Prompt ID: {value}")

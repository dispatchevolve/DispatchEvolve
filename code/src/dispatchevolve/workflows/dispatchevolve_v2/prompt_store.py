"""Versioned prompt registry with bundled task templates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping

from .contracts import jsonable


@dataclass(frozen=True)
class RenderedPrompt:
    role: str
    system: str
    user: str
    prompt_hash: str


class PromptStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or not document.get("version"):
            raise ValueError("invalid V2 prompt JSON header")
        roles = document.get("roles")
        if not isinstance(roles, dict) or not roles:
            raise ValueError("V2 prompt JSON requires roles")
        self.version = str(document["version"])
        self.roles = roles
        self._load_bundled_templates()

    def _load_bundled_templates(self) -> None:
        """Load bundled task templates; custom JSON supplies complete prompts."""
        if self.path.resolve().parent != Path(__file__).resolve().parent / "prompts":
            return
        for role, spec in self.roles.items():
            templates = spec["templates"]
            for part in ("system", "user"):
                spec[part] = self.path.with_name(templates[part]).read_text(encoding="utf-8")
            spec["parameters"] = list(dict.fromkeys(
                match.group("named") or match.group("braced")
                for match in Template.pattern.finditer(spec["user"])
                if match.group("named") or match.group("braced")
            ))

    def render(self, role: str, parameters: Mapping[str, Any]) -> RenderedPrompt:
        spec = self.roles.get(role)
        if not isinstance(spec, dict):
            raise ValueError(f"unknown V2 prompt role: {role}")
        required = tuple(spec.get("parameters", ()))
        if set(parameters) != set(required):
            raise ValueError(
                f"invalid {role} prompt parameters: missing={sorted(set(required)-set(parameters))}, "
                f"unknown={sorted(set(parameters)-set(required))}"
            )
        values = {
            key: value if isinstance(value, str) else json.dumps(
                jsonable(value), ensure_ascii=False, sort_keys=True
            )
            for key, value in parameters.items()
        }
        system = str(spec["system"])
        user = Template(str(spec["user"])).substitute(values)
        digest = hashlib.sha256(
            json.dumps({"version": self.version, "role": role, "system": system, "user": user}, sort_keys=True).encode()
        ).hexdigest()
        return RenderedPrompt(role, system, user, digest)

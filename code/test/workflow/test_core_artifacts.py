"""Long-term tests for V2's workflow-independent artifact dependency.

Related files: dispatchevolve.artifacts and full_dispatch.evolution_adapter.
Covered behavior: canonical hashing and atomic JSON replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

from dispatchevolve.artifacts import atomic_write_json, canonical_hash
from dispatchevolve.tasks.full_dispatch.evolution_adapter import (
    evaluator_implementation_fingerprint,
)


def test_core_artifacts_are_deterministic_and_full_dispatch_is_importable(
    tmp_path: Path,
) -> None:
    left = {"path": Path("engine.py"), "values": {"b": 2, "a": 1}}
    right = {"values": {"a": 1, "b": 2}, "path": Path("engine.py")}
    assert canonical_hash(left) == canonical_hash(right)

    destination = tmp_path / "state" / "artifact.json"
    atomic_write_json(destination, left)
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "path": "engine.py",
        "values": {"a": 1, "b": 2},
    }
    assert len(evaluator_implementation_fingerprint()) == 64

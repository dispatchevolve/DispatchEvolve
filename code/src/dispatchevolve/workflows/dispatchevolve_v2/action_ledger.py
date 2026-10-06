"""Durable reserve/commit ledger for provider and evaluator actions."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from .contracts import content_hash, jsonable
from .protocol import atomic_json


class ActionLedger:
    def __init__(self, root: Path):
        self.root = root / "actions"
        self._lock = threading.Lock()

    @staticmethod
    def _next_failed_attempt(path: Path) -> Path:
        """Return a lossless, monotonically numbered physical-attempt artifact."""
        numbers = []
        for artifact in path.glob("failed_attempt_*.json"):
            suffix = artifact.stem.removeprefix("failed_attempt_")
            if suffix.isdigit():
                numbers.append(int(suffix))
        return path / f"failed_attempt_{max(numbers, default=0) + 1:06d}.json"

    def execute(
        self, action_id: str, identity: Any, callback: Callable[[], Any],
        *, recover: Callable[[], Any | None] | None = None,
    ) -> Any:
        key = content_hash({"action_id": action_id, "identity": identity})
        path = self.root / key
        committed = path / "commit.json"
        with self._lock:
            if committed.is_file():
                document = json.loads(committed.read_text(encoding="utf-8"))
                if "error" in document:
                    # A failed physical attempt is audit evidence, not a reusable
                    # semantic result.  Resume must execute the same action again.
                    committed.replace(self._next_failed_attempt(path))
                else:
                    return document["result"]
            reservation = path / "reservation.json"
            if reservation.is_file():
                result_path = path / "result.json"
                recovered = (json.loads(result_path.read_text(encoding="utf-8"))["result"]
                             if result_path.is_file() else (recover() if recover else None))
                if recovered is None:
                    raise RuntimeError(f"ambiguous reserved action requires review: {action_id}")
                atomic_json(committed, {"action_id": action_id, "identity": identity,
                                        "state": "recovered", "result": jsonable(recovered)})
                reservation.unlink(missing_ok=True)
                return recovered
            atomic_json(reservation, {"action_id": action_id, "identity": identity, "state": "reserved"})
        try:
            result = callback()
        except Exception as exc:
            failed = {"action_id": action_id, "identity": identity, "state": "failed",
                      "error": {"type": type(exc).__name__, "message": str(exc)}}
            with self._lock:
                atomic_json(self._next_failed_attempt(path), failed)
                reservation.unlink(missing_ok=True)
            raise
        atomic_json(path / "result.json", {"result": jsonable(result)})
        atomic_json(committed, {"action_id": action_id, "identity": identity, "state": "committed", "result": jsonable(result)})
        reservation.unlink(missing_ok=True)
        return result

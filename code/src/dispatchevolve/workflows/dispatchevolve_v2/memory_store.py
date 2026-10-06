"""Append-only scene memory with one compact current record per scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import content_hash, jsonable


class SceneMemoryStore:
    def __init__(self, root: Path, current: dict[str, dict[str, Any]]):
        self.path = root / "scene_memory" / "records.jsonl"
        self.current = current
        self._record_ids: set[str] = set()
        self._opportunity_current: dict[tuple[str, str], dict[str, Any]] = {}
        self._scene_without_opportunity: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._record_ids.add(str(record["record_id"]))
                    scene_hash = str(record["scene_hash"])
                    self.current[scene_hash] = record
                    opportunity = record.get("opportunity") or {}
                    opportunity_id = opportunity.get("opportunity_id")
                    if opportunity_id:
                        self._opportunity_current[(scene_hash, str(opportunity_id))] = record
                    else:
                        self._scene_without_opportunity[scene_hash] = record

    def append(self, scene_hash: str, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = jsonable(payload)
        incoming_opportunity = normalized.get("opportunity") or {}
        incoming_opportunity_id = incoming_opportunity.get("opportunity_id")
        previous = (self._opportunity_current.get((scene_hash, str(incoming_opportunity_id)))
                    if incoming_opportunity_id else self.current.get(scene_hash))
        record_id = content_hash({"scene": scene_hash, "kind": kind, "payload": normalized})
        if record_id in self._record_ids:
            return previous or {"record_id": record_id, "scene_hash": scene_hash, "kind": kind, **normalized}
        carried = {
            key: value for key, value in (previous or {}).items()
            if key not in {"record_id", "scene_hash", "kind", "previous_record_id"}
        }
        record = {
            "record_id": record_id,
            "scene_hash": scene_hash, "kind": kind,
            "previous_record_id": previous.get("record_id") if previous else None,
            **carried, **normalized,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.current[scene_hash] = record
        if incoming_opportunity_id:
            self._opportunity_current[(scene_hash, str(incoming_opportunity_id))] = record
        else:
            self._scene_without_opportunity[scene_hash] = record
        self._record_ids.add(record_id)
        return record

    @staticmethod
    def _prompt_record(record: Mapping[str, Any]) -> dict[str, Any]:
        summary = dict(record.get("scenario_summary") or {})
        evidence = list(summary.pop("evidence_ids", ()) or ())
        return {
            "scene_hash": str(record["scene_hash"]),
            "query": ((record.get("scenario_rule") or {}).get("scenario") or {}).get("canonical"),
            "engine_id": record.get("engine_id"),
            "round": record.get("round"),
            "evidence_hash": record.get("evidence_hash"),
            "evidence_count": len(evidence),
            "opportunity": record.get("opportunity"),
            "llm_scene_summary": record.get("llm_scene_summary"),
            "opportunity_status": record.get("opportunity_status"),
            "scenario_summary": summary or None,
            "assessment": record.get("assessment"),
            "evolution_result": record.get("evolution_result"),
            "local_outcome": {
                name: record.get(name)
                for name in ("success", "evaluated_count", "best_delta")
                if name in record
            } or None,
        }

    def prompt_view(self) -> list[dict[str, Any]]:
        return [self._prompt_record(self.current[key]) for key in sorted(self.current)]

    def lifecycle_view(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("lifecycle view limit must be positive")
        records = [
            *self._opportunity_current.values(),
            *(item for item in self._scene_without_opportunity.values()
              if item.get("opportunity_status") == "NO_OPPORTUNITY"),
        ]
        records.sort(key=lambda item: (int(item.get("round") or 0), str(item.get("record_id") or "")))
        return [self._prompt_record(item) for item in records[-limit:]]

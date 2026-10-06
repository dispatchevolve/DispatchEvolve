"""Immutable raw local-task records for a later versioned DPO Dataset Builder."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import Opportunity, content_hash
from .shared_genetic_adapter import LocalEvolutionResult


def raw_task_records(
    opportunities: list[Opportunity], outcomes: Mapping[str, LocalEvolutionResult], *,
    protocol_hash: str, critic_prompts: Mapping[str, Mapping[str, Any]],
    raw_sample_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Describe observed tasks without inventing labels or preference responses.

    Pairing, thresholds, balancing, and chosen/rejected construction deliberately
    belong to the deferred DPO Dataset Builder and are absent from this schema.
    """
    records: list[dict[str, Any]] = []
    for opportunity in opportunities:
        outcome = outcomes.get(opportunity.opportunity_id)
        if outcome is None:
            continue
        records.append({
            "schema_version": "dpo-raw-local-task-v1", "protocol_hash": protocol_hash,
            "record_id": content_hash({"protocol": protocol_hash, "opportunity": opportunity.opportunity_id}),
            "opportunity_id": opportunity.opportunity_id,
            "scenario": opportunity.scenario.canonical, "objective": opportunity.objective,
            "critic_prompt": dict(critic_prompts.get(opportunity.opportunity_id, {})),
            "observed_local_evolution": {
                             "success": bool(outcome.success),
                             "evaluated_count": outcome.evaluated_count,
                             "best_delta": outcome.best_delta, "attempts": list(outcome.attempts)},
            "raw_task_artifact": dict((raw_sample_artifacts or {}).get(opportunity.opportunity_id, {})),
            "dataset_builder_status": "pending",
        })
    return records

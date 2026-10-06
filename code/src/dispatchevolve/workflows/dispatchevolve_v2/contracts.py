"""Frozen contracts and canonical identities for DispatchEvolve V2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


OBJECTIVE_KEYS = (
    "order_ar", "mean_gmv", "mean_eta", "mean_pcaa", "mean_dcaa",
    "mean_fqs",
)
GUARDRAIL_KEYS = ("order_br",)
METRIC_KEYS = (*OBJECTIVE_KEYS, *GUARDRAIL_KEYS)
OBJECTIVE_DIRECTIONS: Mapping[str, int] = {
    "order_ar": 1, "mean_gmv": 1, "mean_eta": -1, "mean_pcaa": -1,
    "mean_dcaa": -1, "mean_fqs": 1,
}
METRIC_DIRECTIONS: Mapping[str, int] = {
    **OBJECTIVE_DIRECTIONS,
    "order_br": 1,
}


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("V2 artifacts cannot contain non-finite numbers")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class MetricVector:
    raw: Mapping[str, float]
    oriented_delta: Mapping[str, float]
    baseline_id: str
    data_scope_hash: str

    def __post_init__(self) -> None:
        if tuple(self.raw) != METRIC_KEYS or tuple(self.oriented_delta) != METRIC_KEYS:
            raise ValueError("metric vectors must use the frozen metric order")


@dataclass(frozen=True)
class ScenarioPredicate:
    expression: Mapping[str, Any]
    canonical: str
    predicate_hash: str


@dataclass(frozen=True)
class ScenarioRule:
    """LLM-1 output before deterministic scenario aggregation."""

    scenario: ScenarioPredicate
    rationale: str


@dataclass(frozen=True)
class DecisionTrace:
    trace_id: str
    batch_id: str
    input_reference: Mapping[str, Any]
    ordered_policy_events: tuple[Mapping[str, Any], ...]
    final_result: Mapping[str, Any]
    engine_id: str


@dataclass(frozen=True)
class ScenarioSummary:
    scenario: ScenarioPredicate
    matched_rows: int
    matched_batches: int
    coverage: float
    metrics: Mapping[str, float]
    global_metrics: Mapping[str, float]
    total_rows: int
    total_batches: int
    policy_statistics: Mapping[str, Any]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    scenario: ScenarioPredicate
    objective: str
    related_policies: tuple[str, ...]
    rationale: str
    improvement_plan: str | None = None
    evidence_basis: str | None = None


@dataclass(frozen=True)
class CriticAssessment:
    decision: Literal["ACCEPT", "REJECT"]
    confidence: float
    reason: str
    model: str
    prompt_hash: str


@dataclass(frozen=True)
class LocalCandidate:
    candidate_id: str
    engine_dir: Path
    opportunity_id: str
    scenario: ScenarioPredicate
    objective: str
    policy_files: tuple[str, ...]
    metrics: Mapping[str, float]
    oriented_delta: Mapping[str, float]
    lineage: tuple[str, ...]
    diff_summary: str
    introduction: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scenario_hash(self) -> str:
        return self.scenario.predicate_hash


@dataclass(frozen=True)
class RelationEdge:
    left: str
    right: str
    relation: Literal["hard", "unresolved", "clear", "soft"]
    overlap: float
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    model: str | None = None
    prompt_hash: str | None = None


@dataclass(frozen=True)
class LLMPriorityPlan:
    groups: tuple[tuple[str, ...], ...]
    ordered_candidate_ids: tuple[str, ...]
    rationale: str
    model: str
    prompt_hash: str
    precedence_edges: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class CombinationProposal:
    proposal_id: str
    candidate_ids: tuple[str, ...]
    priority_plan: LLMPriorityPlan
    rationale: str
    incremental_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchiveEntry:
    engine_id: str
    engine_dir: Path
    metrics: Mapping[str, float]
    oriented_delta: Mapping[str, float]
    round_index: int
    candidate_ids: tuple[str, ...] = ()
    online_probability: float = 1.0
    change_size: int = 0
    candidate_summaries: tuple[Mapping[str, Any], ...] = ()
    online_eligibility: str | None = None
    online_decision_source: str | None = None
    online_llm_called: bool = False
    conflict_resolution_plan: str = "NONE"
    precedence_edges: tuple[Mapping[str, Any], ...] = ()


@dataclass
class RunState:
    schema_version: int
    protocol_hash: str
    run_id: str
    mode: str
    stage: str = "initialized"
    completed_rounds: int = 0
    incumbent_engine_id: str = ""
    incumbent_engine_dir: str = ""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    scene_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    archive: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    priority_objective: str | None = None
    priority_objective_evidence: dict[str, Any] = field(default_factory=dict)
    archive_replay_session: str | None = None
    archive_replay_through_round: int = 0

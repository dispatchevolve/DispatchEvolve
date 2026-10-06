"""Method-independent contracts for repository-adapted baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, TypeAlias


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    search_path: Path
    heldout_path: Path | None
    search_period: str | None
    heldout_period: str | None
    dataset_version: str | None
    sample_rate: float | None
    expected_search_sha256: str | None
    expected_heldout_sha256: str | None
    aggregation_weight: float | None


@dataclass(frozen=True)
class ModelProfile:
    key: str
    provider: str
    model: str
    api_base: str | None
    api_key_reference: str
    main_enabled: bool

    def public_dict(self) -> dict[str, str | bool | None]:
        """Return serializable connection metadata without resolving a secret."""

        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_reference": self.api_key_reference,
            "main_enabled": self.main_enabled,
        }


@dataclass(frozen=True)
class RunKey:
    dataset: str
    method: str
    model: str
    seed: int
    evaluator_backend: str
    budget_profile: str
    semantic_fingerprint: str


@dataclass(frozen=True)
class E0Key:
    dataset: str
    evaluator_backend: str
    seed_candidate_sha256: str
    engine_batch_mode: str
    evaluator_fingerprint: str
    column_policy_fingerprint: str
    split_manifest_sha256: str


@dataclass(frozen=True)
class SelectionKey:
    dataset: str
    method: str
    model: str
    evaluator_backend: str
    selection_fingerprint: str


@dataclass(frozen=True)
class E0FinalizationKey:
    dataset: str
    evaluator_backend: str
    finalization_fingerprint: str


WorkUnitKey: TypeAlias = E0Key | RunKey | SelectionKey | E0FinalizationKey


@dataclass(frozen=True)
class CandidateSandboxConfig:
    backend: Literal["local_process"]


@dataclass(frozen=True)
class MatrixPlan:
    e0_fixtures: tuple[E0Key, ...]
    search_cells: tuple[RunKey, ...]
    selection_groups: tuple[SelectionKey, ...]
    e0_finalizations: tuple[E0FinalizationKey, ...]


@dataclass(frozen=True)
class EngineCandidate:
    """Materialized engine with native lineage and a separate transport base."""

    candidate_id: str
    engine_dir: Path
    # Native algorithm parents/inspirations; this is not the patch apply base.
    parent_ids: tuple[str, ...]
    native_step: int
    # Repository snapshot the proposal was applied to, retained for provenance.
    base_candidate_id: str | None = None


@dataclass(frozen=True)
class EvaluationFeedback:
    candidate_id: str
    valid: bool
    feasible: bool
    reward: float
    utility: float
    vector_fitness: tuple[float, float] | None
    total_violation: float
    metrics: Mapping[str, float]
    invalid_reason: str | None
    artifact_ref: str | None


@dataclass(frozen=True)
class BaselineCheckpoint:
    method: str
    native_step: int
    state: Mapping[str, object]
    rng_state: Mapping[str, object]


@dataclass(frozen=True)
class CandidateProposal:
    """Method proposal whose apply base is independent of semantic parents."""

    method: str
    # Transport-only repository snapshot used to materialize the patch.
    base_candidate_id: str
    # Native algorithm lineage/inspirations reported without transport invention.
    parent_ids: tuple[str, ...]
    native_step: int
    raw_text: str


@dataclass(frozen=True)
class SuiteConfig:
    datasets: tuple[str, ...]
    methods: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    mode: Literal["smoke", "full", "main"]
    evaluator_backend: str
    budget_profile: str
    candidate_sandbox: CandidateSandboxConfig
    suite_config_hash: str
    seed_candidate_sha256: str
    engine_batch_mode: str
    evaluator_fingerprint: str
    column_policy_fingerprint: str
    objective_fingerprint: str


@dataclass(frozen=True)
class RunContext:
    run_key: RunKey
    seed_candidate: EngineCandidate
    services: Mapping[str, object]


@dataclass(frozen=True)
class StepResult:
    evaluations: tuple[tuple[EngineCandidate, EvaluationFeedback], ...]
    finished: bool


@dataclass(frozen=True)
class BaselineResult:
    representative: EngineCandidate
    feedback: EvaluationFeedback
    evaluator_submissions: int
    valid_evaluations: int
    physical_expensive_evaluations: int
    cache_hits: int
    method_diagnostics: Mapping[str, object] = field(default_factory=dict)


class BaselineMethod(Protocol):
    def initialize(self, context: RunContext) -> None: ...

    def step(self) -> StepResult: ...

    def is_finished(self) -> bool: ...

    def checkpoint(self) -> BaselineCheckpoint: ...

    def restore(self, checkpoint: BaselineCheckpoint) -> None: ...

    def result(self) -> BaselineResult: ...


BaselineFactory: TypeAlias = Callable[[Mapping[str, Any]], BaselineMethod]

"""Validated YAML configuration for DispatchEvolve V2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import (
    GUARDRAIL_KEYS,
    METRIC_KEYS,
    OBJECTIVE_DIRECTIONS,
    OBJECTIVE_KEYS,
)


DEFAULT_PROMPT_JSON = Path(__file__).resolve().parent / "prompts" / "prompts.json"
DEFAULT_PRIORITY_OBJECTIVE_KEYS = (
    "order_ar", "mean_pcaa", "mean_dcaa", "mean_eta",
)



@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    transport_max_retries: int | None = None


@dataclass(frozen=True)
class BudgetConfig:
    full_replay_equivalents: float = 30.0
    outer_rounds: int = 10
    proposal_attempts_per_round: int = 30
    admitted_opportunities_target: int = 10
    max_evolved_directions_per_scene: int = 2
    local_iterations_per_opportunity: int = 30
    local_evaluations_per_opportunity: int = 30
    parallel_opportunities: int = 5
    population_size: int = 10
    archive_size: int = 3
    retained_candidates: int = 1
    num_islands: int = 1
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    feature_dimensions: tuple[str, str] = ("complexity", "diversity")
    feature_bins: int = 10
    diversity_reference_size: int = 20
    migration_interval: int = 50
    migration_rate: float = 0.1
    combination_proposals_per_round: int = 10
    combination_evaluations_per_round: int = 3
    reference_limit: int = 14
    relation_pairs_per_call: int = 30
    combination_feedback_limit: int = 30
    local_early_stopping_patience: int | None = None
    outer_early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.full_replay_equivalents, bool) or not math.isfinite(self.full_replay_equivalents) or self.full_replay_equivalents < 1:
            raise ValueError("full_replay_equivalents must be finite and at least 1")
        object.__setattr__(self, "feature_dimensions", tuple(self.feature_dimensions))
        for name, value in vars(self).items():
            if name in {"feature_dimensions"}:
                continue
            if value is not None and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"budget.{name} must be positive")
        if self.retained_candidates > self.archive_size:
            raise ValueError("retained candidates cannot exceed local archive size")
        if abs(self.elite_selection_ratio + self.exploration_ratio + self.exploitation_ratio - 1.0) > 1e-12:
            raise ValueError("local selection ratios must sum to one")
        if self.feature_dimensions != ("complexity", "diversity"):
            raise ValueError("V2 feature dimensions are frozen to complexity and diversity")


@dataclass(frozen=True)
class DPOCollectionConfig:
    """DPO collection goal plus independent local-search quality controls."""

    target_records: int = 210
    proposal_attempt_multiplier: int = 3
    max_iterations_per_scene_objective: int = 10
    max_evaluations_per_scene_objective: int = 10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"dpo_collection.{name} must be a positive integer")
    @property
    def max_scene_proposal_attempts(self) -> int:
        # One valid Scene may yield only one Opportunity, so the safety budget
        # must scale with the requested record count rather than a fixed
        # Scene-to-Objective ratio.
        return self.target_records * self.proposal_attempt_multiplier


@dataclass(frozen=True)
class V2Config:
    run_id: str
    mode: str
    engine_dir: Path
    data_path: Path
    test_path: Path | None
    backend: str
    output_dir: Path
    cache_root: Path
    log_root: Path
    model: ModelConfig
    budget: BudgetConfig
    critic_model: ModelConfig | None = None
    online_model: ModelConfig | None = None
    prompt_json: Path = DEFAULT_PROMPT_JSON
    dpo_collection: DPOCollectionConfig = field(default_factory=DPOCollectionConfig)
    rho: float = 0.005
    epsilon: float = 1e-12
    comparison_tolerance: float = 1e-12
    coverage_min_exclusive: float = 0.01
    coverage_max_inclusive: float = 0.20
    coverage_max_is_exclusive: bool = False
    csv_chunk_size: int = 25_000
    trace_evidence_per_policy: int = 3
    trace_transport_max_bytes: int = 268_435_456
    candidate_process_workers: int = 1
    local_candidate_process_workers: int = 1
    candidate_process_worker_budget: int = 5
    online_uplift_enabled: bool = False
    evaluate_test_baseline: bool = False
    candidate_independence_strategy: str = "incumbent_search_fixed_e0_admission"
    experiment_metric_keys: tuple[str, ...] = METRIC_KEYS
    composition_priority_metric: str = "order_ar"
    priority_objective_keys: tuple[str, ...] | None = None
    debug_max_batches: int | None = None
    persist_debug_frames: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in {"main", "dpo_data_collection", "local_debug"}:
            raise ValueError("unsupported V2 mode")
        if self.mode == "main":
            if self.test_path is None:
                raise ValueError("main mode requires test_path")
            if self.data_path.resolve() == self.test_path.resolve():
                raise ValueError("evolution and test paths must differ")
            if not self.evaluate_test_baseline:
                raise ValueError("main mode requires evaluate_test_baseline for test reporting")
        if self.mode == "dpo_data_collection" and self.test_path is not None:
            raise ValueError("dpo_data_collection must not receive a sealed test path")
        if self.backend != "local":
            raise ValueError("backend must be local")
        if not 0 <= self.rho < 1:
            raise ValueError("rho must be in [0, 1)")
        if not 0 <= self.coverage_min_exclusive < self.coverage_max_inclusive <= 1:
            raise ValueError("invalid coverage interval")
        if self.trace_evidence_per_policy <= 0:
            raise ValueError("trace_evidence_per_policy must be positive")
        if self.trace_transport_max_bytes <= 0:
            raise ValueError("trace_transport_max_bytes must be positive")
        if isinstance(self.candidate_process_workers, bool) or self.candidate_process_workers <= 0:
            raise ValueError("candidate_process_workers must be a positive integer")
        if (
            isinstance(self.local_candidate_process_workers, bool)
            or self.local_candidate_process_workers <= 0
        ):
            raise ValueError("local_candidate_process_workers must be a positive integer")
        if (
            isinstance(self.candidate_process_worker_budget, bool)
            or self.candidate_process_worker_budget <= 0
        ):
            raise ValueError("candidate_process_worker_budget must be a positive integer")
        required_worker_budget = (
            self.local_candidate_process_workers * self.budget.parallel_opportunities
        )
        if required_worker_budget > self.candidate_process_worker_budget:
            raise ValueError(
                "candidate process demand exceeds the per-experiment worker budget: "
                f"{required_worker_budget} > {self.candidate_process_worker_budget}"
            )
        if tuple(OBJECTIVE_DIRECTIONS) != OBJECTIVE_KEYS:
            raise AssertionError("objective order drift")
        object.__setattr__(self, "experiment_metric_keys", tuple(self.experiment_metric_keys))
        if not self.experiment_metric_keys:
            raise ValueError("experiment_metric_keys must not be empty")
        if len(set(self.experiment_metric_keys)) != len(self.experiment_metric_keys):
            raise ValueError("experiment_metric_keys must not contain duplicates")
        if any(key not in METRIC_KEYS for key in self.experiment_metric_keys):
            raise ValueError(
                "experiment_metric_keys must contain only fixed evaluator metrics"
            )
        if not self.experiment_objective_keys:
            raise ValueError(
                "experiment_metric_keys must contain at least one reward objective"
            )
        if self.composition_priority_metric not in self.experiment_objective_keys:
            raise ValueError(
                "composition_priority_metric must be an active reward objective"
            )
        if self.budget.reference_limit < len(self.experiment_objective_keys):
            raise ValueError(
                "reference limit must cover every active reward objective"
            )
        if self.priority_objective_keys is None:
            priority_objective_keys = tuple(
                key
                for key in DEFAULT_PRIORITY_OBJECTIVE_KEYS
                if key in self.experiment_objective_keys
            )
            if not priority_objective_keys:
                priority_objective_keys = self.experiment_objective_keys
        else:
            priority_objective_keys = tuple(self.priority_objective_keys)
        object.__setattr__(self, "priority_objective_keys", priority_objective_keys)
        if not self.priority_objective_keys:
            raise ValueError("priority_objective_keys must not be empty")
        if len(set(self.priority_objective_keys)) != len(self.priority_objective_keys):
            raise ValueError("priority_objective_keys must not contain duplicates")
        if any(key not in self.experiment_objective_keys for key in self.priority_objective_keys):
            raise ValueError(
                "priority_objective_keys must be a subset of active reward objectives"
            )
        if self.candidate_independence_strategy not in {
            "incumbent_search_fixed_e0_admission", "evolve_from_fixed_e0",
        }:
            raise ValueError("unsupported candidate independence strategy")

    @property
    def experiment_objective_keys(self) -> tuple[str, ...]:
        return tuple(
            key for key in self.experiment_metric_keys if key in OBJECTIVE_KEYS
        )

    @property
    def experiment_guardrail_keys(self) -> tuple[str, ...]:
        return tuple(
            key for key in self.experiment_metric_keys if key in GUARDRAIL_KEYS
        )


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> V2Config:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V2 config must be a mapping")
    if "ablation_variant" in raw:
        raise ValueError("ablation_variant is not supported by the main-experiment release")
    base = config_path.parent
    model = ModelConfig(**dict(raw["model"]))
    budget_values = dict(raw.get("budget", {}))
    legacy_target = budget_values.pop("accepted_opportunities_target", None)
    if legacy_target is not None:
        if "admitted_opportunities_target" in budget_values:
            raise ValueError(
                "budget cannot set both admitted_opportunities_target and its legacy alias "
                "accepted_opportunities_target"
            )
        budget_values["admitted_opportunities_target"] = legacy_target
    budget = BudgetConfig(**budget_values)
    dpo_collection = DPOCollectionConfig(**dict(raw.get("dpo_collection", {})))
    raw_priority_objective_keys = raw.get("priority_objective_keys")
    return V2Config(
        run_id=str(raw["run_id"]), mode=str(raw.get("mode", "main")),
        engine_dir=_path(raw["engine_dir"], base), data_path=_path(raw["data_path"], base),
        test_path=_path(raw["test_path"], base) if raw.get("test_path") else None,
        backend=str(raw.get("backend", "local")), output_dir=_path(raw["output_dir"], base),
        cache_root=_path(raw["cache_root"], base), log_root=_path(raw["log_root"], base),
        prompt_json=(_path(raw["prompt_json"], base)
                     if raw.get("prompt_json") else DEFAULT_PROMPT_JSON),
        model=model, budget=budget,
        critic_model=(ModelConfig(**dict(raw["critic_model"]))
                      if raw.get("critic_model") is not None else None),
        online_model=(ModelConfig(**dict(raw["online_model"]))
                      if raw.get("online_model") is not None else None),
        dpo_collection=dpo_collection, rho=float(raw.get("rho", 0.005)),
        epsilon=float(raw.get("epsilon", 1e-12)),
        comparison_tolerance=float(raw.get("comparison_tolerance", 1e-12)),
        coverage_min_exclusive=float(raw.get("coverage_min_exclusive", .01)),
        coverage_max_inclusive=float(raw.get("coverage_max_inclusive", .20)),
        coverage_max_is_exclusive=bool(raw.get("coverage_max_is_exclusive", False)),
        csv_chunk_size=int(raw.get("csv_chunk_size", 25_000)),
        trace_evidence_per_policy=int(raw.get("trace_evidence_per_policy", 3)),
        trace_transport_max_bytes=int(raw.get("trace_transport_max_bytes", 268_435_456)),
        candidate_process_workers=int(raw.get("candidate_process_workers", 1)),
        local_candidate_process_workers=int(raw.get(
            "local_candidate_process_workers", raw.get("candidate_process_workers", 1),
        )),
        candidate_process_worker_budget=int(raw.get(
            "candidate_process_worker_budget",
            int(raw.get(
                "local_candidate_process_workers", raw.get("candidate_process_workers", 1),
            )) * budget.parallel_opportunities,
        )),
        online_uplift_enabled=bool(raw.get("online_uplift_enabled", False)),
        evaluate_test_baseline=bool(raw.get("evaluate_test_baseline", False)),
        candidate_independence_strategy=str(raw.get(
            "candidate_independence_strategy", "incumbent_search_fixed_e0_admission"
        )),
        experiment_metric_keys=tuple(raw.get("experiment_metric_keys", METRIC_KEYS)),
        composition_priority_metric=str(raw.get(
            "composition_priority_metric", "order_ar"
        )),
        priority_objective_keys=(
            tuple(raw_priority_objective_keys)
            if raw_priority_objective_keys is not None
            else None
        ),
        debug_max_batches=raw.get("debug_max_batches"),
        persist_debug_frames=bool(raw.get("persist_debug_frames", False)),
        random_seed=int(raw.get("random_seed", 42)),
    )

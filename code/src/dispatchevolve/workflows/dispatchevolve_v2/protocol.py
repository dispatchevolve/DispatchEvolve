"""Protocol hashing and atomic V2 artifacts."""

from __future__ import annotations

import hashlib
import csv
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import canonical_json, jsonable


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    return hashlib.sha256(canonical_json([
        (path.relative_to(root).as_posix(), file_hash(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]).encode()).hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    files = [
        {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path),
         "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    return {"root_name": root.name, "file_count": len(files), "files": files,
            "tree_hash": hashlib.sha256(canonical_json(files).encode()).hexdigest()}


def data_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        columns = next(csv.reader(handle))
    return {"name": path.name, "size_bytes": path.stat().st_size,
            "sha256": file_hash(path), "columns": columns,
            "schema_hash": hashlib.sha256(canonical_json(columns).encode()).hexdigest()}


def protocol_document(config: Any) -> dict[str, Any]:
    workflow_root = Path(__file__).resolve().parent
    prompt_document = json.loads(config.prompt_json.read_text(encoding="utf-8"))
    local_task_context = workflow_root / "prompts" / "local_policy_evolution_task_context.txt"
    from dispatchevolve.tasks.full_dispatch import evaluator as evaluator_module
    from .prompt_store import PromptStore
    from .batch_query import FEATURE_CATALOG_VERSION, QUERY_DSL_VERSION
    task_root = Path(evaluator_module.__file__).resolve().parent
    genetic_root = workflow_root.parents[1] / "optimizer" / "genetic"
    if config.mode == "dpo_data_collection":
        # Collection size and its proposal safety multiplier are stopping controls,
        # not sample semantics. Excluding them lets a compatible run grow in place.
        budget = {
            "max_iterations_per_scene_objective": config.dpo_collection.max_iterations_per_scene_objective,
            "max_evaluations_per_scene_objective": config.dpo_collection.max_evaluations_per_scene_objective,
            "parallel_opportunities": config.budget.parallel_opportunities,
            "population_size": config.budget.population_size,
            "archive_size": config.budget.archive_size,
            "retained_candidates": config.budget.retained_candidates,
            "num_islands": config.budget.num_islands,
            "elite_selection_ratio": config.budget.elite_selection_ratio,
            "exploration_ratio": config.budget.exploration_ratio,
            "exploitation_ratio": config.budget.exploitation_ratio,
            "feature_dimensions": config.budget.feature_dimensions,
            "feature_bins": config.budget.feature_bins,
            "diversity_reference_size": config.budget.diversity_reference_size,
            "migration_interval": config.budget.migration_interval,
            "migration_rate": config.budget.migration_rate,
        }
    else:
        budget = jsonable(config.budget)
    document = {
        "schema_version": 1,
        "method": "dispatchevolve_v2",
        "run_id": config.run_id,
        "mode": config.mode,
        "evaluator_metric_keys": [
            "order_ar", "mean_gmv", "mean_eta", "mean_pcaa", "mean_dcaa",
            "mean_fqs", "order_br",
        ],
        "experiment_metric_keys": config.experiment_metric_keys,
        "experiment_objective_keys": config.experiment_objective_keys,
        "experiment_guardrail_keys": config.experiment_guardrail_keys,
        "composition_priority_metric": config.composition_priority_metric,
        "objectives": ["order_ar", "mean_gmv", "mean_eta", "mean_pcaa", "mean_dcaa", "mean_fqs"],
        "objective_directions": {
            "order_ar": 1, "mean_gmv": 1, "mean_eta": -1, "mean_pcaa": -1,
            "mean_dcaa": -1, "mean_fqs": 1,
        },
        "guardrails": {
            "order_br": {
                "source_metric": "order_br",
                "definition": "distinct matched orders divided by distinct source orders",
                "maximum_normalized_regression": config.rho,
                "local_scale_source": "absolute scene baseline order_br",
                "global_scale_source": "absolute full-search E0 order_br",
                "rewarded_in_pareto_or_hypervolume": False,
                "eligible_as_local_target": False,
                "eligible_as_priority_objective": False,
            },
        },
        "rho": config.rho, "epsilon": config.epsilon,
        "comparison_tolerance": config.comparison_tolerance,
        "coverage": {
            "lower": config.coverage_min_exclusive,
            "lower_exclusive": True,
            "upper": config.coverage_max_inclusive,
            "upper_exclusive": config.coverage_max_is_exclusive,
        },
        "scene_query_contract": {
            "dsl_version": QUERY_DSL_VERSION,
            "feature_catalog_version": FEATURE_CATALOG_VERSION,
            "membership_unit": "complete_batch",
            "aggregate_null_rule": "ignore_nulls",
            "empty_aggregate_rule": "NA_except_count_nunique_sum",
            "division_by_zero": 0.0,
        },
        "format_repair_contract": {
            "version": "format-only-repair-v1",
            "maximum_repairs_per_semantic_action": 1,
            "consumes_business_proposal_budget": False,
            "semantic_reconsideration_allowed": False,
            "invalid_candidate_relation_after_repair": (
                "deterministic_conservative_unresolved_relation"
            ),
        },
        "data_loading": {"csv_chunk_size": config.csv_chunk_size,
                         "debug_max_batches": config.debug_max_batches},
        "matching_backend": "offline_local_maximum_weight_bipartite",
        "trace_evidence_per_policy": config.trace_evidence_per_policy,
        "trace_transport_max_bytes": config.trace_transport_max_bytes,
        "candidate_process_workers": config.candidate_process_workers,
        "local_candidate_process_workers": config.local_candidate_process_workers,
        "candidate_process_worker_budget": config.candidate_process_worker_budget,
        "online_uplift_enabled": config.online_uplift_enabled,
        "online_selection_rule": "pairwise_wins_then_hypervolume_contribution_then_engine_id",
        "archive_admission": "global_feasibility_before_pareto_update",
        "evaluate_test_baseline": config.evaluate_test_baseline,
        "candidate_independence_strategy": config.candidate_independence_strategy,
        "priority_objective_keys": config.priority_objective_keys,
        "random_seed": config.random_seed,
        "engine_manifest": tree_manifest(config.engine_dir),
        "data_manifest": data_manifest(config.data_path),
        "test_data_manifest": (data_manifest(config.test_path)
                               if config.mode == "main" and config.test_path is not None else None),
        "workflow_source_hash": tree_hash(workflow_root),
        "full_dispatch_task_source_hash": tree_hash(task_root),
        "shared_genetic_source_hash": tree_hash(genetic_root),
        "prompt_path": str(config.prompt_json),
        "prompt_version": prompt_document.get("version"),
        "prompt_hash": file_hash(config.prompt_json),
        "effective_prompt_hash": hashlib.sha256(
            canonical_json(PromptStore(config.prompt_json).roles).encode()
        ).hexdigest(),
        "local_policy_evolution_task_context_path": str(local_task_context),
        "local_policy_evolution_task_context_hash": file_hash(local_task_context),
        "model": jsonable(config.model),
        "backend": config.backend, "budget": budget,
    }
    if config.critic_model is not None:
        document["critic_model"] = jsonable(config.critic_model)
    if config.online_model is not None:
        document["online_model"] = jsonable(config.online_model)
    return document


def semantic_protocol_document(config: Any) -> dict[str, Any]:
    document = protocol_document(config)
    document.pop("run_id", None)
    document.pop("prompt_path", None)
    document.pop("local_policy_evolution_task_context_path", None)
    document["engine_manifest"].pop("root_name", None)
    document["data_manifest"].pop("name", None)
    if document.get("test_data_manifest"):
        document["test_data_manifest"].pop("name", None)
    return document


def protocol_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json(semantic_protocol_document(config)).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(jsonable(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_gzip_json(path: Path, value: Any) -> str:
    """Atomically persist compact deterministic gzip JSON and return payload SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                handle.write(payload)
            raw.flush(); os.fsync(raw.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return file_hash(path)

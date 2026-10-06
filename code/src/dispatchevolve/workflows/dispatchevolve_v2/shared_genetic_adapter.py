"""DispatchEvolve adapter over the shared genetic optimizer."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping

import pandas as pd

from dispatchevolve.baselines.candidates import RepositoryGenomeCodec
from dispatchevolve.baselines.mutations import escape_repository_source
from dispatchevolve.optimizer.genetic.api import EvolutionResult, run_evolution
from dispatchevolve.optimizer.genetic.config import Config as GeneticConfig
from dispatchevolve.optimizer.genetic.config import LLMModelConfig
from dispatchevolve.optimizer.genetic.controller import (
    is_complete_checkpoint,
    prune_complete_checkpoints,
)
from dispatchevolve.optimizer.genetic.database import Program

from .config import V2Config
from .candidate_semantics import build_candidate_introduction
from .contracts import (LocalCandidate, METRIC_DIRECTIONS, METRIC_KEYS,
                        OBJECTIVE_KEYS, Opportunity)
from .local_evaluator import OBJECTIVE_SEMANTICS, V2Evaluator
from .protocol import atomic_json, file_hash
from .shared_genetic_evaluator import candidate_key, repository_files


TASK_CONTEXT_TEMPLATE = Path(__file__).with_name("prompts") / "local_policy_evolution_task_context.txt"


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize a process-local input without duplicating bytes when possible."""
    source = source.resolve()
    if destination.exists():
        if destination.resolve() == source:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@dataclass(frozen=True)
class LocalEvolutionResult:
    candidates: tuple[LocalCandidate, ...]
    success: bool
    evaluated_count: int
    best_delta: Mapping[str, float] | None
    attempts: tuple[Mapping[str, Any], ...]


def render_task_context(
    opportunity: Opportunity,
    *,
    scene_description: str,
    relative_policies: tuple[str, ...],
    rho: float,
    baseline_metrics: Mapping[str, float],
    scales: Mapping[str, float],
    experiment_metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    """Render the single Dispatch-specific template injected into shared genetic."""
    del baseline_metrics, scales
    metric_rows = []
    for name in experiment_metric_keys:
        role = "primary objective" if name == opportunity.objective else "protected metric"
        direction = "maximize" if METRIC_DIRECTIONS[name] > 0 else "minimize"
        metric_rows.append(
            f"{name} | {role} | {direction} | {OBJECTIVE_SEMANTICS[name]['meaning']}"
        )
    local_task = "\n".join((
        f"Scene Query: {opportunity.scenario.canonical}",
        f"Scene Description: {scene_description}",
        f"Requested Adjustment: {opportunity.improvement_plan or opportunity.rationale}",
    ))
    local_contract = "\n".join((
        f"Primary metric: {opportunity.objective}",
        "",
        "metric | role | business_direction | meaning",
        *metric_rows,
        "",
        "A Program is locally feasible only when:",
        "1. normalized improvement in the Primary metric is positive; and",
        (
            "2. normalized improvement in every protected experiment metric is "
            f"at least -{float(rho):.12g}."
        ),
        "",
        (
            "For a locally feasible Program, fitness is the Primary metric's "
            "normalized improvement. The deterministic evaluator calculates all "
            "values by replaying complete Scene batches. Do not claim improvement "
            "without measured feedback."
        ),
    ))
    return Template(TASK_CONTEXT_TEMPLATE.read_text(encoding="utf-8")).substitute(
        local_task=local_task,
        local_policy_evolution_contract=local_contract,
        editable_policy_paths="\n".join(f"- {path}" for path in relative_policies),
    )


def _model(config: V2Config) -> LLMModelConfig:
    model_name = config.model.model
    if config.model.provider == "gemini" and not model_name.startswith("gemini/"):
        model_name = f"gemini/{model_name}"
    elif config.model.provider == "vertex_ai" and not model_name.startswith("vertex_ai/"):
        model_name = f"vertex_ai/{model_name}"
    api_key = os.environ.get(config.model.api_key_env or "", "") or None
    return LLMModelConfig(
        name=model_name,
        api_base=config.model.api_base,
        api_key=api_key,
        retries=config.model.transport_max_retries,
    )


def _genetic_config(
    config: V2Config, *, task_context: str, iterations: int,
    source_size: int, output_dir: Path, random_seed: int,
) -> GeneticConfig:
    genetic = GeneticConfig()
    genetic.max_iterations = iterations
    genetic.checkpoint_interval = 1
    genetic.durable_serial_iterations = True
    genetic.random_seed = random_seed
    genetic.max_code_length = max(10_000, source_size * 3)
    genetic.log_dir = str(
        config.log_root / config.run_id / "local_policy_evolution" / output_dir.parent.name / output_dir.name
    )
    genetic.early_stopping_patience = config.budget.local_early_stopping_patience
    genetic.llm.models = [_model(config)]
    genetic.llm.evaluator_models = genetic.llm.models.copy()
    genetic.llm.api_base = config.model.api_base
    genetic.llm.api_key = genetic.llm.models[0].api_key
    genetic.prompt.system_message = task_context
    genetic.prompt.template_dir = str(
        TASK_CONTEXT_TEMPLATE.with_name("local_genetic")
    )
    genetic.prompt.use_template_stochasticity = False
    genetic.prompt.history_programs_as_changes_description = True
    genetic.prompt.prompt_metric_keys = [
        "combined_score",
        "locally_feasible",
        "constraint_max_violation",
        "constraint_violation_count",
        "constraint_total_violation",
        *(
            f"{prefix}_{name}"
            for prefix in ("raw", "normalized_improvement")
            for name in config.experiment_metric_keys
        ),
    ]
    genetic.prompt.deduplicate_prompt_history = True
    genetic.prompt.omit_empty_feature_coordinates = True
    genetic.prompt.compact_action_context = True
    genetic.prompt.include_artifacts = False
    genetic.prompt.suggest_simplification_after_chars = None
    genetic.prompt.code_length_threshold = None
    genetic.prompt.num_top_programs = min(3, config.budget.diversity_reference_size)
    genetic.prompt.num_diverse_programs = min(
        2, config.budget.diversity_reference_size
    )
    genetic.database.population_size = config.budget.population_size
    genetic.database.archive_size = config.budget.archive_size
    genetic.database.num_islands = config.budget.num_islands
    genetic.database.elite_selection_ratio = config.budget.elite_selection_ratio
    genetic.database.exploration_ratio = config.budget.exploration_ratio
    genetic.database.exploitation_ratio = config.budget.exploitation_ratio
    genetic.database.feature_dimensions = list(config.budget.feature_dimensions)
    genetic.database.feature_bins = config.budget.feature_bins
    genetic.database.diversity_reference_size = config.budget.diversity_reference_size
    genetic.database.migration_interval = config.budget.migration_interval
    genetic.database.migration_rate = config.budget.migration_rate
    genetic.database.random_seed = random_seed
    genetic.evaluator.cascade_evaluation = False
    genetic.evaluator.timeout = None
    return genetic


def _nondominated(
    candidates: list[LocalCandidate],
    tolerance: float,
    objective_keys: tuple[str, ...] = OBJECTIVE_KEYS,
) -> list[LocalCandidate]:
    def dominates(left: LocalCandidate, right: LocalCandidate) -> bool:
        return all(left.oriented_delta[key] >= right.oriented_delta[key] - tolerance for key in objective_keys) and any(
            left.oriented_delta[key] > right.oriented_delta[key] + tolerance for key in objective_keys
        )
    return [item for item in candidates if not any(other.candidate_id != item.candidate_id and dominates(other, item) for other in candidates)]


def _program_genome_from_code(code: str) -> str:
    genome = code.replace("# EVOLVE-BLOCK-START\n", "", 1)
    if "# EVOLVE-BLOCK-END" in genome:
        genome = genome[: genome.rfind("# EVOLVE-BLOCK-END")]
    return genome.rstrip() + "\n"


def _ordered_repository_genome(
    files: Mapping[str, str], editable_policies: tuple[str, ...]
) -> str:
    """Put editable files first so an unscoped duplicate SEARCH stays in scope."""
    ordered = (*editable_policies, *sorted(set(files) - set(editable_policies)))
    return "".join(
        f"<<<FILE {path}>>>\n{escape_repository_source(files[path])}<<<END FILE>>>\n"
        for path in ordered
    )


def _all_checkpoint_programs(
    result: EvolutionResult, checkpoint_root: Path,
) -> tuple[Program, ...]:
    """Recover every evaluated Program, including those later evicted from MAP-Elites.

    V2 checkpoints every iteration.  The newly evaluated child is protected
    through that iteration's population cleanup, so the union of complete
    checkpoints is the lossless evaluated-program history for resume and DPO.
    """
    programs = {program.id: program for program in result.programs}
    checkpoints = sorted(
        (path for path in checkpoint_root.glob("checkpoint_*") if is_complete_checkpoint(path)),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    for checkpoint in checkpoints:
        for path in sorted((checkpoint / "programs").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            program = Program.from_dict(document)
            programs.setdefault(program.id, program)
    return tuple(sorted(
        programs.values(),
        key=lambda item: (int(item.iteration_found), float(item.timestamp), item.id),
    ))


def evolve_opportunity_with_shared_genetic(
    *,
    config: V2Config,
    opportunity: Opportunity,
    scenario_summary: Mapping[str, Any],
    incumbent_dir: Path,
    scene_frame: pd.DataFrame | None = None,
    metric_rows: pd.DataFrame | None = None,
    scene_frame_path: Path | None = None,
    metric_rows_path: Path | None = None,
    baseline_metrics: Mapping[str, float],
    scales: Mapping[str, float],
    evaluator: V2Evaluator,
    round_dir: Path,
) -> LocalEvolutionResult:
    """Jointly evolve all related Policies in one complete-repository Program."""
    root = round_dir / "local_policy_evolution" / opportunity.opportunity_id
    root.mkdir(parents=True, exist_ok=True)
    frame_path = root / "scene_frame.pkl"
    if scene_frame_path is not None:
        _link_or_copy(scene_frame_path, frame_path)
    elif scene_frame is not None:
        scene_frame.to_pickle(frame_path)
    else:
        raise ValueError("scene_frame or scene_frame_path is required")
    local_metric_rows_path = root / "metric_rows.pkl" if metric_rows_path is not None or metric_rows is not None else None
    if metric_rows_path is not None and local_metric_rows_path is not None:
        _link_or_copy(metric_rows_path, local_metric_rows_path)
    elif metric_rows is not None and local_metric_rows_path is not None:
        metric_rows.to_pickle(local_metric_rows_path)

    policies = tuple(dict.fromkeys(opportunity.related_policies))
    if not policies:
        raise ValueError("opportunity has no editable Policy")
    for relative_policy in policies:
        source_path = incumbent_dir / relative_policy
        if not source_path.is_file():
            raise ValueError(f"editable Policy does not exist: {relative_policy}")
    iterations = min(
        config.budget.local_iterations_per_opportunity,
        config.budget.local_evaluations_per_opportunity,
    )
    if iterations <= 0:
        outcome = LocalEvolutionResult((), False, 0, None, ())
        atomic_json(root / "outcome.json", outcome)
        return outcome

    codec = RepositoryGenomeCodec()
    baseline_files = repository_files(codec.encode(incumbent_dir))
    baseline_genome = _ordered_repository_genome(baseline_files, policies)
    joint_root = root / "joint_repository"
    joint_root.mkdir(parents=True, exist_ok=True)
    initial_program = joint_root / "initial_program.py"
    initial_program.write_text(
        f"# EVOLVE-BLOCK-START\n{baseline_genome.rstrip()}\n# EVOLVE-BLOCK-END\n",
        encoding="utf-8",
    )
    context_path = joint_root / "evaluator_context.json"
    context = {
        "incumbent_dir": str(incumbent_dir),
        "relative_policies": list(policies),
        "candidate_root": str(joint_root / "candidates"),
        "scene_frame_path": str(frame_path),
        "metric_rows_path": str(local_metric_rows_path) if local_metric_rows_path else None,
        "runner_root": str(joint_root / "runner"),
        "cache_root": str(evaluator.cache_root),
        "protocol_hash": evaluator.protocol_hash,
        "replay_budget_path": str(evaluator.replay_budget_path) if evaluator.replay_budget_path else None,
        "backend": evaluator.backend,
        "objective": opportunity.objective,
        "experiment_metric_keys": list(config.experiment_metric_keys),
        "experiment_objective_keys": list(config.experiment_objective_keys),
        "experiment_guardrail_keys": list(config.experiment_guardrail_keys),
        "baseline_metrics": dict(baseline_metrics),
        "scales": dict(scales),
        "rho": config.rho,
        "comparison_tolerance": config.comparison_tolerance,
        "trace_transport_max_bytes": config.trace_transport_max_bytes,
        "candidate_process_workers": config.local_candidate_process_workers,
    }
    atomic_json(context_path, context)
    evaluator_wrapper = joint_root / "genetic_evaluator.py"
    evaluator_wrapper.write_text(
        "from dispatchevolve.workflows.dispatchevolve_v2.shared_genetic_evaluator import evaluate_with_context\n"
        f"CONTEXT_PATH = {str(context_path)!r}\n\n"
        "def evaluate(program_path):\n"
        "    return evaluate_with_context(program_path, CONTEXT_PATH)\n",
        encoding="utf-8",
    )
    task_context = render_task_context(
        opportunity,
        scene_description=str(
            scenario_summary.get("scene_description")
            or scenario_summary.get("llm_scene_summary")
            or "the validated business Scene represented by this local task"
        ),
        relative_policies=policies,
        rho=config.rho,
        baseline_metrics=baseline_metrics,
        scales=scales,
        experiment_metric_keys=config.experiment_metric_keys,
    )
    genetic_config = _genetic_config(
        config,
        task_context=task_context,
        iterations=iterations,
        source_size=len(baseline_genome),
        output_dir=joint_root,
        random_seed=config.random_seed + int(hashlib.sha256(
            f"{opportunity.opportunity_id}\0{'|'.join(policies)}".encode()
        ).hexdigest()[:8], 16),
    )
    genetic_config.replay_budget_path = str(evaluator.replay_budget_path) if evaluator.replay_budget_path else None
    checkpoint_root = joint_root / "optimizer" / "checkpoints"
    checkpoints = sorted(
        (path for path in checkpoint_root.glob("checkpoint_*") if is_complete_checkpoint(path)),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    try:
        result: EvolutionResult = run_evolution(
            initial_program,
            evaluator_wrapper,
            config=genetic_config,
            iterations=iterations,
            output_dir=str(joint_root / "optimizer"),
            cleanup=False,
            checkpoint_path=str(checkpoints[-1]) if checkpoints else None,
        )
    except Exception:
        if evaluator.replay_budget_path:
            from .replay_budget import ReplayBudget
            ReplayBudget(evaluator.replay_budget_path).check_stopped()
        raise
    candidates: list[LocalCandidate] = []
    attempts: list[Mapping[str, Any]] = []
    evaluated_count = 0
    for program in _all_checkpoint_programs(result, checkpoint_root):
        evolved = _program_genome_from_code(program.code)
        if evolved == baseline_genome:
            continue
        evolved_files = repository_files(evolved)
        changed_policies = tuple(
            relative for relative in policies
            if evolved_files[relative] != baseline_files[relative]
        )
        if not changed_policies:
            continue
        required_metric_fields = tuple(
            f"{prefix}_{name}"
            for prefix in ("raw", "normalized_improvement")
            for name in METRIC_KEYS
        )
        missing_metric_fields = tuple(
            field for field in required_metric_fields if field not in program.metrics
        )
        if missing_metric_fields:
            attempts.append({
                "program_id": program.id,
                "parent_id": program.parent_id,
                "generation": program.generation,
                "iteration_found": program.iteration_found,
                "policies": changed_policies,
                "evaluated": False,
                "accepted": False,
                "skip_reason": "legacy_checkpoint_missing_current_metrics",
                "missing_metric_fields": missing_metric_fields,
            })
            continue
        evaluated_count += 1
        delta = {name: float(program.metrics[f"normalized_improvement_{name}"]) for name in METRIC_KEYS}
        accepted = float(program.metrics.get("locally_feasible", 0.0)) == 1.0
        record = {
            "program_id": program.id,
            "parent_id": program.parent_id,
            "generation": program.generation,
            "iteration_found": program.iteration_found,
            "policies": changed_policies,
            "evaluated": True,
            "normalized_improvement": delta,
            "accepted": accepted,
        }
        program_artifact = root / "program_artifacts" / f"{program.id}.json"
        atomic_json(program_artifact, {
            "schema_version": "v2-local-evaluated-program-v1",
            "program": program.to_dict(),
            "changed_policies": changed_policies,
            "normalized_improvement": delta,
            "locally_feasible": accepted,
        })
        record["program_artifact"] = {
            "path": program_artifact.relative_to(round_dir).as_posix(),
            "sha256": file_hash(program_artifact),
        }
        attempts.append(record)
        if not accepted:
            continue
        key = candidate_key(evolved)
        engine_dir = joint_root / "candidates" / key / "engine"
        metrics = {name: float(program.metrics[f"raw_{name}"]) for name in METRIC_KEYS}
        candidate_id = codec.candidate_id(engine_dir)
        diffs = []
        for relative_policy in changed_policies:
            diffs.append("".join(difflib.unified_diff(
                baseline_files[relative_policy].splitlines(True),
                evolved_files[relative_policy].splitlines(True),
                fromfile=f"a/{relative_policy}", tofile=f"b/{relative_policy}", n=2,
            )))
        lineage = tuple(item for item in (program.parent_id,) if item)
        behavior_description = program.changes_description or opportunity.improvement_plan or "validated shared genetic mutation"
        introduction = build_candidate_introduction(
            candidate_id=candidate_id,
            opportunity=opportunity,
            scene_description=str(
                scenario_summary.get("scene_description")
                or scenario_summary.get("llm_scene_summary")
                or "the validated business Scene represented by this Local Candidate"
            ),
            policy_files=changed_policies,
            baseline_sources={item: baseline_files[item] for item in changed_policies},
            evolved_sources={item: evolved_files[item] for item in changed_policies},
            behavior_description=behavior_description,
            evaluated_batches=int(scenario_summary.get("matched_batches", 0)),
            scene_coverage=float(scenario_summary.get("coverage", 0.0)),
        )
        candidates.append(LocalCandidate(
            candidate_id,
            engine_dir,
            opportunity.opportunity_id,
            opportunity.scenario,
            opportunity.objective,
            changed_policies,
            metrics,
            delta,
            lineage,
            behavior_description + "\n" + "".join(diffs)[:4000],
            introduction,
        ))
    retained = sorted(
        _nondominated(
            candidates,
            config.comparison_tolerance,
            config.experiment_objective_keys,
        ),
        key=lambda item: (-item.oriented_delta[opportunity.objective], item.candidate_id),
    )[: config.budget.retained_candidates]
    best_delta = max(
        (item["normalized_improvement"] for item in attempts
         if "normalized_improvement" in item),
        key=lambda value: value[opportunity.objective],
        default=None,
    )
    outcome = LocalEvolutionResult(tuple(retained), bool(retained), evaluated_count, best_delta, tuple(attempts))
    atomic_json(root / "outcome.json", outcome)
    frame_path.unlink(missing_ok=True)
    if local_metric_rows_path is not None:
        local_metric_rows_path.unlink(missing_ok=True)
    prune_complete_checkpoints(checkpoint_root)
    return outcome

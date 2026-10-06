"""Subprocess evaluator for V2 Local Policy Evolution genetic Programs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from dispatchevolve.baselines.candidates import RepositoryGenomeCodec
from dispatchevolve.baselines.mutations import CandidateValidationError, parse_repository_patch

from .contracts import GUARDRAIL_KEYS, METRIC_KEYS, OBJECTIVE_KEYS
from .integration import copy_engine
from .local_evaluator import V2Evaluator, feasible, metric_values, oriented_delta


def program_genome(program_path: str | Path) -> str:
    """Extract the complete repository genome from a shared-genetic Program file."""
    source = Path(program_path).read_text(encoding="utf-8")
    source = source.replace("# EVOLVE-BLOCK-START\n", "", 1)
    marker = "# EVOLVE-BLOCK-END"
    if marker in source:
        source = source[: source.rfind(marker)]
    return source.rstrip() + "\n"


def candidate_key(genome: str) -> str:
    return hashlib.sha256(genome.encode("utf-8")).hexdigest()


def repository_files(genome: str) -> dict[str, str]:
    """Parse a complete repository genome into validated source files."""
    patch = parse_repository_patch(genome)
    if any(mutation.content is None for mutation in patch.mutations):
        raise CandidateValidationError("a complete repository genome cannot delete files")
    files = {
        mutation.path: mutation.content
        for mutation in patch.mutations
        if mutation.content is not None
    }
    if "engine.py" not in files:
        raise CandidateValidationError("complete repository genome omitted engine.py")
    for relative, source in files.items():
        compile(source, relative, "exec")
    return files


def _public_symbols(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    }


def evaluate_with_context(program_path: str | Path, context_path: str | Path) -> dict[str, Any]:
    """Materialize one joint repository Program and replay its complete engine."""
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    genome = program_genome(program_path)
    files = repository_files(genome)
    incumbent_dir = Path(context["incumbent_dir"])
    baseline_files = repository_files(RepositoryGenomeCodec().encode(incumbent_dir))
    if set(files) != set(baseline_files):
        missing = sorted(set(baseline_files) - set(files))
        added = sorted(set(files) - set(baseline_files))
        raise ValueError(f"genetic Program changed repository file membership: missing={missing}, added={added}")
    editable_policies = tuple(str(item) for item in context["relative_policies"])
    changed = tuple(path for path in sorted(files) if files[path] != baseline_files[path])
    forbidden = tuple(path for path in changed if path not in editable_policies)
    if forbidden:
        raise ValueError(f"genetic Program changed files outside related policies: {list(forbidden)}")
    for relative_policy in changed:
        if _public_symbols(files[relative_policy]) != _public_symbols(baseline_files[relative_policy]):
            raise ValueError(f"genetic Program changed the Policy public callable interface: {relative_policy}")
    key = candidate_key(genome)
    candidate_dir = Path(context["candidate_root"]) / key / "engine"
    if not candidate_dir.is_dir():
        copy_engine(incumbent_dir, candidate_dir)
        for relative_policy in changed:
            target = candidate_dir / relative_policy
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(files[relative_policy], encoding="utf-8")

    frame = pd.read_pickle(context["scene_frame_path"])
    metric_rows_path = context.get("metric_rows_path")
    metric_rows = pd.read_pickle(metric_rows_path) if metric_rows_path else None
    evaluator = V2Evaluator(
        runner_root=Path(context["runner_root"]),
        cache_root=Path(context["cache_root"]),
        protocol_hash=str(context["protocol_hash"]),
        backend=str(context["backend"]),
        replay_budget_path=context.get("replay_budget_path"),
        trace_transport_max_bytes=int(context.get("trace_transport_max_bytes", 268_435_456)),
        candidate_process_workers=int(context.get("candidate_process_workers", 1)),
    )
    result = evaluator.evaluate(
        candidate_dir,
        frame,
        output_dir=candidate_dir.parent / "evaluation",
        metric_rows=metric_rows,
        trace=False,
        metrics_only=True,
    )
    metrics = metric_values(result)
    delta = oriented_delta(metrics, context["baseline_metrics"], context["scales"])
    accepted = feasible(
        delta,
        target=str(context["objective"]),
        rho=float(context["rho"]),
        tolerance=float(context["comparison_tolerance"]),
        metric_keys=tuple(context.get("experiment_metric_keys", METRIC_KEYS)),
        objective_keys=tuple(context.get("experiment_objective_keys", OBJECTIVE_KEYS)),
        guardrail_keys=tuple(context.get("experiment_guardrail_keys", GUARDRAIL_KEYS)),
    )
    objective = str(context["objective"])
    tolerance = float(context["comparison_tolerance"])
    rho = float(context["rho"])
    experiment_metric_keys = tuple(context.get("experiment_metric_keys", METRIC_KEYS))
    violations = {
        name: (max(0.0, tolerance - float(value)) if name == objective
               else max(0.0, -rho - float(value)))
        for name in experiment_metric_keys
        for value in (delta[name],)
    }
    maximum_violation = max(violations.values(), default=0.0)
    violation_count = sum(value > 0.0 for value in violations.values())
    total_violation = sum(violations.values())
    if accepted:
        combined_score = float(delta[objective])
    else:
        # Keep every infeasible Program below every feasible Program while
        # exposing a stable near-feasibility gradient to the shared optimizer.
        # The dominant term is maximum single-constraint violation, followed by
        # violation count and total violation; target gain is only a final tie-break.
        combined_score = (
            -1.0 - maximum_violation
            - 1e-9 * violation_count
            - 1e-12 * math.log1p(total_violation)
            + 1e-15 * math.tanh(float(delta[objective]))
        )
    return {
        "combined_score": combined_score,
        "locally_feasible": float(accepted),
        "constraint_max_violation": float(maximum_violation),
        "constraint_violation_count": float(violation_count),
        "constraint_total_violation": float(total_violation),
        **{f"raw_{name}": float(value) for name, value in metrics.items()},
        **{f"normalized_improvement_{name}": float(value) for name, value in delta.items()},
    }

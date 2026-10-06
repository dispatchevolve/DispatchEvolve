"""Durable end-to-end DispatchEvolve V2 state machine."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml
from dispatchevolve.baselines.candidates import RepositoryGenomeCodec

from .action_ledger import ActionLedger
from .batch_query import (BatchQueryError, build_feature_catalog, feature_catalog_identity,
                          parse_batch_query, render_feature_catalog)
from .candidate_semantics import objective_catalog
from .combination_search import canonical_conflict_plan, propose_combination
from .config import ModelConfig, V2Config
from .contracts import (ArchiveEntry, CriticAssessment, LocalCandidate, METRIC_DIRECTIONS,
                        METRIC_KEYS, OBJECTIVE_KEYS, Opportunity, RunState,
                        ScenarioPredicate, ScenarioRule, content_hash)
from .dpo_dataset import raw_task_records
from .integration import (copy_engine, engine_frozen_scenarios, engine_policy_callables,
                          engine_result_fields, integrate, instrument_variant_call_probe,
                          policy_call_symbols, variant_policy_path)
from .engine_instrumentation import instrument_engine
from .format_repair import ResponseFormatError, format_repair_prompt
from .llm import V2LLM
from .local_evaluator import (OBJECTIVE_SEMANTICS, V2Evaluator, feasible,
                              local_metric_scales, metric_values, oriented_delta,
                              passes_global_acceptance)
from .shared_genetic_adapter import LocalEvolutionResult, evolve_opportunity_with_shared_genetic
from .memory_store import SceneMemoryStore
from .pareto_archive import (
    reference_reasons,
    select_incumbent,
    select_references,
    update_archive,
)
from .prompt_store import PromptStore, RenderedPrompt
from .prompt_ids import PromptIdRegistry
from .protocol import (atomic_gzip_json, atomic_json, file_hash, protocol_document,
                       protocol_hash, tree_manifest)
from .relation_graph import build_relation_graph
from .scenario_predicate import coverage_ratio, parse_predicate, scenario_evaluation_scope, scenario_target_mask
from .trace_analysis import normalize_traces, prompt_trace_summary, scenario_evidence_ids, summarize_scenario
from .policy_trace import render_policy_trace_text


_SCENARIO_QUERY = re.compile(
    r"\A## Query\n+([^\n]+)\n+## Discovery Purpose\n+([\s\S]+)\Z"
)
_OPPORTUNITY_DISCOVERY = re.compile(
    r"\A## Scene Summary\n+([^\n]+)\n+## Opportunity Status\n+"
    r"(OPPORTUNITY_FOUND|NO_OPPORTUNITY)\n+## Improvement Opportunities\n+([\s\S]+)\Z"
)
_CRITIC = re.compile(r"\A## Decision\n+(ACCEPT|REJECT)\n+## Confidence\n+([01](?:\.\d+)?)\n+## Reason\n+([\s\S]+)\Z")


def _evolve_opportunity_worker(job: Mapping[str, Any]) -> tuple[str, LocalEvolutionResult, int]:
    """Run exactly one Opportunity in a fresh process-owned evaluator context."""
    opportunity = job["opportunity"]
    worker_root = Path(job["round_dir"]) / "local_policy_evolution" / opportunity.opportunity_id
    evaluator = V2Evaluator(
        runner_root=worker_root / "worker_runner",
        cache_root=Path(job["cache_root"]),
        protocol_hash=str(job["protocol_hash"]),
        backend=str(job["backend"]),
        replay_budget_path=(job["config"].output_dir / job["config"].run_id / "replay_budget.sqlite"
                            if job["config"].mode != "dpo_data_collection" else None),
        wall_timeout_seconds=float(job["wall_timeout_seconds"]),
        trace_transport_max_bytes=int(job["trace_transport_max_bytes"]),
        candidate_process_workers=int(job["candidate_process_workers"]),
    )
    result = evolve_opportunity_with_shared_genetic(
        config=job["config"], opportunity=opportunity,
        scenario_summary=job["scenario_summary"],
        incumbent_dir=Path(job["incumbent_dir"]),
        scene_frame_path=Path(job["scene_frame_path"]),
        metric_rows_path=(Path(job["metric_rows_path"])
                          if job.get("metric_rows_path") else None),
        baseline_metrics=job["baseline_metrics"],
        scales=job["scales"], evaluator=evaluator, round_dir=Path(job["round_dir"]),
    )
    return opportunity.opportunity_id, result, os.getpid()


def _run_opportunity_process_pool(
    jobs: list[Mapping[str, Any]], max_workers: int,
    *, worker: Callable[[Mapping[str, Any]], Any] = _evolve_opportunity_worker,
) -> list[Any]:
    """Use spawn and one task per child so Opportunity state cannot leak."""
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=context, max_tasks_per_child=1,
    ) as executor:
        return list(executor.map(worker, jobs))


def _atomic_pickle(frame: pd.DataFrame, path: Path) -> None:
    """Replace a shared process input without mutating existing hard links."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        frame.to_pickle(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _combination_search_space_exhausted(
    active_candidate_count: int,
    seen_proposals: set[str],
) -> bool:
    """Detect the only safely enumerable exhausted composition space.

    One active candidate has exactly one non-empty selection and cannot require
    a precedence edge. Once that valid proposal identity has been seen, another
    LLM call can only repeat it.
    """
    return active_candidate_count == 1 and bool(seen_proposals)


def _persist_opportunity_parallelism(
    path: Path,
    *,
    configured_max_workers: int,
    pending_opportunities: int,
    effective_max_workers: int,
    worker_pids: Mapping[str, int],
    resumed_outcomes: int,
) -> dict[str, Any]:
    """Preserve the original worker evidence on an outcome-only resume."""
    if pending_opportunities == 0 and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    document = {
        "configured_max_workers": configured_max_workers,
        "pending_opportunities": pending_opportunities,
        "effective_max_workers": effective_max_workers,
        "executor": "spawn_process_per_opportunity",
        "worker_pids": dict(worker_pids),
        "distinct_worker_processes": len(set(worker_pids.values())),
        "resumed_outcomes": resumed_outcomes,
    }
    atomic_json(path, document)
    return document


def _unique_opportunities(opportunities: list[Opportunity]) -> list[Opportunity]:
    """Keep each LLM-proposed Opportunity once while preserving discovery order."""
    return list({item.opportunity_id: item for item in opportunities}.values())


def _select_resumed_opportunities(
    accepted_records: list[dict[str, Any]], round_dir: Path, maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce an interrupted round without discarding completed successful work."""
    successful: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    completed_unsuccessful: list[str] = []
    for record in accepted_records:
        opportunity_id = str(record["opportunity"]["opportunity_id"])
        outcome_path = round_dir / "local_policy_evolution" / opportunity_id / "outcome.json"
        if not outcome_path.is_file():
            pending.append(record)
            continue
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        if outcome.get("candidates"):
            successful.append(record)
        else:
            completed_unsuccessful.append(opportunity_id)
    selected = (successful + pending)[:maximum]
    selected_ids = [str(item["opportunity"]["opportunity_id"]) for item in selected]
    return selected, {
        "strategy": "completed_successes_then_pending",
        "configured_limit": maximum,
        "successful_outcomes_available": [
            str(item["opportunity"]["opportunity_id"]) for item in successful
        ],
        "pending_outcomes_available": [
            str(item["opportunity"]["opportunity_id"]) for item in pending
        ],
        "completed_unsuccessful_outcomes": completed_unsuccessful,
        "selected_opportunity_ids": selected_ids,
        "skipped_opportunity_ids": [
            str(item["opportunity"]["opportunity_id"])
            for item in accepted_records
            if str(item["opportunity"]["opportunity_id"]) not in set(selected_ids)
        ],
    }


def _load_debug_frame(config: V2Config) -> pd.DataFrame:
    data_identity = file_hash(config.data_path)[:16]
    cache = config.cache_root / "debug_frames" / f"{config.data_path.stem}__{data_identity}__{config.debug_max_batches or 'all'}.pkl"
    if config.persist_debug_frames and cache.is_file():
        return pd.read_pickle(cache)
    selected: list[str] = []
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(config.data_path, chunksize=config.csv_chunk_size, low_memory=False):
        if config.debug_max_batches is None:
            pieces.append(chunk)
            continue
        keys = chunk["batch_id"].astype("string").fillna("<NA>").astype(str)
        for key in keys.unique():
            if len(selected) >= config.debug_max_batches:
                break
            if key not in selected:
                selected.append(key)
        kept = chunk.loc[keys.isin(selected)]
        if not kept.empty:
            pieces.append(kept.copy())
    if not pieces:
        raise ValueError("dataset produced no rows")
    frame = pd.concat(pieces, ignore_index=True)
    if config.debug_max_batches is not None:
        frame = frame.loc[frame["batch_id"].astype(str).isin(set(selected))].reset_index(drop=True)
    if config.persist_debug_frames:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache)
    return frame


def _archive_from_documents(items: list[Mapping[str, Any]]) -> list[ArchiveEntry]:
    return [ArchiveEntry(
                item["engine_id"], Path(item["engine_dir"]), item["metrics"], item["oriented_delta"],
                item["round_index"], tuple(item.get("candidate_ids", ())),
                float(item.get("online_probability", 1.0)), int(item.get("change_size", 0)),
                tuple(item.get("candidate_summaries", ())),
                item.get("online_eligibility"), item.get("online_decision_source"),
                bool(item.get("online_llm_called", False)),
                str(item.get("conflict_resolution_plan", "NONE")),
                tuple(item.get("precedence_edges", ())),
            )
            for item in items]


def _archive_from_state(state: RunState) -> list[ArchiveEntry]:
    return _archive_from_documents(state.archive)


def _local_candidate_from_dict(item: Mapping[str, Any]) -> LocalCandidate:
    scenario = ScenarioPredicate(**item["scenario"])
    return LocalCandidate(item["candidate_id"], Path(item["engine_dir"]), item["opportunity_id"], scenario,
                          item["objective"], tuple(item["policy_files"]), item["metrics"], item["oriented_delta"],
                          tuple(item["lineage"]), item["diff_summary"], dict(item.get("introduction", {})))


def _relation_edge_from_dict(item: Mapping[str, Any]):
    from .contracts import RelationEdge
    return RelationEdge(
        str(item["left"]), str(item["right"]), str(item["relation"]),
        float(item["overlap"]), str(item["reason"]), dict(item.get("evidence", {})),
        item.get("model"), item.get("prompt_hash"),
    )


def _historical_relation_edges(run_root: Path) -> tuple[Any, ...]:
    """Load the latest durable classification for every historical pair."""
    by_pair: dict[tuple[str, str], Any] = {}
    for path in sorted(run_root.glob("round_*/relation_graph.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for item in document.get("edges", ()):
            edge = _relation_edge_from_dict(item)
            by_pair[tuple(sorted((edge.left, edge.right)))] = edge
    return tuple(by_pair[key] for key in sorted(by_pair))


def _opportunity_from_dict(item: Mapping[str, Any]) -> Opportunity:
    return Opportunity(str(item["opportunity_id"]), ScenarioPredicate(**item["scenario"]), str(item["objective"]),
                       tuple(item["related_policies"]), str(item["rationale"]),
                       item.get("improvement_plan"), item.get("evidence_basis"))


@dataclass(frozen=True)
class OpportunityDiscovery:
    scene_summary: str
    status: str
    opportunities: tuple[Opportunity, ...]


@dataclass(frozen=True)
class ScenarioQueryAttempt:
    scenario_rule: ScenarioRule | None
    raw_query: str | None
    failure: Mapping[str, Any] | None


def _table_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "/").replace("\n", " ").strip()


def _objective_performance_table(
    summary: Any, metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    summary_value = asdict(summary) if hasattr(summary, "__dataclass_fields__") else dict(summary)
    statistics = dict(summary_value.get("policy_statistics", {}) or {})
    evidence = dict(statistics.get("performance_evidence") or {})
    scene_evidence = dict(evidence.get("scene") or {})
    global_evidence = dict(evidence.get("global") or {})
    scene_counts = dict(scene_evidence.get("counts") or {})
    global_counts = dict(global_evidence.get("counts") or {})
    batch_quantiles = dict(scene_evidence.get("batch_quantiles") or {})
    include_counts = any(
        scene_counts.get(key) is not None or global_counts.get(key) is not None
        for key in metric_keys
    )
    include_quantiles = any(
        any(dict(batch_quantiles.get(key) or {}).get(name) is not None
            for name in ("p10", "p50", "p90"))
        for key in metric_keys
    )
    header = ["metric", "scene_value", "global_value"]
    if include_counts:
        header.extend(("scene_contributors", "global_contributors"))
    if include_quantiles:
        header.extend(("scene_batch_p10", "scene_batch_p50", "scene_batch_p90"))
    rows = [" | ".join(header)]
    for key in metric_keys:
        row = [
            key,
            f"{float(summary_value['metrics'][key]):.12g}",
            f"{float(summary_value['global_metrics'][key]):.12g}",
        ]
        if include_counts:
            row.extend(
                "NA" if value is None else str(value)
                for value in (scene_counts.get(key), global_counts.get(key))
            )
        if include_quantiles:
            quantiles = dict(batch_quantiles.get(key) or {})
            row.extend(
                "NA" if quantiles.get(name) is None
                else f"{float(quantiles[name]):.12g}"
                for name in ("p10", "p50", "p90")
            )
        rows.append(" | ".join(row))
    return "\n".join(rows)


def _history_status(record: Mapping[str, Any]) -> tuple[str, str, str]:
    assessment = record.get("assessment") or {}
    critic = str(assessment.get("decision") or "PENDING")
    reason = str(assessment.get("reason") or "")
    local_outcome = record.get("local_outcome") or {}
    if record.get("evolution_result"):
        evolution = str(record["evolution_result"])
    elif "success" in local_outcome:
        evolution = "SUCCEEDED" if bool(local_outcome.get("success")) else "FAILED"
    elif critic == "REJECT":
        evolution = "NOT_RUN"
    else:
        evolution = "PENDING"
    return critic, reason, evolution


def _history_table(
    memory: SceneMemoryStore, *, review: bool, exclude_opportunity_id: str | None = None,
    exclude_scene_hash: str | None = None,
    limit: int = 100,
) -> str:
    rows: list[str] = []
    for record in memory.lifecycle_view(limit=limit):
        if exclude_scene_hash and record.get("scene_hash") == exclude_scene_hash:
            continue
        opportunity = record.get("opportunity") or {}
        if exclude_opportunity_id and opportunity.get("opportunity_id") == exclude_opportunity_id:
            continue
        scenario = opportunity.get("scenario") or {}
        query = scenario.get("canonical") or record.get("query") or ""
        summary = record.get("llm_scene_summary") or ""
        objective = opportunity.get("objective") or "NONE"
        scope = ";".join(opportunity.get("related_policies") or ()) or "NONE"
        critic, reason, evolution = _history_status(record)
        values = [query, summary, objective, scope, critic]
        if review:
            values.append(reason)
        values.append(evolution)
        rows.append(" | ".join(_table_cell(value) for value in values))
    return "\n".join(rows) if rows else "NONE"


def _scene_discovery_memory_table(memory: SceneMemoryStore, *, limit: int = 100) -> str:
    rows = ["query | scene_summary | critic_accepted | evolution_succeeded"]
    for record in memory.lifecycle_view(limit=limit):
        opportunity = record.get("opportunity") or {}
        scenario = opportunity.get("scenario") or record.get("scenario_rule", {}).get("scenario") or {}
        query = scenario.get("canonical") or record.get("query") or "NA"
        summary = record.get("llm_scene_summary") or "NA"
        assessment = record.get("assessment") or {}
        local = record.get("local_outcome") or {}
        critic = "NA" if not assessment else str(assessment.get("decision") == "ACCEPT").lower()
        evolved = "NA" if "success" not in local else str(bool(local.get("success"))).lower()
        rows.append(" | ".join(_table_cell(value) for value in (query, summary, critic, evolved)))
    return "\n".join(rows)


def _parse_opportunity_discovery(
    response: str, *, scenario: ScenarioPredicate, allowed_policies: set[str],
    identity: Mapping[str, Any],
    allowed_objectives: set[str] | None = None,
) -> OpportunityDiscovery | None:
    match = _OPPORTUNITY_DISCOVERY.fullmatch(response.strip())
    if not match:
        raise ResponseFormatError("Opportunity Discovery response violates the section contract")
    scene_summary = match.group(1).strip()
    status = match.group(2)
    lines = [line.strip() for line in match.group(3).splitlines() if line.strip()]
    if not lines:
        raise ResponseFormatError("Opportunity Discovery table is empty")
    header = tuple(cell.strip().lower() for cell in lines.pop(0).strip("|").split("|"))
    if header != ("objective", "related_policies", "improvement_plan", "evidence_basis"):
        raise ResponseFormatError("Opportunity Discovery table header is invalid")
    if lines and all(set(cell.strip()) <= {"-", ":"} for cell in lines[0].strip("|").split("|")):
        lines.pop(0)
    parsed: list[tuple[str, tuple[str, ...], str, str]] = []
    for line in lines:
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        if len(cells) != 4 or any(not cell for cell in cells):
            raise ResponseFormatError("Opportunity Discovery table row must contain four cells")
        objective, policy_text, improvement_plan, evidence_basis = cells
        related = (() if policy_text == "NONE" else tuple(dict.fromkeys(
            item.strip() for item in policy_text.split(";") if item.strip()
        )))
        parsed.append((objective, related, improvement_plan, evidence_basis))
    if status == "NO_OPPORTUNITY":
        if len(parsed) != 1 or parsed[0][:3] != ("NONE", (), "NONE"):
            raise ResponseFormatError("NO_OPPORTUNITY requires the frozen NONE row")
        return OpportunityDiscovery(scene_summary, status, ())
    if not parsed:
        raise ResponseFormatError("OPPORTUNITY_FOUND requires at least one table row")
    opportunities: list[Opportunity] = []
    semantic_keys: set[tuple[str, tuple[str, ...], str]] = set()
    for index, (objective, related, improvement_plan, evidence_basis) in enumerate(parsed):
        objectives = (
            set(OBJECTIVE_KEYS)
            if allowed_objectives is None
            else allowed_objectives
        )
        if objective not in objectives or not related or not set(related) <= allowed_policies:
            raise ValueError("Opportunity Discovery names an invalid Objective or Policy")
        semantic_key = (objective, tuple(sorted(related)), " ".join(improvement_plan.lower().split()))
        if semantic_key in semantic_keys:
            raise ValueError("Opportunity Discovery repeats the same semantic Opportunity")
        semantic_keys.add(semantic_key)
        opportunity_id = content_hash({**identity, "row": index, "objective": objective,
                                       "policies": related, "plan": improvement_plan})[:16]
        rationale = f"{improvement_plan} Evidence: {evidence_basis}"
        opportunities.append(Opportunity(
            opportunity_id, scenario, objective, related, rationale,
            improvement_plan, evidence_basis,
        ))
    return OpportunityDiscovery(scene_summary, status, tuple(opportunities))






def _attempt_opportunity_entries(
    record: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]]:
    """Read both legacy one-Opportunity and current multi-Opportunity artifacts."""
    if record.get("opportunity"):
        return [(record["opportunity"], record.get("assessment"), record.get("critic_prompt"))]
    assessments = {
        str(item.get("opportunity_id")): item for item in record.get("assessments", ())
    }
    prompts = record.get("critic_prompts") or {}
    return [
        (item, assessments.get(str(item.get("opportunity_id"))),
         prompts.get(str(item.get("opportunity_id"))))
        for item in record.get("opportunities", ())
    ]


def _select_accepted_for_scene(
    items: list[tuple[Opportunity, CriticAssessment, dict[str, Any]]], *,
    maximum: int, remaining: int, priority_objective: str | None = None,
) -> list[tuple[Opportunity, CriticAssessment, dict[str, Any]]]:
    """Prioritize the prior round's weakest metric, then Critic confidence."""
    ordered = sorted(items, key=lambda item: (
        0 if priority_objective and item[0].objective == priority_objective else 1,
        -item[1].confidence,
        item[0].opportunity_id,
    ))
    return ordered[:min(maximum, max(0, remaining))]


def _next_priority_objective(
    feedback: list[Mapping[str, Any]], *, selected_engine_id: str,
    incumbent_updated: bool, pareto_engine_ids: set[str],
    priority_objective_keys: tuple[str, ...],
) -> tuple[str | None, dict[str, Any]]:
    """Find the weakest priority metric of the round's primary reference."""
    valid = [
        item for item in feedback
        if item.get("evaluated") is True and not item.get("runtime_failure")
        and isinstance(item.get("oriented_delta"), Mapping)
        and all(key in item["oriented_delta"] for key in METRIC_KEYS)
    ]
    if not valid:
        return None, {}
    selected_incumbent = next(
        (item for item in valid if str(item.get("engine_id")) == selected_engine_id),
        None,
    )
    if incumbent_updated and selected_incumbent is not None:
        source = selected_incumbent
        source_rule = "new_incumbent"
    else:
        pareto = [
            item for item in valid
            if str(item.get("engine_id")) in pareto_engine_ids
        ] or valid
        source = max(pareto, key=lambda item: (
            sum(max(float(item["oriented_delta"][key]), 0.0) for key in OBJECTIVE_KEYS),
            sum(float(item["oriented_delta"][key]) for key in OBJECTIVE_KEYS),
            str(item.get("engine_id", "")),
        ))
        source_rule = "primary_pareto_reference_by_positive_improvement_strength"
    delta = {key: float(source["oriented_delta"][key]) for key in METRIC_KEYS}
    objective = min(
        priority_objective_keys,
        key=lambda key: (delta[key], priority_objective_keys.index(key)),
    )
    return objective, {
        "source": source_rule,
        "round_engine_id": source.get("engine_id"),
        "proposal_attempt": source.get("proposal_attempt"),
        "oriented_delta": delta,
        "priority_objective_keys": priority_objective_keys,
        "weakest_value": delta[objective],
        "positive_improvement_strength": sum(
            max(delta[key], 0.0) for key in OBJECTIVE_KEYS
        ),
    }


def _local_result_from_dict(item: Mapping[str, Any]) -> LocalEvolutionResult:
    return LocalEvolutionResult(
        tuple(_local_candidate_from_dict(candidate) for candidate in item.get("candidates", ())),
        bool(item.get("success")), int(item.get("evaluated_count", 0)),
        item.get("best_delta"), tuple(item.get("attempts", ())),
    )


def _append_jsonl(path: Path, value: Any) -> None:
    from .contracts import jsonable
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl_once(path: Path, value: Any) -> bool:
    target = content_hash(value)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and content_hash(json.loads(line)) == target:
                return False
    _append_jsonl(path, value)
    return True


class DispatchEvolveV2Orchestrator:
    def __init__(self, config: V2Config):
        self.config = config
        self.protocol_hash = protocol_hash(config)
        self.run_root = config.output_dir / config.run_id
        from .replay_budget import ReplayBudget
        self.replay_budget = (ReplayBudget(self.run_root / "replay_budget.sqlite")
                              if config.mode != "dpo_data_collection" else None)
        self.state_path = self.run_root / "state" / "run_state.json"
        self.prompt_ids = PromptIdRegistry(
            self.run_root / "manifests" / "prompt_ids.json"
        )
        self.prompts = PromptStore(config.prompt_json)
        log_dir = config.log_root / config.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"dispatchevolve.v2.{config.run_id}")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_dir / "workflow.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler); self.logger.setLevel(logging.INFO)
        self.llm = V2LLM(config.model)
        self.critic_llm = V2LLM(config.critic_model) if config.critic_model else self.llm
        self.ledger = ActionLedger(self.run_root)
        self._counter_lock = threading.Lock()
        self.evaluator = V2Evaluator(runner_root=self.run_root / "candidate_runner",
                                     cache_root=config.cache_root / "evaluations",
                                     protocol_hash=self.protocol_hash, backend=config.backend,
                                     replay_budget_path=self.replay_budget.path if self.replay_budget else None,
                                     trace_transport_max_bytes=config.trace_transport_max_bytes,
                                     candidate_process_workers=config.candidate_process_workers)
        self.local_candidate_evaluator = V2Evaluator(
            runner_root=self.run_root / "local_candidate_runner",
            cache_root=config.cache_root / "evaluations",
            protocol_hash=self.protocol_hash,
            backend=config.backend,
            trace_transport_max_bytes=config.trace_transport_max_bytes,
            candidate_process_workers=config.local_candidate_process_workers,
            replay_budget_path=self.replay_budget.path if self.replay_budget else None,
        )

    def _register_query(self, scenario: ScenarioPredicate, round_index: int, attempt: int) -> bool:
        """Permanently reserve a canonical Query for this run and its resumes."""
        path = self.run_root / "query_registry" / f"{scenario.predicate_hash}.json"
        action = {"round": round_index, "attempt": attempt}
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            return existing.get("first_action") == action
        atomic_json(path, {
            "query_hash": scenario.predicate_hash,
            "canonical_query": scenario.canonical,
            "first_action": action,
        })
        return True

    def _register_raw_query(self, query: str, round_index: int, attempt: int) -> bool:
        """Reserve even an invalid Query so an unchanged retry is permanently duplicate."""
        normalized = " ".join(query.strip().split())
        identity = content_hash({"raw_query": normalized})
        path = self.run_root / "query_registry" / f"raw_{identity}.json"
        action = {"round": round_index, "attempt": attempt}
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            return existing.get("first_action") == action
        atomic_json(path, {
            "raw_query_hash": identity, "raw_query": normalized,
            "first_action": action,
        })
        return True

    def _candidate_library(self, *, through_round: int | None = None) -> list[LocalCandidate]:
        path = self.run_root / "candidate_library" / "admissions.jsonl"
        if not path.is_file():
            return []
        ordered: dict[str, LocalCandidate] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            admission_round = int(record.get("round", 0))
            if through_round is not None and admission_round > through_round:
                continue
            candidate = _local_candidate_from_dict(record["candidate"])
            if not candidate.engine_dir.is_dir():
                raise FileNotFoundError(
                    f"candidate library artifact is missing: {candidate.engine_dir}"
                )
            ordered.setdefault(candidate.candidate_id, candidate)
        return list(ordered.values())

    def _register_composition(
        self, proposal: Any, *, fixed_e0: Path, round_index: int, attempt: int,
    ) -> str | None:
        identity = content_hash({
            "protocol_hash": self.protocol_hash,
            "fixed_e0": RepositoryGenomeCodec().candidate_id(fixed_e0),
            "selected_candidate_ids": sorted(proposal.candidate_ids),
            "conflict_resolution_plan": canonical_conflict_plan(
                proposal.priority_plan.precedence_edges
            ),
        })
        path = self.run_root / "composition_identity_registry" / f"{identity}.json"
        action = {"round": round_index, "attempt": attempt}
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            self.prompt_ids.get("attempt", identity)
            return identity if existing.get("first_action") == action else None
        atomic_json(path, {
            "composition_identity": identity, "first_action": action,
            "proposal_id": proposal.proposal_id,
            "selected_candidate_ids": sorted(proposal.candidate_ids),
            "incremental_candidate_ids": sorted(proposal.incremental_candidate_ids),
            "conflict_resolution_plan": canonical_conflict_plan(
                proposal.priority_plan.precedence_edges
            ),
            "precedence_edges": proposal.priority_plan.precedence_edges,
            "attempt_status": "reserved",
        })
        self.prompt_ids.get("attempt", identity)
        return identity

    def _update_composition_status(self, identity: str, status: str, **details: Any) -> None:
        allowed = {
            "reserved", "validation_rejected", "materialization_failed", "runtime_failed",
            "evaluated", "evaluated_infeasible", "evaluated_feasible",
        }
        if status not in allowed:
            raise ValueError(f"invalid Composition attempt status: {status}")
        path = self.run_root / "composition_identity_registry" / f"{identity}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Composition identity is not reserved: {identity}")
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(attempt_status=status, **details)
        atomic_json(path, document)

    def _composition_registry(self) -> list[Mapping[str, Any]]:
        root = self.run_root / "composition_identity_registry"
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in root.glob("*.json")
        ]
        records.sort(key=lambda item: (
            int(item.get("first_action", {}).get("round", 0)),
            int(item.get("first_action", {}).get("attempt", 0)),
            str(item.get("composition_identity")),
        ))
        for item in records:
            self.prompt_ids.get("attempt", str(item["composition_identity"]))
        return records

    def _candidate_prompt_ids(
        self, candidates: list[LocalCandidate] | tuple[LocalCandidate, ...],
    ) -> dict[str, str]:
        return self.prompt_ids.get_many(
            "candidate", tuple(item.candidate_id for item in candidates),
        )

    def _composition_prompt_maps(
        self,
        *,
        candidates: list[LocalCandidate],
        references: list[Mapping[str, Any]],
        composition_registry: list[Mapping[str, Any]],
        feedback: list[Mapping[str, Any]],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        candidate_ids = list(dict.fromkeys([
            *(item.candidate_id for item in candidates),
            *(
                str(identifier)
                for reference in references
                for identifier in reference.get("candidate_ids", ())
            ),
            *(
                str(identifier)
                for record in composition_registry
                for identifier in record.get("selected_candidate_ids", ())
            ),
            *(
                str(identifier)
                for record in feedback
                for identifier in record.get("selected_candidate_ids", ())
            ),
        ]))
        reference_ids = [
            str(item["engine_id"])
            for item in references if item.get("engine_id") is not None
        ]
        attempt_ids = list(dict.fromkeys([
            *(
                str(item["composition_identity"])
                for item in composition_registry
                if item.get("composition_identity") is not None
            ),
            *(
                str(item.get("composition_identity") or item.get("proposal"))
                for item in feedback
                if item.get("composition_identity") is not None
                or item.get("proposal") is not None
            ),
        ]))
        return (
            self.prompt_ids.get_many("candidate", tuple(candidate_ids)),
            self.prompt_ids.get_many("reference", tuple(reference_ids)),
            self.prompt_ids.get_many("attempt", tuple(attempt_ids)),
        )

    def _reference_documents(
        self,
        archive: list[ArchiveEntry],
        references: list[ArchiveEntry] | tuple[ArchiveEntry, ...],
    ) -> list[dict[str, Any]]:
        selected = list(references)
        reasons = reference_reasons(
            archive,
            selected,
            reference=-self.config.rho,
            objective_keys=self.config.experiment_objective_keys,
        )
        return [
            {**asdict(item), "reference_reason": reasons[item.engine_id]}
            for item in selected
        ]

    @staticmethod
    def _candidate_archive_summary(candidate: LocalCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "scenario": candidate.scenario.canonical,
            "objective": candidate.objective,
            "policy_files": candidate.policy_files,
            "local_delta": candidate.oriented_delta,
        }

    def _archive_entry_from_feedback(
        self,
        state: RunState,
        *,
        round_index: int,
        round_dir: Path,
        feedback: Mapping[str, Any],
        candidate_map: Mapping[str, LocalCandidate],
    ) -> ArchiveEntry | None:
        if feedback.get("evaluated") is not True or feedback.get("runtime_failure"):
            return None
        attempt = feedback.get("proposal_attempt")
        engine_id = feedback.get("engine_id")
        if attempt is None or not engine_id:
            return None
        proposal_dir = round_dir / "combinations" / f"proposal_{int(attempt):03d}"
        engine_dir = Path(feedback.get("engine_dir") or proposal_dir / "engine")
        if not engine_dir.is_dir():
            return None
        metrics_path = proposal_dir / "evaluation" / "metrics.json"
        raw_metrics = feedback.get("metrics")
        if isinstance(raw_metrics, Mapping) and all(key in raw_metrics for key in METRIC_KEYS):
            metrics = {key: float(raw_metrics[key]) for key in METRIC_KEYS}
            delta = oriented_delta(metrics, state.baseline_metrics, state.scales)
        elif metrics_path.is_file():
            metrics = metric_values(json.loads(metrics_path.read_text(encoding="utf-8")))
            delta = oriented_delta(metrics, state.baseline_metrics, state.scales)
        else:
            # Legacy and test evaluators may have persisted only the normalized
            # result in durable round feedback.
            raw_delta = feedback.get("oriented_delta")
            if not isinstance(raw_delta, Mapping) or any(
                key not in raw_delta for key in METRIC_KEYS
            ):
                return None
            delta = {key: float(raw_delta[key]) for key in METRIC_KEYS}
            metrics = {
                key: float(state.baseline_metrics[key])
                + METRIC_DIRECTIONS[key] * delta[key] * float(state.scales[key])
                for key in METRIC_KEYS
            }
        candidate_ids = tuple(map(str, feedback.get("selected_candidate_ids", ())))
        selected = [candidate_map[item] for item in candidate_ids if item in candidate_map]
        return ArchiveEntry(
            str(engine_id), engine_dir, metrics, delta, round_index, candidate_ids,
            conflict_resolution_plan=str(feedback.get("conflict_resolution_plan") or "NONE"),
            precedence_edges=tuple(feedback.get("precedence_edges", ())),
            change_size=sum(item.diff_summary.count("\n") for item in selected),
            candidate_summaries=tuple(self._candidate_archive_summary(item) for item in selected),
        )

    def _reconcile_pareto_archive(self, state: RunState) -> None:
        """Rebuild the non-dominated archive from every successful full-D evaluation.

        Restore feasible entries from completed rounds and the current durable
        combination checkpoint before selecting an engine.
        """
        if not state.baseline_metrics:
            return
        zero = {key: 0.0 for key in METRIC_KEYS}
        e0_dir = self.run_root / "engines" / "e0"
        e0_id = RepositoryGenomeCodec().candidate_id(e0_dir)
        archive = [ArchiveEntry(e0_id, e0_dir, state.baseline_metrics, zero, 0)]
        previous_archive_ids = {
            str(item["engine_id"]) for item in state.archive if item.get("engine_id")
        }
        candidate_map = {item.candidate_id: item for item in self._candidate_library()}
        considered = 0
        replay_through_round = (
            state.archive_replay_through_round
            if state.archive_replay_session is not None else 0
        )
        for round_index in range(replay_through_round + 1, state.completed_rounds + 2):
            round_dir = self.run_root / f"round_{round_index:03d}"
            result_path = round_dir / "round_result.json"
            if not result_path.is_file():
                result_path = round_dir / "state" / "combination_search.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for feedback in result.get("global_feedback", ()):
                entry = self._archive_entry_from_feedback(
                    state, round_index=round_index, round_dir=round_dir,
                    feedback=feedback, candidate_map=candidate_map,
                )
                if entry is None:
                    continue
                archive, update = update_archive(
                    archive, entry, tolerance=self.config.comparison_tolerance,
                    objective_keys=self.config.experiment_objective_keys,
                    rho=self.config.rho, guardrail_keys=self.config.experiment_guardrail_keys,
                )
                acceptance_passed = feedback.get("acceptance_passed")
                if acceptance_passed is None:
                    acceptance_passed = passes_global_acceptance(
                        entry.oriented_delta, rho=self.config.rho,
                        tolerance=self.config.comparison_tolerance,
                        objective_keys=self.config.experiment_objective_keys,
                        guardrail_keys=self.config.experiment_guardrail_keys,
                    )
                acceptance_passed = bool(acceptance_passed)
                _append_jsonl_once(self.run_root / "pareto_archive" / "events.jsonl", {
                    **update,
                    "round": round_index,
                    "proposal_attempt": int(feedback["proposal_attempt"]),
                    "acceptance_passed": acceptance_passed,
                    "source": "full_search_evaluation_reconciliation",
                })
                considered += 1
        replay_results = (
            sorted(
                (self.run_root / "pareto_recombination" / state.archive_replay_session)
                .glob("round_*/round_result.json")
            )
            if state.archive_replay_session is not None
            else sorted(
                (self.run_root / "pareto_recombination")
                .glob("*/round_*/round_result.json")
            )
        )
        for result_path in replay_results:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            round_index = int(result["round"])
            if state.archive_replay_session is not None and round_index > replay_through_round:
                continue
            candidate_map = {item.candidate_id: item for item in self._candidate_library()}
            for feedback in result.get("global_feedback", ()):
                entry = self._archive_entry_from_feedback(
                    state, round_index=round_index, round_dir=result_path.parent,
                    feedback=feedback, candidate_map=candidate_map,
                )
                if entry is None:
                    continue
                archive, update = update_archive(
                    archive, entry, tolerance=self.config.comparison_tolerance,
                    objective_keys=self.config.experiment_objective_keys,
                    rho=self.config.rho, guardrail_keys=self.config.experiment_guardrail_keys,
                )
                acceptance_passed = feedback.get("acceptance_passed")
                if acceptance_passed is None:
                    acceptance_passed = passes_global_acceptance(
                        entry.oriented_delta, rho=self.config.rho,
                        tolerance=self.config.comparison_tolerance,
                        objective_keys=self.config.experiment_objective_keys,
                        guardrail_keys=self.config.experiment_guardrail_keys,
                    )
                _append_jsonl_once(self.run_root / "pareto_archive" / "events.jsonl", {
                    **update,
                    "round": round_index,
                    "proposal_attempt": int(feedback["proposal_attempt"]),
                    "acceptance_passed": bool(acceptance_passed),
                    "source": "pareto_recombination_reconciliation",
                })
                considered += 1
        state.archive = [asdict(item) for item in archive]
        reconciled_archive_ids = {item.engine_id for item in archive}
        archive_changed = reconciled_archive_ids != previous_archive_ids
        if archive_changed:
            state.counters["outer_no_progress_rounds"] = 0
        atomic_json(self.run_root / "pareto_archive" / "head.json", {"entries": archive})
        atomic_json(self.run_root / "pareto_archive" / "reconciliation.json", {
            "schema_version": "dispatchevolve-v2-pareto-reconciliation-v1",
            "rule": "all_successful_full_search_evaluations_update_non_dominated_archive",
            "feasibility_required_for_archive": True,
            "evaluations_considered": considered,
            "retained_entries": len(archive),
            "archive_changed": archive_changed,
            "outer_no_progress_reset": archive_changed,
        })
        self._save(state)

    def _admit_candidate(
        self, candidate: LocalCandidate, *, fixed_e0: Path, frame: pd.DataFrame,
        round_dir: Path, replay: bool,
    ) -> LocalCandidate | None:
        artifact_root = self.run_root / "candidate_library" / "artifacts" / candidate.candidate_id
        engine_dir = artifact_root / "engine"
        copy_engine(fixed_e0, engine_dir)
        for relative in candidate.policy_files:
            source = candidate.engine_dir / relative
            if not source.is_file():
                raise FileNotFoundError(f"candidate Policy artifact is missing: {source}")
            target = engine_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        independent_id = RepositoryGenomeCodec().candidate_id(engine_dir)
        existing = {item.candidate_id for item in self._candidate_library()}
        if independent_id in existing:
            atomic_json(artifact_root / "admission_manifest.json", {
                "status": "duplicate_artifact", "source_candidate_id": candidate.candidate_id,
                "candidate_id": independent_id,
            })
            return None
        if replay:
            scene, metric_rows = scenario_evaluation_scope(candidate.scenario, frame)
            baseline = metric_values(self._evaluate(
                f"library-admission:baseline:{candidate.scenario_hash}", fixed_e0, scene,
                artifact_root / "fixed_e0_baseline", metric_rows,
                trace=False, metrics_only=True, evaluator=self.local_candidate_evaluator,
            ))
            evaluation = self._evaluate(
                f"library-admission:candidate:{independent_id}", engine_dir, scene,
                artifact_root / "candidate_evaluation", metric_rows,
                trace=False, metrics_only=True, evaluator=self.local_candidate_evaluator,
            )
            metrics = metric_values(evaluation)
            admission_scales = local_metric_scales(
                baseline, self._active_state.scales, epsilon=self.config.epsilon,
            )
            delta = oriented_delta(metrics, baseline, admission_scales)
            if independent_id == candidate.candidate_id:
                inconsistent = {
                    key: {"local": float(candidate.oriented_delta[key]), "recheck": delta[key]}
                    for key in METRIC_KEYS
                    if key in candidate.oriented_delta
                    and abs(float(candidate.oriented_delta[key]) - delta[key])
                    > self.config.comparison_tolerance
                }
                if inconsistent:
                    atomic_json(artifact_root / "admission_manifest.json", {
                        "status": "replay_inconsistent", "strategy": self.config.candidate_independence_strategy,
                        "source_candidate_id": candidate.candidate_id, "candidate_id": independent_id,
                        "metrics": metrics, "oriented_delta": delta,
                        "local_oriented_delta": candidate.oriented_delta,
                        "inconsistent_metrics": inconsistent, "policy_files": candidate.policy_files,
                    })
                    raise RuntimeError(
                        f"candidate admission replay diverged from identical local evaluation: {inconsistent}"
                    )
            admitted = feasible(
                delta, target=candidate.objective, rho=self.config.rho,
                tolerance=self.config.comparison_tolerance,
                metric_keys=self.config.experiment_metric_keys,
                objective_keys=self.config.experiment_objective_keys,
                guardrail_keys=self.config.experiment_guardrail_keys,
            )
        else:
            metrics = dict(candidate.metrics)
            delta = dict(candidate.oriented_delta)
            admitted = feasible(
                delta, target=candidate.objective, rho=self.config.rho,
                tolerance=self.config.comparison_tolerance,
                metric_keys=self.config.experiment_metric_keys,
                objective_keys=self.config.experiment_objective_keys,
                guardrail_keys=self.config.experiment_guardrail_keys,
            )
        atomic_json(artifact_root / "admission_manifest.json", {
            "status": "admitted" if admitted else "rejected_fixed_e0_gate",
            "strategy": self.config.candidate_independence_strategy,
            "source_candidate_id": candidate.candidate_id,
            "candidate_id": independent_id, "metrics": metrics,
            "oriented_delta": delta, "policy_files": candidate.policy_files,
            "protocol_migrated_metric_keys": [
                key for key in METRIC_KEYS if key not in candidate.oriented_delta
            ],
        })
        if not admitted:
            return None
        return replace(
            candidate, candidate_id=independent_id, engine_dir=engine_dir,
            metrics=metrics, oriented_delta=delta,
            lineage=tuple(dict.fromkeys((*candidate.lineage, candidate.candidate_id))),
        )

    def _llm_call(
        self, action_id: str, prompt: RenderedPrompt, *, semantic_action: bool = True,
        llm: V2LLM | None = None, model_config: ModelConfig | None = None,
    ) -> str:
        if self.replay_budget and hasattr(self, "_evolution_frame"):
            self.replay_budget.check_stopped()
        selected_llm = llm or self.llm
        selected_model = model_config or self.config.model
        if semantic_action:
            self._count("logical_llm_calls")
        def invoke() -> str:
            self._count("physical_llm_attempts")
            return selected_llm.complete(prompt)
        return str(self.ledger.execute(f"llm:{action_id}", {"protocol_hash": self.protocol_hash,
                                       "prompt_hash": prompt.prompt_hash,
                                       "model": selected_model.model},
                                       invoke))

    def _repair_llm_format(
        self, action_id: str, prompt: RenderedPrompt, response: str,
        error: ResponseFormatError, *, llm: V2LLM | None = None,
        model_config: ModelConfig | None = None,
    ) -> str:
        """Spend at most one extra physical LLM action, not a business proposal."""
        repair = format_repair_prompt(prompt, response, error)
        self._count("format_repair_calls")
        return self._llm_call(
            f"{action_id}:format-repair", repair, semantic_action=False,
            llm=llm, model_config=model_config,
        )

    def _evaluate(self, action_id: str, engine: Path, frame: pd.DataFrame, output: Path,
                  metric_rows: pd.DataFrame | None = None, *, trace: bool = True,
                  metrics_only: bool = False,
                  evaluator: V2Evaluator | None = None, charge_replay: bool = True) -> dict[str, Any]:
        self._count("logical_evaluator_calls")
        selected_evaluator = evaluator or self.evaluator
        evaluation_identity = selected_evaluator.evaluation_identity(
            engine, frame, metric_rows, trace=trace, metrics_only=metrics_only,
        )
        identity = evaluation_identity.ledger_identity(self.protocol_hash)
        return dict(self.ledger.execute(
            f"evaluate:{action_id}", identity,
            lambda: selected_evaluator.evaluate(
                engine, frame, output_dir=output, metric_rows=metric_rows,
                identity=evaluation_identity, trace=trace, metrics_only=metrics_only,
                charge_replay=charge_replay,
            ),
            recover=lambda: selected_evaluator.recover_cached(
                engine, frame, metric_rows, identity=evaluation_identity,
                trace=trace, metrics_only=metrics_only,
            ),
        ))

    def _count(self, key: str) -> None:
        # Counters are persisted at the next state transition; the ledger remains authoritative on resume.
        if hasattr(self, "_active_state"):
            with self._counter_lock:
                self._active_state.counters[key] = self._active_state.counters.get(key, 0) + 1

    def _load_or_initialize(
        self, resume: bool, *, allow_protocol_migration: bool = False,
    ) -> RunState:
        legacy = self.run_root / "state.json"
        source = self.state_path if self.state_path.is_file() else legacy
        if resume and source.is_file():
            state = RunState(**json.loads(source.read_text(encoding="utf-8")))
            if state.protocol_hash != self.protocol_hash:
                if not allow_protocol_migration:
                    raise ValueError("resume protocol hash mismatch")
                old_hash = state.protocol_hash
                history = self.run_root / "protocol_history"
                history.mkdir(parents=True, exist_ok=True)
                protocol_path = self.run_root / "protocol.json"
                config_path = self.run_root / "run_config.yaml"
                if protocol_path.is_file():
                    (history / f"protocol__{old_hash}.json").write_text(
                        protocol_path.read_text(encoding="utf-8"), encoding="utf-8",
                    )
                if config_path.is_file():
                    (history / f"run_config__{old_hash}.yaml").write_text(
                        config_path.read_text(encoding="utf-8"), encoding="utf-8",
                    )
                protocol = protocol_document(self.config)
                protocol["semantic_protocol_hash"] = self.protocol_hash
                atomic_json(protocol_path, protocol)
                from .contracts import jsonable
                config_path.write_text(
                    yaml.safe_dump(jsonable(self.config), sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                _append_jsonl(self.run_root / "protocol_migrations.jsonl", {
                    "authorization": "explicit_cli_allow_protocol_migration",
                    "reason": "resume after an explicitly approved source or budget adjustment",
                    "stage_at_migration": state.stage,
                    "old_protocol_hash": old_hash,
                    "new_protocol_hash": self.protocol_hash,
                    "old_protocol_artifact": f"protocol_history/protocol__{old_hash}.json",
                    "old_config_artifact": f"protocol_history/run_config__{old_hash}.yaml",
                    "new_budget": protocol["budget"],
                })
                state.protocol_hash = self.protocol_hash
                # Metric-contract migrations must refresh the fixed E0 vector
                # from the durable evaluator artifact.  Historical Pareto
                # entries are then rebuilt by _reconcile_pareto_archive from
                # their full evaluation artifacts under the new metric key.
                if state.baseline_metrics and any(
                    key not in state.baseline_metrics for key in METRIC_KEYS
                ):
                    baseline_path = self.run_root / "baseline" / "metrics.json"
                    if not baseline_path.is_file():
                        raise FileNotFoundError(
                            "protocol migration requires the durable baseline metrics artifact"
                        )
                    state.baseline_metrics = metric_values(json.loads(
                        baseline_path.read_text(encoding="utf-8")
                    ))
                    state.scales = {
                        key: max(abs(value), self.config.epsilon)
                        for key, value in state.baseline_metrics.items()
                    }
                self._save(state)
            return state
        if resume:
            raise FileNotFoundError(
                f"cannot resume run_id {self.config.run_id!r}: no V2 state exists"
            )
        if not resume and any(path.exists() for path in (
            self.state_path, legacy, self.run_root / "actions", self.run_root / "scene_memory",
            self.run_root / "pareto_archive",
        )):
            raise FileExistsError(
                f"run_id {self.config.run_id!r} already has V2 state; use --resume or choose a new run_id"
            )
        self.run_root.mkdir(parents=True, exist_ok=True)
        seed = self.run_root / "engines" / "e0"
        copy_engine(self.config.engine_dir, seed)
        source_manifest = tree_manifest(self.config.engine_dir)
        snapshot_manifest = tree_manifest(seed)
        if source_manifest["tree_hash"] != snapshot_manifest["tree_hash"]:
            raise ValueError("V2 E0 snapshot differs from the source engine")
        atomic_json(self.run_root / "manifests" / "e0_engine_snapshot.json", {
            "source": source_manifest, "snapshot": snapshot_manifest,
        })
        instrument_engine(seed)
        engine_id = RepositoryGenomeCodec().candidate_id(seed)
        state = RunState(2, self.protocol_hash, self.config.run_id, self.config.mode,
                         incumbent_engine_id=engine_id, incumbent_engine_dir=str(seed))
        protocol = protocol_document(self.config)
        protocol["semantic_protocol_hash"] = self.protocol_hash
        atomic_json(self.run_root / "protocol.json", protocol)
        from .contracts import jsonable
        (self.run_root / "run_config.yaml").write_text(
            yaml.safe_dump(jsonable(self.config), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        self._save(state)
        self._write_run_manifest(state)
        return state

    def _save(self, state: RunState) -> None:
        atomic_json(self.state_path, state)

    def _artifact_ref(self, path: Path) -> dict[str, Any]:
        """Return a stable reference without copying the referenced artifact."""
        relative = path.relative_to(self.run_root).as_posix()
        if not path.exists():
            return {"path": relative, "exists": False}
        if path.is_file():
            return {
                "path": relative, "exists": True, "kind": "file",
                "sha256": file_hash(path), "size_bytes": path.stat().st_size,
            }
        manifest = tree_manifest(path)
        return {
            "path": relative, "exists": True, "kind": "directory",
            "tree_hash": manifest["tree_hash"], "file_count": manifest["file_count"],
        }

    def _write_run_manifest(self, state: RunState) -> None:
        heads = [
            self.run_root / "protocol.json",
            self.run_root / "run_config.yaml",
            self.state_path,
            self.run_root / "manifests" / "e0_engine_snapshot.json",
            self.run_root / "manifests" / "feature_catalog.json",
            self.run_root / "manifests" / "prompt_ids.json",
            self.run_root / "candidate_library" / "admissions.jsonl",
            self.run_root / "pareto_archive" / "head.json",
            self.run_root / "pareto_archive" / "events.jsonl",
            self.run_root / "combinations" / "feedback.jsonl",
            self.run_root / "online_uplift" / "assessments.jsonl",
            self.run_root / "dpo_data" / "raw_task_records.jsonl",
            self.run_root / "final" / "selected_engine_manifest.json",
            self.run_root / "final" / "test_result.json",
        ]
        heads.extend(sorted(self.run_root.glob("round_*/relation_graph.json")))
        heads.extend(sorted((self.run_root / "composition_identity_registry").glob("*.json")))
        atomic_json(self.run_root / "run_manifest.json", {
            "schema_version": "dispatchevolve-v2-run-manifest-v1",
            "producer": "DispatchEvolveV2Orchestrator._write_run_manifest",
            "run_id": self.config.run_id,
            "mode": self.config.mode,
            "stage": state.stage,
            "completed_rounds": state.completed_rounds,
            "semantic_protocol_hash": self.protocol_hash,
            "selected_engine_id": state.incumbent_engine_id,
            "durable_store_heads": [self._artifact_ref(path) for path in heads],
        })

    def _write_selected_engine_manifest(self, state: RunState) -> None:
        selected_id = state.incumbent_engine_id
        selected_engine = Path(state.incumbent_engine_dir)
        archive_entry = next(
            (item for item in state.archive if str(item.get("engine_id")) == selected_id), None,
        )
        candidate_ids = tuple(map(str, (archive_entry or {}).get("candidate_ids", ())))
        admissions = self._candidate_library()
        candidate_refs = []
        for candidate in admissions:
            if candidate.candidate_id in candidate_ids:
                candidate_refs.append({
                    "candidate_id": candidate.candidate_id,
                    "opportunity_id": candidate.opportunity_id,
                    "scenario_hash": candidate.scenario_hash,
                    "objective": candidate.objective,
                    "policy_files": candidate.policy_files,
                    "lineage": candidate.lineage,
                    "engine_artifact": self._artifact_ref(candidate.engine_dir),
                    "admission_manifest": self._artifact_ref(
                        candidate.engine_dir.parent / "admission_manifest.json"
                    ),
                })
        composition = next(
            (item for item in self._composition_registry()
             if str(item.get("engine_id", "")) == selected_id), None,
        )
        integration_manifest = None
        for path in sorted(self.run_root.glob("round_*/combinations/proposal_*/integration_manifest.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            if str(document.get("engine_id")) == selected_id:
                integration_manifest = self._artifact_ref(path)
                break
        relation_refs = [
            self._artifact_ref(path) for path in sorted(self.run_root.glob("round_*/relation_graph.json"))
        ]
        llm_action_refs = [
            self._artifact_ref(path) for path in sorted((self.run_root / "actions").rglob("committed.json"))
            if "llm" in path.as_posix()
        ]
        atomic_json(self.run_root / "final" / "selected_engine_manifest.json", {
            "schema_version": "dispatchevolve-v2-selected-engine-lineage-v1",
            "producer": "DispatchEvolveV2Orchestrator._write_selected_engine_manifest",
            "run_id": self.config.run_id,
            "semantic_protocol_hash": self.protocol_hash,
            "selected_engine_id": selected_id,
            "selected_engine_source": self._artifact_ref(selected_engine),
            "selected_engine_source_manifest": tree_manifest(selected_engine),
            "fixed_e0_snapshot": self._artifact_ref(
                self.run_root / "manifests" / "e0_engine_snapshot.json"
            ),
            "archive_entry": archive_entry,
            "composition": composition,
            "conflict_resolution_plan": (
                (composition or {}).get("conflict_resolution_plan")
                or (archive_entry or {}).get("conflict_resolution_plan")
                or "NONE"
            ),
            "precedence_edges": (composition or {}).get("precedence_edges", ()),
            "local_candidates": candidate_refs,
            "relation_graphs": relation_refs,
            "deterministic_compiler_manifest": integration_manifest,
            "llm_action_commits": llm_action_refs,
            "full_search_evaluations": self._artifact_ref(
                self.run_root / "combinations" / "feedback.jsonl"
            ),
            "online_uplift_decisions": self._artifact_ref(
                self.run_root / "online_uplift" / "assessments.jsonl"
            ),
            "sealed_test_result": self._artifact_ref(
                self.run_root / "final" / "test_result.json"
            ),
        })

    def _scenario_rule_proposal(
        self, frame: pd.DataFrame, traces: tuple[Any, ...], memory: SceneMemoryStore,
        round_index: int, attempt_index: int, incumbent_metrics: Mapping[str, float],
        previous_feedback: Mapping[str, Any] | None = None,
    ) -> ScenarioQueryAttempt:
        del traces, incumbent_metrics
        target = frame.loc[pd.to_numeric(frame["product_id"], errors="coerce").eq(1)]
        catalog = build_feature_catalog(target, excluded={"product_id"})
        identity = feature_catalog_identity(catalog)
        catalog_path = self.run_root / "manifests" / "feature_catalog.json"
        if catalog_path.is_file():
            previous = json.loads(catalog_path.read_text(encoding="utf-8"))
            if previous.get("catalog_hash") != identity["catalog_hash"]:
                raise ValueError("Feature Catalog changed within one V2 run")
        else:
            atomic_json(catalog_path, identity)
        prompt = self.prompts.render("scenario_discovery_query", {
            "coverage_min_exclusive": f"{self.config.coverage_min_exclusive:.12g}",
            "coverage_max_inclusive": f"{self.config.coverage_max_inclusive:.12g}",
            "coverage_upper_relation": ("less than" if self.config.coverage_max_is_exclusive
                                        else "less than or equal to"),
            "feature_catalog": render_feature_catalog(catalog),
            "scene_memory_limit": 100,
            "scene_memory": _scene_discovery_memory_table(memory),
            "query_feedback": previous_feedback if previous_feedback is not None else "null",
        })
        response = self._llm_call(f"scenario-query:{round_index}:{attempt_index}", prompt)
        match = _SCENARIO_QUERY.fullmatch(response.strip())
        if not match:
            error = ResponseFormatError(
                "response must contain Query and Discovery Purpose sections"
            )
            response = self._repair_llm_format(
                f"scenario-query:{round_index}:{attempt_index}", prompt, response, error,
            )
            match = _SCENARIO_QUERY.fullmatch(response.strip())
            if not match:
                return ScenarioQueryAttempt(None, None, {
                    "category": "INVALID_RESPONSE_CONTRACT",
                    "explanation": "response remained invalid after one format-only repair",
                    "measured_coverage": None,
                })
        raw_query, purpose = match.group(1).strip(), match.group(2).strip()
        if not self._register_raw_query(raw_query, round_index, attempt_index):
            return ScenarioQueryAttempt(None, raw_query, {
                "query": raw_query, "category": "DUPLICATE_QUERY",
                "explanation": "the same Query text was already proposed in this run",
                "measured_coverage": None,
            })
        schema = {item.name: item.type for item in catalog}
        try:
            predicate = parse_batch_query(raw_query, schema)
        except BatchQueryError as exc:
            return ScenarioQueryAttempt(None, raw_query, {
                "query": raw_query, "category": exc.code,
                "explanation": str(exc), "measured_coverage": None,
            })
        if not self._register_query(predicate, round_index, attempt_index):
            return ScenarioQueryAttempt(None, raw_query, {
                "query": raw_query, "category": "DUPLICATE_QUERY",
                "explanation": "an equivalent canonical Query was already proposed in this run",
                "measured_coverage": None,
            })
        try:
            ratio = coverage_ratio(predicate, frame)
        except BatchQueryError as exc:
            return ScenarioQueryAttempt(None, raw_query, {
                "query": raw_query, "category": exc.code,
                "explanation": str(exc), "measured_coverage": None,
            })
        if ratio <= 0:
            category = "EMPTY_MATCH"
        elif ratio <= self.config.coverage_min_exclusive:
            category = "COVERAGE_TOO_LOW"
        elif (ratio >= self.config.coverage_max_inclusive
              if self.config.coverage_max_is_exclusive
              else ratio > self.config.coverage_max_inclusive):
            category = "COVERAGE_TOO_HIGH"
        else:
            return ScenarioQueryAttempt(ScenarioRule(predicate, purpose), raw_query, None)
        return ScenarioQueryAttempt(None, raw_query, {
            "query": raw_query, "category": category,
            "explanation": (
                f"measured coverage {ratio:.12g} is outside "
                f"({self.config.coverage_min_exclusive:.12g}, {self.config.coverage_max_inclusive:.12g}"
                f"{')' if self.config.coverage_max_is_exclusive else ']'}"
            ),
            "measured_coverage": ratio,
        })

    def _opportunity_proposal(
        self, scenario_rule: ScenarioRule, summary: Any, memory: SceneMemoryStore,
        round_index: int, attempt_index: int, engine_dir: Path,
        priority_objective: str | None = None,
    ) -> OpportunityDiscovery | None:
        del priority_objective
        summary_value = asdict(summary) if hasattr(summary, "__dataclass_fields__") else dict(summary)
        policy_statistics = dict(summary_value.get("policy_statistics", {}))
        policy_statistics.pop("policy_trace_text", None)
        policy_trace_evidence = policy_statistics.pop("policy_trace_evidence")
        policy_trace_text = render_policy_trace_text(
            policy_trace_evidence,
            metric_keys=self.config.experiment_metric_keys,
        )
        observed = {
            str(item["policy"]) for item in policy_trace_evidence.get("policies", ())
        }
        available = {
            path.relative_to(engine_dir).as_posix()
            for path in (engine_dir / "policies").rglob("*.py") if path.name != "__init__.py"
        }
        allowed = sorted(observed & available)
        if not allowed:
            return None
        prompt = self.prompts.render("opportunity_discovery", {
            "query": scenario_rule.scenario.canonical,
            "matched_batch_count": summary.matched_batches,
            "total_batch_count": summary.total_batches,
            "batch_coverage": f"{summary.coverage:.2%}",
            "scene_od_pair_count": summary.matched_rows,
            "total_od_pair_count": summary.total_rows,
            "row_coverage": f"{summary.matched_rows / summary.total_rows:.2%}",
            "experiment_metric_catalog": objective_catalog(
                metric_keys=self.config.experiment_metric_keys,
            ),
            "scene_and_global_performance": _objective_performance_table(
                summary, self.config.experiment_metric_keys,
            ),
            "policy_trace_text": policy_trace_text,
            "scene_history": _history_table(memory, review=False),
        })
        response = self._llm_call(f"opportunity:{round_index}:{attempt_index}", prompt)
        identity = {"round": round_index, "predicate": scenario_rule.scenario}
        def parse(value: str) -> OpportunityDiscovery:
            return _parse_opportunity_discovery(
                value, scenario=scenario_rule.scenario, allowed_policies=set(allowed),
                identity=identity,
                allowed_objectives=set(self.config.experiment_objective_keys),
            )
        try:
            discovery = parse(response)
        except ResponseFormatError as error:
            response = self._repair_llm_format(
                f"opportunity:{round_index}:{attempt_index}", prompt, response, error,
            )
            try:
                discovery = parse(response)
            except (ResponseFormatError, ValueError):
                return None
        except ValueError:
            # Invalid semantic content is not a format error and gets no repair.
            return None
        return discovery

    def _critic_prompt(
        self, opportunity: Opportunity, summary: Any, scene_summary: str,
        memory: SceneMemoryStore,
    ) -> RenderedPrompt:
        summary_value = asdict(summary) if hasattr(summary, "__dataclass_fields__") else dict(summary)
        policy_statistics = dict(summary_value.get("policy_statistics", {}))
        trace_evidence = policy_statistics.get("policy_trace_evidence")
        if not isinstance(trace_evidence, Mapping):
            raise ValueError("Opportunity critic requires canonical Policy Trace evidence")
        policy_trace_text = render_policy_trace_text(
            trace_evidence,
            policy_scope=opportunity.related_policies,
            metric_keys=self.config.experiment_metric_keys,
        )
        return self.prompts.render("opportunity_critic", {
            "query": opportunity.scenario.canonical,
            "scene_summary": scene_summary,
            "matched_batch_count": summary_value["matched_batches"],
            "total_batch_count": summary_value["total_batches"],
            "batch_coverage": f"{float(summary_value['coverage']):.2%}",
            "scene_od_pair_count": summary_value["matched_rows"],
            "total_od_pair_count": summary_value["total_rows"],
            "row_coverage": f"{int(summary_value['matched_rows']) / int(summary_value['total_rows']):.2%}",
            "objective": opportunity.objective,
            "related_policies": "\n".join(f"- {path}" for path in opportunity.related_policies),
            "improvement_plan": opportunity.improvement_plan or opportunity.rationale,
            "evidence_basis": opportunity.evidence_basis or opportunity.rationale,
            "experiment_metric_catalog": objective_catalog(
                metric_keys=self.config.experiment_metric_keys,
            ),
            "scene_and_global_performance": _objective_performance_table(
                summary, self.config.experiment_metric_keys,
            ),
            "policy_trace_text": policy_trace_text,
            "review_history": _history_table(
                memory, review=True, exclude_opportunity_id=opportunity.opportunity_id,
                exclude_scene_hash=opportunity.scenario.predicate_hash,
            ),
        })

    def _critic(
        self, opportunity: Opportunity, summary: Any, scene_summary: str,
        memory: SceneMemoryStore,
    ) -> tuple[CriticAssessment, dict[str, Any]]:
        prompt = self._critic_prompt(opportunity, summary, scene_summary, memory)
        critic_model = self.config.critic_model or self.config.model
        response = self._llm_call(
            f"critic:{opportunity.opportunity_id}", prompt,
            llm=self.critic_llm, model_config=critic_model,
        )
        match = _CRITIC.fullmatch(response.strip())
        if not match:
            error = ResponseFormatError("Opportunity Critic response violates the section contract")
            response = self._repair_llm_format(
                f"critic:{opportunity.opportunity_id}", prompt, response, error,
                llm=self.critic_llm, model_config=critic_model,
            )
            match = _CRITIC.fullmatch(response.strip())
        assessment = (CriticAssessment("REJECT", 0.0, "invalid critic response", critic_model.model, prompt.prompt_hash)
                      if not match else CriticAssessment(match.group(1), float(match.group(2)), match.group(3).strip(), critic_model.model, prompt.prompt_hash))
        return assessment, {"system": prompt.system, "user": prompt.user, "prompt_hash": prompt.prompt_hash}

    def _finish_no_progress(self, state: RunState, round_index: int) -> bool:
        state.completed_rounds = round_index; state.stage = "round_complete"
        state.counters["outer_no_progress_rounds"] = state.counters.get("outer_no_progress_rounds", 0) + 1
        patience = self.config.budget.outer_early_stopping_patience or 1
        if patience is not None and state.counters["outer_no_progress_rounds"] >= patience:
            state.counters["outer_early_stopped"] = 1
        self._save(state)
        return state.counters.get("outer_early_stopped") == 1

    def _preflight_engine(
        self, engine: Path, incumbent: Path, frame: pd.DataFrame,
        proposal: Any, candidate_map: Mapping[str, LocalCandidate],
    ) -> None:
        if "batch_id" not in frame:
            raise ValueError("V2 integration preflight requires batch_id")
        priority = tuple(proposal.priority_plan.ordered_candidate_ids)
        selected_ids = tuple(proposal.candidate_ids)
        precedence = tuple(proposal.priority_plan.precedence_edges)
        selected_batches: list[Any] = []
        masks: dict[str, pd.Series] = {}
        for candidate_id in priority:
            mask = scenario_target_mask(candidate_map[candidate_id].scenario, frame)
            masks[candidate_id] = mask
            if not bool(mask.any()):
                raise ValueError(f"integration preflight cannot find a matching batch for {candidate_id}")
            batch_id = frame.loc[mask, "batch_id"].iloc[0]
            if batch_id not in selected_batches:
                selected_batches.append(batch_id)
        for left_index, left in enumerate(priority):
            for right in priority[left_index + 1:]:
                overlap = masks[left] & masks[right]
                if bool(overlap.any()):
                    batch_id = frame.loc[overlap, "batch_id"].iloc[0]
                    if batch_id not in selected_batches:
                        selected_batches.append(batch_id)
        reachable_batches: dict[str, Any] = {}
        for batch_id, batch in frame.groupby("batch_id", sort=False, dropna=False):
            matched = {
                candidate_id for candidate_id in selected_ids
                if bool(masks[candidate_id].loc[batch.index].any())
            }
            active: list[str] = []
            for candidate_id in priority:
                if candidate_id in matched and not any(
                    str(edge["lower"]) == candidate_id and str(edge["higher"]) in active
                    for edge in precedence
                ):
                    active.append(candidate_id)
            for candidate_id in active:
                reachable_batches.setdefault(candidate_id, batch_id)
        missing_reachability = sorted(set(selected_ids) - set(reachable_batches))
        if missing_reachability:
            raise ValueError(
                f"Conflict Resolution Plan makes selected candidates unreachable: {missing_reachability}"
            )
        for batch_id in reachable_batches.values():
            if batch_id not in selected_batches:
                selected_batches.append(batch_id)
        schema = {name: str(dtype) for name, dtype in frame.dtypes.items()}
        prior_masks: list[pd.Series] = []
        for canonical in engine_frozen_scenarios(incumbent).values():
            prior_mask = scenario_target_mask(parse_predicate(canonical, schema), frame)
            prior_masks.append(prior_mask)
            if bool(prior_mask.any()):
                batch_id = frame.loc[prior_mask, "batch_id"].iloc[0]
                if batch_id not in selected_batches:
                    selected_batches.append(batch_id)
        union = pd.Series(False, index=frame.index)
        for mask in masks.values():
            union |= mask
        target_rows = pd.to_numeric(frame["product_id"], errors="coerce").eq(1)
        prior_union = pd.Series(False, index=frame.index)
        for mask in prior_masks:
            prior_union |= mask
        unmatched_batches = frame.loc[~union & ~prior_union & target_rows, "batch_id"]
        if len(unmatched_batches):
            batch_id = unmatched_batches.iloc[0]
            if batch_id not in selected_batches:
                selected_batches.append(batch_id)
        sample = frame.loc[frame["batch_id"].isin(selected_batches)].copy()
        sample["__v2_probe_id"] = [f"probe-{index}" for index in range(len(sample))]
        expected_callables: dict[str, dict[str, str]] = {
            module: {
                symbol: f"incumbent|{module.replace('.', '/')}.py|{symbol}"
                for symbol in symbols
            }
            for module, symbols in engine_policy_callables(incumbent).items()
        }
        for identifier in priority:
            for relative in candidate_map[identifier].policy_files:
                variant_file = variant_policy_path(identifier, relative)
                module = variant_file[:-3].replace("/", ".")
                expected_callables[module] = {
                    symbol: f"{identifier}|{variant_file}|{symbol}"
                    for symbol in policy_call_symbols(candidate_map[identifier].engine_dir, relative)
                }
        probe_engine = instrument_variant_call_probe(
            engine, engine.parent / f".{engine.name}_system_probe", expected_callables,
            engine_result_fields(incumbent),
        )
        output, traces = self.evaluator.runner(probe_engine, sample, True, None)
        incumbent_output, _ = self.evaluator.runner(incumbent, sample, True, None)
        if not isinstance(output, pd.DataFrame) or not isinstance(traces, list):
            raise TypeError("integrated engine failed the run_batch output contract")

        def validate_row_evidence(value: Any, label: str) -> None:
            if not isinstance(value, Mapping) or isinstance(value.get("count"), bool):
                raise ValueError(f"{label} must be exact count+bitmap evidence")
            count = value.get("count")
            bitmap_hex = value.get("bitmap_hex")
            if not isinstance(count, int) or count < 0 or not isinstance(bitmap_hex, str):
                raise ValueError(f"{label} must be exact count+bitmap evidence")
            try:
                bitmap = bytes.fromhex(bitmap_hex)
            except ValueError as exc:
                raise ValueError(f"{label} bitmap_hex is invalid") from exc
            if sum(byte.bit_count() for byte in bitmap) != count:
                raise ValueError(f"{label} bitmap count does not match bitmap_hex")

        expected_passthrough = int((~pd.to_numeric(sample["product_id"], errors="coerce").eq(1)).sum())
        actual_passthrough = int((~pd.to_numeric(output["product_id"], errors="coerce").eq(1)).sum()) if "product_id" in output else -1
        if actual_passthrough != expected_passthrough:
            raise ValueError("integrated engine changed non-target-category passthrough rows")
        if not traces or any(not isinstance(item.get("policy_events"), list) for item in traces):
            raise ValueError("integrated engine trace omits ordered policy_events")
        required_trace_fields = {
            "input_rows", "output_rows", "target_row_count", "target_output_rows",
            "passthrough_row_count", "filtered_count", "filter_rule_counts",
            "filter_policy_counts", "input_target_rows", "final_eligible_target_rows",
        }
        for trace_item in traces:
            missing_fields = sorted(required_trace_fields - set(trace_item))
            if missing_fields:
                raise ValueError(f"integrated engine trace omits next-round fields: {missing_fields}")
            validate_row_evidence(trace_item["input_target_rows"], "input_target_rows")
            validate_row_evidence(trace_item["final_eligible_target_rows"], "final_eligible_target_rows")
            for event in trace_item["policy_events"]:
                if not all(event.get(key) is not None for key in ("policy_call", "policy_file", "input_row_count")):
                    raise ValueError("integrated engine policy event omits required identity fields")
                validate_row_evidence(event.get("eligible_before_rows"), "eligible_before_rows")
                validate_row_evidence(event.get("applicable_before_rows"), "applicable_before_rows")
                validate_row_evidence(event.get("newly_filtered_rows"), "newly_filtered_rows")
                modified = event.get("modified_rows_by_field") or {}
                if not isinstance(modified, Mapping):
                    raise ValueError("modified_rows_by_field must be a mapping")
                for field, evidence in modified.items():
                    validate_row_evidence(evidence, f"modified_rows_by_field.{field}")
        expected_routes: dict[tuple[str, int], tuple[str, ...]] = {}
        expected_probe_routes: dict[str, tuple[str, ...]] = {}
        sample_masks = {
            identifier: scenario_target_mask(candidate_map[identifier].scenario, sample)
            for identifier in priority
        }
        for batch_id, batch in sample.groupby("batch_id", sort=False, dropna=False):
            matched = {
                candidate_id for candidate_id in selected_ids
                if bool(sample_masks[candidate_id].loc[batch.index].any())
            }
            active: list[str] = []
            for candidate_id in priority:
                if candidate_id not in matched:
                    continue
                if not any(
                    str(edge["lower"]) == candidate_id and str(edge["higher"]) in active
                    for edge in precedence
                ):
                    active.append(candidate_id)
            active_tuple = tuple(active)
            for local_order, (index, row) in enumerate(batch.iterrows()):
                route = active_tuple if bool(pd.to_numeric(row["product_id"], errors="coerce") == 1) else ()
                expected_routes[(str(batch_id), local_order)] = route
                expected_probe_routes[str(row["__v2_probe_id"])] = route
        observed_routes: dict[tuple[str, int], tuple[str, ...]] = {}
        executed: set[str] = set()
        executed_files: dict[str, set[str]] = {}
        executed_symbols: dict[tuple[str, str], set[str]] = {}
        system_counts: dict[str, int] = {}
        reported_effects: dict[str, list[str]] = {}
        observed_effects: dict[str, list[str]] = {}
        known_effect_keys = {key for symbols in expected_callables.values() for key in symbols.values()}
        for trace_item in traces:
            batch_id = str(trace_item.get("batch_id"))
            routing = trace_item.get("v2_routing")
            if not isinstance(routing, list):
                raise ValueError("integrated engine trace omits v2_routing evidence")
            for route in routing:
                active_ids = route.get("active_candidate_ids")
                if not isinstance(active_ids, list):
                    raise ValueError("integrated engine trace omits active Local Candidate IDs")
                observed_routes[(batch_id, int(route["input_row_order"]))] = tuple(map(str, active_ids))
            for event in trace_item["policy_events"]:
                policy_file = str(event.get("policy_file", ""))
                symbol = str(event.get("policy_call", "")).rsplit(".", 1)[-1]
                if event.get("candidate_id") in priority:
                    identifier = str(event["candidate_id"])
                    executed.add(identifier)
                    executed_files.setdefault(identifier, set()).add(policy_file)
                    executed_symbols.setdefault((identifier, policy_file), set()).add(
                        symbol
                    )
                    effect_key = f"{identifier}|{policy_file}|{symbol}"
                else:
                    effect_key = f"incumbent|{policy_file}|{symbol}"
                if effect_key not in known_effect_keys:
                    raise ValueError(f"integrated engine reports an unknown policy callable: {effect_key}")
                reported_effects.setdefault(effect_key, []).append(content_hash({
                    "newly_filtered_rows": event.get("newly_filtered_rows"),
                    "modified_rows_by_field": event.get("modified_rows_by_field") or {},
                }))
            for key, value in (trace_item.get("_v2_system_call_counts") or {}).items():
                system_counts[str(key)] = system_counts.get(str(key), 0) + int(value)
            for event in trace_item.get("_v2_system_observed_events") or []:
                key = str(event.get("key"))
                observed_effects.setdefault(key, []).append(content_hash({
                    "newly_filtered_rows": event.get("newly_filtered_rows"),
                    "modified_rows_by_field": event.get("modified_rows_by_field") or {},
                }))
        if observed_routes != expected_routes:
            raise ValueError("integrated engine runtime routing does not implement the frozen predicates and priority order")
        expected_executed = {identifier for route in expected_routes.values() for identifier in route}
        if expected_executed != set(selected_ids):
            missing_routes = sorted(set(selected_ids) - expected_executed)
            raise ValueError(f"frozen priority makes selected candidates unreachable: {missing_routes}")
        if not expected_executed <= executed:
            raise ValueError("integrated engine trace does not prove execution of every routed variant")
        for identifier in expected_executed:
            expected_files = {
                variant_policy_path(identifier, relative)
                for relative in candidate_map[identifier].policy_files
            }
            if not expected_files <= executed_files.get(identifier, set()):
                raise ValueError(f"integrated engine trace omits exact variant policy files for {identifier}")
            missing_calls = []
            for path in sorted(expected_files):
                module = path[:-3].replace("/", ".")
                for symbol, key in expected_callables[module].items():
                    if symbol not in executed_symbols.get((identifier, path), set()) or system_counts.get(key, 0) <= 0:
                        missing_calls.append(key)
            if missing_calls:
                raise ValueError(f"system call probe did not execute routed variants: {missing_calls}")
        selected_variant_keys = {
            key for module, symbols in expected_callables.items() if module.startswith("policies.v2_")
            for key in symbols.values() if not key.startswith("incumbent|")
        }
        for module_symbols in expected_callables.values():
            for key in module_symbols.values():
                if key in selected_variant_keys and system_counts.get(key, 0) <= 0:
                    raise ValueError(f"system call probe did not execute required policy callable: {key}")
                if sorted(reported_effects.get(key, [])) != sorted(observed_effects.get(key, [])):
                    raise ValueError(f"self-reported policy effects differ from system observation: {key}")
        unmatched_probe_ids = {probe for probe, route in expected_probe_routes.items() if not route}
        if list(output.columns) != list(incumbent_output.columns):
            raise ValueError("integrated engine changed the incumbent output schema or column order")
        columns = list(output.columns)
        left = output.loc[output["__v2_probe_id"].isin(unmatched_probe_ids), columns].sort_values("__v2_probe_id").reset_index(drop=True)
        right = incumbent_output.loc[incumbent_output["__v2_probe_id"].isin(unmatched_probe_ids), columns].sort_values("__v2_probe_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_like=True)
        for candidate_id in expected_executed:
            candidate_output, _ = self.evaluator.runner(candidate_map[candidate_id].engine_dir, sample, True, None)
            if list(output.columns) != list(candidate_output.columns):
                raise ValueError(f"integrated engine output schema differs from candidate {candidate_id}")
            probe_ids = {probe for probe, route in expected_probe_routes.items() if route == (candidate_id,)}
            if not probe_ids:
                continue
            integrated_rows = output.loc[output["__v2_probe_id"].isin(probe_ids), columns].sort_values("__v2_probe_id").reset_index(drop=True)
            candidate_rows = candidate_output.loc[candidate_output["__v2_probe_id"].isin(probe_ids), columns].sort_values("__v2_probe_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(integrated_rows, candidate_rows, check_dtype=False, check_like=True)


    def run(
        self, *, resume: bool = False, rounds: int | None = None,
        allow_protocol_migration: bool = False,
        stop_after_candidate_admission_round: int | None = None,
    ) -> RunState:
        from .replay_budget import ReplayBudgetExhausted
        try:
            return self._run_search(resume=resume, rounds=rounds,
                allow_protocol_migration=allow_protocol_migration,
                stop_after_candidate_admission_round=stop_after_candidate_admission_round)
        except ReplayBudgetExhausted:
            state = self._active_state
            state.counters["replay_budget_stopped"] = 1
            state.counters["outer_early_stopped"] = 1
            self._reconcile_pareto_archive(state)
            archive = _archive_from_state(state)
            if self.config.online_uplift_enabled:
                from .pairwise_ranking import rank_engines
                frame = self._evolution_frame
                ranked, report = rank_engines(
                    archive, llm=V2LLM(self.config.online_model) if self.config.online_model else self.critic_llm,
                    context={"dataset_rows": len(frame), "metric_names": list(self.config.experiment_metric_keys),
                             "predictive_feature_summary": frame.loc[:, [name for name in
                                 ("eta", "dar", "pcaa", "dcaa") if name in frame]].describe().to_dict()},
                    objective_keys=self.config.experiment_objective_keys, reference=-self.config.rho)
                selected = ranked[0]
                atomic_json(self.run_root / "online_uplift" / "pairwise_ranking.json", report)
            else:
                selected, _ = select_incumbent(archive, rho=self.config.rho,
                    tolerance=self.config.comparison_tolerance, current_engine_id=state.incumbent_engine_id,
                    objective_keys=self.config.experiment_objective_keys, guardrail_keys=self.config.experiment_guardrail_keys)
            state.incumbent_engine_id = selected.engine_id
            state.incumbent_engine_dir = str(selected.engine_dir)
            state.stage = "complete"
            self._save(state)
            return self._finalize_run(state)

    def _run_search(
        self, *, resume: bool = False, rounds: int | None = None,
        allow_protocol_migration: bool = False,
        stop_after_candidate_admission_round: int | None = None,
    ) -> RunState:
        state = self._load_or_initialize(
            resume, allow_protocol_migration=allow_protocol_migration,
        )
        resume_stage = state.stage
        self._active_state = state
        _append_jsonl(self.run_root / "effective_config_history.jsonl", {
            "resume": resume, "semantic_protocol_hash": self.protocol_hash,
            "effective_config": self.config,
            "derived": {
                "dpo_max_scene_proposal_attempts": self.config.dpo_collection.max_scene_proposal_attempts,
            },
            "source_precedence": "core defaults < explicit YAML < supported launcher overrides",
        })
        raw_records_path = self.run_root / "dpo_data" / "raw_task_records.jsonl"

        def dpo_record_count() -> int:
            if not raw_records_path.is_file():
                return 0
            return sum(1 for line in raw_records_path.read_text(encoding="utf-8").splitlines() if line.strip())

        if self.config.mode == "dpo_data_collection":
            plan_path = self.run_root / "dpo_data" / "collection_plan.json"
            previous_plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
            collected_before = dpo_record_count()
            plan = {
                "target_records": self.config.dpo_collection.target_records,
                "record_target_semantics": "minimum_llm_proposed_opportunities",
                "scene_opportunity_cardinality": "llm_determined",
                "execution_order": "discover_review_evolve_commit_one_scene_then_repeat",
                "resume_order": "finish_discovered_incomplete_scenes_until_target_before_new_discovery",
                "proposal_attempt_multiplier": self.config.dpo_collection.proposal_attempt_multiplier,
                "max_scene_proposal_attempts": self.config.dpo_collection.max_scene_proposal_attempts,
                "max_iterations_per_scene_objective": self.config.dpo_collection.max_iterations_per_scene_objective,
                "max_evaluations_per_scene_objective": self.config.dpo_collection.max_evaluations_per_scene_objective,
                "collected_records_before_run": collected_before,
            }
            _append_jsonl_once(self.run_root / "dpo_data" / "collection_plan_history.jsonl", plan)
            atomic_json(plan_path, plan)
            if resume and collected_before < self.config.dpo_collection.target_records:
                # A compatible larger target reopens the single cumulative collection pass.
                state.completed_rounds = 0
        self.logger.info("run_start run_id=%s mode=%s resume=%s completed_rounds=%s",
                         self.config.run_id, self.config.mode, resume, state.completed_rounds)
        memory = SceneMemoryStore(self.run_root, state.scene_memory)
        frame = _load_debug_frame(self.config)
        self._evolution_frame = frame
        if self.replay_budget:
            self.replay_budget.initialize(len(frame), self.config.budget.full_replay_equivalents)
        incumbent = Path(state.incumbent_engine_dir)
        if not state.baseline_metrics:
            state.stage = "baseline_evaluation"; self._save(state)
            baseline = self._evaluate("baseline", incumbent, frame, self.run_root / "baseline")
            state.baseline_metrics = metric_values(baseline)
            state.scales = {key: max(abs(value), self.config.epsilon) for key, value in state.baseline_metrics.items()}
            state.archive = [asdict(ArchiveEntry(state.incumbent_engine_id, incumbent, state.baseline_metrics,
                                                 {key: 0.0 for key in METRIC_KEYS}, 0))]
            self._save(state)
        if self.config.mode != "dpo_data_collection":
            self._reconcile_pareto_archive(state)
        if self.config.mode == "dpo_data_collection":
            stop = 1
        else:
            stop = min(self.config.budget.outer_rounds, state.completed_rounds + rounds) if rounds else self.config.budget.outer_rounds
        while state.completed_rounds < stop and not state.counters.get("outer_early_stopped"):
            if self.replay_budget:
                self.replay_budget.check_stopped()
            round_index = state.completed_rounds + 1
            self.logger.info("round_start round=%s incumbent=%s", round_index, state.incumbent_engine_id)
            round_dir = self.run_root / f"round_{round_index:03d}"
            archive_ids_before = {item["engine_id"] for item in state.archive}; incumbent_before = state.incumbent_engine_id
            incumbent_eval = self._evaluate(f"round:{round_index}:incumbent", incumbent, frame, round_dir / "incumbent_full")
            traces = normalize_traces(incumbent, frame, incumbent_eval)
            atomic_gzip_json(self.run_root / "traces" / f"round_{round_index:03d}.json.gz", {"traces": traces})
            accepted_path = round_dir / "opportunities" / "accepted.jsonl"
            attempts_path = round_dir / "opportunities" / "attempts.jsonl"
            result_dir = round_dir / "opportunities" / "results"
            attempt_records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(result_dir.glob("attempt_*.json"))
            ]
            attempt_entries = [
                (record, opportunity, assessment, critic_prompt)
                for record in attempt_records
                for opportunity, assessment, critic_prompt in _attempt_opportunity_entries(record)
            ]
            accepted_records: list[dict[str, Any]] = []
            for record, opportunity, assessment, critic_prompt in attempt_entries:
                selected_ids = record.get("selected_opportunity_ids")
                selected = (selected_ids is None or
                            str(opportunity.get("opportunity_id")) in set(map(str, selected_ids)))
                if assessment and assessment.get("decision") == "ACCEPT" and critic_prompt and selected:
                    accepted_records.append({"opportunity": opportunity, "critic_prompt": critic_prompt})
            resume_selection_path = round_dir / "state" / "resume_opportunity_selection.json"
            if resume and resume_selection_path.is_file():
                selection = json.loads(resume_selection_path.read_text(encoding="utf-8"))
                selected_ids = set(map(str, selection["selected_opportunity_ids"]))
                accepted_records = [
                    item for item in accepted_records
                    if str(item["opportunity"]["opportunity_id"]) in selected_ids
                ]
            elif (
                resume and resume_stage == "local_policy_evolution"
                and len(accepted_records) > self.config.budget.admitted_opportunities_target
            ):
                accepted_records, selection = _select_resumed_opportunities(
                    accepted_records, round_dir,
                    self.config.budget.admitted_opportunities_target,
                )
                atomic_json(resume_selection_path, selection)
            for item in attempt_records:
                _append_jsonl_once(attempts_path, {key: value for key, value in item.items() if key != "critic_prompt"})
            for item in accepted_records:
                _append_jsonl_once(accepted_path, item)
            accepted: list[Opportunity] = [_opportunity_from_dict(item["opportunity"]) for item in accepted_records]
            critic_prompts: dict[str, dict[str, Any]] = {
                str(opportunity["opportunity_id"]): critic_prompt
                for _, opportunity, _, critic_prompt in attempt_entries
                if critic_prompt
            }
            state.stage = "scenario_discovery_query"; self._save(state)
            prior_attempts = len(attempt_records)
            discovered: list[Opportunity] = [
                _opportunity_from_dict(opportunity)
                for _, opportunity, _, _ in attempt_entries
            ]
            scenario_summaries: dict[str, Mapping[str, Any]] = {
                str(opportunity["opportunity_id"]): record["summary"]
                for record, opportunity, _, _ in attempt_entries if record.get("summary")
            }
            scene_descriptions: dict[str, str] = {
                str(opportunity["opportunity_id"]): str(record.get("llm_scene_summary") or "")
                for record, opportunity, _, _ in attempt_entries
            }

            def evolve_dpo_scene(opportunities: list[Opportunity]) -> int:
                """Evolve and commit one Scene's LLM-proposed Opportunities."""
                scene_opportunities = _unique_opportunities(opportunities)
                committed_opportunity_ids = {
                    str(document["opportunity_id"])
                    for document in (
                        json.loads(line) for line in raw_records_path.read_text(
                            encoding="utf-8"
                        ).splitlines() if line.strip()
                    )
                } if raw_records_path.is_file() else set()
                scene_opportunities = [
                    opportunity for opportunity in scene_opportunities
                    if opportunity.opportunity_id not in committed_opportunity_ids
                ]
                if not scene_opportunities:
                    return dpo_record_count()
                scene_hashes = {item.scenario.predicate_hash for item in scene_opportunities}
                if len(scene_hashes) != 1:
                    raise ValueError("streaming DPO evolution requires exactly one Scene per batch")
                scene_hash = next(iter(scene_hashes))
                state.stage = "local_policy_evolution"; self._save(state)
                self.logger.info(
                    "dpo_scene_evolution_start round=%s scene=%s opportunities=%s",
                    round_index, scene_hash, len(scene_opportunities),
                )
                fixed_e0 = self.run_root / "engines" / "e0"
                outcomes: dict[str, LocalEvolutionResult] = {}
                opportunity_jobs: list[Mapping[str, Any]] = []
                scene, metric_rows = scenario_evaluation_scope(
                    scene_opportunities[0].scenario, frame,
                )
                search_base = (fixed_e0
                               if self.config.candidate_independence_strategy == "evolve_from_fixed_e0"
                               else incumbent)
                shared_root = round_dir / "state" / "local_evolution_inputs" / scene_hash
                shared_root.mkdir(parents=True, exist_ok=True)
                scene_frame_path = shared_root / "scene_frame.pkl"
                metric_rows_path = shared_root / "metric_rows.pkl" if metric_rows is not None else None
                if not scene_frame_path.is_file():
                    _atomic_pickle(scene, scene_frame_path)
                if metric_rows_path is not None and not metric_rows_path.is_file():
                    _atomic_pickle(metric_rows, metric_rows_path)
                for opportunity in scene_opportunities:
                    outcome_path = (
                        round_dir / "local_policy_evolution" /
                        opportunity.opportunity_id / "outcome.json"
                    )
                    if outcome_path.is_file():
                        outcomes[opportunity.opportunity_id] = _local_result_from_dict(
                            json.loads(outcome_path.read_text(encoding="utf-8"))
                        )
                        continue
                    scene_baseline = metric_values(self._evaluate(
                        f"round:{round_index}:local-baseline:{opportunity.opportunity_id}",
                        search_base,
                        scene,
                        round_dir / "local_policy_evolution" /
                        opportunity.opportunity_id / "baseline",
                        metric_rows,
                        trace=False,
                        metrics_only=True,
                        evaluator=self.local_candidate_evaluator,
                    ))
                    local_config = replace(self.config, budget=replace(
                        self.config.budget,
                        local_iterations_per_opportunity=(
                            self.config.dpo_collection.max_iterations_per_scene_objective
                        ),
                        local_evaluations_per_opportunity=(
                            self.config.dpo_collection.max_evaluations_per_scene_objective
                        ),
                    ))
                    opportunity_jobs.append({
                        "config": local_config,
                        "opportunity": opportunity,
                        "scenario_summary": {
                            **scenario_summaries[opportunity.opportunity_id],
                            "scene_description": scene_descriptions.get(
                                opportunity.opportunity_id, "",
                            ),
                        },
                        "incumbent_dir": search_base,
                        "scene_frame_path": scene_frame_path,
                        "metric_rows_path": metric_rows_path,
                        "baseline_metrics": scene_baseline,
                        "scales": local_metric_scales(
                            scene_baseline, state.scales, epsilon=self.config.epsilon,
                        ),
                        "round_dir": round_dir,
                        "cache_root": self.evaluator.cache_root,
                        "protocol_hash": self.evaluator.protocol_hash,
                        "backend": self.evaluator.backend,
                        "wall_timeout_seconds": self.evaluator.runner.wall_timeout_seconds,
                        "trace_transport_max_bytes": self.evaluator.runner.max_trace_bytes,
                        "candidate_process_workers": self.config.local_candidate_process_workers,
                    })
                effective_parallel = min(
                    self.config.budget.parallel_opportunities, len(opportunity_jobs),
                ) if opportunity_jobs else 0
                worker_pids: dict[str, int] = {}
                if opportunity_jobs:
                    for opportunity_id, result, worker_pid in _run_opportunity_process_pool(
                        opportunity_jobs, effective_parallel,
                    ):
                        outcomes[opportunity_id] = result
                        worker_pids[opportunity_id] = worker_pid
                scene_frame_path.unlink(missing_ok=True)
                if metric_rows_path is not None:
                    metric_rows_path.unlink(missing_ok=True)
                parallelism_path = (
                    round_dir / "state" / "streaming_opportunity_parallelism" /
                    f"{scene_hash}.json"
                )
                parallelism = _persist_opportunity_parallelism(
                    parallelism_path,
                    configured_max_workers=self.config.budget.parallel_opportunities,
                    pending_opportunities=len(opportunity_jobs),
                    effective_max_workers=effective_parallel,
                    worker_pids=worker_pids,
                    resumed_outcomes=len(scene_opportunities) - len(opportunity_jobs),
                )
                state.counters["effective_parallel_opportunities"] = int(
                    parallelism["effective_max_workers"]
                )
                raw_sample_artifacts: dict[str, dict[str, Any]] = {}
                for opportunity in scene_opportunities:
                    outcome = outcomes[opportunity.opportunity_id]
                    memory.append(scene_hash, "local_policy_evolution_outcome", {
                        "engine_id": state.incumbent_engine_id, "round": round_index,
                        "opportunity": asdict(opportunity), "success": outcome.success,
                        "evaluated_count": outcome.evaluated_count,
                        "best_delta": outcome.best_delta,
                    })
                    best_reward = (float(outcome.best_delta[opportunity.objective])
                                   if outcome.best_delta is not None else None)
                    feasible_deltas = [
                        item["normalized_improvement"] for item in outcome.attempts
                        if item.get("accepted") and
                        item.get("normalized_improvement") is not None
                    ]
                    best_feasible_delta = max(
                        feasible_deltas,
                        key=lambda value: value[opportunity.objective],
                        default=None,
                    )
                    raw_path = (
                        self.run_root / "dpo_data" / "local_task_traces" /
                        f"{opportunity.opportunity_id}.json"
                    )
                    atomic_json(raw_path, {
                        "schema_version": 1,
                        "protocol_hash": self.protocol_hash,
                        "opportunity": opportunity,
                        "critic_prompt": critic_prompts[opportunity.opportunity_id],
                        "scenario_summary": scenario_summaries[opportunity.opportunity_id],
                        "local_policy_evolution_trace": list(outcome.attempts),
                        "best_reward": {
                            "objective": opportunity.objective,
                            "value": best_reward,
                            "oriented_delta": outcome.best_delta,
                            "selection_scope": "all_evaluated_candidates",
                        },
                        "best_feasible_reward": {
                            "objective": opportunity.objective,
                            "value": (float(best_feasible_delta[opportunity.objective])
                                      if best_feasible_delta is not None else None),
                            "oriented_delta": best_feasible_delta,
                        },
                        "real_outcome": {
                            "success": outcome.success,
                            "evaluated_count": outcome.evaluated_count,
                            "retained_candidates": outcome.candidates,
                        },
                        "dataset_builder": {
                            "status": "pending", "labels_or_pairs_present": False,
                        },
                    })
                    raw_sample_artifacts[opportunity.opportunity_id] = {
                        "uri": raw_path.relative_to(self.run_root).as_posix(),
                        "sha256": file_hash(raw_path),
                    }
                records = raw_task_records(
                    scene_opportunities, outcomes, protocol_hash=self.protocol_hash,
                    critic_prompts=critic_prompts,
                    raw_sample_artifacts=raw_sample_artifacts,
                )
                existing_documents = [
                    json.loads(line) for line in raw_records_path.read_text(
                        encoding="utf-8"
                    ).splitlines() if line.strip()
                ] if raw_records_path.is_file() else []
                existing_ids = {item["record_id"] for item in existing_documents}
                existing_opportunity_ids = {
                    str(item["opportunity_id"]) for item in existing_documents
                }
                for record in records:
                    if (record["record_id"] not in existing_ids and
                            str(record["opportunity_id"]) not in existing_opportunity_ids):
                        _append_jsonl(raw_records_path, record)
                        existing_documents.append(record)
                        existing_ids.add(record["record_id"])
                        existing_opportunity_ids.add(str(record["opportunity_id"]))
                collected = len(existing_ids)
                state.counters["dpo_target_records"] = self.config.dpo_collection.target_records
                state.counters["dpo_collected_records"] = collected
                state.counters["dpo_collected_scenes"] = len({
                    item["scenario"] for item in existing_documents
                })
                state.counters["dpo_record_shortfall"] = max(
                    0, self.config.dpo_collection.target_records - collected,
                )
                state.stage = "scenario_discovery_query"; self._save(state)
                self.logger.info(
                    "dpo_scene_evolution_complete round=%s scene=%s records=%s",
                    round_index, scene_hash, collected,
                )
                return collected

            if self.config.mode == "dpo_data_collection":
                # Resume first drains every discovered Scene that lacks committed
                # raw records until the target is met, before asking the LLM for
                # another Scene. Lowering the target never deletes prior records.
                for record in attempt_records:
                    if dpo_record_count() >= self.config.dpo_collection.target_records:
                        break
                    scene_items = [
                        _opportunity_from_dict(item)
                        for item, _, _ in _attempt_opportunity_entries(record)
                    ]
                    if scene_items:
                        evolve_dpo_scene(scene_items)
            proposal_limit = (self.config.dpo_collection.max_scene_proposal_attempts
                              if self.config.mode == "dpo_data_collection"
                              else self.config.budget.proposal_attempts_per_round)
            previous_query_feedback: Mapping[str, Any] | None = None
            if prior_attempts:
                previous_document = attempt_records[-1]
                previous_query_feedback = previous_document.get("query_feedback")
            for attempt in range(prior_attempts, proposal_limit):
                if (self.config.mode != "dpo_data_collection" and
                        len(accepted) >= self.config.budget.admitted_opportunities_target):
                    break
                if (self.config.mode == "dpo_data_collection" and
                        dpo_record_count() >= self.config.dpo_collection.target_records):
                    break
                state.stage = "scenario_discovery_query"; self._save(state)
                state.counters["scenario_attempts"] = state.counters.get("scenario_attempts", 0) + 1
                proposal_path = round_dir / "opportunities" / "proposals" / f"attempt_{attempt:03d}.json"
                if proposal_path.is_file():
                    proposal_document = json.loads(proposal_path.read_text(encoding="utf-8"))
                    scenario_attempt = ScenarioQueryAttempt(ScenarioRule(
                        ScenarioPredicate(**proposal_document["scenario_rule"]["scenario"]),
                        str(proposal_document["scenario_rule"]["rationale"]),
                    ), proposal_document.get("raw_query"), None) if proposal_document.get("status") == "valid" else ScenarioQueryAttempt(
                        None, proposal_document.get("raw_query"), proposal_document.get("query_feedback")
                    )
                else:
                    scenario_attempt = self._scenario_rule_proposal(
                        frame, traces, memory, round_index, attempt, metric_values(incumbent_eval),
                        previous_feedback=previous_query_feedback,
                    )
                    atomic_json(proposal_path, {
                        "status": "valid", "scenario_rule": scenario_attempt.scenario_rule,
                        "raw_query": scenario_attempt.raw_query,
                    } if scenario_attempt.scenario_rule is not None else {
                        "status": "invalid", "raw_query": scenario_attempt.raw_query,
                        "query_feedback": scenario_attempt.failure,
                    })
                scenario_rule = scenario_attempt.scenario_rule
                if scenario_rule is None:
                    previous_query_feedback = scenario_attempt.failure
                    document = {"attempt": attempt, "status": "invalid_scenario_query",
                                "query_feedback": scenario_attempt.failure}
                    atomic_json(result_dir / f"attempt_{attempt:03d}.json", document)
                    _append_jsonl_once(attempts_path, document)
                    continue
                previous_query_feedback = None
                scene_hash = scenario_rule.scenario.predicate_hash
                previous = state.scene_memory.get(scene_hash)
                evidence_hash = content_hash(scenario_evidence_ids(scenario_rule.scenario, frame, traces))
                if previous and previous.get("engine_id") == state.incumbent_engine_id and previous.get("evidence_hash") == evidence_hash:
                    document = {"attempt": attempt, "status": "duplicate", "scene": scenario_rule.scenario.canonical}
                    atomic_json(result_dir / f"attempt_{attempt:03d}.json", document)
                    _append_jsonl_once(attempts_path, document)
                    continue
                scene, metric_rows = scenario_evaluation_scope(scenario_rule.scenario, frame)
                scene_eval = self._evaluate(f"round:{round_index}:scene:{scene_hash}:incumbent", incumbent, scene,
                                            round_dir / "opportunities" / scene_hash[:16] / "scene_evaluation",
                                            metric_rows)
                summary = summarize_scenario(scenario_rule.scenario, frame, traces, scene_eval,
                                             incumbent_eval, state.scales)
                state.stage = "opportunity_discovery"; self._save(state)
                discovery = self._opportunity_proposal(
                    scenario_rule, summary, memory, round_index, attempt, incumbent,
                    state.priority_objective,
                )
                if discovery is None:
                    memory.append(scene_hash, "analysis", {"engine_id": state.incumbent_engine_id,
                                  "evidence_hash": content_hash(summary.evidence_ids), "round": round_index,
                                  "scenario_rule": asdict(scenario_rule), "scenario_summary": asdict(summary),
                                  "opportunity_status": "invalid"})
                    document = {"attempt": attempt, "status": "invalid_opportunity",
                                "scenario_rule": scenario_rule, "summary": summary}
                    atomic_json(result_dir / f"attempt_{attempt:03d}.json", document)
                    _append_jsonl_once(attempts_path, document)
                    continue
                if not discovery.opportunities:
                    memory.append(scene_hash, "analysis", {"engine_id": state.incumbent_engine_id,
                                  "evidence_hash": content_hash(summary.evidence_ids), "round": round_index,
                                  "scenario_rule": asdict(scenario_rule), "scenario_summary": asdict(summary),
                                  "llm_scene_summary": discovery.scene_summary,
                                  "opportunity_status": "NO_OPPORTUNITY"})
                    document = {"attempt": attempt, "status": "no_opportunity",
                                "scenario_rule": scenario_rule, "summary": summary,
                                "llm_scene_summary": discovery.scene_summary,
                                "opportunity_status": discovery.status, "opportunities": []}
                    atomic_json(result_dir / f"attempt_{attempt:03d}.json", document)
                    _append_jsonl_once(attempts_path, document)
                    continue
                memory.append(scene_hash, "analysis", {"engine_id": state.incumbent_engine_id,
                              "evidence_hash": content_hash(summary.evidence_ids), "round": round_index,
                              "scenario_rule": asdict(scenario_rule), "scenario_summary": asdict(summary),
                              "llm_scene_summary": discovery.scene_summary,
                              "opportunities": [asdict(item) for item in discovery.opportunities],
                              "opportunity_status": discovery.status})
                assessments: list[CriticAssessment] = []
                attempt_prompts: dict[str, dict[str, Any]] = {}
                accepted_in_scene: list[tuple[Opportunity, CriticAssessment, dict[str, Any]]] = []
                for opportunity in discovery.opportunities:
                    assessment, critic_prompt = self._critic(
                        opportunity, summary, discovery.scene_summary, memory,
                    )
                    assessments.append(assessment)
                    discovered.append(opportunity)
                    scenario_summaries[opportunity.opportunity_id] = asdict(summary)
                    scene_descriptions[opportunity.opportunity_id] = discovery.scene_summary
                    critic_prompts[opportunity.opportunity_id] = critic_prompt
                    attempt_prompts[opportunity.opportunity_id] = critic_prompt
                    memory.append(scene_hash, "critic", {"engine_id": state.incumbent_engine_id,
                                  "evidence_hash": content_hash(summary.evidence_ids), "round": round_index,
                                  "opportunity": asdict(opportunity),
                                  "llm_scene_summary": discovery.scene_summary,
                                  "scenario_summary": asdict(summary), "assessment": asdict(assessment)})
                    if assessment.decision == "ACCEPT":
                        accepted_in_scene.append((opportunity, assessment, critic_prompt))
                if self.config.mode != "dpo_data_collection":
                    critic_accepted_in_scene = list(accepted_in_scene)
                    accepted_in_scene = _select_accepted_for_scene(
                        accepted_in_scene,
                        maximum=self.config.budget.max_evolved_directions_per_scene,
                        remaining=self.config.budget.admitted_opportunities_target - len(accepted),
                        priority_objective=state.priority_objective,
                    )
                    selected_ids = {item[0].opportunity_id for item in accepted_in_scene}
                    for opportunity, assessment, _ in critic_accepted_in_scene:
                        if opportunity.opportunity_id not in selected_ids:
                            memory.append(scene_hash, "pre_evolution_selection", {
                                "engine_id": state.incumbent_engine_id, "round": round_index,
                                "opportunity": asdict(opportunity), "assessment": asdict(assessment),
                                "llm_scene_summary": discovery.scene_summary,
                                "evolution_result": "NOT_RUN",
                            })
                for opportunity, _, critic_prompt in accepted_in_scene:
                    accepted.append(opportunity)
                    _append_jsonl_once(accepted_path, {
                        "opportunity": opportunity, "critic_prompt": critic_prompt,
                    })
                document = {"attempt": attempt, "scenario_rule": scenario_rule,
                            "summary": summary, "llm_scene_summary": discovery.scene_summary,
                            "opportunity_status": discovery.status,
                            "opportunities": discovery.opportunities,
                            "selected_opportunity_ids": [item[0].opportunity_id for item in accepted_in_scene],
                            "assessments": [
                                {"opportunity_id": opportunity.opportunity_id, **asdict(assessment)}
                                for opportunity, assessment in zip(discovery.opportunities, assessments)
                            ],
                            "critic_prompts": attempt_prompts}
                atomic_json(result_dir / f"attempt_{attempt:03d}.json", document)
                _append_jsonl_once(attempts_path, {
                    key: value for key, value in document.items() if key != "critic_prompts"
                })
                if self.config.mode == "dpo_data_collection":
                    evolve_dpo_scene(list(discovery.opportunities))
                if (self.config.mode != "dpo_data_collection" and
                        len(accepted) >= self.config.budget.admitted_opportunities_target):
                    break
            if self.config.mode == "dpo_data_collection":
                state.completed_rounds = round_index
                state.stage = "round_complete"; self._save(state)
                continue
            if not accepted:
                atomic_json(round_dir / "round_result.json", {"reason": "no accepted opportunities", "selected_engine_id": state.incumbent_engine_id})
                if self._finish_no_progress(state, round_index): break
                continue
            state.stage = "local_policy_evolution"; self._save(state)
            self.logger.info("local_policy_evolution_start round=%s opportunities=%s", round_index, len(accepted))
            fixed_e0 = self.run_root / "engines" / "e0"

            outcomes: dict[str, LocalEvolutionResult] = {}
            opportunity_jobs: list[Mapping[str, Any]] = []
            shared_scene_inputs: dict[str, tuple[Path, Path | None]] = {}
            for opportunity in accepted:
                outcome_path = round_dir / "local_policy_evolution" / opportunity.opportunity_id / "outcome.json"
                if outcome_path.is_file():
                    outcomes[opportunity.opportunity_id] = _local_result_from_dict(
                        json.loads(outcome_path.read_text(encoding="utf-8"))
                    )
                    continue
                scene, metric_rows = scenario_evaluation_scope(opportunity.scenario, frame)
                search_base = (fixed_e0 if self.config.candidate_independence_strategy == "evolve_from_fixed_e0"
                               else incumbent)
                scene_baseline = metric_values(self._evaluate(
                    f"round:{round_index}:local-baseline:{opportunity.opportunity_id}",
                    search_base,
                    scene,
                    round_dir / "local_policy_evolution" / opportunity.opportunity_id / "baseline",
                    metric_rows,
                    trace=False,
                    metrics_only=True,
                    evaluator=self.local_candidate_evaluator,
                ))
                scene_key = opportunity.scenario.predicate_hash
                shared_input = shared_scene_inputs.get(scene_key)
                if shared_input is None:
                    shared_root = round_dir / "state" / "local_evolution_inputs" / scene_key
                    shared_root.mkdir(parents=True, exist_ok=True)
                    scene_frame_path = shared_root / "scene_frame.pkl"
                    _atomic_pickle(scene, scene_frame_path)
                    metric_rows_path = shared_root / "metric_rows.pkl" if metric_rows is not None else None
                    if metric_rows_path is not None:
                        _atomic_pickle(metric_rows, metric_rows_path)
                    shared_input = (scene_frame_path, metric_rows_path)
                    shared_scene_inputs[scene_key] = shared_input
                opportunity_jobs.append({
                    "config": self.config,
                    "opportunity": opportunity,
                    "scenario_summary": {
                        **scenario_summaries[opportunity.opportunity_id],
                        "scene_description": scene_descriptions.get(opportunity.opportunity_id, ""),
                    },
                    "incumbent_dir": search_base,
                    "scene_frame_path": shared_input[0],
                    "metric_rows_path": shared_input[1],
                    "baseline_metrics": scene_baseline,
                    "scales": local_metric_scales(
                        scene_baseline, state.scales, epsilon=self.config.epsilon,
                    ),
                    "round_dir": round_dir,
                    "cache_root": self.evaluator.cache_root,
                    "protocol_hash": self.evaluator.protocol_hash,
                    "backend": self.evaluator.backend,
                    "wall_timeout_seconds": self.evaluator.runner.wall_timeout_seconds,
                    "trace_transport_max_bytes": self.evaluator.runner.max_trace_bytes,
                    "candidate_process_workers": self.config.local_candidate_process_workers,
                })

            effective_parallel = min(
                self.config.budget.parallel_opportunities, len(opportunity_jobs),
            ) if opportunity_jobs else 0
            worker_pids: dict[str, int] = {}
            if opportunity_jobs:
                for opportunity_id, result, worker_pid in _run_opportunity_process_pool(
                    opportunity_jobs, effective_parallel,
                ):
                    outcomes[opportunity_id] = result
                    worker_pids[opportunity_id] = worker_pid
            for scene_frame_path, metric_rows_path in shared_scene_inputs.values():
                scene_frame_path.unlink(missing_ok=True)
                if metric_rows_path is not None:
                    metric_rows_path.unlink(missing_ok=True)
            parallelism_path = round_dir / "state" / "opportunity_parallelism.json"
            parallelism = _persist_opportunity_parallelism(
                parallelism_path,
                configured_max_workers=self.config.budget.parallel_opportunities,
                pending_opportunities=len(opportunity_jobs),
                effective_max_workers=effective_parallel,
                worker_pids=worker_pids,
                resumed_outcomes=len(accepted) - len(opportunity_jobs),
            )
            effective_parallel = int(parallelism["effective_max_workers"])
            state.counters["effective_parallel_opportunities"] = effective_parallel
            for opportunity in accepted:
                outcome = outcomes[opportunity.opportunity_id]
                memory.append(opportunity.scenario.predicate_hash, "local_policy_evolution_outcome", {
                    "engine_id": state.incumbent_engine_id, "round": round_index,
                    "opportunity": asdict(opportunity), "success": outcome.success,
                    "evaluated_count": outcome.evaluated_count, "best_delta": outcome.best_delta,
                })
            local_candidates = [candidate for outcome in outcomes.values() for candidate in outcome.candidates]
            self.logger.info("local_policy_evolution_complete round=%s retained_candidates=%s", round_index, len(local_candidates))
            atomic_json(round_dir / "local_candidates.json", {"candidates": local_candidates})
            if self.replay_budget:
                self.replay_budget.check_stopped()
            admitted_candidates: list[LocalCandidate] = []
            for candidate in local_candidates:
                admitted = self._admit_candidate(
                    candidate, fixed_e0=fixed_e0, frame=frame, round_dir=round_dir,
                    replay=(self.config.candidate_independence_strategy ==
                            "incumbent_search_fixed_e0_admission"),
                )
                if admitted is not None:
                    admitted_candidates.append(admitted)
                    _append_jsonl_once(self.run_root / "candidate_library" / "admissions.jsonl", {
                        "round": round_index, "admission_index": len(self._candidate_library()),
                        "candidate": admitted,
                    })
                    self.prompt_ids.get("candidate", admitted.candidate_id)
            local_candidates = admitted_candidates
            library = self._candidate_library()
            incumbent_entry = next(
                (item for item in _archive_from_state(state)
                 if item.engine_id == state.incumbent_engine_id),
                None,
            )
            incumbent_candidate_ids = tuple(
                incumbent_entry.candidate_ids if incumbent_entry is not None else ()
            )
            library_map = {item.candidate_id: item for item in library}
            missing_incumbent = sorted(set(incumbent_candidate_ids) - set(library_map))
            if missing_incumbent:
                raise FileNotFoundError(
                    f"incumbent candidate artifacts are missing: {missing_incumbent}"
                )
            incumbent_candidates = [library_map[item] for item in incumbent_candidate_ids]
            active_candidates = [
                item for item in library if item.candidate_id not in incumbent_candidate_ids
            ]
            incumbent_precedence_edges = tuple(
                incumbent_entry.precedence_edges if incumbent_entry is not None else ()
            )
            atomic_json(round_dir / "candidate_library_view.json", {
                "new_candidates": local_candidates,
                "active_candidates": active_candidates,
                "incumbent_integrated_candidates": incumbent_candidates,
                "incumbent_precedence_edges": incumbent_precedence_edges,
                "stored_candidate_count": len(library),
                "selection_scope": "cumulative_candidates_excluding_incumbent_with_incumbent_context",
            })
            if stop_after_candidate_admission_round == round_index:
                state.stage = "candidate_admission_complete"
                self._save(state)
                atomic_json(round_dir / "state" / "planned_pause.json", {
                    "round": round_index,
                    "stage": state.stage,
                    "preserved_opportunity_outcomes": len(outcomes),
                    "admitted_candidates": len(active_candidates),
                    "reason": "prepare sequential Pareto recombination replay",
                })
                self.logger.info(
                    "planned_pause_after_candidate_admission round=%s candidates=%s",
                    round_index, len(active_candidates),
                )
                return state
            if not active_candidates:
                atomic_json(round_dir / "round_result.json", {"accepted_opportunities": accepted, "local_candidates": [],
                            "reason": "candidate library is empty after fixed-E0 admission", "selected_engine_id": state.incumbent_engine_id})
                if self._finish_no_progress(state, round_index): break
                continue

            state.stage = "relation_graph"; self._save(state)
            relation_candidates = [*incumbent_candidates, *active_candidates]
            all_edges = (build_relation_graph(
                             relation_candidates, frame=frame, llm=self.llm, prompts=self.prompts,
                             pairs_per_call=self.config.budget.relation_pairs_per_call,
                             llm_call=lambda action, prompt:
                             self._llm_call(f"round:{round_index}:{action}", prompt),
                             format_repair_call=lambda action, prompt, response, error:
                             self._repair_llm_format(
                                 f"round:{round_index}:{action}", prompt, response, error,
                             ),
                             existing_edges=_historical_relation_edges(self.run_root),
                             fixed_e0=fixed_e0,
                             prompt_candidate_ids=self._candidate_prompt_ids(
                                 relation_candidates
                             ),
                             experiment_metric_keys=self.config.experiment_metric_keys,
                         ))
            relation_ids = {item.candidate_id for item in relation_candidates}
            edges = tuple(edge for edge in all_edges
                          if edge.left in relation_ids and edge.right in relation_ids)
            atomic_json(round_dir / "relation_graph.json", {
                "selection_scope": "all_incumbent_and_cumulative_selectable_candidates",
                "selectable_candidate_ids": [item.candidate_id for item in active_candidates],
                "incumbent_candidate_ids": incumbent_candidate_ids,
                "edges": edges,
            })
            candidate_map = {item.candidate_id: item for item in relation_candidates}
            combination_checkpoint = round_dir / "state" / "combination_search.json"
            if combination_checkpoint.is_file():
                checkpoint = json.loads(combination_checkpoint.read_text(encoding="utf-8"))
                global_feedback = list(checkpoint["global_feedback"])
                evaluated = int(checkpoint["evaluated"])
                seen_proposals = set(checkpoint["seen_proposals"])
                seen_engines = set(checkpoint["seen_engines"])
                start_attempt = int(checkpoint["next_attempt"])
                # A pre-migration checkpoint may contain the old acceptance-
                # filtered archive. Merge its durable evaluations into the
                # reconciled state frontier instead of letting it overwrite it.
                archive = _archive_from_state(state)
                for checkpoint_entry in _archive_from_documents(checkpoint["archive"]):
                    archive, _ = update_archive(
                        archive, checkpoint_entry,
                        tolerance=self.config.comparison_tolerance,
                        objective_keys=self.config.experiment_objective_keys,
                        rho=self.config.rho, guardrail_keys=self.config.experiment_guardrail_keys,
                    )
                for feedback in global_feedback:
                    entry = self._archive_entry_from_feedback(
                        state, round_index=round_index, round_dir=round_dir,
                        feedback=feedback, candidate_map=candidate_map,
                    )
                    if entry is None:
                        continue
                    archive, archive_update = update_archive(
                        archive, entry, tolerance=self.config.comparison_tolerance,
                        objective_keys=self.config.experiment_objective_keys,
                        rho=self.config.rho, guardrail_keys=self.config.experiment_guardrail_keys,
                    )
                    acceptance_passed = feedback.get("acceptance_passed")
                    if acceptance_passed is None:
                        acceptance_passed = passes_global_acceptance(
                            entry.oriented_delta, rho=self.config.rho,
                            tolerance=self.config.comparison_tolerance,
                            objective_keys=self.config.experiment_objective_keys,
                            guardrail_keys=self.config.experiment_guardrail_keys,
                        )
                    _append_jsonl_once(self.run_root / "pareto_archive" / "events.jsonl", {
                        **archive_update,
                        "round": round_index,
                        "proposal_attempt": int(feedback["proposal_attempt"]),
                        "acceptance_passed": bool(acceptance_passed),
                        "source": "in_progress_combination_checkpoint_reconciliation",
                    })
            else:
                archive = _archive_from_state(state); global_feedback = []
                evaluated = 0; seen_proposals = set(); seen_engines = set(); start_attempt = 0

            def save_combination_checkpoint(next_attempt: int) -> None:
                atomic_json(combination_checkpoint, {
                    "next_attempt": next_attempt, "evaluated": evaluated,
                    "seen_proposals": sorted(seen_proposals), "seen_engines": sorted(seen_engines),
                    "global_feedback": global_feedback, "archive": [asdict(item) for item in archive],
                })

            for attempt in range(start_attempt, self.config.budget.combination_proposals_per_round):
                if evaluated >= self.config.budget.combination_evaluations_per_round:
                    break
                if _combination_search_space_exhausted(len(active_candidates), seen_proposals):
                    self.logger.info(
                        "combination_search_space_exhausted round=%s candidates=%s seen=%s",
                        round_index, len(active_candidates), len(seen_proposals),
                    )
                    break
                proposal = None
                composition_identity: str | None = None
                references = (
                    select_references(
                        archive, self.config.budget.reference_limit, -self.config.rho,
                        self.config.experiment_objective_keys,
                    )
                )
                try:
                    reference_documents = self._reference_documents(
                        archive, references
                    )
                    composition_registry = self._composition_registry()
                    feedback_window = global_feedback[
                        -self.config.budget.combination_feedback_limit:
                    ]
                    prompt_candidate_ids, prompt_reference_ids, prompt_attempt_ids = (
                        self._composition_prompt_maps(
                            candidates=relation_candidates,
                            references=reference_documents,
                            composition_registry=composition_registry,
                            feedback=feedback_window,
                        )
                    )
                    proposal = propose_combination(
                        active_candidates, edges, llm=self.llm, prompts=self.prompts,
                        incumbent_candidates=incumbent_candidates,
                        incumbent_precedence_edges=incumbent_precedence_edges,
                        global_feedback=feedback_window,
                        references=reference_documents,
                        scales=state.scales, rho=self.config.rho,
                        comparison_tolerance=self.config.comparison_tolerance,
                        composition_registry=composition_registry,
                        fixed_e0_id="E0",
                        experiment_metric_keys=self.config.experiment_metric_keys,
                        composition_priority_metric=self.config.composition_priority_metric,
                        prompt_candidate_ids=prompt_candidate_ids,
                        prompt_reference_ids=prompt_reference_ids,
                        prompt_attempt_ids=prompt_attempt_ids,
                        llm_call=lambda action, prompt:
                        self._llm_call(
                            f"round:{round_index}:{action}:{attempt}", prompt,
                        ),
                        format_repair_call=lambda action, prompt, response, error:
                        self._repair_llm_format(
                            f"round:{round_index}:{action}:{attempt}",
                            prompt, response, error,
                        ),
                    )
                    if proposal.proposal_id in seen_proposals:
                        raise ValueError("duplicate combination proposal")
                    seen_proposals.add(proposal.proposal_id)
                    composition_identity = self._register_composition(
                        proposal, fixed_e0=fixed_e0, round_index=round_index, attempt=attempt,
                    )
                    if composition_identity is None:
                        raise ValueError("composition identity was already attempted in an earlier action")
                except Exception as exc:
                    # A malformed, illegal, or already-reserved LLM proposal has no new
                    # Composition identity.  It consumes one proposal attempt but never
                    # masquerades as an engine materialization or full-D evaluation.
                    feedback = {
                        "proposal_attempt": attempt,
                        "evaluated": False,
                        "rejection_stage": "proposal_validation",
                        "rejection": f"{type(exc).__name__}: {exc}",
                    }
                    if proposal is not None:
                        feedback.update(
                            selected_candidate_ids=proposal.candidate_ids,
                            incremental_candidate_ids=proposal.incremental_candidate_ids,
                            conflict_resolution_plan=canonical_conflict_plan(
                                proposal.priority_plan.precedence_edges
                            ),
                            precedence_edges=proposal.priority_plan.precedence_edges,
                            composition_identity=composition_identity,
                        )
                    global_feedback.append(feedback)
                    _append_jsonl_once(self.run_root / "combinations" / "feedback.jsonl", feedback)
                    save_combination_checkpoint(attempt + 1)
                    continue
                try:
                    engine_dir, engine_id = integrate(fixed_e0, candidate_map, proposal, edges,
                        round_dir / "combinations" / f"proposal_{attempt:03d}" / "engine",
                        llm=self.llm, prompts=self.prompts,
                        llm_call=lambda action, prompt: self._llm_call(f"round:{round_index}:{action}", prompt),
                        protocol_hash=self.protocol_hash)
                except Exception as exc:
                    feedback = {
                        "proposal_attempt": attempt,
                        "evaluated": False,
                        "rejection_stage": "materialization",
                        "rejection": f"{type(exc).__name__}: {exc}",
                        "composition_identity": composition_identity,
                        "selected_candidate_ids": proposal.candidate_ids,
                        "incremental_candidate_ids": proposal.incremental_candidate_ids,
                        "conflict_resolution_plan": canonical_conflict_plan(
                            proposal.priority_plan.precedence_edges
                        ),
                        "precedence_edges": proposal.priority_plan.precedence_edges,
                    }
                    self._update_composition_status(
                        composition_identity, "materialization_failed",
                        failure=feedback["rejection"],
                    )
                    global_feedback.append(feedback)
                    _append_jsonl_once(self.run_root / "combinations" / "feedback.jsonl", feedback)
                    save_combination_checkpoint(attempt + 1)
                    continue
                try:
                    self._preflight_engine(engine_dir, fixed_e0, frame, proposal, candidate_map)
                    if engine_id in seen_engines:
                        raise ValueError("duplicate integrated engine")
                    seen_engines.add(engine_id)
                except Exception as exc:
                    feedback = {
                        "proposal_attempt": attempt,
                        "evaluated": False,
                        "rejection_stage": "integration_preflight",
                        "rejection": f"{type(exc).__name__}: {exc}",
                        "composition_identity": composition_identity,
                        "selected_candidate_ids": proposal.candidate_ids,
                        "incremental_candidate_ids": proposal.incremental_candidate_ids,
                        "conflict_resolution_plan": canonical_conflict_plan(
                            proposal.priority_plan.precedence_edges
                        ),
                        "precedence_edges": proposal.priority_plan.precedence_edges,
                    }
                    self._update_composition_status(
                        composition_identity, "validation_rejected",
                        engine_id=engine_id, failure=feedback["rejection"],
                    )
                    global_feedback.append(feedback)
                    _append_jsonl_once(self.run_root / "combinations" / "feedback.jsonl", feedback)
                    save_combination_checkpoint(attempt + 1)
                    continue
                try:
                    evaluation = self._evaluate(f"round:{round_index}:combination:{engine_id}", engine_dir, frame,
                                                round_dir / "combinations" / f"proposal_{attempt:03d}" / "evaluation")
                    metrics = metric_values(evaluation); delta = oriented_delta(metrics, state.baseline_metrics, state.scales)
                    acceptance_passed = passes_global_acceptance(
                        delta, rho=self.config.rho, tolerance=self.config.comparison_tolerance,
                        objective_keys=self.config.experiment_objective_keys,
                        guardrail_keys=self.config.experiment_guardrail_keys,
                    )
                    archive, archive_update = update_archive(
                        archive,
                        ArchiveEntry(
                            engine_id, engine_dir, metrics, delta, round_index,
                            proposal.candidate_ids,
                            conflict_resolution_plan=canonical_conflict_plan(
                                proposal.priority_plan.precedence_edges
                            ),
                            precedence_edges=proposal.priority_plan.precedence_edges,
                            change_size=sum(
                                candidate_map[item].diff_summary.count("\n")
                                for item in proposal.candidate_ids
                            ),
                            candidate_summaries=tuple(
                                self._candidate_archive_summary(candidate_map[item])
                                for item in proposal.candidate_ids
                            ),
                        ),
                        tolerance=self.config.comparison_tolerance,
                        objective_keys=self.config.experiment_objective_keys,
                        rho=self.config.rho, guardrail_keys=self.config.experiment_guardrail_keys,
                    )
                    feedback = {"proposal_attempt": attempt, "proposal": proposal.proposal_id,
                                "composition_identity": composition_identity, "engine_id": engine_id,
                                "selected_candidate_ids": proposal.candidate_ids,
                                "incremental_candidate_ids": proposal.incremental_candidate_ids,
                                "conflict_resolution_plan": canonical_conflict_plan(
                                    proposal.priority_plan.precedence_edges
                                ),
                                "precedence_edges": proposal.priority_plan.precedence_edges,
                                "evaluated": True, "metrics": metrics,
                                "oriented_delta": delta,
                                "acceptance_passed": acceptance_passed,
                                "archive_update": archive_update["status"]}
                    _append_jsonl_once(self.run_root / "pareto_archive" / "events.jsonl", {
                        **archive_update,
                        "round": round_index,
                        "proposal_attempt": attempt,
                        "acceptance_passed": acceptance_passed,
                        "source": "full_search_evaluation",
                    })
                    self._update_composition_status(
                        composition_identity, "evaluated",
                        engine_id=engine_id, oriented_delta=delta,
                        acceptance_passed=acceptance_passed,
                        archive_update=archive_update["status"],
                    )
                except Exception as exc:
                    feedback = {"proposal_attempt": attempt, "engine_id": engine_id, "evaluated": True,
                                "composition_identity": composition_identity,
                                "selected_candidate_ids": proposal.candidate_ids,
                                "incremental_candidate_ids": proposal.incremental_candidate_ids,
                                "conflict_resolution_plan": canonical_conflict_plan(
                                    proposal.priority_plan.precedence_edges
                                ),
                                "precedence_edges": proposal.priority_plan.precedence_edges,
                                "runtime_failure": f"{type(exc).__name__}: {exc}"}
                    self._update_composition_status(
                        composition_identity, "runtime_failed",
                        engine_id=engine_id, failure=feedback["runtime_failure"],
                    )
                evaluated += 1; global_feedback.append(feedback)
                _append_jsonl_once(self.run_root / "combinations" / "feedback.jsonl", feedback)
                save_combination_checkpoint(attempt + 1)
            if evaluated == 0:
                self.logger.warning("combination_search_no_valid_evaluation round=%s", round_index)
                atomic_json(round_dir / "round_result.json", {"reason": "no valid full-D combination", "global_feedback": global_feedback})
                if self._finish_no_progress(state, round_index): break
                continue

            state.stage = "online_uplift"; self._save(state)
            if self.config.online_uplift_enabled:
                from .pairwise_ranking import rank_engines
                ranked, assessment = rank_engines(
                    archive, llm=V2LLM(self.config.online_model) if self.config.online_model else self.critic_llm,
                    context={"dataset_rows": len(frame), "metric_names": list(self.config.experiment_metric_keys),
                             "predictive_feature_summary": frame.loc[:, [name for name in
                                 ("eta", "dar", "pcaa", "dcaa") if name in frame]].describe().to_dict()},
                    objective_keys=self.config.experiment_objective_keys, reference=-self.config.rho,
                )
                selected = ranked[0] if ranked else None
                incumbent_hypervolumes = assessment["hypervolume_contributions"]
                atomic_json(self.run_root / "online_uplift" / "pairwise_ranking.json", assessment)
                selection_rule = "pairwise_wins_then_hypervolume_contribution_then_engine_id"
            else:
                selected, incumbent_hypervolumes = select_incumbent(
                    archive, rho=self.config.rho, tolerance=self.config.comparison_tolerance,
                    current_engine_id=state.incumbent_engine_id,
                    objective_keys=self.config.experiment_objective_keys,
                    guardrail_keys=self.config.experiment_guardrail_keys,
                )
                selection_rule = "offline_fallback_max_individual_hypervolume"
            _append_jsonl_once(self.run_root / "online_uplift" / "assessments.jsonl", {
                "round": round_index, "selection_rule": selection_rule,
                "llm_called": self.config.online_uplift_enabled and len(archive) > 1,
            })
            if selected is None:
                raise RuntimeError("archive contains no selectable engine")
            incumbent = selected.engine_dir; state.incumbent_engine_id = selected.engine_id; state.incumbent_engine_dir = str(selected.engine_dir)
            state.archive = [asdict(item) for item in archive]
            atomic_json(self.run_root / "pareto_archive" / "head.json", {"entries": archive})
            next_priority, priority_evidence = _next_priority_objective(
                global_feedback, selected_engine_id=selected.engine_id,
                incumbent_updated=selected.engine_id != incumbent_before,
                pareto_engine_ids={item.engine_id for item in archive},
                priority_objective_keys=self.config.priority_objective_keys,
            )
            state.priority_objective = next_priority
            state.priority_objective_evidence = priority_evidence
            atomic_json(round_dir / "round_result.json", {"accepted_opportunities": accepted, "local_candidates": local_candidates,
                        "relations": edges, "global_feedback": global_feedback,
                        "selected_engine_id": selected.engine_id,
                        "incumbent_selection_rule": selection_rule,
                        "incumbent_hypervolumes": incumbent_hypervolumes,
                        "next_priority_objective": next_priority,
                        "priority_objective_evidence": priority_evidence})
            state.completed_rounds = round_index; state.stage = "round_complete"
            archive_ids_after = {item["engine_id"] for item in state.archive}
            state.counters["outer_no_progress_rounds"] = 0 if archive_ids_after - archive_ids_before else state.counters.get("outer_no_progress_rounds", 0) + 1
            patience = self.config.budget.outer_early_stopping_patience or 1
            if patience is not None and state.counters["outer_no_progress_rounds"] >= patience:
                state.counters["outer_early_stopped"] = 1
            self._save(state)
            self.logger.info("round_complete round=%s selected=%s archive_size=%s",
                             round_index, state.incumbent_engine_id, len(state.archive))

        if self.config.mode == "dpo_data_collection":
            collected = dpo_record_count()
            state.counters["dpo_target_records"] = self.config.dpo_collection.target_records
            state.counters["dpo_collected_records"] = collected
            state.counters["dpo_record_shortfall"] = max(
                0, self.config.dpo_collection.target_records - collected
            )
            state.stage = ("complete" if collected >= self.config.dpo_collection.target_records
                           else "collection_shortfall")
        else:
            state.stage = "complete" if state.completed_rounds >= self.config.budget.outer_rounds or state.counters.get("outer_early_stopped") else "round_complete"
        return self._finalize_run(state)

    def _finalize_run(self, state: RunState) -> RunState:
        if self.replay_budget:
            cost = self.replay_budget.snapshot()
            if cost["denied_calls"] or cost["remaining"] == 0:
                state.counters["replay_budget_stopped"] = 1
                state.counters["outer_early_stopped"] = 1
                state.stage = "complete"
        if (state.stage == "complete" and self.config.mode == "main" and
                self.config.test_path is not None and not state.counters.get("final_test_evaluations")):
            state.stage = "final_test"; self._save(state)
            test_frame = _load_debug_frame(replace(self.config, mode="local_debug", data_path=self.config.test_path, test_path=None, debug_max_batches=None))
            baseline_test_metrics = None
            normalized_percentage_improvement = None
            if self.config.evaluate_test_baseline:
                baseline_test_result = self._evaluate(
                    "sealed-final-test-current-engine",
                    self.run_root / "engines" / "e0",
                    test_frame,
                    self.run_root / "final" / "test_current_engine", charge_replay=False,
                )
                baseline_test_metrics = metric_values(baseline_test_result)
            final_result = self._evaluate("sealed-final-test", Path(state.incumbent_engine_dir), test_frame, self.run_root / "final" / "test", charge_replay=False)
            final_metrics = metric_values(final_result)
            if baseline_test_metrics is not None:
                test_scales = {
                    key: max(abs(float(baseline_test_metrics[key])), self.config.epsilon)
                    for key in METRIC_KEYS
                }
                normalized_percentage_improvement = {
                    key: 100.0 * value
                    for key, value in oriented_delta(
                        final_metrics, baseline_test_metrics, test_scales,
                    ).items()
                }
            from .paper_metrics import paper_report
            report = (paper_report(final_metrics, baseline_test_metrics,
                                   rho=self.config.rho, epsilon=self.config.epsilon,
                                   tolerance=self.config.comparison_tolerance)
                      if baseline_test_metrics is not None else None)
            atomic_json(self.run_root / "final" / "test_result.json", {
                "engine_id": state.incumbent_engine_id,
                "metrics": final_metrics,
                "current_engine_metrics": baseline_test_metrics,
                "normalized_percentage_improvement_vs_current_engine":
                    normalized_percentage_improvement,
                "normalization": (
                    "100 * direction * (variant - current) / max(abs(current), epsilon)"
                    if baseline_test_metrics is not None else None
                ),
                "test_path": self.config.test_path,
                "feedback_to_search": False,
                "data_split": "test",
                "paper_report": report,
            })
            state.counters["final_test_evaluations"] = 1; state.stage = "complete"
        atomic_json(self.run_root / "final" / "selected_engine.json", {"engine_id": state.incumbent_engine_id,
                    "engine_dir": state.incumbent_engine_dir, "protocol_hash": self.protocol_hash})
        if self.replay_budget:
            atomic_json(self.run_root / "final" / "replay_cost.json", self.replay_budget.snapshot())
        atomic_json(self.run_root / "final" / "transport_summary.json", state.counters)
        self._save(state)
        self._write_selected_engine_manifest(state)
        self._write_run_manifest(state)
        self.logger.info("run_end stage=%s completed_rounds=%s selected=%s",
                         state.stage, state.completed_rounds, state.incumbent_engine_id)
        return state

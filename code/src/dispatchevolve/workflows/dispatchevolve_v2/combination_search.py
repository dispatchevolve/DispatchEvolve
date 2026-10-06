"""Frozen LLM-6 Composition proposal and policy-scoped precedence validation."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from .candidate_semantics import (
    normalized_introduction,
    objective_catalog,
    render_candidate_policy_mapping,
    render_candidate_summaries,
    safe_cell,
)
from .contracts import (CombinationProposal, GUARDRAIL_KEYS, LLMPriorityPlan,
                        LocalCandidate, METRIC_DIRECTIONS, METRIC_KEYS, OBJECTIVE_KEYS,
                        RelationEdge, content_hash)
from .format_repair import ResponseFormatError
from .llm import V2LLM
from .prompt_store import PromptStore
from .relation_graph import required_precedence_pairs, validate_conflict_plan


_RESPONSE = re.compile(
    r"\A## Selected Local Candidate IDs\n+([^\n]+)\n+"
    r"## Conflict Resolution Plan\n+([\s\S]+?)\n+## Rationale\n+([^\n]+)\Z"
)


def _ids(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def canonical_conflict_plan(edges: tuple[Mapping[str, Any], ...]) -> str:
    if not edges:
        return "NONE"
    return ";".join(
        f"{item['higher']}>{item['lower']}@{','.join(sorted(map(str, item['policies'])))}"
        for item in sorted(edges, key=lambda value: (str(value["higher"]), str(value["lower"])))
    )


def _parse_edges(text: str) -> tuple[Mapping[str, Any], ...]:
    if text.strip() == "NONE":
        return ()
    edges = []
    for line in text.strip().splitlines():
        cells = [item.strip() for item in line.split("|")]
        if len(cells) != 3 or not all(cells):
            raise ResponseFormatError(
                "Conflict Resolution Plan lines require HIGHER|LOWER|POLICIES"
            )
        policies = tuple(dict.fromkeys(item.strip() for item in cells[2].split(",") if item.strip()))
        if not policies:
            raise ValueError("Conflict Resolution Plan edge requires shared Policies")
        edges.append({"higher": cells[0], "lower": cells[1], "policies": policies})
    return tuple(edges)


def _candidate_performance(
    candidates: list[LocalCandidate],
    scales: Mapping[str, float],
    metric_keys: tuple[str, ...] = METRIC_KEYS,
    candidate_ids: Mapping[str, str] | None = None,
) -> str:
    rows = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            str(candidate_ids[item.candidate_id])
            if candidate_ids is not None
            else item.candidate_id
        ),
    ):
        introduction = normalized_introduction(candidate)
        batches = int(introduction.get("evaluated_batches", 0))
        coverage = float(introduction.get("scene_coverage", 0.0))
        display_id = (
            str(candidate_ids[candidate.candidate_id])
            if candidate_ids is not None
            else candidate.candidate_id
        )
        for metric in metric_keys:
            if metric not in candidate.oriented_delta or metric not in candidate.metrics:
                # A protocol migration may retain local candidates evaluated
                # before the current guardrail metric existed.  Keep the row
                # visible as unknown; every new full-D composition is still
                # deterministically evaluated under the active guardrail.
                rows.append(" | ".join(safe_cell(value) for value in (
                    display_id, metric, "NA", "NA", "NA", "NA",
                    batches, f"{coverage:.12g}",
                )))
                continue
            normalized = float(candidate.oriented_delta[metric])
            candidate_raw = float(candidate.metrics[metric])
            if metric in GUARDRAIL_KEYS:
                # Local guardrails are normalized by their own Scene baseline.
                # For order_br (positive direction), d=(c-b)/b, so the
                # exact baseline is recoverable from the persisted candidate.
                denominator = 1.0 + normalized
                baseline_raw = (
                    candidate_raw / denominator
                    if abs(denominator) > 1e-12 else candidate_raw
                )
                raw_change = candidate_raw - baseline_raw
            else:
                raw_change = normalized * float(scales[metric]) / METRIC_DIRECTIONS[metric]
                baseline_raw = candidate_raw - raw_change
            rows.append(" | ".join(safe_cell(value) for value in (
                display_id, metric, f"{baseline_raw:.12g}", f"{candidate_raw:.12g}",
                f"{raw_change:.12g}", f"{normalized:.12g}", batches, f"{coverage:.12g}",
            )))
    return "\n".join(rows)


def _relation_evidence(
    edges: tuple[RelationEdge, ...],
    candidate_ids: Mapping[str, str] | None = None,
) -> str:
    rows = []
    visible_edges = []
    for edge in edges:
        if edge.relation == "clear":
            continue
        relation = "hard_conflict" if edge.relation == "hard" else "unresolved"
        evidence = edge.evidence
        left = str(candidate_ids[edge.left]) if candidate_ids is not None else edge.left
        right = str(candidate_ids[edge.right]) if candidate_ids is not None else edge.right
        left, right = sorted((left, right))
        visible_edges.append((left, right, " | ".join(safe_cell(value) for value in (
            left, right, relation,
            ",".join(map(str, evidence.get("shared_policy_files", ()))) or "NONE",
            evidence.get("predicate_intersection_status", "unknown"),
            f"{float(evidence.get('scenario_overlap', {}).get('jaccard', edge.overlap)):.12g}",
            evidence.get("relationship_description", edge.reason),
        ))))
    rows.extend(item[2] for item in sorted(visible_edges))
    return "\n".join(rows) if rows else "NONE"


def _incumbent_strategies(
    candidates: list[LocalCandidate], precedence_edges: tuple[Mapping[str, Any], ...],
    candidate_ids: Mapping[str, str] | None = None,
) -> str:
    if not candidates:
        return "NONE"
    return (
        render_candidate_summaries(candidates, candidate_ids=candidate_ids)
        + "\nINCUMBENT_CONFLICT_RESOLUTION_PLAN="
        + _prompt_conflict_plan(precedence_edges, candidate_ids)
    )


def _metrics(
    value: Mapping[str, Any], metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    return ",".join(
        f"{key}={float(value[key]):.12g}" for key in metric_keys if key in value
    ) or "NONE"


def _prompt_conflict_plan(
    edges: tuple[Mapping[str, Any], ...],
    candidate_ids: Mapping[str, str] | None,
) -> str:
    if not edges:
        return "NONE"
    def display(identifier: Any) -> str:
        value = str(identifier)
        return str(candidate_ids[value]) if candidate_ids is not None else value
    return ";".join(
        f"{display(item['higher'])}>{display(item['lower'])}@"
        f"{','.join(sorted(map(str, item['policies'])))}"
        for item in sorted(
            edges, key=lambda value: (str(value["higher"]), str(value["lower"]))
        )
    )


def _legacy_prompt_conflict_plan(
    plan: Any,
    candidate_ids: Mapping[str, str] | None,
) -> str:
    """Map a legacy canonical plan without exposing unmapped identities."""
    text = str(plan or "NONE").strip()
    if text == "NONE":
        return "NONE"
    if candidate_ids is None:
        return text
    rendered: list[str] = []
    for raw_edge in text.split(";"):
        edge = raw_edge.strip()
        if not edge or ">" not in edge or "@" not in edge:
            return "UNKNOWN"
        higher, remainder = edge.split(">", 1)
        lower, policies = remainder.split("@", 1)
        higher = higher.strip()
        lower = lower.strip()
        if (
            higher not in candidate_ids
            or lower not in candidate_ids
            or not policies.strip()
        ):
            return "UNKNOWN"
        rendered.append(
            f"{candidate_ids[higher]}>{candidate_ids[lower]}@{policies.strip()}"
        )
    return ";".join(rendered) if rendered else "UNKNOWN"


def _pareto_references(
    references: list[dict],
    active_ids: set[str],
    *,
    candidate_ids: Mapping[str, str] | None = None,
    reference_ids: Mapping[str, str] | None = None,
    metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    rows = []
    for item in references:
        selected = tuple(map(str, item.get("candidate_ids", ())))
        summaries = item.get("candidate_summaries", ())
        plan = item.get("conflict_resolution_plan") or "NONE"
        unavailable = sorted(set(selected) - active_ids)
        display_selected = sorted([
            str(candidate_ids[value]) if candidate_ids is not None else value
            for value in selected
        ])
        display_unavailable = sorted([
            str(candidate_ids[value]) if candidate_ids is not None else value
            for value in unavailable
        ])
        engine_id = str(item.get("engine_id"))
        reference_id = (
            str(reference_ids[engine_id]) if reference_ids is not None else engine_id
        )
        rows.append(" | ".join(safe_cell(value) for value in (
            reference_id,
            ",".join(display_selected) or "NONE",
            ",".join(display_unavailable) or "NONE",
            _prompt_conflict_plan(tuple(item.get("precedence_edges", ())), candidate_ids)
            if item.get("precedence_edges")
            else _legacy_prompt_conflict_plan(plan, candidate_ids),
            _metrics(item.get("metrics", {}), metric_keys),
            _metrics(item.get("oriented_delta", {}), metric_keys),
            item.get("reference_reason", "selected by the deterministic Pareto reference policy"),
        )))
    return "\n".join(rows) if rows else "NONE"


def _identity_registry(
    records: list[Mapping[str, Any]],
    *,
    candidate_ids: Mapping[str, str] | None = None,
    attempt_ids: Mapping[str, str] | None = None,
) -> str:
    rows = []
    for item in records:
        acceptance = item.get("acceptance_passed")
        if acceptance is None and item.get("attempt_status") in {"evaluated_feasible", "evaluated_infeasible"}:
            acceptance = item.get("attempt_status") == "evaluated_feasible"
        identity = str(item.get("composition_identity"))
        selected = tuple(map(str, item.get("selected_candidate_ids", ())))
        rows.append(" | ".join(safe_cell(value) for value in (
            str(attempt_ids[identity]) if attempt_ids is not None else identity,
            ",".join(sorted(
                str(candidate_ids[value]) if candidate_ids is not None else value
                for value in selected
            )) or "NONE",
            _prompt_conflict_plan(tuple(item.get("precedence_edges", ())), candidate_ids)
            if item.get("precedence_edges")
            else _legacy_prompt_conflict_plan(
                item.get("conflict_resolution_plan", "NONE"),
                candidate_ids,
            ),
            item.get("attempt_status", "reserved"),
            "NA" if acceptance is None else str(bool(acceptance)).lower(),
        )))
    return "\n".join(rows) if rows else "NONE"


def _feedback(
    records: list[dict],
    *,
    candidate_ids: Mapping[str, str] | None = None,
    attempt_ids: Mapping[str, str] | None = None,
    metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    rows = []
    for item in records:
        acceptance = item.get("acceptance_passed", item.get("feasible"))
        completed = item.get("evaluated") is True and not item.get("runtime_failure")
        if item.get("violated_constraints"):
            assessment = item["violated_constraints"]
        elif completed:
            assessment = "global_acceptance_passed" if acceptance else "global_acceptance_not_passed"
        else:
            assessment = item.get("rejection") or item.get("runtime_failure") or "not_evaluated"
        archive_update = item.get("archive_update", "legacy_result" if completed else "NONE")
        identity = str(item.get("composition_identity") or item.get("proposal"))
        selected = tuple(map(str, item.get("selected_candidate_ids", ())))
        rows.append(" | ".join(safe_cell(value) for value in (
            str(attempt_ids[identity]) if attempt_ids is not None else identity,
            ",".join(sorted(
                str(candidate_ids[value]) if candidate_ids is not None else value
                for value in selected
            )) or "NONE",
            _prompt_conflict_plan(tuple(item.get("precedence_edges", ())), candidate_ids)
            if item.get("precedence_edges")
            else _legacy_prompt_conflict_plan(
                item.get("conflict_resolution_plan", "NONE"),
                candidate_ids,
            ),
            _metrics(item.get("oriented_delta", {}), metric_keys), assessment,
            item.get("feedback") or item.get("rejection") or item.get("runtime_failure") or (
                f"evaluated; archive_update={archive_update}; acceptance_passed={bool(acceptance)}"
            ),
        )))
    return "\n".join(rows) if rows else "NONE"


def propose_combination(
    candidates: list[LocalCandidate], edges: tuple[RelationEdge, ...], *,
    incumbent_candidates: list[LocalCandidate] | None = None,
    incumbent_precedence_edges: tuple[Mapping[str, Any], ...] = (),
    llm: V2LLM, prompts: PromptStore, global_feedback: list[dict], references: list[dict],
    scales: Mapping[str, float], rho: float, comparison_tolerance: float,
    composition_registry: list[Mapping[str, Any]], fixed_e0_id: str,
    llm_call: Callable[[str, object], str] | None = None,
    format_repair_call: Callable[[str, object, str, ResponseFormatError], str] | None = None,
    experiment_metric_keys: tuple[str, ...] = METRIC_KEYS,
    composition_priority_metric: str = "order_ar",
    prompt_candidate_ids: Mapping[str, str] | None = None,
    prompt_reference_ids: Mapping[str, str] | None = None,
    prompt_attempt_ids: Mapping[str, str] | None = None,
) -> CombinationProposal:
    incumbent_candidates = list(incumbent_candidates or ())
    incumbent_ids = tuple(item.candidate_id for item in incumbent_candidates)
    if set(incumbent_ids) & {item.candidate_id for item in candidates}:
        raise ValueError("incumbent and new candidate sets must be disjoint")
    active_ids = {item.candidate_id for item in candidates} | set(incumbent_ids)
    displayed = {
        identifier: (
            str(prompt_candidate_ids[identifier])
            if prompt_candidate_ids is not None
            else identifier
        )
        for identifier in active_ids
    }
    if len(set(displayed.values())) != len(displayed):
        raise ValueError("Prompt Candidate IDs must be unique")
    prompt_to_internal = {value: key for key, value in displayed.items()}
    reward_keys = tuple(
        key for key in experiment_metric_keys if key in OBJECTIVE_KEYS
    )
    guardrail_keys = tuple(
        key for key in experiment_metric_keys if key in GUARDRAIL_KEYS
    )
    if composition_priority_metric not in reward_keys:
        raise ValueError("Composition priority metric must be an active reward")
    reward_text = ",".join(reward_keys) or "NONE"
    guardrail_text = ",".join(guardrail_keys) or "NONE"
    prompt = prompts.render("combination", {
        "experiment_metric_catalog": objective_catalog(
            display_names=False,
            metric_keys=experiment_metric_keys,
        ),
        "composition_priority_metric": composition_priority_metric,
        "incumbent_integrated_strategies": _incumbent_strategies(
            incumbent_candidates, incumbent_precedence_edges, displayed,
        ),
        "local_candidate_summary": render_candidate_summaries(
            candidates, candidate_ids=displayed,
        ),
        "candidate_policy_mapping": render_candidate_policy_mapping(
            candidates,
            candidate_ids=displayed,
            include_flow_steps=False,
        ),
        "candidate_performance": _candidate_performance(
            candidates, scales, experiment_metric_keys, displayed,
        ),
        "local_candidate_evaluation_contract": (
            f"{fixed_e0_id} | each candidate's complete Scene Batches | evaluator-defined mean metrics | "
            "fixed absolute full-search E0 scales for reward metrics and each Scene E0 baseline for active guardrails | "
            "direction_sign * (candidate_raw - baseline_raw) / scale | "
            f"designated primary reward metric > {comparison_tolerance:.12g}; every other active metric "
            f">= -{rho:.12g}; active rewards={reward_text}; active guardrails={guardrail_text}"
        ),
        "full_composition_evaluation_contract": (
            f"{fixed_e0_id} | full Search D | evaluator-defined mean metrics | fixed absolute E0 scale | "
            "direction_sign * (composition_raw - baseline_raw) / scale | "
            "every successfully evaluated Composition updates the non-dominated Pareto archive; "
            f"acceptance_passed only when every active normalized reward >= -{comparison_tolerance:.12g}, "
            f"at least one active reward > {comparison_tolerance:.12g}, and every active guardrail >= -{rho:.12g}; "
            f"active rewards={reward_text}; active guardrails={guardrail_text}; "
            "guardrails are excluded from Pareto dominance, hypervolume, and optimization reward; "
            "acceptance never gates archive maintenance, "
            "search continuation, or references; incumbent promotion requires acceptance and then maximizes "
            "individual active-reward hypervolume from E0"
        ),
        "relation_evidence": _relation_evidence(edges, displayed),
        "pareto_references": _pareto_references(
            references,
            active_ids,
            candidate_ids=prompt_candidate_ids,
            reference_ids=prompt_reference_ids,
            metric_keys=experiment_metric_keys,
        ),
        "composition_attempt_registry": _identity_registry(
            composition_registry,
            candidate_ids=prompt_candidate_ids,
            attempt_ids=prompt_attempt_ids,
        ),
        "recent_composition_attempt_feedback": _feedback(
            global_feedback,
            candidate_ids=prompt_candidate_ids,
            attempt_ids=prompt_attempt_ids,
            metric_keys=experiment_metric_keys,
        ),
    })
    response = llm_call("pareto-composition", prompt) if llm_call else llm.complete(prompt)
    repaired = False
    match = _RESPONSE.fullmatch(response.strip())
    if not match:
        error = ResponseFormatError("Pareto Composition response has an invalid section contract")
        if format_repair_call is None:
            raise error
        response = format_repair_call("pareto-composition", prompt, response, error)
        repaired = True
        match = _RESPONSE.fullmatch(response.strip())
        if not match:
            raise ResponseFormatError(
                "Pareto Composition response remained invalid after one format-only repair"
            )
    selected_prompt = _ids(match.group(1))
    selectable_prompt_ids = {
        displayed[item.candidate_id]: item.candidate_id for item in candidates
    }
    if not selected_prompt or not set(selected_prompt) <= set(selectable_prompt_ids):
        raise ValueError("Pareto Composition selected no candidate or an unknown candidate")
    selected = tuple(selectable_prompt_ids[value] for value in selected_prompt)
    try:
        prompt_precedence = _parse_edges(match.group(2))
    except ResponseFormatError as error:
        if format_repair_call is None or repaired:
            raise
        response = format_repair_call("pareto-composition", prompt, response, error)
        match = _RESPONSE.fullmatch(response.strip())
        if not match:
            raise ResponseFormatError(
                "Pareto Composition response remained invalid after one format-only repair"
            )
        selected_prompt = _ids(match.group(1))
        if not selected_prompt or not set(selected_prompt) <= set(selectable_prompt_ids):
            raise ValueError("Pareto Composition selected no candidate or an unknown candidate")
        selected = tuple(selectable_prompt_ids[value] for value in selected_prompt)
        prompt_precedence = _parse_edges(match.group(2))
    precedence = tuple({
        "higher": prompt_to_internal[str(item["higher"])],
        "lower": prompt_to_internal[str(item["lower"])],
        "policies": item["policies"],
    } for item in prompt_precedence if (
        str(item["higher"]) in prompt_to_internal
        and str(item["lower"]) in prompt_to_internal
    ))
    if len(precedence) != len(prompt_precedence):
        raise ValueError("Conflict Resolution Plan names an unknown Candidate ID")
    combined_ids = tuple(dict.fromkeys((*incumbent_ids, *selected)))
    required = required_precedence_pairs(combined_ids, edges)
    preserved = {
        frozenset((str(item["higher"]), str(item["lower"]))): item
        for item in incumbent_precedence_edges
    }
    if not set(preserved) <= set(required):
        raise ValueError("incumbent precedence contains an edge absent from the cumulative relation graph")
    newly_required = set(required) - set(preserved)
    observed = {
        frozenset((str(item["higher"]), str(item["lower"]))): item
        for item in precedence
    }
    if set(observed) != newly_required:
        raise ValueError("Conflict Resolution Plan must cover every and only newly introduced competition")
    cumulative_precedence = tuple((*incumbent_precedence_edges, *precedence))
    topological_order = validate_conflict_plan(combined_ids, cumulative_precedence, edges)
    rationale = match.group(3).strip()
    plan = LLMPriorityPlan(
        (), topological_order, rationale, llm.config.model, prompt.prompt_hash,
        cumulative_precedence,
    )
    proposal_id = content_hash({
        "selected": sorted(combined_ids),
        "incremental": sorted(selected),
        "conflict_resolution_plan": canonical_conflict_plan(cumulative_precedence),
    })
    return CombinationProposal(proposal_id, combined_ids, plan, rationale, selected)

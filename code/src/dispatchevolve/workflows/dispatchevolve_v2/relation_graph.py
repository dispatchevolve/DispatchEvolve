"""Cumulative deterministic and LLM-described Local Candidate relations."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from .candidate_semantics import (
    normalized_introduction,
    objective_catalog,
    render_candidate_policy_mapping,
    render_candidate_summaries,
    render_engine_policy_flow,
    safe_cell,
)
from .contracts import METRIC_KEYS, LocalCandidate, RelationEdge
from .format_repair import ResponseFormatError
from .llm import V2LLM
from .prompt_store import PromptStore
from .scenario_predicate import predicate_overlap, predicates_provably_disjoint


_LINE = re.compile(r"^([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)$")
_VARIANT_PREFIX = re.compile(r"^v2_[0-9a-f]{12}_(.+)$")


def logical_policy_file(relative: str) -> str:
    name = relative.rsplit("/", 1)[-1]
    while (match := _VARIANT_PREFIX.match(name)) is not None:
        name = match.group(1)
    return f"policies/{name}"


def _policy_change_map(candidate: LocalCandidate) -> dict[str, Mapping[str, object]]:
    introduction = normalized_introduction(candidate)
    return {
        logical_policy_file(str(item["policy"])): item
        for item in introduction.get("policy_mapping", ())
    }


def _structural_evidence(left: LocalCandidate, right: LocalCandidate) -> tuple[list[str], str]:
    left_changes, right_changes = _policy_change_map(left), _policy_change_map(right)
    shared_policies = sorted(set(left_changes) & set(right_changes))
    facts: list[str] = []
    for policy in shared_policies:
        left_item, right_item = left_changes[policy], right_changes[policy]
        left_writes, right_writes = set(left_item.get("writes", ())), set(right_item.get("writes", ()))
        left_reads, right_reads = set(left_item.get("reads", ())), set(right_item.get("reads", ()))
        write_overlap = sorted(left_writes & right_writes)
        dependencies = sorted((left_writes & right_reads) | (right_writes & left_reads))
        if write_overlap:
            facts.append(f"{policy} both write {','.join(write_overlap)}")
        if dependencies:
            facts.append(f"{policy} has read-after-write fields {','.join(dependencies)}")
        if bool(left_item.get("may_filter_rows")) or bool(right_item.get("may_filter_rows")):
            facts.append(f"{policy} may change row eligibility")
    if not facts:
        left_writes = {field for item in left_changes.values() for field in item.get("writes", ())}
        right_writes = {field for item in right_changes.values() for field in item.get("writes", ())}
        left_reads = {field for item in left_changes.values() for field in item.get("reads", ())}
        right_reads = {field for item in right_changes.values() for field in item.get("reads", ())}
        dependency = sorted((left_writes & right_reads) | (right_writes & left_reads) | (left_writes & right_writes))
        facts.append(
            f"cross-Policy shared data fields {','.join(dependency)}" if dependency
            else "no direct shared write or read-after-write fact was proven"
        )
    return shared_policies, "; ".join(facts)


def _pair_record(left: LocalCandidate, right: LocalCandidate, frame: pd.DataFrame) -> dict[str, object]:
    overlap = predicate_overlap(left.scenario, right.scenario, frame)
    shared_policies, structural = _structural_evidence(left, right)
    logical_disjoint = predicates_provably_disjoint(left.scenario, right.scenario)
    predicate_status = "satisfiable" if (
        int(overlap["intersection_batches"]) > 0 or left.scenario_hash == right.scenario_hash
    ) else ("disjoint" if logical_disjoint else "unknown")
    observed = str(overlap["relation"])
    if observed == "disjoint":
        observed = "observed_disjoint"
    evidence = {
        "shared_policy_files": shared_policies,
        "scenario_overlap": overlap,
        "predicate_intersection_status": predicate_status,
        "observed_batch_relation": observed,
        "structural_interaction_evidence": structural,
    }
    return {"left": left.candidate_id, "right": right.candidate_id, "evidence": evidence}


def _render_unresolved_pairs(
    records: list[Mapping[str, object]],
    candidate_ids: Mapping[str, str],
) -> str:
    rows = []
    for item in records:
        evidence = item["evidence"]
        overlap = evidence["scenario_overlap"]
        left = candidate_ids[str(item["left"])]
        right = candidate_ids[str(item["right"])]
        left_batches = overlap["left_batches"]
        right_batches = overlap["right_batches"]
        observed_relation = evidence["observed_batch_relation"]
        if left > right:
            left, right = right, left
            left_batches, right_batches = right_batches, left_batches
            observed_relation = {
                "left_contains_right": "right_contains_left",
                "right_contains_left": "left_contains_right",
            }.get(str(observed_relation), observed_relation)
        rows.append(" | ".join(safe_cell(value) for value in (
            left, right,
            left_batches, right_batches,
            overlap["intersection_batches"], f"{float(overlap['jaccard']):.12g}",
            observed_relation, evidence["predicate_intersection_status"],
            ",".join(evidence["shared_policy_files"]) or "NONE",
            evidence["structural_interaction_evidence"],
        )))
    return "\n".join(rows)


def _parse_relation_chunk(
    response: str, *, chunk: list[Mapping[str, object]],
    by_id: Mapping[str, LocalCandidate],
    classified: Mapping[tuple[str, str], RelationEdge],
    model: str, prompt_hash: str,
    prompt_to_internal: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], RelationEdge]:
    """Validate one complete LLM relation response without partial mutation."""
    response_lines = response.strip().splitlines()
    if not response_lines or any(not _LINE.fullmatch(line.strip()) for line in response_lines):
        raise ResponseFormatError(
            "Candidate Interaction lines require LEFT|RIGHT|AFFECTED_POLICIES|DESCRIPTION"
        )
    parsed: dict[tuple[str, str], RelationEdge] = {}
    for line in response_lines:
        match = _LINE.fullmatch(line.strip())
        if match is None:  # Narrow the optional match before accessing groups.
            raise ResponseFormatError(
                "Candidate Interaction lines require LEFT|RIGHT|AFFECTED_POLICIES|DESCRIPTION"
            )
        raw_ids = (match.group(1).strip(), match.group(2).strip())
        if prompt_to_internal is not None:
            if any(identifier not in prompt_to_internal for identifier in raw_ids):
                raise ResponseFormatError(
                    "Candidate Interaction response names an unknown Candidate ID"
                )
            raw_ids = tuple(prompt_to_internal[identifier] for identifier in raw_ids)
        key = tuple(sorted(raw_ids))
        source = next(
            (item for item in chunk if (item["left"], item["right"]) == key),
            None,
        )
        if source is None:
            raise ResponseFormatError(
                "Candidate Interaction response contains a pair absent from the supplied chunk"
            )
        if key in parsed or key in classified:
            raise ResponseFormatError(
                "Candidate Interaction response repeats an already analyzed pair"
            )
        affected_text = match.group(3).strip()
        affected = (
            []
            if affected_text == "NONE"
            else [
                logical_policy_file(item.strip())
                for item in affected_text.split(",")
                if item.strip()
            ]
        )
        allowed = {
            logical_policy_file(relative)
            for identifier in key for relative in by_id[identifier].policy_files
        }
        unknown = sorted(set(affected) - allowed)
        if unknown:
            raise ResponseFormatError(
                "Candidate Interaction response names an unknown logical Policy not available to the pair: "
                + ",".join(unknown)
            )
        description = match.group(4).strip()
        if not description or description.lower() in {"hard", "clear", "soft", "unresolved"}:
            raise ResponseFormatError(
                "Candidate Interaction must provide a concrete relationship description"
            )
        evidence = dict(source["evidence"])
        evidence.update(affected_policies=affected, relationship_description=description)
        parsed[key] = RelationEdge(
            *key, "unresolved", float(evidence["scenario_overlap"]["jaccard"]),
            description, evidence, model, prompt_hash,
        )
    expected = {(str(item["left"]), str(item["right"])) for item in chunk}
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        raise ResponseFormatError(
            "Candidate Interaction did not analyze every supplied pair exactly once; missing="
            + ",".join(f"{left}:{right}" for left, right in missing)
        )
    return parsed


def _fallback_relation_chunk(
    *, chunk: list[Mapping[str, object]], model: str, prompt_hash: str,
    validation_error: ResponseFormatError,
) -> dict[tuple[str, str], RelationEdge]:
    """Preserve a conservative complete graph when the one repair is still invalid."""
    parsed: dict[tuple[str, str], RelationEdge] = {}
    for source in chunk:
        key = (str(source["left"]), str(source["right"]))
        evidence = dict(source["evidence"])
        affected = list(evidence.get("shared_policy_files", ()))
        structural = str(evidence.get("structural_interaction_evidence", "unavailable"))
        description = (
            "Conservative unresolved relation retained after the LLM format repair "
            f"remained invalid; supplied structural evidence: {structural}"
        )
        evidence.update(
            affected_policies=affected,
            relationship_description=description,
            relationship_description_source="deterministic_contract_fallback",
            llm_validation_error=str(validation_error),
            attempted_llm_model=model,
        )
        parsed[key] = RelationEdge(
            *key, "unresolved", float(evidence["scenario_overlap"]["jaccard"]),
            description, evidence, "deterministic_contract_fallback", prompt_hash,
        )
    return parsed


def build_relation_graph(
    candidates: list[LocalCandidate], *, frame: pd.DataFrame,
    llm: V2LLM, prompts: PromptStore,
    pairs_per_call: int = 30,
    llm_call: Callable[[str, object], str] | None = None,
    format_repair_call: Callable[[str, object, str, ResponseFormatError], str] | None = None,
    existing_edges: tuple[RelationEdge, ...] = (),
    fixed_e0: Path | None = None,
    prompt_candidate_ids: Mapping[str, str] | None = None,
    experiment_metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> tuple[RelationEdge, ...]:
    """Build each new pair once; only system-unresolved pairs consume an LLM call."""
    if pairs_per_call <= 0:
        raise ValueError("pairs_per_call must be positive")
    ordered_candidates = sorted(candidates, key=lambda item: item.candidate_id)
    candidate_ids = {item.candidate_id for item in ordered_candidates}
    displayed = {
        identifier: (
            str(prompt_candidate_ids[identifier])
            if prompt_candidate_ids is not None
            else identifier
        )
        for identifier in candidate_ids
    }
    if len(set(displayed.values())) != len(displayed):
        raise ValueError("Prompt Candidate IDs must be unique")
    prompt_to_internal = {value: key for key, value in displayed.items()}
    classified: dict[tuple[str, str], RelationEdge] = {
        tuple(sorted((edge.left, edge.right))): edge
        for edge in existing_edges
        if edge.left in candidate_ids and edge.right in candidate_ids
    }
    unresolved: list[dict[str, object]] = []
    for left, right in combinations(ordered_candidates, 2):
        key = (left.candidate_id, right.candidate_id)
        if key in classified:
            continue
        record = _pair_record(left, right, frame)
        evidence = record["evidence"]
        shared = list(evidence["shared_policy_files"])
        if left.scenario_hash == right.scenario_hash and shared:
            classified[key] = RelationEdge(
                *key, "hard", float(evidence["scenario_overlap"]["jaccard"]),
                "identical canonical Scenes modify at least one same logical Policy",
                evidence, None, None,
            )
        elif evidence["predicate_intersection_status"] == "disjoint":
            classified[key] = RelationEdge(
                *key, "clear", float(evidence["scenario_overlap"]["jaccard"]),
                "canonical Scene predicates are provably disjoint",
                evidence, None, None,
            )
        else:
            unresolved.append(record)
    by_id = {item.candidate_id: item for item in ordered_candidates}
    context_engine = fixed_e0 or (ordered_candidates[0].engine_dir if ordered_candidates else None)
    for chunk_index, start in enumerate(range(0, len(unresolved), pairs_per_call)):
        chunk = unresolved[start:start + pairs_per_call]
        ids = sorted({str(item[key]) for item in chunk for key in ("left", "right")})
        selected = [by_id[item] for item in ids]
        if context_engine is None:
            raise ValueError("relation construction requires a fixed E0 engine")
        prompt = prompts.render("candidate_relation", {
            "experiment_metric_catalog": objective_catalog(
                metric_keys=experiment_metric_keys,
            ),
            "local_candidate_summary": render_candidate_summaries(
                selected, candidate_ids=displayed,
            ),
            "candidate_policy_mapping": render_candidate_policy_mapping(
                selected,
                engine_dir=context_engine,
                candidate_ids=displayed,
            ),
            "engine_policy_flow": render_engine_policy_flow(context_engine),
            "unresolved_pairs": _render_unresolved_pairs(chunk, displayed),
        })
        action_id = f"candidate-interaction:{chunk_index:03d}"
        response = llm_call(action_id, prompt) if llm_call else llm.complete(prompt)
        try:
            parsed = _parse_relation_chunk(
                response, chunk=chunk, by_id=by_id, classified=classified,
                model=llm.config.model, prompt_hash=prompt.prompt_hash,
                prompt_to_internal=prompt_to_internal,
            )
        except ResponseFormatError as error:
            if format_repair_call is None:
                raise
            response = format_repair_call(action_id, prompt, response, error)
            try:
                parsed = _parse_relation_chunk(
                    response, chunk=chunk, by_id=by_id, classified=classified,
                    model=llm.config.model, prompt_hash=prompt.prompt_hash,
                    prompt_to_internal=prompt_to_internal,
                )
            except ResponseFormatError as repaired_error:
                parsed = _fallback_relation_chunk(
                    chunk=chunk, model=llm.config.model, prompt_hash=prompt.prompt_hash,
                    validation_error=repaired_error,
                )
        classified.update(parsed)
    expected_pairs = set(combinations(sorted(candidate_ids), 2))
    if set(classified) != expected_pairs:
        raise ValueError("relation graph is incomplete")
    return tuple(classified[key] for key in sorted(classified))


def validate_selection(candidate_ids: tuple[str, ...], edges: tuple[RelationEdge, ...]) -> None:
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Composition Local Candidate IDs must be nonempty and unique")
    selected = set(candidate_ids)
    for edge in edges:
        if edge.relation == "hard" and {edge.left, edge.right} <= selected:
            raise ValueError("Composition contains a hard-conflict pair")


def required_precedence_pairs(
    candidate_ids: tuple[str, ...], edges: tuple[RelationEdge, ...],
) -> dict[frozenset[str], tuple[str, ...]]:
    selected = set(candidate_ids)
    required: dict[frozenset[str], tuple[str, ...]] = {}
    for edge in edges:
        if edge.left not in selected or edge.right not in selected or edge.relation == "clear":
            continue
        shared = tuple(sorted(map(str, edge.evidence.get("shared_policy_files", ()))))
        status = str(edge.evidence.get("predicate_intersection_status", "unknown"))
        if shared and status != "disjoint":
            required[frozenset((edge.left, edge.right))] = shared
    return required


def validate_conflict_plan(
    candidate_ids: tuple[str, ...], precedence_edges: tuple[Mapping[str, object], ...],
    relations: tuple[RelationEdge, ...],
) -> tuple[str, ...]:
    validate_selection(candidate_ids, relations)
    selected = set(candidate_ids)
    required = required_precedence_pairs(candidate_ids, relations)
    observed: dict[frozenset[str], Mapping[str, object]] = {}
    adjacency = {item: set() for item in candidate_ids}
    indegree = {item: 0 for item in candidate_ids}
    for edge in precedence_edges:
        higher, lower = str(edge["higher"]), str(edge["lower"])
        if higher == lower or higher not in selected or lower not in selected:
            raise ValueError("Conflict Resolution Plan contains an invalid candidate edge")
        pair = frozenset((higher, lower))
        if pair in observed or pair not in required:
            raise ValueError("Conflict Resolution Plan contains a duplicate or unrelated edge")
        policies = tuple(sorted(map(str, edge.get("policies", ()))))
        if policies != required[pair]:
            raise ValueError("Conflict Resolution Plan must name every and only governed shared Policy")
        observed[pair] = edge
        adjacency[higher].add(lower)
        indegree[lower] += 1
    if set(observed) != set(required):
        raise ValueError("Conflict Resolution Plan omits a required same-Policy competition")
    ready = sorted(item for item, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0); order.append(node)
        for child in sorted(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child); ready.sort()
    if len(order) != len(candidate_ids):
        raise ValueError("Conflict Resolution Plan must be acyclic")
    return tuple(order)


def validate_priority_plan(candidate_ids: tuple[str, ...], ordered_ids: tuple[str, ...],
                           groups: tuple[tuple[str, ...], ...], edges: tuple[RelationEdge, ...]) -> None:
    """Backward-compatible validator for pre-migration fixtures only."""
    validate_selection(candidate_ids, edges)
    if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != set(candidate_ids):
        raise ValueError("legacy priority order must be a complete permutation")
    if groups:
        expected = {frozenset(pair) for pair in required_precedence_pairs(candidate_ids, edges)}
        if {frozenset(group) for group in groups} != expected:
            raise ValueError("legacy overlap groups are incomplete")

"""Code-free, validated Local Candidate semantics for LLM-5 and LLM-6."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import (
    GUARDRAIL_KEYS,
    METRIC_DIRECTIONS,
    METRIC_KEYS,
    LocalCandidate,
    Opportunity,
)
from .local_evaluator import OBJECTIVE_SEMANTICS


def safe_cell(value: Any, *, structural: bool = False) -> str:
    text = " ".join(str(value if value is not None else "NA").split())
    text = text.replace("|", "/")
    if structural:
        text = (text.replace("{", "(").replace("}", ")")
                .replace(";", ",").replace("=", ":"))
    return text or "NA"


def objective_catalog(
    *,
    display_names: bool = False,
    metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    names = {
        "order_ar": "AR", "mean_gmv": "GMV", "mean_eta": "ETA",
        "mean_pcaa": "PCAA", "mean_dcaa": "DCAA",
        "mean_fqs": "CR",
    }
    rows = []
    for name in metric_keys:
        direction = "maximize" if METRIC_DIRECTIONS[name] > 0 else "minimize"
        role = "guardrail" if name in GUARDRAIL_KEYS else "reward"
        values = [name, role]
        if display_names:
            values.append(names.get(name, name))
        values.extend((direction, OBJECTIVE_SEMANTICS[name]["meaning"]))
        rows.append(" | ".join(safe_cell(item) for item in values))
    return "\n".join(rows)


def _subscript_field(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.value, ast.Name) and node.value.id.isupper():
        return None
    value = node.slice
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _assigned_fields(target: ast.AST) -> set[str]:
    fields: set[str] = set()
    direct = _subscript_field(target)
    if direct:
        fields.add(direct)
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
        # frame.loc[mask, "field"] / frame.at[index, "field"]
        selector = target.slice
        if isinstance(selector, ast.Tuple) and selector.elts:
            last = selector.elts[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, str):
                fields.add(last.value)
    for child in ast.iter_child_nodes(target):
        fields.update(_assigned_fields(child))
    return fields


def policy_contract(source: str, *, relative: str) -> Mapping[str, Any]:
    tree = ast.parse(source, relative)
    writes: set[str] = set()
    reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                writes.update(_assigned_fields(target))
        if isinstance(node, ast.Subscript):
            field = _subscript_field(node)
            if field and isinstance(node.ctx, ast.Load):
                reads.add(field)
            if isinstance(node.value, ast.Attribute):
                selector = node.slice
                if isinstance(selector, ast.Tuple) and selector.elts:
                    last = selector.elts[-1]
                    if isinstance(last, ast.Constant) and isinstance(last.value, str):
                        reads.add(last.value)
    may_filter = bool({"is_filtered", "filter_rule", "filter_policy"} & writes) or "is_filtered" in source
    if may_filter:
        role = "filter"
    elif any("lock" in name.lower() or "stage" in name.lower() for name in writes):
        role = "state_control"
    elif {"weight", "matching_score", "score"} & writes or "weight" in source:
        role = "value_model"
    else:
        role = "policy_transform"
    return {
        "logical_policy": relative,
        "role": role,
        "reads": tuple(sorted(reads)),
        "writes": tuple(sorted(writes)),
        "may_filter_rows": may_filter,
    }


def build_candidate_introduction(
    *, candidate_id: str, opportunity: Opportunity, scene_description: str,
    policy_files: tuple[str, ...], baseline_sources: Mapping[str, str],
    evolved_sources: Mapping[str, str], behavior_description: str,
    evaluated_batches: int = 0, scene_coverage: float = 0.0,
) -> Mapping[str, Any]:
    del behavior_description
    description = safe_cell(scene_description or "the validated business Scene represented by this candidate")
    intent = safe_cell(opportunity.improvement_plan or opportunity.rationale)
    policy_mapping = []
    for relative in policy_files:
        before_contract = policy_contract(baseline_sources[relative], relative=relative)
        after_contract = policy_contract(evolved_sources[relative], relative=relative)
        reads = sorted(set(before_contract["reads"]) | set(after_contract["reads"]))
        writes = sorted(set(before_contract["writes"]) | set(after_contract["writes"]))
        policy_mapping.append({
            "policy": relative,
            "role": str(after_contract["role"]),
            "reads": reads,
            "writes": writes,
            "may_filter_rows": bool(after_contract["may_filter_rows"]),
        })
    return {
        "local_candidate_id": candidate_id,
        "scene_description": description,
        "objective": opportunity.objective,
        "opportunity_intent": intent,
        "policy_mapping": policy_mapping,
        "evaluated_batches": int(evaluated_batches),
        "scene_coverage": float(scene_coverage),
    }


def normalized_introduction(candidate: LocalCandidate) -> Mapping[str, Any]:
    if candidate.introduction:
        value = dict(candidate.introduction)
        value["local_candidate_id"] = candidate.candidate_id
        if "opportunity_intent" not in value:
            value["opportunity_intent"] = value.get(
                "intervention_summary",
                "the accepted Opportunity that produced this Local Candidate",
            )
        if "policy_mapping" not in value:
            value["policy_mapping"] = value.get("policy_changes", ())
        return value
    # Compatibility for test fixtures and pre-migration artifacts.  Crucially,
    # this fallback never renders ``diff_summary`` or any source-derived text.
    policy_mapping = []
    for relative in candidate.policy_files:
        contract = policy_contract(
            (candidate.engine_dir / relative).read_text(encoding="utf-8"), relative=relative,
        )
        policy_mapping.append({
            "policy": relative, "role": contract["role"],
            "reads": contract["reads"], "writes": contract["writes"],
            "may_filter_rows": contract["may_filter_rows"],
        })
    return {
        "local_candidate_id": candidate.candidate_id,
        "scene_description": "the validated business Scene represented by this Local Candidate",
        "objective": candidate.objective,
        "opportunity_intent": "the accepted Opportunity that produced this Local Candidate",
        "policy_mapping": policy_mapping,
    }


def _display_id(candidate: LocalCandidate, candidate_ids: Mapping[str, str] | None) -> str:
    return (
        str(candidate_ids[candidate.candidate_id])
        if candidate_ids is not None
        else candidate.candidate_id
    )


def render_candidate_summaries(
    candidates: Iterable[LocalCandidate],
    *,
    candidate_ids: Mapping[str, str] | None = None,
) -> str:
    rows = []
    for candidate in sorted(
        candidates, key=lambda item: _display_id(item, candidate_ids)
    ):
        item = normalized_introduction(candidate)
        rows.append(" | ".join(safe_cell(value) for value in (
            _display_id(candidate, candidate_ids),
            item["scene_description"],
            item["objective"],
            item["opportunity_intent"],
        )))
    return "\n".join(rows) if rows else "NONE"


def render_candidate_introductions(
    candidates: Iterable[LocalCandidate],
    *,
    candidate_ids: Mapping[str, str] | None = None,
) -> str:
    """Compatibility alias for the code-free Candidate summary."""
    return render_candidate_summaries(candidates, candidate_ids=candidate_ids)


def _local_function_nodes(tree: ast.Module, operation: str) -> list[ast.AST] | None:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    root = functions.get(operation)
    if root is None:
        return None
    selected: list[ast.AST] = []
    pending = [root]
    seen: set[str] = set()
    while pending:
        node = pending.pop()
        name = getattr(node, "name", "")
        if name in seen:
            continue
        seen.add(name)
        selected.append(node)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                helper = functions.get(child.func.id)
                if helper is not None and helper.name not in seen:
                    pending.append(helper)
    return selected


def policy_operation_contract(
    source: str, *, relative: str, operation: str,
) -> Mapping[str, Any]:
    tree = ast.parse(source, relative)
    nodes = _local_function_nodes(tree, operation)
    if nodes is None:
        return {
            "logical_policy": relative,
            "operation": operation,
            "role": "UNKNOWN",
            "reads": "UNKNOWN",
            "writes": "UNKNOWN",
            "may_filter_rows": "UNKNOWN",
        }
    fragment = ast.Module(body=[
        node for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ], type_ignores=[])
    contract = policy_contract(ast.unparse(fragment), relative=relative)
    dynamic_reads: set[str] = set()
    field_helpers = {"_numeric", "_flag", "_strings"}
    alias_helpers = {"_numeric_alias"}
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in field_helpers and len(node.args) >= 2:
                value = node.args[1]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    dynamic_reads.add(value.value)
            elif node.func.id in alias_helpers and len(node.args) >= 2:
                value = node.args[1]
                if isinstance(value, (ast.Tuple, ast.List)):
                    dynamic_reads.update(
                        str(item.value)
                        for item in value.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    )
    return {
        **contract,
        "operation": operation,
        "reads": tuple(sorted(set(contract["reads"]) | dynamic_reads)),
    }


def engine_policy_flow(engine_dir: Path) -> tuple[Mapping[str, Any], ...]:
    engine_tree = ast.parse((engine_dir / "engine.py").read_text(encoding="utf-8"), "engine.py")
    aliases: dict[str, str] = {}
    for node in engine_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "policies":
            aliases.update({alias.asname or alias.name: f"policies/{alias.name}.py" for alias in node.names})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("policies."):
                    aliases[alias.asname or alias.name.split(".")[-1]] = alias.name.replace(".", "/") + ".py"
    calls = sorted((
        node for node in ast.walk(engine_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in aliases
    ), key=lambda node: (node.lineno, node.col_offset))
    flow = []
    for position, call in enumerate(calls, 1):
        relative = aliases[call.func.value.id]
        path = engine_dir / relative
        operation = call.func.attr
        contract = (
            policy_operation_contract(
                path.read_text(encoding="utf-8"),
                relative=relative,
                operation=operation,
            )
            if path.is_file()
            else {
                "role": "UNKNOWN",
                "reads": "UNKNOWN",
                "writes": "UNKNOWN",
                "may_filter_rows": "UNKNOWN",
            }
        )
        flow.append({
            "step": f"{position:02d}",
            "policy": relative,
            "operation": operation,
            "role": contract["role"],
            "reads": contract["reads"],
            "writes": contract["writes"],
            "may_filter_rows": contract["may_filter_rows"],
        })
    return tuple(flow)


def render_engine_policy_flow(engine_dir: Path) -> str:
    rows = []
    for item in engine_policy_flow(engine_dir):
        reads = (
            item["reads"]
            if isinstance(item["reads"], str)
            else ",".join(map(str, item["reads"])) or "NONE"
        )
        writes = (
            item["writes"]
            if isinstance(item["writes"], str)
            else ",".join(map(str, item["writes"])) or "NONE"
        )
        may_filter = (
            item["may_filter_rows"]
            if isinstance(item["may_filter_rows"], str)
            else str(bool(item["may_filter_rows"])).lower()
        )
        rows.append(" | ".join(safe_cell(value) for value in (
            item["step"], item["policy"], item["operation"],
            item["role"], reads, writes, may_filter,
        )))
    return "\n".join(rows) if rows else "NONE"


def render_candidate_policy_mapping(
    candidates: Iterable[LocalCandidate],
    *,
    engine_dir: Path | None = None,
    candidate_ids: Mapping[str, str] | None = None,
    include_flow_steps: bool = True,
) -> str:
    flow = engine_policy_flow(engine_dir) if engine_dir is not None else ()
    rows = []
    for candidate in sorted(
        candidates, key=lambda item: _display_id(item, candidate_ids)
    ):
        for relative in candidate.policy_files:
            steps = ",".join(
                str(item["step"]) for item in flow if item["policy"] == relative
            )
            values = [_display_id(candidate, candidate_ids), relative]
            if include_flow_steps:
                values.append(steps or "UNKNOWN")
            rows.append(" | ".join(safe_cell(value) for value in values))
    return "\n".join(rows) if rows else "NONE"


def render_policy_interaction_context(
    engine_dir: Path, candidates: Iterable[LocalCandidate],
) -> str:
    """Compatibility alias for the complete ordered operation-level flow."""
    del candidates
    return render_engine_policy_flow(engine_dir)

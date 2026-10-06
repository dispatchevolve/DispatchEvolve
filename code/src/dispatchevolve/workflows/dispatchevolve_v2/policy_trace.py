"""Deterministic Policy attribution and Prompt-text production for V2 scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from dispatchevolve.tasks.full_dispatch.utils import summarize_matches

from .contracts import DecisionTrace, METRIC_KEYS


@dataclass(frozen=True)
class PolicySemantics:
    policy_type: str
    purpose: str
    decisions: Mapping[str, str]


# Public adapters infer generic semantics from observed effects. No production
# policy names or descriptions are needed by the core trace summarizer.
POLICY_SEMANTICS: Mapping[str, PolicySemantics] = {}


def _observed_semantics(events):
    effects = set()
    decisions = {}
    for event in events:
        fields = event.get("modified_rows_by_field") or {}
        if "weight" in fields: effects.add("score")
        if "stage" in fields: effects.add("stage")
        if (event.get("newly_filtered_rows") or {}).get("count", 0): effects.add("filter")
        decisions[str(event.get("policy_call") or "")] = "observed decision"
    kind = next(iter(effects)) if len(effects) == 1 else "mixed" if effects else "filter"
    return PolicySemantics(kind, "Policy effects inferred from execution traces.", decisions)


def _decode_row_evidence(evidence: Any, universe_size: int, label: str) -> set[int]:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{label} must be row evidence")
    count = int(evidence.get("count", -1))
    bitmap_hex = evidence.get("bitmap_hex")
    if count < 0 or not isinstance(bitmap_hex, str):
        raise ValueError(f"{label} has an invalid count or bitmap")
    try:
        bitmap = bytes.fromhex(bitmap_hex)
    except ValueError as exc:
        raise ValueError(f"{label} bitmap is not hexadecimal") from exc
    positions = {
        position
        for position in range(universe_size)
        if position // 8 < len(bitmap) and bitmap[position // 8] & (1 << (position % 8))
    }
    overflow = any(
        bitmap[position // 8] & (1 << (position % 8))
        for position in range(universe_size, len(bitmap) * 8)
    )
    if overflow or len(positions) != count:
        raise ValueError(f"{label} bitmap does not match its universe or count")
    return positions


def _trace_frame(frame: pd.DataFrame, trace: DecisionTrace) -> pd.DataFrame:
    batch = frame.loc[frame["batch_id"].astype(str).eq(trace.batch_id)]
    scope = trace.input_reference.get("row_order_scope")
    if scope == "target_category_subsequence":
        batch = batch.loc[pd.to_numeric(batch["product_id"], errors="coerce").eq(1)]
    elif scope != "full_batch":
        raise ValueError(f"unsupported trace row-order scope: {scope}")
    expected = int(trace.final_result.get("input_rows") or 0)
    if len(batch) != expected:
        raise ValueError(
            f"trace batch {trace.batch_id} row universe differs: frame={len(batch)} trace={expected}"
        )
    return batch.reset_index(drop=True)


def _changed_positions(event: Mapping[str, Any], semantics: PolicySemantics, universe: int) -> set[int]:
    if semantics.policy_type == "mixed":
        positions = _decode_row_evidence(event.get("newly_filtered_rows"), universe, "newly_filtered_rows")
        for field, evidence in (event.get("modified_rows_by_field") or {}).items():
            if field in {"weight", "stage"}:
                positions |= _decode_row_evidence(evidence, universe, f"modified.{field}")
        return positions
    if semantics.policy_type == "filter":
        return _decode_row_evidence(event.get("newly_filtered_rows"), universe, "newly_filtered_rows")
    field = "weight" if semantics.policy_type == "score" else "stage"
    modified = event.get("modified_rows_by_field") or {}
    evidence = modified.get(field)
    return set() if evidence is None else _decode_row_evidence(evidence, universe, f"modified.{field}")


def _quantiles(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return []
    return [float(numeric.quantile(q)) for q in (0.1, 0.5, 0.9)]


def _cohort_metrics(
    batch_ids: set[str],
    batch_frames: Mapping[str, pd.DataFrame],
    matched_positions: Mapping[str, set[int]],
) -> dict[str, float]:
    source_parts = [
        batch_frames[key].loc[
            pd.to_numeric(batch_frames[key]["product_id"], errors="coerce").eq(1)
        ]
        for key in sorted(batch_ids)
    ]
    match_parts = [
        batch_frames[key].iloc[sorted(matched_positions[key])].loc[
            lambda value: pd.to_numeric(value["product_id"], errors="coerce").eq(1)
        ]
        for key in sorted(batch_ids)
    ]
    source = pd.concat(source_parts, ignore_index=True) if source_parts else pd.DataFrame()
    matches = pd.concat(match_parts, ignore_index=True) if match_parts else source.iloc[0:0]
    metrics = summarize_matches(matches, source_df=source)
    missing = [key for key in METRIC_KEYS if metrics.get(key) is None]
    if missing:
        raise ValueError(f"Policy cohort evaluator omitted objectives: {missing}")
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def build_policy_trace_evidence(
    frame: pd.DataFrame,
    traces: tuple[DecisionTrace, ...],
) -> dict[str, Any]:
    """Build exact scene Policy effects and observational performance associations."""
    if "batch_id" not in frame:
        raise ValueError("Policy trace evidence requires batch_id")
    trace_by_batch = {trace.batch_id: trace for trace in traces}
    if len(trace_by_batch) != len(traces):
        raise ValueError("Policy trace evidence requires one trace per batch")
    batch_frames = {key: _trace_frame(frame, trace) for key, trace in trace_by_batch.items()}
    matched_positions = {
        key: _decode_row_evidence(
            trace.final_result.get("final_matched_target_rows"),
            len(batch_frames[key]),
            "final_matched_target_rows",
        )
        for key, trace in trace_by_batch.items()
    }
    eligible_positions = {
        key: _decode_row_evidence(
            trace.final_result.get("final_eligible_target_rows"),
            len(batch_frames[key]),
            "final_eligible_target_rows",
        )
        for key, trace in trace_by_batch.items()
    }

    events_by_policy = {}
    for trace in trace_by_batch.values():
        for event in trace.ordered_policy_events:
            events_by_policy.setdefault(str(event.get("policy_file") or ""), []).append(event)
    inferred = {policy: _observed_semantics(events) for policy, events in events_by_policy.items()}
    policies: dict[str, dict[str, Any]] = {}
    for batch_id, trace in trace_by_batch.items():
        batch = batch_frames[batch_id]
        final_positions = matched_positions[batch_id]
        for event in trace.ordered_policy_events:
            policy = str(event.get("policy_file") or "")
            semantics = POLICY_SEMANTICS.get(policy) or inferred[policy]
            if semantics is None or semantics.policy_type == "lock":
                continue
            policy_call = str(event.get("policy_call") or "")
            decision = semantics.decisions.get(policy_call)
            if decision is None:
                raise ValueError(f"Policy Catalog omits decision semantics for {policy_call}")
            universe = len(batch)
            applicable = _decode_row_evidence(
                event.get("applicable_before_rows"), universe, "applicable_before_rows"
            )
            eligible_before = _decode_row_evidence(
                event.get("eligible_before_rows"), universe, "eligible_before_rows"
            )
            newly_filtered = _decode_row_evidence(
                event.get("newly_filtered_rows"), universe, "newly_filtered_rows"
            )
            if not applicable <= eligible_before or not newly_filtered <= applicable:
                raise ValueError(f"Policy event sets are inconsistent for {policy_call}")
            changed = _changed_positions(event, semantics, universe)
            if not changed <= eligible_before:
                raise ValueError(f"Policy changed rows outside eligible-before scope for {policy_call}")

            item = policies.setdefault(policy, {
                "policy": policy,
                "policy_type": semantics.policy_type,
                "purpose": semantics.purpose,
                "applicable_batches": set(),
                "binding_batches": set(),
                "decisions": {},
            })
            if applicable:
                item["applicable_batches"].add(batch_id)
            if changed:
                item["binding_batches"].add(batch_id)
            decision_item = item["decisions"].setdefault(policy_call, {
                "decision": decision,
                "binding_batches": set(),
                "applicable_pairs": set(),
                "changed_pairs": set(),
                "matched_changed_pairs": set(),
                "orders_with_candidates_before": set(),
                "orders_losing_all_candidates": set(),
            })
            if changed:
                decision_item["binding_batches"].add(batch_id)
            decision_item["applicable_pairs"].update((batch_id, position) for position in applicable)
            decision_item["changed_pairs"].update((batch_id, position) for position in changed)
            decision_item["matched_changed_pairs"].update(
                (batch_id, position) for position in changed & final_positions
            )
            if semantics.policy_type == "filter":
                order_ids = batch["order_id"].astype("string").fillna("<NA>")
                before_by_order: dict[str, set[int]] = {}
                after_by_order: dict[str, set[int]] = {}
                for position in eligible_before:
                    order = str(order_ids.iloc[position])
                    before_by_order.setdefault(order, set()).add(position)
                    if position not in newly_filtered:
                        after_by_order.setdefault(order, set()).add(position)
                decision_item["orders_with_candidates_before"].update(
                    (batch_id, order) for order in before_by_order
                )
                decision_item["orders_losing_all_candidates"].update(
                    (batch_id, order) for order in before_by_order if not after_by_order.get(order)
                )

    rendered_policies: list[dict[str, Any]] = []
    for policy in sorted(policies):
        item = policies[policy]
        binding = set(item["binding_batches"])
        applicable_nonbinding = set(item["applicable_batches"]) - binding
        association = None
        if binding and applicable_nonbinding:
            binding_metrics = _cohort_metrics(binding, batch_frames, matched_positions)
            nonbinding_metrics = _cohort_metrics(applicable_nonbinding, batch_frames, matched_positions)
            association = {
                "binding_batches": len(binding),
                "applicable_nonbinding_batches": len(applicable_nonbinding),
                "binding_raw": binding_metrics,
                "applicable_nonbinding_raw": nonbinding_metrics,
                "binding_minus_nonbinding_raw": {
                    key: binding_metrics[key] - nonbinding_metrics[key]
                    for key in METRIC_KEYS
                },
            }
        decisions = []
        for policy_call, decision_item in item["decisions"].items():
            if not decision_item["applicable_pairs"] and not decision_item["changed_pairs"]:
                continue
            decisions.append({
                "policy_call": policy_call,
                "decision": decision_item["decision"],
                "binding_batches": len(decision_item["binding_batches"]),
                "applicable_od_pairs": len(decision_item["applicable_pairs"]),
                "changed_od_pairs": len(decision_item["changed_pairs"]),
                "matched_changed_od_pairs": len(decision_item["matched_changed_pairs"]),
                "orders_with_candidates_before": len(decision_item["orders_with_candidates_before"]),
                "orders_losing_all_candidates": len(decision_item["orders_losing_all_candidates"]),
            })
        if not decisions:
            continue
        rendered_policies.append({
            "policy": policy,
            "policy_type": item["policy_type"],
            "purpose": item["purpose"],
            "binding_batches": len(binding),
            "applicable_nonbinding_batches": len(applicable_nonbinding),
            "decisions": decisions,
            "performance_association": association,
        })

    source_order_instances: set[tuple[str, str]] = set()
    eligible_order_instances: set[tuple[str, str]] = set()
    matched_order_instances: set[tuple[str, str]] = set()
    source_driver_instances: set[tuple[str, str]] = set()
    eligible_driver_instances: set[tuple[str, str]] = set()
    order_degrees: list[int] = []
    driver_degrees: list[int] = []
    source_od_pair_count = 0
    for batch_id, batch in batch_frames.items():
        source = batch.loc[pd.to_numeric(batch["product_id"], errors="coerce").eq(1)]
        source_od_pair_count += len(source)
        source_orders = source["order_id"].astype("string").fillna("<NA>")
        source_drivers = source["driver_id"].astype("string").fillna("<NA>")
        source_order_instances.update((batch_id, str(value)) for value in source_orders.unique())
        source_driver_instances.update((batch_id, str(value)) for value in source_drivers.unique())
        eligible = batch.iloc[sorted(eligible_positions[batch_id])]
        matched = batch.iloc[sorted(matched_positions[batch_id])]
        eligible_orders = eligible["order_id"].astype("string").fillna("<NA>")
        eligible_drivers = eligible["driver_id"].astype("string").fillna("<NA>")
        matched_orders = matched["order_id"].astype("string").fillna("<NA>")
        eligible_order_instances.update((batch_id, str(value)) for value in eligible_orders.unique())
        eligible_driver_instances.update((batch_id, str(value)) for value in eligible_drivers.unique())
        matched_order_instances.update((batch_id, str(value)) for value in matched_orders.unique())
        if len(eligible):
            order_degrees.extend(eligible_orders.value_counts(sort=False).astype(int).tolist())
            driver_degrees.extend(eligible_drivers.value_counts(sort=False).astype(int).tolist())

    return {
        "scene_batches": len(trace_by_batch),
        "policies": rendered_policies,
        "matching": {
            "source_od_pairs": source_od_pair_count,
            "source_order_instances": len(source_order_instances),
            "source_driver_instances": len(source_driver_instances),
            "eligible_od_pairs": sum(len(value) for value in eligible_positions.values()),
            "eligible_order_instances": len(eligible_order_instances),
            "eligible_driver_instances": len(eligible_driver_instances),
            "orders_without_eligible_candidates": len(source_order_instances - eligible_order_instances),
            "eligible_drivers_per_order_p10_p50_p90": _quantiles(pd.Series(order_degrees, dtype=float)),
            "eligible_orders_per_driver_p10_p50_p90": _quantiles(pd.Series(driver_degrees, dtype=float)),
            "matched_od_pairs": sum(len(value) for value in matched_positions.values()),
            "matched_order_instances": len(matched_order_instances),
            "unmatched_orders_with_eligible_candidates": len(eligible_order_instances - matched_order_instances),
        },
    }


def _number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:+.1f}"
    if abs(value) >= 1:
        return f"{value:+.3f}"
    return f"{value:+.6f}"


def render_policy_trace_text(
    evidence: Mapping[str, Any],
    *,
    policy_scope: tuple[str, ...] | None = None,
    metric_keys: tuple[str, ...] = METRIC_KEYS,
) -> str:
    """Render one canonical Trace block, optionally scoped to exact Policy paths."""
    scene_batches = int(evidence["scene_batches"])
    policies = list(evidence["policies"])
    if policy_scope is not None:
        requested = tuple(dict.fromkeys(str(path) for path in policy_scope))
        available = {str(item["policy"]) for item in policies}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"Policy Trace scope is not present in scene evidence: {missing}")
        requested_set = set(requested)
        policies = [item for item in policies if str(item["policy"]) in requested_set]
    lines = [
        "### Policy Trace Evidence",
        "",
        "The evidence is observational: it identifies Policy activity and associated performance, not the result of changing a Policy.",
    ]
    sections = (("filter", "Filter Policy Evidence"), ("score", "Score Policy Evidence"), ("stage", "Stage Policy Evidence"), ("mixed", "Mixed Policy Evidence"))
    for policy_type, title in sections:
        items = [item for item in policies if item["policy_type"] == policy_type]
        if not items:
            continue
        lines.extend(["", f"#### {title}"])
        if policy_type == "filter":
            lines.extend(["", "Decision fields: decision | binding_batches/scene_batches | applicable_od_pairs | first_filtered_od_pairs/applicable_od_pairs | orders_losing_all_candidates/orders_with_candidates_before"])
        else:
            changed_name = {"score": "weight_changed_od_pairs", "stage": "stage_changed_od_pairs", "mixed": "changed_od_pairs"}[policy_type]
            lines.extend(["", f"Decision fields: decision | binding_batches/scene_batches | applicable_od_pairs | {changed_name}/applicable_od_pairs | finally_matched_changed_od_pairs"])
        for item in items:
            lines.extend(["", f"Policy: {item['policy']}", f"Purpose: {item['purpose']}"])
            for decision in item["decisions"]:
                common = (
                    f"- {decision['decision']} | {decision['binding_batches']}/{scene_batches} | "
                    f"{decision['applicable_od_pairs']} | {decision['changed_od_pairs']}/{decision['applicable_od_pairs']} | "
                )
                if policy_type == "filter":
                    common += (
                        f"{decision['orders_losing_all_candidates']}/"
                        f"{decision['orders_with_candidates_before']} | "
                    )
                lines.append(common.rstrip(" |") if policy_type == "filter"
                             else common + str(decision["matched_changed_od_pairs"]))
            association = item.get("performance_association")
            if association:
                vector = ",".join(
                    f"{key}:{_number(float(association['binding_minus_nonbinding_raw'][key]))}"
                    for key in metric_keys
                )
                lines.append(
                    "Performance association: "
                    f"binding_batches={association['binding_batches']};"
                    f"applicable_nonbinding_batches={association['applicable_nonbinding_batches']};"
                    f"binding_minus_nonbinding_raw={vector}"
                )

    matching = evidence["matching"]
    lines.extend([
        "",
        "#### Scene Matching Evidence",
        "",
        f"complete_batches={scene_batches}",
        f"source_od_pairs={matching['source_od_pairs']}",
        f"source_order_instances={matching['source_order_instances']}",
        f"source_driver_instances={matching['source_driver_instances']}",
        f"eligible_od_pairs={matching['eligible_od_pairs']}",
        f"eligible_order_instances={matching['eligible_order_instances']}",
        f"eligible_driver_instances={matching['eligible_driver_instances']}",
        f"orders_without_eligible_candidates={matching['orders_without_eligible_candidates']}",
        "eligible_drivers_per_order_p10_p50_p90=" + ",".join(map(str, matching["eligible_drivers_per_order_p10_p50_p90"])),
        "eligible_orders_per_driver_p10_p50_p90=" + ",".join(map(str, matching["eligible_orders_per_driver_p10_p50_p90"])),
        f"matched_od_pairs={matching['matched_od_pairs']}",
        f"matched_order_instances={matching['matched_order_instances']}",
        f"unmatched_orders_with_eligible_candidates={matching['unmatched_orders_with_eligible_candidates']}",
        "",
        "### Policy Trace Input Semantics",
        "",
        "Each input row is one candidate order-driver pair. A complete batch is one joint dispatch decision. The scene contains all target rows from every complete batch selected by the Query. All counts and statistics use the full scene without sampling.",
        "",
        "A Policy path is the editable repository-relative file. Purpose states what that Policy controls. Each decision row is one independently attributable branch inside that file. binding_batches counts complete scene batches where that decision changed at least one still-eligible row.",
        "",
        "applicable_od_pairs reached the decision while still eligible and satisfied its business branch and preconditions. A Filter's first_filtered_od_pairs were first rejected at that decision; rows rejected earlier are not counted again. orders_losing_all_candidates counts batch-order instances whose eligible-driver count changed from positive to zero at that decision. finally_matched_changed_od_pairs is the exact intersection between changed rows and final Matching output.",
        "",
        "A Policy binding batch contains at least one row whose eligibility, weight, or stage changed. An applicable non-binding batch contains applicable rows but no such change. Batches where the Policy was not applicable belong to neither cohort.",
        "",
        "binding_minus_nonbinding_raw contains the active experiment metrics evaluated on the union of complete binding batches minus the same metrics on the union of complete applicable non-binding batches. It is shown only when both cohorts contain at least one complete batch. It is an observational association, because the two cohorts may differ in demand, supply, and candidate composition.",
        "",
        "Only metrics listed in the current Experiment Metric Catalog are projected into this performance-association vector; the Trace structure and Policy evidence are otherwise unchanged.",
        "",
        "An order instance or driver instance is identified within one batch. eligible rows remain after every Filter Policy. An order without eligible candidates has no remaining driver before Matching. An unmatched order with eligible candidates retained at least one pair but was not selected by the one-to-one batch assignment.",
    ])
    return "\n".join(lines)


__all__ = [
    "POLICY_SEMANTICS",
    "PolicySemantics",
    "build_policy_trace_evidence",
    "render_policy_trace_text",
]

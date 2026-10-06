"""Decision-trace normalization and deterministic scenario summaries."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from dispatchevolve.baselines.candidates import RepositoryGenomeCodec

from .contracts import METRIC_KEYS, DecisionTrace, ScenarioPredicate, ScenarioSummary, content_hash
from .local_evaluator import metric_values
from .policy_trace import build_policy_trace_evidence, render_policy_trace_text
from .scenario_predicate import coverage_ratio, scenario_batch_ids, scenario_target_mask


def normalize_traces(engine_dir: Path, frame: pd.DataFrame, evaluation: Mapping[str, Any]) -> tuple[DecisionTrace, ...]:
    engine_id = RepositoryGenomeCodec().candidate_id(engine_dir)
    raw_traces = evaluation.get("batch_traces")
    if raw_traces is None and evaluation.get("batch_traces_artifact"):
        with gzip.open(Path(str(evaluation["batch_traces_artifact"])), "rt", encoding="utf-8") as handle:
            raw_traces = json.load(handle)
    raw_by_batch = {str(item.get("batch_id")): dict(item) for item in (raw_traces or [])}
    traces: list[DecisionTrace] = []
    for batch_id, batch in frame.groupby("batch_id", sort=False, dropna=False):
        target_count = int(pd.to_numeric(batch["product_id"], errors="coerce").eq(1).sum())
        if target_count == 0:
            continue
        raw = raw_by_batch.get(str(batch_id), {})
        raw_events = raw.get("policy_events")
        if not isinstance(raw_events, list):
            raise ValueError(
                "V2 requires runtime policy_events; static AST reconstruction is not a valid decision trace"
            )
        events = tuple({"order": index, **dict(event)} for index, event in enumerate(raw_events))
        stable_fields = [name for name in ("batch_id", "uuid", "order_id", "driver_id", "product_id") if name in batch]
        row_reference_hash = content_hash([
            {name: None if pd.isna(row[name]) else str(row[name]) for name in stable_fields}
            for _, row in batch.iterrows()
        ])
        input_reference = {
            "batch_id": str(batch_id), "row_count": int(len(batch)),
            "target_category_rows": target_count,
            "row_order_scope": ("full_batch" if int(raw.get("input_rows") or 0) == len(batch)
                                else "target_category_subsequence"
                                if int(raw.get("input_rows") or 0) == target_count else "unknown"),
            "schema_hash": content_hash([(str(name), str(dtype)) for name, dtype in batch.dtypes.items()]),
            "row_reference_hash": row_reference_hash,
        }
        if input_reference["row_order_scope"] == "unknown":
            raise ValueError("trace input row count does not identify its row-order coordinate system")
        final = {key: raw.get(key) for key in (
            "input_rows", "output_rows", "target_row_count", "target_output_rows",
            "passthrough_row_count", "filtered_count", "filter_rule_counts", "filter_policy_counts",
            "input_target_rows", "final_eligible_target_rows",
            "final_match_references", "final_match_count", "final_matched_target_rows",
        )}
        trace_id = content_hash({"engine": engine_id, "batch": str(batch_id), "input": input_reference, "events": events, "final": final})
        traces.append(DecisionTrace(trace_id, str(batch_id), input_reference, events, final, engine_id))
    return tuple(traces)


def prompt_trace_summary(
    traces: tuple[DecisionTrace, ...], *, evidence_per_policy: int,
) -> dict[str, Any]:
    """Build a bounded-by-policy, deterministic aggregate without truncating raw traces."""
    policy_totals: dict[str, dict[str, Any]] = {}
    evidence: dict[str, list[tuple[int, str]]] = {}
    final_counts: Counter[str] = Counter()
    for trace in traces:
        final_counts.update({
            "input_rows": int(trace.final_result.get("input_rows") or 0),
            "output_rows": int(trace.final_result.get("output_rows") or 0),
            "target_rows": int(trace.final_result.get("target_row_count") or 0),
            "target_output_rows": int(trace.final_result.get("target_output_rows") or 0),
            "final_matches": int(trace.final_result.get("final_match_count") or 0),
        })
        for event in trace.ordered_policy_events:
            name = str(event.get("policy_file") or event.get("policy_call"))
            total = policy_totals.setdefault(name, {
                "calls": 0, "input_rows": 0, "newly_filtered_rows": 0,
                "policy_calls": Counter(), "modified_rows_by_field": Counter(),
                "filter_rule_counts": Counter(),
            })
            total["calls"] += 1
            total["policy_calls"].update([str(event.get("policy_call"))])
            total["input_rows"] += int(event.get("input_row_count") or 0)
            affected_evidence = event.get("newly_filtered_rows") or {}
            affected = int(affected_evidence.get("count") or len(event.get("newly_filtered_row_orders") or []))
            total["newly_filtered_rows"] += affected
            modified = event.get("modified_rows_by_field") or event.get("modified_row_orders") or {}
            total["modified_rows_by_field"].update({str(key): int(value.get("count", 0))
                if isinstance(value, Mapping) else len(value) for key, value in modified.items()})
            total["filter_rule_counts"].update({
                str(key): int(value) for key, value in (event.get("filter_rule_counts") or {}).items()
            })
            evidence.setdefault(name, []).append((affected, trace.trace_id))
    normalized = {}
    representative = {}
    for name in sorted(policy_totals):
        item = policy_totals[name]
        normalized[name] = {
            **{key: value for key, value in item.items() if not isinstance(value, Counter)},
            "policy_calls": dict(sorted(item["policy_calls"].items())),
            "modified_rows_by_field": dict(sorted(item["modified_rows_by_field"].items())),
            "filter_rule_counts": dict(sorted(item["filter_rule_counts"].items())),
        }
        representative[name] = [trace_id for _, trace_id in sorted(evidence[name], key=lambda pair: (-pair[0], pair[1]))[:evidence_per_policy]]
    return {
        "trace_count": len(traces), "engine_id": traces[0].engine_id if traces else None,
        "final_counts": dict(final_counts), "policy_effects": normalized,
        "representative_evidence_ids": representative,
        "raw_trace_artifact": "traces/<round>.json.gz",
    }


def summarize_scenario(
    predicate: ScenarioPredicate, frame: pd.DataFrame, traces: tuple[DecisionTrace, ...],
    scene_evaluation: Mapping[str, Any], incumbent_evaluation: Mapping[str, Any], scales: Mapping[str, float],
) -> ScenarioSummary:
    # A discovery-time scene has no candidate comparator. Supply raw scene and
    # global incumbent performance without manufacturing a zero improvement vector.
    del scales
    matched = scenario_target_mask(predicate, frame)
    batch_ids = scenario_batch_ids(predicate, frame)
    matched_full_orders: dict[str, set[int]] = {}
    matched_target_orders: dict[str, set[int]] = {}
    if "batch_id" in frame:
        for batch_id, batch in frame.groupby("batch_id", sort=False, dropna=False):
            target_batch = batch.loc[pd.to_numeric(batch["product_id"], errors="coerce").eq(1)]
            matched_full_orders[str(batch_id)] = {
                local_order for local_order, index in enumerate(batch.index) if bool(matched.loc[index])
            }
            matched_target_orders[str(batch_id)] = {
                local_order for local_order, index in enumerate(target_batch.index) if bool(matched.loc[index])
            }
    evidence = scenario_evidence_ids(predicate, frame, traces)
    policy_counts: Counter[str] = Counter()
    observed_policy_files: set[str] = set()
    for trace in traces:
        scope = trace.input_reference.get("row_order_scope")
        selected_orders = (matched_full_orders if scope == "full_batch" else matched_target_orders).get(
            trace.batch_id, set()
        )
        if not selected_orders:
            continue
        for event in trace.ordered_policy_events:
            policy_file = event.get("policy_file")
            if policy_file:
                observed_policy_files.add(str(policy_file))
            row_evidence = event.get("newly_filtered_rows") or {}
            bitmap_hex = row_evidence.get("bitmap_hex") if isinstance(row_evidence, Mapping) else None
            if not isinstance(bitmap_hex, str):
                raise ValueError("V2 scenario summaries require exact compact row bitmaps")
            bitmap = bytes.fromhex(bitmap_hex)
            affected = sum(
                1 for order in selected_orders
                if order // 8 < len(bitmap) and bitmap[order // 8] & (1 << (order % 8))
            )
            if affected:
                policy_counts[str(event.get("policy_file") or event.get("policy_call"))] += affected
    metrics = metric_values(scene_evaluation)
    incumbent_metrics = (
        metric_values(incumbent_evaluation)
        if isinstance(incumbent_evaluation.get("metrics"), Mapping)
        else {key: float(incumbent_evaluation[key]) for key in METRIC_KEYS}
    )
    scene_trace_ids = set(evidence)
    scene_traces = tuple(trace for trace in traces if trace.trace_id in scene_trace_ids)
    scene_frame = frame.loc[frame["batch_id"].astype(str).isin(batch_ids)].copy()
    policy_trace_evidence = build_policy_trace_evidence(scene_frame, scene_traces)
    policy_trace_text = render_policy_trace_text(policy_trace_evidence)
    eligible_rows = pd.to_numeric(frame["product_id"], errors="coerce").eq(1)
    eligible_batches = frame.loc[eligible_rows, "batch_id"].astype(str).nunique()
    return ScenarioSummary(
        predicate, int(matched.sum()), len(batch_ids), coverage_ratio(predicate, frame), metrics,
        {key: float(incumbent_metrics[key]) for key in METRIC_KEYS},
        int(eligible_rows.sum()), int(eligible_batches),
        {"scenario_newly_filtered_by_policy": dict(sorted(policy_counts.items())),
         "scenario_observed_policy_files": sorted(observed_policy_files),
         "trace_count": len(evidence), "membership_unit": "batch",
         "metric_scope": "all_target_category_rows_in_matched_batches",
         "performance_evidence": {
             "scene": scene_evaluation.get("objective_prompt_evidence"),
             "global": incumbent_evaluation.get("objective_prompt_evidence"),
         },
         "policy_trace_evidence": policy_trace_evidence,
         "policy_trace_text": policy_trace_text}, evidence,
    )


def scenario_evidence_ids(
    predicate: ScenarioPredicate, frame: pd.DataFrame, traces: tuple[DecisionTrace, ...],
) -> tuple[str, ...]:
    batch_ids = scenario_batch_ids(predicate, frame)
    return tuple(item.trace_id for item in traces if item.batch_id in batch_ids)

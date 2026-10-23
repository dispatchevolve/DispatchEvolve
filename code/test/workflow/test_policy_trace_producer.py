"""Long-term tests for deterministic V2 Policy-trace evidence production.

Purpose: verify exact row evidence, Policy attribution cohorts, seven-objective
association, and compact Prompt rendering. Related core files are the Full
Dispatch evaluator and ``dispatchevolve_v2.policy_trace``. These tests protect
trace semantics without external model calls and should remain with V2.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dispatchevolve.tasks.full_dispatch.evaluator import _final_match_row_evidence
from dispatchevolve.workflows.dispatchevolve_v2.contracts import DecisionTrace, METRIC_KEYS
from dispatchevolve.workflows.dispatchevolve_v2.policy_trace import (
    build_policy_trace_evidence,
    render_policy_trace_text,
)


def _bitmap(*positions: int, universe: int = 2) -> dict[str, object]:
    bitmap = bytearray((universe + 7) // 8)
    for position in positions:
        bitmap[position // 8] |= 1 << (position % 8)
    return {"count": len(positions), "bitmap_hex": bytes(bitmap).hex()}


def _trace(batch_id: str, *, binding: bool) -> DecisionTrace:
    changed = _bitmap(1) if binding else _bitmap()
    final_eligible = _bitmap(0) if binding else _bitmap(0, 1)
    event = {
        "policy_call": "travel_cost.apply_linear",
        "policy_file": "policies/travel_cost.py",
        "eligible_before_rows": _bitmap(0, 1),
        "applicable_before_rows": _bitmap(0, 1),
        "newly_filtered_rows": changed,
        "modified_rows_by_field": {},
    }
    return DecisionTrace(
        f"trace-{batch_id}", batch_id,
        {"row_order_scope": "full_batch"}, (event,),
        {
            "input_rows": 2,
            "final_eligible_target_rows": final_eligible,
            "final_matched_target_rows": _bitmap(0),
        },
        "engine",
    )


def test_final_match_rows_join_back_to_exact_batch_positions() -> None:
    source = pd.DataFrame({
        "batch_id": ["B1", "B1", "B2"],
        "order_id": [1, 1, 2], "driver_id": [10, 11, 20],
        "product_id": [1, 1, 1], "uuid": ["a", "b", "c"],
    })
    matches = source.iloc[[1, 2]].copy()
    evidence = _final_match_row_evidence(source, matches)
    assert evidence == {
        "B1": {"count": 1, "bitmap_hex": "02"},
        "B2": {"count": 1, "bitmap_hex": "01"},
    }


def test_policy_trace_builds_real_cohorts_and_compact_grouped_text() -> None:
    frame = pd.DataFrame({
        "batch_id": ["B1", "B1", "B2", "B2"],
        "order_id": [1, 1, 2, 2], "driver_id": [10, 11, 20, 21],
        "product_id": [1, 1, 1, 1],
        "eta": [100.0, 500.0, 200.0, 600.0],
        "gmv": [20.0, 10.0, 30.0, 15.0],
        "cr": [0.8, 0.5, 0.7, 0.4],
        "dar": [0.9, 0.5, 0.8, 0.4],
        "pcaa": [0.1, 0.4, 0.2, 0.5],
        "dcaa": [0.05, 0.3, 0.1, 0.4],
        "is_broadcasted": [1, 1, 1, 1],
    })
    evidence = build_policy_trace_evidence(
        frame, (_trace("B1", binding=True), _trace("B2", binding=False))
    )
    policy = evidence["policies"][0]
    association = policy["performance_association"]
    assert policy["binding_batches"] == 1
    assert policy["applicable_nonbinding_batches"] == 1
    assert tuple(association["binding_minus_nonbinding_raw"]) == METRIC_KEYS
    assert association["binding_minus_nonbinding_raw"]["mean_eta"] == -100.0

    rendered = render_policy_trace_text(evidence)
    assert rendered.count("Policy: policies/travel_cost.py") == 1
    assert "first_filtered_od_pairs/applicable_od_pairs" in rendered
    assert "binding_minus_nonbinding_raw=" in rendered
    assert "observational association" in rendered

    second = {**policy, "policy": "policies/other_filter.py"}
    multi_policy_evidence = {**evidence, "policies": [policy, second]}
    scoped = render_policy_trace_text(
        multi_policy_evidence,
        policy_scope=("policies/travel_cost.py",),
    )
    assert "Policy: policies/travel_cost.py" in scoped
    assert "Policy: policies/other_filter.py" not in scoped
    assert "#### Scene Matching Evidence" in scoped
    assert "### Policy Trace Input Semantics" in scoped
    with pytest.raises(ValueError, match="not present in scene evidence"):
        render_policy_trace_text(
            evidence, policy_scope=("policies/missing.py",),
        )


def test_generic_policy_semantics_use_all_batches_and_preserve_mixed_effects():
    from dispatchevolve.workflows.dispatchevolve_v2.policy_trace import _observed_semantics, _changed_positions
    empty = {"policy_call": "toy.apply", "newly_filtered_rows": {"count": 0, "bitmap_hex": "00"}, "modified_rows_by_field": {}}
    changed = {**empty, "modified_rows_by_field": {"weight": {"count": 1, "bitmap_hex": "01"}, "stage": {"count": 1, "bitmap_hex": "02"}}}
    semantics = _observed_semantics([empty, changed])
    assert semantics.policy_type == 'mixed'
    assert _changed_positions(changed, semantics, 2) == {0, 1}

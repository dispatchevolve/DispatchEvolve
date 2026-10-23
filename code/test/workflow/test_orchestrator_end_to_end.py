"""Long-term one-round V2 orchestrator test with fake provider/evaluator.

Related program: dispatchevolve_v2.orchestrator and every Stage-I/Stage-II
module it coordinates. The test is local and deterministic; it validates the
complete control flow, including deterministic compilation, without external
external model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dispatchevolve.workflows.dispatchevolve_v2.config import BudgetConfig, ModelConfig, V2Config
from dispatchevolve.workflows.dispatchevolve_v2.contracts import LocalCandidate, METRIC_KEYS, OBJECTIVE_KEYS
from dispatchevolve.workflows.dispatchevolve_v2.orchestrator import DispatchEvolveV2Orchestrator
from dispatchevolve.workflows.dispatchevolve_v2.shared_genetic_adapter import LocalEvolutionResult
from dispatchevolve.workflows.dispatchevolve_v2.trace_analysis import normalize_traces
from dispatchevolve.tasks.full_dispatch.candidate_executor import LocalProcessCandidateRunner
import dispatchevolve.workflows.dispatchevolve_v2.llm as llm_module
import dispatchevolve.workflows.dispatchevolve_v2.local_evaluator as evaluator_module
import dispatchevolve.workflows.dispatchevolve_v2.orchestrator as orchestrator_module


REPO = Path(__file__).resolve().parents[2]


def _first_table_id(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = lines.index(heading) + 2
    return lines[start].split("|", 1)[0].strip()


def test_one_round_executes_local_search_and_deterministic_compilation(
    tmp_path: Path, monkeypatch,
) -> None:
    data = pd.DataFrame({
        "batch_id": list(range(100)), "product_id": [1] * 100,
        "order_id": list(range(100)), "driver_id": list(range(100)),
        "eta": list(range(100)), "weight": [1.0] * 100,
    })
    data_path = tmp_path / "train.csv"; data.to_csv(data_path, index=False)
    history = tmp_path / "history.yaml"; history.write_text("schema_version: test\nrecords: []\n")
    engine_dir = tmp_path / "seed_engine"
    (engine_dir / "policies").mkdir(parents=True)
    (engine_dir / "policies/__init__.py").write_text("")
    (engine_dir / "README.md").write_text("test engine\n")
    (engine_dir / "policies/candidate_filter.py").write_text(
        "import pandas as pd\n"
        "def trace_applicable_mask(frame, policy_symbol):\n"
        "    return pd.Series(True, index=frame.index, dtype=bool)\n"
        "def apply(frame):\n    return frame\n"
    )
    (engine_dir / "engine.py").write_text(
        "from collections import Counter\n"
        "import pandas as pd\n"
        "from policies import candidate_filter\n"
        "RESULT_FIELDS = ('is_filtered', 'filter_rule', 'filter_policy')\n"
        "def _run_target_policies(target):\n    candidate_filter.apply(target)\n"
        "def run_batch(batch_data, *, trace=True):\n"
        "    result = batch_data.copy()\n"
        "    result['__engine_row_order'] = range(len(result))\n"
        "    result['is_filtered'] = False\n"
        "    result['filter_rule'] = ''\n"
        "    result['filter_policy'] = ''\n"
        "    target = result.loc[result['product_id'].eq(1)].copy()\n"
        "    if len(target):\n        _run_target_policies(target)\n"
        "    eligible_target = target.loc[~target['is_filtered']].copy()\n"
        "    trace_payload = None\n"
        "    if trace:\n"
        "        trace_payload = {'input_rows': len(result), 'output_rows': len(result), "
        "'target_row_count': len(target), 'target_output_rows': len(target), "
        "'passthrough_row_count': len(result) - len(target), 'filtered_count': 0, "
        "'filter_rule_counts': {}, 'filter_policy_counts': {}}\n"
        "    return result, trace_payload\n"
    )

    def fake_complete(self, prompt):
        if prompt.role == "scenario_discovery_query":
            return "## Query\nmean(eta) < 5\n## Discovery Purpose\ntrace-backed test scenario"
        if prompt.role == "opportunity_discovery":
            return ("## Scene Summary\nA trace-backed test Scene has an actionable filter opportunity.\n"
                    "## Opportunity Status\nOPPORTUNITY_FOUND\n"
                    "## Improvement Opportunities\n"
                    "objective | related_policies | improvement_plan | evidence_basis\n"
                    "order_ar | policies/candidate_filter.py | Adjust candidate filtering to preserve useful candidates | The Policy binds in the Scene")
        if prompt.role == "opportunity_critic":
            return "## Decision\nACCEPT\n## Confidence\n1.0\n## Reason\ncovered by evidence"
        if prompt.role == "combination":
            identifier = _first_table_id(
                prompt.user, "### Selectable Local Candidate Summary"
            )
            assert identifier == "00001"
            assert "sharedgenetic01" not in prompt.user
            return (f"## Selected Local Candidate IDs\n{identifier}\n"
                    "## Conflict Resolution Plan\nNONE\n"
                    "## Rationale\nexpand the non-dominated Pareto archive")
        raise AssertionError(f"unexpected role: {prompt.role}")

    def fake_evaluate(
        self, engine_dir, frame, *, output_dir, metric_rows=None, identity=None,
        **_kwargs,
    ):
        improved = any(part in str(engine_dir) for part in (
            "shared_candidate", "candidate_library", "combinations",
        ))
        baseline_value = 20.0 if "pareto_recombination" in str(output_dir) else 10.0
        metrics = {key: baseline_value for key in METRIC_KEYS}
        if improved:
            metrics["order_ar"] = baseline_value * 1.1
            # A 0.1% regression passes local tolerance but must fail global admission.
            metrics["mean_fqs"] = baseline_value * 0.999
        traces = [{"batch_id": str(batch_id), "input_rows": len(batch), "output_rows": len(batch),
                   "target_row_count": len(batch), "target_output_rows": len(batch),
                   "passthrough_row_count": 0, "filtered_count": 0,
                       "filter_rule_counts": {}, "filter_policy_counts": {},
                       "input_target_rows": {"count": 1, "bitmap_hex": "01"},
                       "final_eligible_target_rows": {"count": 1, "bitmap_hex": "01"},
                       "final_matched_target_rows": {"count": 1, "bitmap_hex": "01"},
                       "policy_events": [{"policy_call": "candidate_filter.apply", "input_row_count": len(batch),
                                          "policy_file": "policies/candidate_filter.py",
                                          "eligible_before_rows": {"count": 1, "bitmap_hex": "01"},
                                          "applicable_before_rows": {"count": 1, "bitmap_hex": "01"},
                                          "newly_filtered_rows": {"count": 0, "bitmap_hex": "00"}}]}
                  for batch_id, batch in frame.groupby("batch_id")]
        return {"metrics": metrics, "batch_traces": traces, "cache_hit": False}

    def fake_shared_genetic(**kwargs):
        candidate_dir = kwargs["round_dir"] / "shared_candidate" / "engine"
        from dispatchevolve.workflows.dispatchevolve_v2.integration import copy_engine
        copy_engine(kwargs["incumbent_dir"], candidate_dir)
        policy = candidate_dir / "policies/candidate_filter.py"
        policy.write_text(policy.read_text() + "# shared genetic test mutation\n")
        delta = {key: 0.0 for key in METRIC_KEYS}; delta["order_ar"] = 0.1
        metrics = {key: 10.0 for key in METRIC_KEYS}; metrics["order_ar"] = 11.0
        candidate = LocalCandidate(
            "sharedgenetic01", candidate_dir, kwargs["opportunity"].opportunity_id,
            kwargs["opportunity"].scenario, "order_ar", ("policies/candidate_filter.py",),
            metrics, delta, (), "shared genetic test mutation",
        )
        return LocalEvolutionResult((candidate,), True, 1, delta, ({
            "program_id": "p1", "policy": "policies/candidate_filter.py", "evaluated": True,
            "normalized_improvement": delta, "accepted": True,
        },))

    monkeypatch.setattr(llm_module.V2LLM, "complete", fake_complete)
    monkeypatch.setattr(evaluator_module.V2Evaluator, "evaluate", fake_evaluate)
    monkeypatch.setattr(orchestrator_module, "evolve_opportunity_with_shared_genetic", fake_shared_genetic)
    monkeypatch.setattr(
        orchestrator_module, "_run_opportunity_process_pool",
        lambda jobs, max_workers: [orchestrator_module._evolve_opportunity_worker(job) for job in jobs],
    )
    budget = BudgetConfig(
        outer_rounds=1, proposal_attempts_per_round=1, admitted_opportunities_target=1,
        local_iterations_per_opportunity=1, local_evaluations_per_opportunity=1,
        parallel_opportunities=1, population_size=1, archive_size=1, retained_candidates=1,
        num_islands=1, combination_proposals_per_round=1, combination_evaluations_per_round=1,
        reference_limit=7,
    )
    config = V2Config(
        run_id="test", mode="local_debug", engine_dir=engine_dir,
        data_path=data_path, test_path=None, backend="local", output_dir=tmp_path / "workspace",
        cache_root=tmp_path / "cache", log_root=tmp_path / "logs",
        model=ModelConfig("gemini", "fake"), budget=budget,
    )
    state = DispatchEvolveV2Orchestrator(config).run()
    assert state.stage == "complete" and state.completed_rounds == 1
    assert len(state.archive) == 1
    assert not any(item["candidate_ids"] for item in state.archive)
    assert (tmp_path / "workspace/test/online_uplift/assessments.jsonl").is_file()
    assert (tmp_path / "workspace/test/pareto_archive/head.json").is_file()
    assert (tmp_path / "workspace/test/final/selected_engine.json").is_file()
    run_manifest = json.loads((tmp_path / "workspace/test/run_manifest.json").read_text())
    prompt_ids = json.loads(
        (tmp_path / "workspace/test/manifests/prompt_ids.json").read_text()
    )
    candidate_prompt_ids = prompt_ids["namespaces"]["candidate"]["internal_to_prompt"]
    assert list(candidate_prompt_ids.values()) == ["00001"]
    assert all(len(identifier) == 64 for identifier in candidate_prompt_ids)
    lineage = json.loads(
        (tmp_path / "workspace/test/final/selected_engine_manifest.json").read_text()
    )
    assert run_manifest["schema_version"] == "dispatchevolve-v2-run-manifest-v1"
    assert run_manifest["stage"] == "complete"
    assert lineage["schema_version"] == "dispatchevolve-v2-selected-engine-lineage-v1"
    # The measured trade-off cannot enter the archive or replace E0.
    assert lineage["local_candidates"] == []
    assert lineage["composition"] is None
    feedback = [json.loads(line) for line in (
        tmp_path / "workspace/test/combinations/feedback.jsonl"
    ).read_text().splitlines()]
    assert feedback[-1]["acceptance_passed"] is False
    assert feedback[-1]["archive_update"] == "infeasible"
    events = [json.loads(line) for line in (
        tmp_path / "workspace/test/pareto_archive/events.jsonl"
    ).read_text().splitlines()]
    assert events[-1]["acceptance_passed"] is False
    assert events[-1]["status"] == "infeasible"
    registry = json.loads(next(
        (tmp_path / "workspace/test/composition_identity_registry").glob("*.json")
    ).read_text())
    assert registry["attempt_status"] == "evaluated"
    assert registry["acceptance_passed"] is False
    integration_manifest = tmp_path / "workspace/test/round_001/combinations/proposal_000/integration_manifest.json"
    assert integration_manifest.is_file()
    frozen = integration_manifest.parent / "engine"
    _, integrated_traces = LocalProcessCandidateRunner(workspace=tmp_path / "next-round-runner")(
        frozen, data, True, None
    )
    normalized = normalize_traces(frozen, data, {"batch_traces": integrated_traces})
    assert len(normalized) == 100
    assert all("bitmap_hex" in event["newly_filtered_rows"]
               for trace in normalized for event in trace.ordered_policy_events)

    # Reconciliation must not resurrect a globally infeasible historical candidate.
    state_path = tmp_path / "workspace/test/state/run_state.json"
    raw_state = json.loads(state_path.read_text())
    raw_state["archive"] = [item for item in raw_state["archive"] if not item["candidate_ids"]]
    state_path.write_text(json.dumps(raw_state))
    resumed = DispatchEvolveV2Orchestrator(config).run(resume=True)
    assert len(resumed.archive) == 1
    reconciliation = json.loads(
        (tmp_path / "workspace/test/pareto_archive/reconciliation.json").read_text()
    )
    assert reconciliation["evaluations_considered"] == 1
    assert reconciliation["retained_entries"] == 1
    assert reconciliation["archive_changed"] is False
    assert reconciliation["outer_no_progress_reset"] is False

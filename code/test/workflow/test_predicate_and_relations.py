from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pandas as pd
import pytest

from dispatchevolve.workflows.dispatchevolve_v2.config import DEFAULT_PROMPT_JSON
from dispatchevolve.workflows.dispatchevolve_v2.batch_query import BatchQueryError, parse_batch_query
from dispatchevolve.workflows.dispatchevolve_v2.contracts import LocalCandidate, METRIC_KEYS, RelationEdge
from dispatchevolve.workflows.dispatchevolve_v2.combination_search import (
    _candidate_performance, propose_combination,
)
from dispatchevolve.workflows.dispatchevolve_v2.prompt_store import PromptStore
from dispatchevolve.workflows.dispatchevolve_v2.relation_graph import (
    build_relation_graph, logical_policy_file, validate_priority_plan,
)
from dispatchevolve.workflows.dispatchevolve_v2.scenario_predicate import (
    coverage_ratio, parse_predicate, scenario_target_mask,
)


REPO = Path(__file__).resolve().parents[2]


def _candidate(identifier: str, scene: str, file: str, root: Path) -> LocalCandidate:
    metrics = {key: 0.0 for key in METRIC_KEYS}
    predicate = parse_predicate(scene, {"eta": "int", "product_id": "int"})
    engine = root / identifier
    (engine / "policies").mkdir(parents=True)
    (engine / file).write_text("import math\ndef apply(frame):\n    return frame\n")
    module = Path(file).stem
    (engine / "engine.py").write_text(
        f"from policies import {module}\n"
        f"def run_batch(frame, *, trace=True):\n    {module}.apply(frame)\n    return frame, {{}}\n"
    )
    return LocalCandidate(identifier, engine, identifier, predicate, "order_ar", (file,), metrics, metrics, (), "diff")


def test_candidate_performance_reconstructs_scene_local_br_baseline(tmp_path: Path) -> None:
    candidate = _candidate("a", "eta LT 5", "policies/x.py", tmp_path)
    metrics = dict(candidate.metrics); metrics["order_br"] = 0.788
    delta = dict(candidate.oriented_delta); delta["order_br"] = -.015
    candidate = replace(candidate, metrics=metrics, oriented_delta=delta)
    scales = {key: 1.0 for key in candidate.metrics}
    scales["order_br"] = 0.84
    rows = _candidate_performance([candidate], scales).splitlines()
    br = next(row for row in rows if " | order_br | " in row)
    cells = [cell.strip() for cell in br.split("|")]
    assert float(cells[2]) == pytest.approx(0.8)
    assert float(cells[3]) == 0.788
    assert float(cells[4]) == pytest.approx(-0.012)


def test_typed_predicate_coverage_uses_target_category_batch_denominator() -> None:
    frame = pd.DataFrame({
        "batch_id": list(range(100)) + list(range(100)),
        "product_id": [1] * 100 + [99] * 100,
        "eta": list(range(100)) * 2,
    })
    predicate = parse_predicate("eta LT 5", {"product_id": "int", "eta": "int"})
    assert coverage_ratio(predicate, frame) == 0.05
    repeated_batch = pd.DataFrame({
        "batch_id": [1, 1, 2], "product_id": [1, 1, 1], "eta": [1, 50, 50],
    })
    repeated_predicate = parse_predicate("eta LT 5", repeated_batch.dtypes)
    assert coverage_ratio(repeated_predicate, repeated_batch) == 0.5
    assert scenario_target_mask(repeated_predicate, repeated_batch).tolist() == [True, True, False]
    assert parse_predicate("eta >= 10", repeated_batch.dtypes).expression == {
        "field": "eta", "op": "ge", "value": 10.0,
    }
    assert parse_predicate("```yaml\neta:\n  ge: 10\n```", repeated_batch.dtypes).expression == {
        "field": "eta", "op": "ge", "value": 10.0,
    }
    with pytest.raises(ValueError):
        parse_predicate("unknown LT 1", {"eta": "int"})


def test_batch_scene_query_uses_aggregates_once_per_complete_batch() -> None:
    frame = pd.DataFrame({
        "batch_id": [1, 1, 2, 2, 3], "product_id": [1, 1, 1, 1, 99],
        "eta": [1, 9, 8, 12, 0], "order_id": [1, 2, 3, 3, 4],
    })
    query = parse_batch_query(
        "mean(eta) < 6 and nunique(order_id) == 2",
        {"eta": "number", "order_id": "number"},
    )
    assert scenario_target_mask(query, frame).tolist() == [True, True, False, False, False]
    assert coverage_ratio(query, frame) == 0.5
    with pytest.raises(BatchQueryError, match="raw feature names"):
        parse_batch_query("eta < 6", {"eta": "number"})
    with pytest.raises(BatchQueryError, match="unknown feature"):
        parse_batch_query("mean(missing) < 6", {"eta": "number"})


def test_nested_variant_paths_share_the_original_logical_policy() -> None:
    assert logical_policy_file("policies/v2_abcdef123456_x.py") == "policies/x.py"
    assert logical_policy_file("policies/v2_111111111111_v2_abcdef123456_x.py") == "policies/x.py"


def test_c09_has_only_exact_scene_hard_conflict_and_requires_llm_order(tmp_path: Path) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("c", "eta LT 8", "policies/x.py", tmp_path),
    ]
    class FakeLLM:
        class Config:
            model = "fake"
        config = Config()
        def complete(self, prompt):
            return ("a|c|policies/x.py|their shared Policy behavior may interact on overlapping batches and requires a deliberate precedence decision\n"
                    "b|c|policies/x.py|their shared Policy behavior may interact on overlapping batches and requires a deliberate precedence decision")
    frame = pd.DataFrame({"batch_id": list(range(10)), "product_id": [1] * 10, "eta": list(range(10))})
    edges = build_relation_graph(
        candidates, frame=frame, llm=FakeLLM(),
        prompts=PromptStore(DEFAULT_PROMPT_JSON),
    )
    relations = {(item.left, item.right): item.relation for item in edges}
    assert relations[("a", "b")] == "hard"
    assert relations[("a", "c")] == "unresolved"
    validate_priority_plan(("a", "c"), ("c", "a"), (("a", "c"),), edges)
    with pytest.raises(ValueError):
        validate_priority_plan(("a", "c"), ("a",), (("a", "c"),), edges)
    with pytest.raises(ValueError):
        validate_priority_plan(("a", "b"), ("a", "b"), (), edges)


def test_relation_graph_batches_worst_case_pairs_deterministically(tmp_path: Path) -> None:
    candidates = [
        _candidate(f"c{index:02d}", f"eta LT {index + 1}", "policies/x.py", tmp_path)
        for index in range(31)
    ]
    calls: list[str] = []

    class FakeLLM:
        class Config:
            model = "fake"
        config = Config()

    def classify(action, prompt):
        calls.append(action)
        lines = prompt.user.splitlines()
        start = lines.index("### Unresolved Local Candidate Pairs") + 2
        pairs = []
        for line in lines[start:]:
            if not line.strip() or line.startswith("### "):
                break
            cells = [item.strip() for item in line.split("|")]
            pairs.append((cells[0], cells[1], cells[8]))
        return "\n".join(
            f"{left}|{right}|{policies}|the candidates may interact through supplied structural evidence and need composition review"
            for left, right, policies in pairs
        )

    frame = pd.DataFrame({"batch_id": list(range(40)), "product_id": [1] * 40, "eta": list(range(40))})
    edges = build_relation_graph(
        candidates, frame=frame, llm=FakeLLM(),
        prompts=PromptStore(DEFAULT_PROMPT_JSON),
        pairs_per_call=30, llm_call=classify,
    )
    assert len(edges) == 465
    assert calls == [f"candidate-interaction:{index:03d}" for index in range(16)]


def test_llm5_format_error_gets_one_format_only_repair(tmp_path: Path) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 8", "policies/x.py", tmp_path),
    ]
    class FakeLLM:
        class Config:
            model = "fake"
        config = Config()

    repairs: list[tuple[str, str]] = []
    edges = build_relation_graph(
        candidates,
        frame=pd.DataFrame({"batch_id": range(10), "product_id": [1] * 10, "eta": range(10)}),
        llm=FakeLLM(), prompts=PromptStore(DEFAULT_PROMPT_JSON),
        llm_call=lambda action, prompt: "malformed response",
        format_repair_call=lambda action, prompt, response, error: (
            repairs.append((action, type(error).__name__)) or
            "a|b|policies/x.py|the supplied shared-Policy evidence requires precedence review"
        ),
    )
    assert repairs == [("candidate-interaction:000", "ResponseFormatError")]
    assert len(edges) == 1 and edges[0].relation == "unresolved"


def test_relation_graph_normalizes_bare_logical_policy_names(tmp_path: Path) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 8", "policies/x.py", tmp_path),
    ]

    class FakeLLM:
        class Config:
            model = "fake"

        config = Config()

    edges = build_relation_graph(
        candidates,
        frame=pd.DataFrame({
            "batch_id": range(10),
            "product_id": [1] * 10,
            "eta": range(10),
        }),
        llm=FakeLLM(),
        prompts=PromptStore(DEFAULT_PROMPT_JSON),
        llm_call=lambda action, prompt: (
            "a|b|x.py|the candidates may interact through their shared logical "
            "Policy and need composition review"
        ),
    )

    assert len(edges) == 1
    assert edges[0].evidence["affected_policies"] == ["policies/x.py"]


def test_relation_graph_still_rejects_unknown_bare_policy_names(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 8", "policies/x.py", tmp_path),
    ]

    class FakeLLM:
        class Config:
            model = "fake"

        config = Config()

    with pytest.raises(ValueError, match="unknown logical Policy"):
        build_relation_graph(
            candidates,
            frame=pd.DataFrame({
                "batch_id": range(10),
                "product_id": [1] * 10,
                "eta": range(10),
            }),
            llm=FakeLLM(),
            prompts=PromptStore(DEFAULT_PROMPT_JSON),
            llm_call=lambda action, prompt: (
                "a|b|unknown.py|the candidates may interact through an "
                "unsupported Policy name"
            ),
        )


def test_relation_graph_repairs_unknown_policy_without_partial_mutation(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 8", "policies/x.py", tmp_path),
        _candidate("c", "eta LT 9", "policies/x.py", tmp_path),
    ]

    class FakeLLM:
        class Config:
            model = "fake"

        config = Config()

    repairs: list[str] = []
    edges = build_relation_graph(
        candidates,
        frame=pd.DataFrame({
            "batch_id": range(10),
            "product_id": [1] * 10,
            "eta": range(10),
        }),
        llm=FakeLLM(),
        prompts=PromptStore(DEFAULT_PROMPT_JSON),
        llm_call=lambda action, prompt: (
            "a|b|policies/x.py|the supplied pair requires composition review\n"
            "a|c|policies/unknown.py|the supplied pair requires composition review\n"
            "b|c|policies/x.py|the supplied pair requires composition review"
        ),
        format_repair_call=lambda action, prompt, response, error: (
            repairs.append(str(error)) or
            "a|b|policies/x.py|the supplied pair requires composition review\n"
            "a|c|policies/x.py|the supplied pair requires composition review\n"
            "b|c|policies/x.py|the supplied pair requires composition review"
        ),
    )

    assert len(edges) == 3
    assert len(repairs) == 1 and "not available" in repairs[0]


def test_relation_graph_repairs_duplicate_pair_and_requires_complete_chunk(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 8", "policies/x.py", tmp_path),
    ]

    class FakeLLM:
        class Config:
            model = "fake"

        config = Config()

    repairs: list[str] = []
    edges = build_relation_graph(
        candidates,
        frame=pd.DataFrame({
            "batch_id": range(10),
            "product_id": [1] * 10,
            "eta": range(10),
        }),
        llm=FakeLLM(),
        prompts=PromptStore(DEFAULT_PROMPT_JSON),
        llm_call=lambda action, prompt: (
            "a|b|policies/x.py|the supplied pair requires composition review\n"
            "a|b|policies/x.py|the supplied pair was accidentally repeated"
        ),
        format_repair_call=lambda action, prompt, response, error: (
            repairs.append(str(error)) or
            "a|b|policies/x.py|the supplied pair requires composition review"
        ),
    )

    assert len(edges) == 1
    assert len(repairs) == 1 and "repeats" in repairs[0]


def test_relation_graph_uses_honest_conservative_fallback_after_bad_repair(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("a", "eta LT 5", "policies/x.py", tmp_path),
        _candidate("b", "eta LT 8", "policies/x.py", tmp_path),
    ]

    class FakeLLM:
        class Config:
            model = "fake"

        config = Config()

    edges = build_relation_graph(
        candidates,
        frame=pd.DataFrame({
            "batch_id": range(10),
            "product_id": [1] * 10,
            "eta": range(10),
        }),
        llm=FakeLLM(),
        prompts=PromptStore(DEFAULT_PROMPT_JSON),
        llm_call=lambda action, prompt: "malformed",
        format_repair_call=lambda action, prompt, response, error: (
            "a|b|policies/unknown.py|the repair still names an unknown Policy"
        ),
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.relation == "unresolved"
    assert edge.model == "deterministic_contract_fallback"
    assert edge.evidence["relationship_description_source"] == (
        "deterministic_contract_fallback"
    )
    assert edge.evidence["affected_policies"] == ["policies/x.py"]
    assert "unknown logical Policy" in edge.evidence["llm_validation_error"]


def test_llm6_format_error_gets_one_format_only_repair(tmp_path: Path) -> None:
    candidate = _candidate("a", "eta LT 5", "policies/x.py", tmp_path)
    class FakeLLM:
        class Config:
            model = "fake"
        config = Config()

    repairs: list[tuple[str, str]] = []
    proposal = propose_combination(
        [candidate], (), llm=FakeLLM(), prompts=PromptStore(DEFAULT_PROMPT_JSON),
        global_feedback=[], references=[], scales={key: 1.0 for key in candidate.metrics},
        rho=.02, comparison_tolerance=1e-9, composition_registry=[], fixed_e0_id="e0",
        llm_call=lambda action, prompt: "missing required sections",
        format_repair_call=lambda action, prompt, response, error: (
            repairs.append((action, type(error).__name__)) or
            "## Selected Local Candidate IDs\na\n\n## Conflict Resolution Plan\nNONE\n\n## Rationale\nSelect the only evaluated candidate."
        ),
    )
    assert repairs == [("pareto-composition", "ResponseFormatError")]
    assert proposal.candidate_ids == ("a",)


def test_incremental_combination_inherits_incumbent_and_requires_new_precedence(tmp_path: Path) -> None:
    old = _candidate("old", "eta LT 8", "policies/x.py", tmp_path)
    new = _candidate("new", "eta LT 5", "policies/x.py", tmp_path)
    evidence = {
        "shared_policy_files": ["policies/x.py"],
        "predicate_intersection_status": "overlap",
        "scenario_overlap": {"jaccard": 0.5},
        "relationship_description": "both variants compete for the same Policy",
    }
    edge = RelationEdge("new", "old", "unresolved", 0.5, "competition", evidence)
    class FakeLLM:
        class Config:
            model = "fake"
        config = Config()

    observed: dict[str, str] = {}
    def complete(_action, prompt):
        observed["prompt"] = prompt.user
        return (
            "## Selected Local Candidate IDs\nnew\n\n"
            "## Conflict Resolution Plan\nnew|old|policies/x.py\n\n"
            "## Rationale\nAdd the new strategy while preserving incumbent context."
        )

    proposal = propose_combination(
        [new], (edge,), incumbent_candidates=[old], incumbent_precedence_edges=(),
        llm=FakeLLM(), prompts=PromptStore(DEFAULT_PROMPT_JSON), global_feedback=[],
        references=[], scales={key: 1.0 for key in new.metrics}, rho=.01,
        comparison_tolerance=1e-9, composition_registry=[], fixed_e0_id="e0",
        llm_call=complete,
    )
    assert proposal.incremental_candidate_ids == ("new",)
    assert proposal.candidate_ids == ("old", "new")
    assert proposal.priority_plan.precedence_edges == ({
        "higher": "new", "lower": "old", "policies": ("policies/x.py",),
    },)
    assert "old" in observed["prompt"] and "new" in observed["prompt"]

"""Long-term smoke tests for the V2 adapter over shared genetic.

Related modules: shared_genetic_adapter and shared_genetic_evaluator.
Covered behavior: one Dispatch task-context template, shared run_evolution use,
single-Policy materialization, and LocalCandidate conversion. No external LLM
or dispatch evaluator is called.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import dispatchevolve.workflows.dispatchevolve_v2.shared_genetic_adapter as adapter
import dispatchevolve.workflows.dispatchevolve_v2.shared_genetic_evaluator as evaluator_adapter
from dispatchevolve.optimizer.genetic.api import EvolutionResult
from dispatchevolve.baselines.candidates import RepositoryGenomeCodec
from dispatchevolve.optimizer.genetic.config import PromptConfig
from dispatchevolve.optimizer.genetic.database import Program
from dispatchevolve.optimizer.genetic.prompt.sampler import PromptSampler
from dispatchevolve.optimizer.genetic.utils.code_utils import apply_diff_blocks
from dispatchevolve.workflows.dispatchevolve_v2.config import BudgetConfig, ModelConfig, V2Config
from dispatchevolve.workflows.dispatchevolve_v2.contracts import METRIC_KEYS, OBJECTIVE_KEYS, Opportunity
from dispatchevolve.workflows.dispatchevolve_v2.local_evaluator import OBJECTIVE_SEMANTICS
from dispatchevolve.workflows.dispatchevolve_v2.scenario_predicate import parse_predicate
from dispatchevolve.workflows.dispatchevolve_v2.shared_genetic_evaluator import candidate_key


def test_shared_genetic_adapter_uses_one_dispatch_template(tmp_path: Path, monkeypatch) -> None:
    incumbent = tmp_path / "incumbent"
    (incumbent / "policies").mkdir(parents=True)
    (incumbent / "engine.py").write_text("def run_batch(frame, *, trace=True):\n    return frame, {}\n")
    (incumbent / "policies/x.py").write_text("def apply(frame):\n    return frame\n")
    (incumbent / "policies/y.py").write_text("def score(frame):\n    return frame\n")
    (incumbent / "policies/unrelated.py").write_text("VALUE = 1\n")
    predicate = parse_predicate("eta LT 5", {"eta": "int", "product_id": "int"})
    opportunity = Opportunity(
        "opportunity-1", predicate, "order_ar",
        ("policies/x.py", "policies/y.py"), "accepted evidence",
    )
    budget = BudgetConfig(
        outer_rounds=1, proposal_attempts_per_round=1, admitted_opportunities_target=1,
        local_iterations_per_opportunity=1, local_evaluations_per_opportunity=1,
        parallel_opportunities=1, population_size=1, archive_size=1,
        retained_candidates=1, num_islands=1, reference_limit=7,
    )
    config = V2Config(
        run_id="test", mode="local_debug", engine_dir=incumbent,
        data_path=tmp_path / "unused.csv", test_path=None, backend="local",
        output_dir=tmp_path / "workspace", cache_root=tmp_path / "cache",
        log_root=tmp_path / "logs",
        model=ModelConfig("gemini", "fake"), budget=budget,
    )
    observed: dict[str, object] = {}

    def fake_run(initial_program, evaluator, *, config, iterations, output_dir, cleanup, checkpoint_path):
        observed["system"] = config.prompt.system_message
        observed["template_dir"] = str(config.prompt.template_dir)
        observed["prompt_metric_keys"] = tuple(config.prompt.prompt_metric_keys or ())
        observed["compact_action_context"] = str(config.prompt.compact_action_context)
        observed["deduplicate_prompt_history"] = str(
            config.prompt.deduplicate_prompt_history
        )
        observed["omit_empty_feature_coordinates"] = str(
            config.prompt.omit_empty_feature_coordinates
        )
        observed["include_artifacts"] = str(config.prompt.include_artifacts)
        observed["simplification_threshold"] = str(
            config.prompt.suggest_simplification_after_chars
        )
        observed["num_diverse_programs"] = str(config.prompt.num_diverse_programs)
        context_path = Path(evaluator).with_name("evaluator_context.json")
        context = json.loads(context_path.read_text())
        initial = Path(initial_program).read_text()
        assert "<<<FILE engine.py>>>" in initial
        assert "<<<FILE policies/unrelated.py>>>" in initial
        assert initial.index("<<<FILE policies/x.py>>>") < initial.index("<<<FILE policies/unrelated.py>>>")
        assert initial.index("<<<FILE policies/y.py>>>") < initial.index("<<<FILE policies/unrelated.py>>>")
        assert context["relative_policies"] == ["policies/x.py", "policies/y.py"]
        prompt_metrics = {
            "combined_score": -1.0,
            "locally_feasible": 0.0,
            "constraint_max_violation": 0.01,
            "constraint_violation_count": 1.0,
            "constraint_total_violation": 0.01,
            **{f"raw_{key}": 1.0 for key in METRIC_KEYS},
            **{f"normalized_improvement_{key}": 0.0 for key in METRIC_KEYS},
        }
        historical = {
            "id": "f" * 64,
            "code": initial + "# historical variant\n",
            "changes_description": "raised the response-quality coefficient",
            "metrics": prompt_metrics,
            "metadata": {"parent_metrics": prompt_metrics},
        }
        rendered = PromptSampler(config.prompt).build_prompt(
            current_program=initial,
            parent_program=initial,
            program_metrics=prompt_metrics,
            previous_programs=[historical],
            top_programs=[historical],
            inspirations=[historical],
            language="python",
            feature_dimensions=["complexity", "diversity"],
            action_context="## Most Recent Failed Mutation\nNONE",
        )
        observed["rendered_system"] = rendered["system"]
        observed["rendered_user"] = rendered["user"]
        evolved_repo = tmp_path / "evolved"
        shutil.copytree(incumbent, evolved_repo)
        (evolved_repo / "policies/x.py").write_text("def apply(frame):\n    return frame.copy()\n")
        (evolved_repo / "policies/y.py").write_text("def score(frame):\n    return frame.copy()\n")
        genome = RepositoryGenomeCodec().encode(evolved_repo)
        engine = Path(context["candidate_root"]) / candidate_key(genome) / "engine"
        shutil.copytree(evolved_repo, engine)
        metrics = {"combined_score": 0.1, "locally_feasible": 1.0}
        metrics.update({f"raw_{key}": 1.0 for key in METRIC_KEYS})
        metrics.update({f"normalized_improvement_{key}": (0.1 if key == "order_ar" else 0.0) for key in METRIC_KEYS})
        program = Program("program-1", genome, "joint copy mutation", metrics=metrics)
        legacy_metrics = {
            key: value for key, value in metrics.items()
            if not key.endswith("_order_br")
        }
        legacy = Program("legacy-program", genome, "legacy mutation", metrics=legacy_metrics)
        return EvolutionResult(program, 0.1, genome, metrics, output_dir, (legacy, program))

    monkeypatch.setattr(adapter, "run_evolution", fake_run)
    monkeypatch.setattr(
        adapter,
        "prune_complete_checkpoints",
        lambda path: observed.setdefault("pruned", str(path)),
    )

    class Evaluator:
        replay_budget_path = None
        cache_root = tmp_path / "cache"
        protocol_hash = "protocol"
        backend = "local"

    shared_frame_path = tmp_path / "shared_scene_frame.pkl"
    pd.DataFrame({"batch_id": [1], "product_id": [1], "eta": [1]}).to_pickle(shared_frame_path)
    result = adapter.evolve_opportunity_with_shared_genetic(
        config=config, opportunity=opportunity, scenario_summary={"coverage": 0.02},
        incumbent_dir=incumbent,
        scene_frame_path=shared_frame_path, metric_rows_path=None,
        baseline_metrics={key: 1.0 for key in METRIC_KEYS},
        scales={key: 1.0 for key in METRIC_KEYS}, evaluator=Evaluator(),
        round_dir=tmp_path / "round",
    )

    assert result.success and result.evaluated_count == 1
    skipped = next(row for row in result.attempts if not row["evaluated"])
    assert skipped["skip_reason"] == "legacy_checkpoint_missing_current_metrics"
    assert set(skipped["missing_metric_fields"]) == {
        "raw_order_br", "normalized_improvement_order_br",
    }
    local_frame_path = tmp_path / "round" / "local_policy_evolution" / opportunity.opportunity_id / "scene_frame.pkl"
    assert not local_frame_path.exists()
    assert pd.read_pickle(shared_frame_path).equals(
        pd.DataFrame({"batch_id": [1], "product_id": [1], "eta": [1]})
    )
    evaluated_attempt = next(row for row in result.attempts if row["evaluated"])
    artifact = tmp_path / "round" / evaluated_attempt["program_artifact"]["path"]
    assert artifact.is_file()
    assert json.loads(artifact.read_text())["program"]["prompts"] is None
    assert result.candidates[0].policy_files == ("policies/x.py", "policies/y.py")
    assert result.candidates[0].engine_dir.joinpath("policies/unrelated.py").read_text() == "VALUE = 1\n"
    assert observed["pruned"].endswith("joint_repository/optimizer/checkpoints")
    assert observed["system"].count("## Local Task") == 1
    assert observed["system"].count("## Objective Function") == 1
    assert observed["system"].count("## Editable Policies") == 1
    assert observed["system"].count("## Hard Constraints") == 1
    assert "every Python file under `policies/`" in observed["system"]
    assert "- policies/x.py\n- policies/y.py" in observed["system"]
    assert "Primary metric: order_ar" in observed["system"]
    assert (
        "mean_eta | protected metric | minimize | "
        f"{OBJECTIVE_SEMANTICS['mean_eta']['meaning']}"
    ) in observed["system"]
    assert observed["system"].count(" | primary objective | ") == 1
    assert observed["system"].count(" | protected metric | ") == len(METRIC_KEYS) - 1
    for forbidden in (
        "opportunity-1",
        "scenario_summary",
        "policy_trace",
        "evidence_ids",
        "predicate_hash",
        "sha256",
        "program_id",
        "Policy Change Summary",
    ):
        assert forbidden not in observed["system"]
    assert observed["template_dir"].endswith("prompts/local_genetic")
    assert observed["compact_action_context"] == "True"
    assert observed["deduplicate_prompt_history"] == "True"
    assert observed["omit_empty_feature_coordinates"] == "True"
    assert observed["include_artifacts"] == "False"
    assert observed["simplification_threshold"] == "None"
    assert observed["num_diverse_programs"] == "2"
    visible = set(observed["prompt_metric_keys"])
    assert {
        *(f"raw_{key}" for key in METRIC_KEYS),
        *(f"normalized_improvement_{key}" for key in METRIC_KEYS),
    } <= visible
    rendered_user = str(observed["rendered_user"])
    assert rendered_user.count("<<<FILE engine.py>>>") == 1
    assert rendered_user.count("<<<FILE policies/x.py>>>") == 1
    assert rendered_user.count("<<<FILE policies/y.py>>>") == 1
    assert rendered_user.count("<<<FILE policies/unrelated.py>>>") == 1
    assert rendered_user.count("raised the response-quality coefficient") == 1
    assert "- Roles: top,inspiration,recent" in rendered_user
    assert "f" * 64 not in rendered_user
    assert "sha256" not in rendered_user
    assert "Feature coordinates" not in rendered_user
    assert "simplif" not in rendered_user.lower()
    assert "## Most Recent Failed Mutation\nNONE" in rendered_user
    assert str(observed["rendered_system"]) == str(observed["system"])


def test_adapter_recovers_programs_evicted_after_each_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "checkpoints"
    retained = Program("retained", "retained code", iteration_found=2)
    evicted = Program("evicted", "evicted code", iteration_found=1)
    for iteration, programs in ((1, (evicted, retained)), (2, (retained,))):
        directory = root / f"checkpoint_{iteration}" / "programs"
        directory.mkdir(parents=True)
        for program in programs:
            (directory / f"{program.id}.json").write_text(json.dumps(program.to_dict()))
    monkeypatch.setattr(adapter, "is_complete_checkpoint", lambda path: True)
    result = EvolutionResult(retained, 0.0, retained.code, {}, None, (retained,))
    recovered = adapter._all_checkpoint_programs(result, root)
    assert [item.id for item in recovered] == ["evicted", "retained"]


def test_objective_semantics_catalog_covers_frozen_objectives() -> None:
    assert tuple(OBJECTIVE_SEMANTICS) == METRIC_KEYS
    assert all(set(fields) == {"meaning", "statistical_unit", "unit"} for fields in OBJECTIVE_SEMANTICS.values())
    assert all(all(value for value in fields.values()) for fields in OBJECTIVE_SEMANTICS.values())


def test_shared_genetic_prompt_embeds_complete_current_policy_source() -> None:
    source = (
        "def apply(frame, context=None):\n"
        "    eligible = frame.loc[frame['eta'] <= 600].copy()\n"
        "    eligible['weight'] = 1.25\n"
        "    return eligible\n"
    )
    prompt = PromptSampler(PromptConfig(
        system_message="DispatchEvolve task context",
        use_template_stochasticity=False,
    )).build_prompt(current_program=source, language="python")
    assert prompt["system"] == "DispatchEvolve task context"
    assert source in prompt["user"]
    assert prompt["user"].count(source) == 1
    assert prompt["user"].index(source) > prompt["user"].index("# Current Program")
    assert "SEARCH/REPLACE" in prompt["user"]


def test_large_repository_history_uses_metrics_and_change_descriptions_without_code_duplication() -> None:
    current = "CURRENT_UNIQUE_SOURCE\n" + "x = 1\n" * 20_000
    historical = "HISTORICAL_DUPLICATE_SOURCE\n" + "y = 2\n" * 20_000
    config = PromptConfig(
        system_message="DispatchEvolve task context",
        use_template_stochasticity=False,
        history_programs_as_changes_description=True,
        num_top_programs=1,
        num_diverse_programs=0,
    )
    prompt = PromptSampler(config).build_prompt(
        current_program=current,
        parent_program=current,
        program_metrics={"combined_score": 0.0},
        previous_programs=[],
        top_programs=[{
            "code": historical,
            "changes_description": None,
            "metadata": {"changes": "raised the response-quality coefficient"},
            "metrics": {"combined_score": -1.02, "constraint_max_violation": 0.02},
        }],
        inspirations=[],
        language="python",
    )
    assert prompt["user"].count(current) == 1
    assert "HISTORICAL_DUPLICATE_SOURCE" not in prompt["user"]
    assert "raised the response-quality coefficient" in prompt["user"]
    assert "constraint_max_violation" in prompt["user"]


def test_v2_history_preserves_ranked_context_without_calling_it_diverse() -> None:
    config = PromptConfig(
        system_message="DispatchEvolve task context",
        use_template_stochasticity=False,
        history_programs_as_changes_description=True,
        deduplicate_prompt_history=True,
        num_top_programs=1,
        num_diverse_programs=1,
    )
    programs = [
        {
            "id": f"program-{index}",
            "code": f"x = {index}\n",
            "changes_description": f"ranked change {index}",
            "metrics": {"combined_score": float(2 - index)},
        }
        for index in (1, 2)
    ]
    prompt = PromptSampler(config).build_prompt(
        current_program="x = 0\n",
        program_metrics={"combined_score": 0.0},
        previous_programs=[],
        top_programs=programs,
        inspirations=[],
        language="python",
    )
    assert "## Additional Ranked Programs" in prompt["user"]
    assert "## Diverse Programs" not in prompt["user"]
    assert prompt["user"].count("ranked change 1") == 1
    assert prompt["user"].count("ranked change 2") == 1


def test_editable_first_repository_order_disambiguates_duplicate_search_text() -> None:
    files = {
        "engine.py": "def run_batch(frame):\n    return frame\n",
        "policies/allowed.py": "LIMIT = 600.0\n",
        "policies/unrelated.py": "LIMIT = 600.0\n",
    }
    genome = adapter._ordered_repository_genome(files, ("policies/allowed.py",))
    evolved, applied = apply_diff_blocks(genome, [("LIMIT = 600.0", "LIMIT = 500.0")])
    parsed = evaluator_adapter.repository_files(evolved)
    assert applied == 1
    assert parsed["policies/allowed.py"] == "LIMIT = 500.0\n"
    assert parsed["policies/unrelated.py"] == "LIMIT = 600.0\n"


def test_shared_genetic_evaluator_materializes_complete_engine(tmp_path: Path, monkeypatch) -> None:
    incumbent = tmp_path / "incumbent"
    (incumbent / "policies").mkdir(parents=True)
    (incumbent / "engine.py").write_text("def run_batch(frame, *, trace=True):\n    return frame, {}\n")
    (incumbent / "policies/x.py").write_text("def apply(frame):\n    return frame\n")
    (incumbent / "policies/y.py").write_text("def score(frame):\n    return frame\n")
    (incumbent / "policies/unrelated.py").write_text("VALUE = 1\n")
    frame_path = tmp_path / "scene.pkl"
    pd.DataFrame({"batch_id": [1], "product_id": [1], "eta": [1]}).to_pickle(frame_path)
    program = tmp_path / "program.py"
    evolved_repo = tmp_path / "evolved"
    shutil.copytree(incumbent, evolved_repo)
    x_source = "def apply(frame):\n    return frame.copy()\n"
    y_source = "def score(frame):\n    return frame.copy()\n"
    (evolved_repo / "policies/x.py").write_text(x_source)
    (evolved_repo / "policies/y.py").write_text(y_source)
    genome = RepositoryGenomeCodec().encode(evolved_repo)
    program.write_text(f"# EVOLVE-BLOCK-START\n{genome}# EVOLVE-BLOCK-END\n")
    context = {
        "incumbent_dir": str(incumbent),
        "relative_policies": ["policies/x.py", "policies/y.py"],
        "candidate_root": str(tmp_path / "candidates"), "scene_frame_path": str(frame_path),
        "metric_rows_path": None, "runner_root": str(tmp_path / "runner"),
        "cache_root": str(tmp_path / "cache"), "protocol_hash": "p", "backend": "local",
        "objective": "order_ar", "baseline_metrics": {key: 1.0 for key in METRIC_KEYS},
        "scales": {key: 1.0 for key in METRIC_KEYS}, "rho": 0.02,
        "comparison_tolerance": 1e-12,
    }
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context))

    def fake_evaluate(
        self, engine_dir, frame, *, output_dir, metric_rows=None, identity=None,
        trace=True, metrics_only=False,
    ):
        assert trace is False and metrics_only is True
        assert (engine_dir / "policies/x.py").read_text() == x_source
        assert (engine_dir / "policies/y.py").read_text() == y_source
        assert (engine_dir / "policies/unrelated.py").read_text() == "VALUE = 1\n"
        metrics = {key: 1.0 for key in METRIC_KEYS}; metrics["order_ar"] = 1.1
        return {"metrics": metrics}

    monkeypatch.setattr(evaluator_adapter.V2Evaluator, "evaluate", fake_evaluate)
    metrics = evaluator_adapter.evaluate_with_context(program, context_path)
    assert metrics["locally_feasible"] == 1.0
    assert metrics["combined_score"] > 0
    (evolved_repo / "policies/x.py").write_text("def renamed(frame):\n    return frame\n")
    invalid_genome = RepositoryGenomeCodec().encode(evolved_repo)
    program.write_text(f"# EVOLVE-BLOCK-START\n{invalid_genome}# EVOLVE-BLOCK-END\n")
    with pytest.raises(ValueError, match="public callable interface"):
        evaluator_adapter.evaluate_with_context(program, context_path)

    (evolved_repo / "policies/x.py").write_text(x_source)
    (evolved_repo / "policies/unrelated.py").write_text("VALUE = 2\n")
    forbidden_genome = RepositoryGenomeCodec().encode(evolved_repo)
    program.write_text(f"# EVOLVE-BLOCK-START\n{forbidden_genome}# EVOLVE-BLOCK-END\n")
    with pytest.raises(ValueError, match="outside related policies"):
        evaluator_adapter.evaluate_with_context(program, context_path)

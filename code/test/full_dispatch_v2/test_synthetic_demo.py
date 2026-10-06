"""Long-term real main-flow test: only LLM text generation is scripted.

Covers synthetic_demo, optimizer, subprocess evaluator, integration and archive.
"""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml


@pytest.mark.parametrize("seed", [42, 2**32, 2**80, -(2**80)])
def test_real_main_flow_on_generated_inputs(tmp_path, seed):
    root=tmp_path/'demo'
    completed=subprocess.run([sys.executable,'-m','dispatchevolve.synthetic_demo',
                              '--output-dir',str(root),'--random-seed',str(seed)],capture_output=True,text=True,timeout=180)
    assert completed.returncode==0, completed.stderr[-5000:]+completed.stdout[-3000:]
    assert yaml.safe_load((root/'config.yaml').read_text())['random_seed'] == seed
    summary=json.loads((root/'summary.json').read_text())
    assert summary['llm_mode']=='scripted'
    assert summary['accepted_combinations']==1
    assert summary['selected_metrics']['order_ar']>summary['baseline_metrics']['order_ar']
    assert summary['selected_metrics']['mean_eta']<summary['baseline_metrics']['mean_eta']
    engine=Path(summary['selected_engine']['engine_dir'])
    assert 'combinations' in engine.parts
    assert (engine/'engine.py').is_file()
    calls=[json.loads(line)['role'] for line in (root/'scripted_llm_calls.jsonl').read_text().splitlines()]
    assert {'scenario','opportunity','critic','local_mutation','combination'}<=set(calls)
    assert list((root/'runs/synthetic').rglob('outcome.json'))
    assert (root/'runs/synthetic/online_uplift/pairwise_ranking.json').is_file()
    assert summary['data_split'] == 'test'
    assert summary['selected_metrics']['mean_gmv'] == 12.
    assert summary['baseline_metrics']['mean_gmv'] == 12.
    assert summary['evolution_metrics']['baseline']['mean_gmv'] == 10.
    assert summary['paper_report']['fai_percent'] == 0.  # Unchanged objectives.
    final=json.loads((root/'runs/synthetic/final/test_result.json').read_text())
    assert final['feedback_to_search'] is False
    assert final['paper_report'] == summary['paper_report']
    assert summary['replay_cost']['used'] < 30
    assert summary['replay_cost']['used'] > 1


    # A budget-stop resume must retain evaluated entries from an unfinished round.
    from dispatchevolve.workflows.dispatchevolve_v2.config import load_config
    from dispatchevolve.workflows.dispatchevolve_v2.orchestrator import DispatchEvolveV2Orchestrator
    workflow=DispatchEvolveV2Orchestrator(load_config(root/'config.yaml'))
    state=workflow._load_or_initialize(True)
    state.completed_rounds=0
    result_file=root/'runs/synthetic/round_001/round_result.json'
    result_file.rename(result_file.with_suffix('.saved'))
    workflow._reconcile_pareto_archive(state)
    assert summary['selected_engine']['engine_id'] in {item['engine_id'] for item in state.archive}


def test_budget_stop_still_reports_test_set(tmp_path):
    for limit in (1, 1.2):
        root=tmp_path/f'budget_{limit}'
        completed=subprocess.run([sys.executable,'-m','dispatchevolve.synthetic_demo',
            '--output-dir',str(root),'--full-replay-equivalents',str(limit)],
            capture_output=True,text=True,timeout=180)
        assert completed.returncode == 0, completed.stderr[-6000:]
        summary=json.loads((root/'summary.json').read_text())
        assert summary['budget_stopped']
        assert summary['replay_cost']['used'] <= limit
        assert summary['data_split'] == 'test'
        assert summary['baseline_metrics']['mean_gmv'] == 12.
        before=summary['replay_cost']['used']
        resumed=subprocess.run([sys.executable,'-m','dispatchevolve.workflows.dispatchevolve_v2.cli',
            '--config',str(root/'config.yaml'),'--resume'],capture_output=True,text=True,timeout=60)
        assert resumed.returncode==0, resumed.stderr[-6000:]
        cost=json.loads((root/'runs/synthetic/final/replay_cost.json').read_text())
        assert cost['used']==before

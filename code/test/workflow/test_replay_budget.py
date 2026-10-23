"""Long-term replay-budget tests: proportional cost, persistence and process races."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pytest
from dispatchevolve.workflows.dispatchevolve_v2.replay_budget import ReplayBudget, ReplayBudgetExhausted


def charge(path):
    try:
        ReplayBudget(path).reserve(10, 'concurrent')
        return True
    except ReplayBudgetExhausted:
        return False


def test_fractional_cost_and_resume(tmp_path):
    ledger = ReplayBudget(tmp_path/'budget.sqlite')
    ledger.initialize(100, 2)
    ledger.reserve(100, 'full')
    ledger.reserve(20, 'scene')
    resumed = ReplayBudget(ledger.path)
    resumed.initialize(100, 2)
    assert resumed.snapshot()['used'] == pytest.approx(1.2)
    with pytest.raises(ReplayBudgetExhausted):
        resumed.reserve(100, 'too-large')
    assert resumed.snapshot()['used'] == pytest.approx(1.2)
    with pytest.raises(ValueError, match='differs'):
        resumed.initialize(100, 3)


def test_parallel_reservations_never_overspend(tmp_path):
    ledger = ReplayBudget(tmp_path/'budget.sqlite')
    ledger.initialize(100, 1)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        accepted = list(pool.map(charge, [ledger.path]*20))
    assert sum(accepted) == 10
    assert ledger.snapshot()['used'] == 1
    assert ledger.snapshot()['denied_calls'] == 10


def test_evaluator_cache_failure_and_test_exclusion(tmp_path, monkeypatch):
    from pathlib import Path
    import shutil
    from dispatchevolve.synthetic_demo import generate_frame
    from dispatchevolve.workflows.dispatchevolve_v2 import local_evaluator
    engine = tmp_path/'engine'
    shutil.copytree(Path(__file__).resolve().parents[2]/'examples/synthetic_engine', engine)
    ledger = ReplayBudget(tmp_path/'budget.sqlite')
    ledger.initialize(80, 3)
    evaluator = local_evaluator.V2Evaluator(runner_root=tmp_path/'runner',
        cache_root=tmp_path/'cache',protocol_hash='test',backend='local',replay_budget_path=ledger.path)
    frame = generate_frame().iloc[:8]
    calls=[]
    def replay(*args, **kwargs):
        calls.append(1)
        return {'metrics': {}, 'batch_traces': []}
    monkeypatch.setattr(local_evaluator,'evaluate_engine_candidate_only',replay)
    evaluator.evaluate(engine,frame,output_dir=tmp_path/'first')
    assert ledger.snapshot()['used'] == pytest.approx(.1)
    result=evaluator.evaluate(engine,frame,output_dir=tmp_path/'cached')
    assert result['cache_hit'] and len(calls)==1
    assert ledger.snapshot()['used'] == pytest.approx(.1)
    evaluator.evaluate(engine,frame,output_dir=tmp_path/'test',metrics_only=True,charge_replay=False)
    assert ledger.snapshot()['used'] == pytest.approx(.1)
    def failed(*args, **kwargs):
        raise RuntimeError('replay failed')
    monkeypatch.setattr(local_evaluator,'evaluate_engine_candidate_only',failed)
    with pytest.raises(RuntimeError,match='replay failed'):
        evaluator.evaluate(engine,frame,output_dir=tmp_path/'failed',trace=False)
    assert ledger.snapshot()['used'] == pytest.approx(.2)

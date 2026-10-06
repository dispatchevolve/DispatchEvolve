"""Long-term paper-contract tests for feasible archive admission and pairwise ranking."""
from dataclasses import replace
from pathlib import Path
import pytest
from dispatchevolve.workflows.dispatchevolve_v2.contracts import ArchiveEntry, METRIC_KEYS
from dispatchevolve.workflows.dispatchevolve_v2.pareto_archive import update_archive
from dispatchevolve.workflows.dispatchevolve_v2.pairwise_ranking import rank_engines


def entry(name, path, **gains):
    delta={key:0. for key in METRIC_KEYS};delta.update(gains)
    return ArchiveEntry(name,path,dict(delta),delta,1,('local',))


def test_archive_rejects_regression_and_uses_configured_guardrail(tmp_path):
    good=entry('good',tmp_path,order_ar=.1)
    bad=entry('bad',tmp_path,order_ar=.2,mean_eta=-.01)
    archive,event=update_archive([good],bad,tolerance=1e-12)
    assert archive==[good] and event['status']=='infeasible'
    guardrail=entry('guardrail',tmp_path,order_ar=.2,order_br=-.015)
    assert update_archive([good],guardrail,tolerance=1e-12,rho=.01)[1]['status']=='infeasible'
    assert update_archive([good],guardrail,tolerance=1e-12,rho=.02)[1]['status']=='retained'


def test_pairwise_wins_then_contribution_then_identifier(tmp_path):
    for name in ('a','b','c'):
        root=tmp_path/name;(root/'policies').mkdir(parents=True)
        (root/'engine.py').write_text('def run_batch(x):\n    return x\n')
        (root/'policies/__init__.py').write_text('')
    entries=[entry('a',tmp_path/'a',order_ar=.2,mean_eta=.1),
             entry('b',tmp_path/'b',order_ar=.1,mean_eta=.2),
             entry('c',tmp_path/'c',order_ar=.15,mean_eta=.15)]
    class LLM:
        def __init__(self,answers): self.answers=iter(answers);self.prompts=[]
        def complete(self,prompt): self.prompts.append(prompt);return next(self.answers)
    llm=LLM(['B','A','A']) # b defeats both; a defeats c.
    ranked,report=rank_engines(entries,llm=llm,context={'synthetic':True},objective_keys=('order_ar','mean_eta'))
    assert [item.engine_id for item in ranked]==['b','a','c']
    assert report['wins']=={'a':1,'b':2,'c':0}
    assert len(llm.prompts)==3
    assert all('historical' not in prompt.user for prompt in llm.prompts)
    cyclic=LLM(['A','B','A'])
    ranked,report=rank_engines(entries,llm=cyclic,context={},objective_keys=('order_ar','mean_eta'))
    expected=sorted(entries,key=lambda e:(-report['hypervolume_contributions'][e.engine_id],e.engine_id))
    assert ranked==expected
    with pytest.raises(ValueError,match='exactly A or B'):
        rank_engines(entries,llm=LLM(['uncertain']),context={},objective_keys=('order_ar','mean_eta'))


def test_default_br_guardrail_matches_paper(tmp_path):
    """Long-term archive boundary check: reject a 0.6% BR loss, allow 0.4%."""
    good = entry('good', tmp_path, order_ar=.1)
    for loss, expected in ((.004, 'retained'), (.006, 'infeasible')):
        candidate = entry('candidate', tmp_path, order_ar=.2, order_br=-loss)
        assert update_archive([good], candidate, tolerance=1e-12)[1]['status'] == expected

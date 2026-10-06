"""Run real evolution on generated matrices with an explicitly scripted LLM.

Only text generation is scripted. Candidate execution, metric computation,
local genetic search, trace collection and engine composition are real.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import threading
import pandas as pd
import yaml
from dispatchevolve.repo_paths import find_repo_root
from dispatchevolve.workflows.dispatchevolve_v2.config import load_config
from dispatchevolve.workflows.dispatchevolve_v2.orchestrator import DispatchEvolveV2Orchestrator


def generate_frame(batches: int = 20) -> pd.DataFrame:
    if batches < 10:
        raise ValueError('at least 10 complete batches are required')
    rows = []
    for batch in range(batches):
        special = batch < max(1, batches // 10)
        for order in range(2):
            for driver in range(2):
                diagonal = order == driver
                eta = (12. if diagonal else 24.) if special else (2. if diagonal else 3.)
                dar = .9 if diagonal else .5
                rows.append(dict(batch_id=batch, order_id=batch*2+order,
                    driver_id=batch*2+driver, product_id=1, eta=eta, dar=dar,
                    pcaa=.05, dcaa=.05, cr=dar*.95*.95, gmv=10., pre_total_fee=10.,
                    weight=1., stage=1,
                    driver_lock_time_s=0., order_lock_time_s=0., if_broadcast=1))
    return pd.DataFrame(rows)


def scripted_response(system: str, user: str) -> tuple[str, str]:
    if 'Scenario Discovery Query Designer' in system:
        return 'scenario', '## Query\nmean(eta) > 10\n## Discovery Purpose\nInspect the generated high-cost batches.'
    if 'Optimization Opportunity Architect' in system:
        return 'opportunity', ('## Scene Summary\nThe toy ranking favors high-cost pairs.\n'
            '## Opportunity Status\nOPPORTUNITY_FOUND\n## Improvement Opportunities\n'
            'objective | related_policies | improvement_plan | evidence_basis\n'
            'order_ar | policies/ranking.py | Prefer low-cost pairs using inverse ETA | Ranking executes on the generated scene')
    if 'Opportunity Quality Gatekeeper' in system:
        return 'critic', '## Decision\nACCEPT\n## Confidence\n1.0\n## Reason\nInspect the deliberately weak toy ranking.'
    if '# Current Program' in user and 'SEARCH/REPLACE' in user:
        return 'local_mutation', "<<<<<<< SEARCH\n    frame['weight'] = frame['eta']\n=======\n    frame['weight'] = 1.0 / (1.0 + frame['eta'])\n>>>>>>> REPLACE"
    if '### Selectable Local Candidate Summary' in user:
        section = user.split('### Selectable Local Candidate Summary',1)[1].split('###',1)[0]
        ids = re.findall(r'^\s*(\d{5})\s*\|', section, flags=re.M)
        if not ids:
            raise ValueError('no selectable candidate in composition prompt')
        return 'combination', f'## Selected Local Candidate IDs\n{ids[0]}\n## Conflict Resolution Plan\nNONE\n## Rationale\nEvaluate the generated inverse-cost policy as a complete engine.'
    if 'pairwise online-uplift' in system.lower():
        return 'online_pairwise', 'A'
    raise ValueError('unsupported scripted role; the demo cannot invent a response')


@contextmanager
def scripted_endpoint(log_path: Path):
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                messages = body['messages']
                system = '\n'.join(m['content'] for m in messages if m['role']=='system')
                user = '\n'.join(m['content'] for m in messages if m['role']=='user')
                role, response = scripted_response(system, user)
                with lock, log_path.open('a') as stream:
                    stream.write(json.dumps({'mode':'scripted_llm', 'role':role})+'\n')
                payload = {'id':'synthetic-response','object':'chat.completion','created':0,
                    'model':'synthetic-scripted','choices':[{'index':0,'finish_reason':'stop',
                    'message':{'role':'assistant','content':response}}],
                    'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}
                self.send_response(200)
            except Exception as exc:
                payload = {'error': {'message':str(exc), 'type':'demo_contract_error'}}
                self.send_response(400)
            self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown(); server.server_close(); thread.join()


def prepare(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    generate_frame().to_csv(root/'synthetic.csv',index=False)
    test_frame = generate_frame()
    for key in ('batch_id', 'order_id', 'driver_id'):
        test_frame[key] += 1000
    test_frame['eta'] *= 1.25
    test_frame['gmv'] = test_frame['pre_total_fee'] = 12.0
    test_frame.to_csv(root/'test.csv',index=False)
    shutil.copytree(find_repo_root()/'examples/synthetic_engine',root/'engine')
    config = dict(run_id='synthetic',mode='main',test_path='test.csv',evaluate_test_baseline=True,rho=0.005,engine_dir='engine',
        data_path='synthetic.csv',backend='local',output_dir='runs',cache_root='cache',
        log_root='logs', online_uplift_enabled=True,
        model=dict(provider='openai_compatible',model='synthetic-scripted',
                   api_base='http://127.0.0.1:1/v1', api_key_env='SYNTHETIC_DEMO_KEY',transport_max_retries=0),
        budget=dict(outer_rounds=1,parallel_opportunities=1,proposal_attempts_per_round=1,
                    admitted_opportunities_target=1,local_iterations_per_opportunity=1,
                    local_evaluations_per_opportunity=1,population_size=2,archive_size=2,
                    retained_candidates=1,combination_proposals_per_round=1,
                    combination_evaluations_per_round=1))
    path=root/'config.yaml'; path.write_text(yaml.safe_dump(config,sort_keys=False))
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('workspaces/synthetic_demo'))
    parser.add_argument('--full-replay-equivalents', type=float, default=30.0)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--live', action='store_true', help='Use a real external model instead of scripted replies')
    parser.add_argument('--model', default=None)
    parser.add_argument('--api-base', default='https://api.openai.com/v1')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    args=parser.parse_args()
    if args.live and not args.model:
        parser.error('--live requires --model')
    root=args.output_dir.resolve(); path=prepare(root)
    prepared=yaml.safe_load(path.read_text())
    prepared['budget']['full_replay_equivalents']=args.full_replay_equivalents
    prepared['random_seed']=args.random_seed
    path.write_text(yaml.safe_dump(prepared,sort_keys=False))
    load_config(path)
    if args.prepare_only:
        print(f'Generated synthetic inputs and config: {path}'); return
    old=os.environ.get('SYNTHETIC_DEMO_KEY'); os.environ['SYNTHETIC_DEMO_KEY']='synthetic-placeholder'
    try:
        with (nullcontext(args.api_base) if args.live else scripted_endpoint(root/'scripted_llm_calls.jsonl')) as endpoint:
            config=yaml.safe_load(path.read_text()); config['model']['api_base']=endpoint
            if args.live:
                config['model'].update(model=args.model, api_key_env=args.api_key_env)
                config['critic_model']=dict(config['model'])
            path.write_text(yaml.safe_dump(config,sort_keys=False))
            state=DispatchEvolveV2Orchestrator(load_config(path)).run()
        selected=json.loads((root/'runs/synthetic/final/selected_engine.json').read_text())
        feedback_path=root/'runs/synthetic/combinations/feedback.jsonl'
        feedback=[json.loads(line) for line in feedback_path.read_text().splitlines()] if feedback_path.exists() else []
        accepted=[item for item in feedback if item.get('acceptance_passed')]
        if not args.live and not accepted and not state.counters.get('replay_budget_stopped'):
            raise RuntimeError('scripted demo failed to produce a measured feasible combination')
        selected_entry=next(item for item in state.archive if item['engine_id']==state.incumbent_engine_id)
        test_result=json.loads((root/'runs/synthetic/final/test_result.json').read_text())
        summary={'input_origin':'generated_from_arithmetic_only','llm_mode':'live' if args.live else 'scripted',
                 'data_split':'test',
                 'baseline_metrics':test_result['current_engine_metrics'],
                 'selected_metrics':test_result['metrics'],
                 'paper_report':test_result['paper_report'],
                 'replay_cost':json.loads((root/'runs/synthetic/final/replay_cost.json').read_text()),
                 'budget_stopped':bool(state.counters.get('replay_budget_stopped')),
                 'evolution_metrics':{'baseline':state.baseline_metrics,'selected':selected_entry['metrics']},
                 'accepted_combinations':len(accepted),
                 'evaluations':'real_local_matching','stage':state.stage,
                 'archive_size':len(state.archive),'selected_engine':selected}
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    finally:
        if old is None: os.environ.pop('SYNTHETIC_DEMO_KEY',None)
        else: os.environ['SYNTHETIC_DEMO_KEY']=old


if __name__=='__main__':
    main()

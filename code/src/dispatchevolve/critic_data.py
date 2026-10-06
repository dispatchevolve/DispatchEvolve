"""Prepare grouped DPO splits from supplied trial outcomes or explicit preferences."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from .preferences import assemble_pair
from .workflows.dispatchevolve_v2.contracts import METRIC_KEYS, OBJECTIVE_KEYS
from .workflows.dispatchevolve_v2.local_evaluator import feasible


def label_record(record, *, kind, rho):
    if not isinstance(record.get('group_id'), str) or not record['group_id'].strip():
        raise ValueError('a nonempty group_id is required for leakage-safe splitting')
    if kind == 'opportunity':
        if record.get('evaluation_status') != 'completed':
            return None  # Infrastructure failures are not negative optimization labels.
        delta = record['oriented_delta']
        if set(delta) != set(METRIC_KEYS) or any(not math.isfinite(float(v)) for v in delta.values()):
            raise ValueError('all finite oriented metric deltas are required')
        if record['objective'] not in OBJECTIVE_KEYS:
            raise ValueError('unknown primary objective')
        label = 'ACCEPT' if feasible(delta, target=record['objective'],rho=rho,tolerance=1e-12) else 'REJECT'
        if set(record['responses']) != {'ACCEPT','REJECT'}:
            raise ValueError('opportunity responses must be named ACCEPT and REJECT')
        for decision, response in record['responses'].items():
            if not isinstance(response, str) or not re.fullmatch(
                r'## Decision\s*\n'+decision+r'\s*\n## Confidence\s*\n(?:0(?:\.\d+)?|1(?:\.0+)?)\s*\n## Reason\s*\n.+',
                response.strip(), re.S,
            ):
                raise ValueError('opportunity response does not match its named runtime decision')
        record = {**record, 'preferred':label}
    else:
        if record.get('preferred') is None:
            return None  # Ambiguous historical decisions are excluded.
        if set(record['responses']) != {'A','B'}:
            raise ValueError('online responses must be named A and B')
        if any(record['responses'][key] != key for key in ('A','B')):
            raise ValueError('online response text must match the exact A/B runtime contract')
    return assemble_pair(record)


def prepare_dataset(records, output_dir, *, kind, rho=0.005, eval_fraction=.2):
    if kind not in {'opportunity','online'} or not 0 <= rho < 1 or not 0 < eval_fraction < 1:
        raise ValueError('invalid dataset kind, rho, or eval fraction')
    pairs=[]; skipped=0; seen={}; groups=set()
    for record in records:
        pair=label_record(record,kind=kind,rho=rho)
        if pair is None:
            skipped+=1; continue
        # Same prompt cannot cross groups with either the same or opposite label.
        key=hashlib.sha256(json.dumps([pair['system'],pair['conversations']],sort_keys=True).encode()).hexdigest()
        if key in seen:
            raise ValueError('duplicate prompt: deduplicate before preparing grouped splits')
        seen[key]=record['group_id'];groups.add(record['group_id']);pairs.append((record['group_id'],pair))
    if len(groups)<2:
        raise ValueError('at least two independent labeled groups are required')
    ordered=sorted(groups,key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    count=max(1,min(len(ordered)-1,round(len(ordered)*eval_fraction)))
    evaluation=set(ordered[:count]); splits={'train':[],'eval':[]}
    for group,pair in pairs:
        splits['eval' if group in evaluation else 'train'].append(pair)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    info={};hashes={}
    for split,rows in splits.items():
        path=output_dir/f'{split}.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        info[f'critic_{split}']={'file_name':path.name,'formatting':'sharegpt','ranking':True,
            'columns':{'messages':'conversations','system':'system','chosen':'chosen','rejected':'rejected'}}
    (output_dir/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
    manifest={'kind':kind,'rho':rho,'input_records':len(records),'pairs':len(pairs),'skipped':skipped,
              'groups':len(groups),'train_groups':len(groups)-count,'eval_groups':count,'overlap_groups':0,
              'train_pairs':len(splits['train']),'eval_pairs':len(splits['eval']),'sha256':hashes}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def synthetic_records(kind):
    """Generate synthetic preference pairs for data-format tests."""
    rows=[]
    for index in range(10):
        base={'group_id':f'synthetic-group-{index}', 'system':'Judge only the supplied synthetic evidence.',
              'user':f'Artificial format example {index}: compare toy decisions.'}
        if kind=='opportunity':
            delta={key:0. for key in METRIC_KEYS};delta['order_ar']=.1 if index%2 else -.1
            base.update(evaluation_status='completed',objective='order_ar',oriented_delta=delta,
                        responses={decision:f'## Decision\n{decision}\n## Confidence\n1.0\n## Reason\nSynthetic format example.' for decision in ('ACCEPT','REJECT')})
        else:
            base.update(responses={'A':'A','B':'B'},preferred='A' if index%2 else 'B')
        rows.append(base)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['opportunity','online'],required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path);source.add_argument('--synthetic',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rho',type=float,default=0.005)
    args=parser.parse_args()
    records=synthetic_records(args.kind) if args.synthetic else [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    result=prepare_dataset(records,args.output_dir,kind=args.kind,rho=args.rho)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

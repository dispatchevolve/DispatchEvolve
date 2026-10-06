"""Validate DPO inputs and launch LLaMA-Factory; dry-run never downloads a model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import yaml

ROOT=Path(__file__).resolve().parents[1]


def resolve(value):
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def validate(config):
    if config.get('stage')!='dpo' or config.get('finetuning_type')!='lora':
        raise ValueError('this entrypoint supports LoRA DPO')
    if config.get('dataset')!='critic_train' or config.get('eval_dataset')!='critic_eval':
        raise ValueError('explicit grouped train/eval datasets are required')
    if config.get('val_size',0) or config.get('max_samples') or config.get('packing',False):
        raise ValueError('do not resplit, sample, or pack the prepared preference data')
    root=resolve(config['dataset_dir']); manifest=json.loads((root/'manifest.json').read_text())
    if manifest['overlap_groups']!=0:
        raise ValueError('train/eval groups overlap')
    rows=[]
    for name in ('train.jsonl','eval.jsonl'):
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256'][name]:
            raise ValueError(f'prepared dataset changed: {name}')
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    info=json.loads((root/'dataset_info.json').read_text())
    for split in ('train','eval'):
        spec=info[f'critic_{split}']
        if spec.get('file_name')!=f'{split}.jsonl' or spec.get('ranking') is not True or spec.get('formatting')!='sharegpt':
            raise ValueError('dataset registry differs from prepared split contract')
    output=resolve(config['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'nonempty training output: {output}')
    return rows


def check_complete_lengths(config, rows):
    # Use the trainer's own template rather than approximating token length.
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    tokenizer=AutoTokenizer.from_pretrained(config['model_name_or_path'],trust_remote_code=False)
    template=get_template_and_fix_tokenizer(tokenizer,DataArguments(template=config['template']))
    maximum=0
    for row in rows:
        messages=[{'role':'user','content':row['conversations'][0]['value']}]
        for side in ('chosen','rejected'):
            prompt,answer=template.encode_oneturn(tokenizer,messages+[{'role':'assistant','content':row[side]['value']}],row['system'],None)
            size=len(prompt)+len(answer)+int(template.efficient_eos)
            maximum=max(maximum,size)
    model_limit=getattr(tokenizer,'model_max_length',None)
    if isinstance(model_limit,int) and model_limit < 1_000_000_000 and maximum > model_limit:
        raise ValueError(f'complete examples exceed tokenizer context limit {model_limit}')
    if maximum>config['cutoff_len']:
        raise ValueError(f'cutoff_len={config["cutoff_len"]} would truncate complete examples; need at least {maximum}')
    return maximum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/critic_dpo.yaml')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    rows=validate(config)
    for key in ('dataset_dir','output_dir'): config[key]=str(resolve(config[key]))
    if args.dry_run:
        print(json.dumps({'pairs':len(rows),'dataset_dir':config['dataset_dir'],
            'output_dir':config['output_dir'],'status':'structural_preflight_only',
            'token_lengths_checked':False,'training_started':False},indent=2));return
    cli=shutil.which('llamafactory-cli')
    if cli is None: raise RuntimeError('install a compatible LLaMA-Factory training environment first')
    maximum=check_complete_lengths(config,rows)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_root=ROOT/'workspaces/training_preflight'/stamp;run_root.mkdir(parents=True)
    path=run_root/'resolved.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    metadata={'max_complete_tokens':maximum,'llamafactory_version':importlib.metadata.version('llamafactory'),
              'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (run_root/'preflight.json').write_text(json.dumps(metadata,indent=2)+'\n')
    logs=ROOT/'logs/scripts/train_critic_dpo';logs.mkdir(parents=True,exist_ok=True)
    log_path=logs/f'train_critic_dpo_{stamp}.log'
    print(f'Training log: {log_path}',flush=True)
    with log_path.open('x') as log:
        result=subprocess.run([cli,'train',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__=='__main__': main()

"""Long-term tests of DPO outcome labels, grouped splits and training preflight."""
import importlib.util
import json
from pathlib import Path
import pytest
from dispatchevolve.critic_data import label_record, prepare_dataset, synthetic_records

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('train_critic_dpo',ROOT/'scripts/train_critic_dpo.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def test_outcome_labels_and_ambiguous_exclusion():
    record=synthetic_records('opportunity')[1]
    assert 'ACCEPT' in label_record(record,kind='opportunity',rho=.02)['chosen']['value']
    record['oriented_delta']['mean_eta']=-.03
    assert 'REJECT' in label_record(record,kind='opportunity',rho=.02)['chosen']['value']
    record['evaluation_status']='transport_failure'
    assert label_record(record,kind='opportunity',rho=.02) is None
    online=synthetic_records('online')[0];online['preferred']=None
    assert label_record(online,kind='online',rho=.02) is None


def test_grouped_splits_and_tamper_detection(tmp_path):
    root=tmp_path/'prepared'
    records=synthetic_records('opportunity')
    records.append({**records[0],'user':'Another prompt from the same synthetic group.'})
    report=prepare_dataset(records,root,kind='opportunity')
    assert report['pairs']==11 and report['groups']==10 and report['overlap_groups']==0
    train=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    eval_rows=[json.loads(line) for line in (root/'eval.jsonl').read_text().splitlines()]
    first=records[0]['user']; second=records[-1]['user']
    for split in (train,eval_rows):
        prompts=[row['conversations'][0]['value'] for row in split]
        assert (first in prompts)==(second in prompts)
    config={'stage':'dpo','finetuning_type':'lora','dataset':'critic_train','eval_dataset':'critic_eval',
            'dataset_dir':str(root),'output_dir':str(tmp_path/'weights')}
    assert len(trainer.validate(config))==11
    (root/'train.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='changed'):
        trainer.validate(config)


def test_duplicate_prompts_are_rejected(tmp_path):
    records=synthetic_records('online')
    records.append({**records[0],'group_id':'different'})
    with pytest.raises(ValueError,match='duplicate prompt'):
        prepare_dataset(records,tmp_path/'output',kind='online')
    assert not (tmp_path/'output').exists()

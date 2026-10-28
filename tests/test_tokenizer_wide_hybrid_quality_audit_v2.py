"""Bound synthetic producer records only; no model or numerical success claim."""
import json

import pytest

from src.research import tokenizer_wide_hybrid_quality_audit_v2 as audit
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_retention_audit import fixture


def bind(root,args):
    return audit.bind_quality_stage(training_token_sha256=digest('synthetic96'),**args)


@pytest.mark.parametrize('boundary',['lr','rank'])
def test_full_stage_binds_every_producer_and_preserves_weights(tmp_path,boundary):
    args=fixture(tmp_path,boundary,producer_v2=True)
    report=bind(tmp_path,args)
    assert report['producer_provenance_bound'] and report['stage_audits_bound']
    assert not report['dispatch_authorized'] and not report['deletion_authorized']
    assert len(report['manifest']['delete'])==12
    assert len(list(tmp_path.rglob('*.pt')))==18


@pytest.mark.parametrize('name',['worker-outputs.json','worker-attempt.json','training-attempt.json',
    'training-outputs.json','training-summary.json','metrics.jsonl','evaluation-attempt.json'])
def test_missing_selected_winner_provenance_blocks_boundary(tmp_path,name):
    args=fixture(tmp_path,producer_v2=True)
    args['evidence'].pop('quality/00/'+name)
    with pytest.raises(ValueError):bind(tmp_path,args)


@pytest.mark.parametrize('target,key,value',[
    ('worker-outputs.json','kind','wide_hybrid_pilot_execution_v2'),
    ('worker-outputs.json','stage_audits_bound',True),
    ('worker-outputs.json','slot',True),
    ('worker-outputs.json','artifacts',{}),
    ('worker-attempt.json','checkpoint_allowance_bytes',1),
    ('training-outputs.json','config_sha256','0'*64),
    ('training-outputs.json','raw',{}),
    ('training-summary.json','completed_steps',1199),
    ('evaluation-attempt.json','training_token_sha256','0'*64),
    ('evaluation-attempt.json','files',{}),
    ('evaluation-attempt.json','sample_settings',{}),
])
def test_changed_producer_fields_even_with_new_file_hashes_block(tmp_path,target,key,value):
    args=fixture(tmp_path,producer_v2=True);name='quality/00/'+target;path=tmp_path/name
    data=json.loads(path.read_text());data[key]=value;path.write_text(json.dumps(data))
    args['evidence'][name]=file_digest(path)
    with pytest.raises(ValueError):bind(tmp_path,args)


@pytest.mark.parametrize('bad',['corpus','metrics','failure','foreign_slot','weight'])
def test_protected_data_and_failed_outputs_block(tmp_path,bad):
    args=fixture(tmp_path,producer_v2=True)
    if bad=='corpus':(tmp_path/'corpus.txt').write_text('changed')
    elif bad=='metrics':
        p=tmp_path/'quality/00/metrics.jsonl';p.write_text('{"loss":1}\n')
        args['evidence']['quality/00/metrics.jsonl']=file_digest(p)
    elif bad=='failure':(tmp_path/'quality/00/worker-failure.json').write_text('{}')
    elif bad=='foreign_slot':args['trials'][0]='quality/01'
    else:(tmp_path/'quality/00/checkpoints/raw.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):bind(tmp_path,args)

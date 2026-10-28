"""Synthetic LR prefixes only; no actual training or numerical audit claim."""
import copy
import json
import os

import pytest

from src.research import tokenizer_wide_hybrid_quality_audit_v2 as audit
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_retention_audit import fixture


def setup(root,count=3):
    args=fixture(root,producer_v2=True,prefix_count=count)
    return dict(root=root,training_token_sha256=digest('synthetic96'),
        requests=list(args['requests'].values()),
        **{k:args[k] for k in ('context','evidence','numerical_ref','projection_ref')})


def inventory(root):
    return {str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('count',range(1,10))
def test_every_prefix_preserves_all_files_without_selecting_or_authorizing(tmp_path,count):
    args=setup(tmp_path,count);before=inventory(tmp_path)
    report=audit.bind_lr_prefix(**args)
    assert len(report['records'])==count
    assert set(report['checkpoint_pairs'])=={f'quality/{i:02d}' for i in range(count)}
    assert report['prefix_audits_bound'] and report['producer_provenance_bound']
    assert report['stage_inventory_complete']
    for key in ('stage_audits_bound','inventory_completeness_verified','frozen_plan_verified',
                'storage_verified','dispatch_authorized','deletion_authorized'):
        assert report[key] is False
    assert report['selection'] is None and report['manifest'] is None
    assert not (tmp_path/'lr-selection.json').exists()
    assert inventory(tmp_path)==before


@pytest.mark.parametrize('bad',['empty','too_many','reordered','config','seed','gap','later','foreign'])
def test_only_exact_independent_ordered_completed_prefix_is_accepted(tmp_path,bad):
    args=setup(tmp_path)
    if bad=='empty':args['requests']=[]
    elif bad=='too_many':args['requests']*=4
    elif bad=='reordered':args['requests'].reverse()
    elif bad=='config':args['requests'][0]['config']['fixture_only']=False
    elif bad=='seed':args['requests'][1]['spec']['seed']+=1
    elif bad=='gap':(tmp_path/'quality/01').rename(tmp_path/'quality/04')
    elif bad=='later':(tmp_path/'quality/03').mkdir()
    else:(tmp_path/'quality/unknown').write_text('unreviewed')
    with pytest.raises(ValueError):audit.bind_lr_prefix(**args)


@pytest.mark.parametrize('slot',[0,2])
@pytest.mark.parametrize('name',['result.json','request.json','tensor-attempt.json','tensor-audit.json',
    'worker-outputs.json','training-outputs.json','resource-audit.json','resource-events.jsonl',
    'process/process.json','process/worker.log'])
def test_every_member_needs_complete_provenance(tmp_path,slot,name):
    args=setup(tmp_path);args['evidence'].pop(f'quality/{slot:02d}/{name}')
    with pytest.raises(ValueError):audit.bind_lr_prefix(**args)


@pytest.mark.parametrize('bad',['numerical_case','numerical_trace','numerical_source','context',
    'token_ids','corpus','unbound','weight','alias','failure','partial','ce'])
def test_prefix_cannot_grant_credit_for_tampered_or_failed_evidence(tmp_path,bad):
    args=setup(tmp_path)
    if bad.startswith('numerical_'):
        p=tmp_path/args['numerical_ref']['path'];value=json.loads(p.read_text())
        if bad=='numerical_case':value['cases'].pop()
        elif bad=='numerical_source':value['source_sha256']='0'*64
        else:(tmp_path/'numerical/trace.txt').write_text('changed')
        p.write_text(json.dumps(value));args['numerical_ref']['sha256']=file_digest(p)
    elif bad=='context':args['context']['plan_sha256']='0'*64
    elif bad=='token_ids':args['training_token_sha256']='0'*64
    elif bad=='corpus':(tmp_path/'corpus.txt').write_text('changed')
    elif bad=='alias':
        p=tmp_path/'quality/00/checkpoints/raw.pt'
        os.link(p,tmp_path/'quality/00/checkpoints/alias.pt')
    elif bad=='ce':
        name='quality/00/result.json';p=tmp_path/name;value=json.loads(p.read_text())
        value['h200']['mean_ce']-=.1;p.write_text(json.dumps(value))
        args['evidence'][name]=file_digest(p)
    else:
        name={'unbound':'unknown.json','weight':'checkpoints/raw.pt',
              'failure':'worker-failure.json','partial':'tensor-audit.json.tmp'}[bad]
        (tmp_path/'quality/00'/name).write_text('changed')
    with pytest.raises(ValueError):audit.bind_lr_prefix(**args)


def test_expected_request_is_not_inferred_from_saved_result(tmp_path):
    args=setup(tmp_path);expected=copy.deepcopy(args['requests'])
    name='quality/00/request.json';p=tmp_path/name;value=json.loads(p.read_text())
    value['config']['fixture_only']=False;p.write_text(json.dumps(value))
    args['evidence'][name]=file_digest(p)
    with pytest.raises(ValueError):audit.bind_lr_prefix(**args)
    assert args['requests']==expected

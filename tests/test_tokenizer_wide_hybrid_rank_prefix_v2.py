"""Synthetic live-anchor/rank audits, without claiming deletion-history verification."""
import copy
import json
import os

import pytest

from src.research import tokenizer_wide_hybrid_quality_audit_v2 as audit
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_retention_audit import fixture,write


def inventory(root):
    return {str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}


def setup(root,count=3,pending=False):
    lr=fixture(root,producer_v2=True)
    args=fixture(root,'rank',producer_v2=True,rank_prefix_count=count)
    # Only disposable byte fixtures, representing a separately reviewed cleanup.
    # This test intentionally supplies no deletion receipts: the audit must NOT
    # claim historical cleanup verification or grant next-stage authorization.
    anchors={i for i,r in enumerate(lr['records']) if r['lr']==lr['rates'][r['family']]}
    for i in set(range(9))-anchors:
        for name in ('raw.pt','h200.pt'):(root/f'quality/{i:02d}/checkpoints'/name).unlink()
    (root/'rank-selection.json').unlink()
    kwargs=dict(root=root,context=args['context'],lr_records=lr['records'],rates=lr['rates'],
        lr_requests=[lr['requests'][i] for i in range(9)],
        rank_requests=[args['requests'][i] for i in range(9,9+count)],
        numerical_ref=args['numerical_ref'],rates_ref=args['selection_refs']['rates'],
        projection_ref=args['projection_ref'],training_token_sha256=digest('synthetic96'))
    if pending:
        spec=audit.rules.rank_specs(kwargs['rates'])[count];i=9+count
        request=dict(spec=spec,config={'fixture_only':True,'data':{'path':str(root/'corpus.txt')}},
            result_path=str(root/f'quality/{i:02d}/result.json'))
        write(root,f'quality/{i:02d}/request.json',request);kwargs['pending_request']=request
    kwargs['evidence']={n:sha for n,sha in inventory(root).items() if not n.endswith('.pt')}
    return kwargs


@pytest.mark.parametrize('count',range(7))
def test_every_rank_prefix_reaudits_anchors_without_selecting_or_authorizing(tmp_path,count):
    args=setup(tmp_path,count);before=inventory(tmp_path)
    checked=audit.bind_rank_prefix(**args)
    assert len(checked['records'])==9+count
    assert checked['audited_indices']==[0,1,2]+list(range(9,9+count))
    assert checked['historical_lr_indices']==list(range(3,9))
    assert set(checked['checkpoint_pairs'])=={f'quality/{i:02d}' for i in checked['audited_indices']}
    assert checked['prefix_audits_bound'] and checked['producer_provenance_bound']
    assert checked['stage_inventory_complete'] and checked['rates']==args['rates']
    for key in ('lr_history_verified','stage_audits_bound','inventory_completeness_verified',
                'frozen_plan_verified','storage_verified','dispatch_authorized','deletion_authorized'):
        assert checked[key] is False
    assert checked['selection'] is None and checked['manifest'] is None
    assert not (tmp_path/'rank-selection.json').exists() and inventory(tmp_path)==before


@pytest.mark.parametrize('count',range(6))
def test_pending_request_receives_no_completed_record_or_checkpoint_credit(tmp_path,count):
    args=setup(tmp_path,count,pending=True);before=inventory(tmp_path)
    checked=audit.bind_rank_prefix(**args)
    assert len(checked['records'])==9+count and f'quality/{9+count:02d}' not in checked['checkpoint_pairs']
    assert inventory(tmp_path)==before


@pytest.mark.parametrize('bad',['lr_order','lr_count','lr_ce','lr_request_config','rates','rates_file',
    'rank_order','rank_lr','rank_count','gap','future','unknown','pending_extra','pending_spec','pending_config'])
def test_only_exact_frozen_conditional_prefix_and_saved_rates_are_accepted(tmp_path,bad):
    args=setup(tmp_path,pending=bad.startswith('pending'))
    if bad=='lr_order':args['lr_records'].reverse()
    elif bad=='lr_count':args['lr_records'].pop()
    elif bad=='lr_ce':args['lr_records'][0]['ce']-=.2
    elif bad=='lr_request_config':args['lr_requests'][0]['config']['fixture_only']=False
    elif bad=='rates':args['rates']['hybrid']=.00125
    elif bad=='rates_file':
        args['rates_ref']=write(tmp_path,'lr-selection.json',dict(args['rates'],kda=.0025))
        args['evidence']['lr-selection.json']=args['rates_ref']['sha256']
    elif bad=='rank_order':args['rank_requests'].reverse()
    elif bad=='rank_lr':args['rank_requests'][0]['spec']['lr']=.0025
    elif bad=='rank_count':args['rank_requests']*=3
    elif bad=='gap':(tmp_path/'quality/09').rename(tmp_path/'quality/15')
    elif bad=='future':(tmp_path/'quality/15').mkdir()
    elif bad=='unknown':(tmp_path/'quality/09/unknown').mkdir()
    elif bad=='pending_extra':write(tmp_path,'quality/12/result.json',{})
    elif bad=='pending_spec':args['pending_request']['spec']['divisor']=4
    else:args['pending_request']['config']['fixture_only']=False
    with pytest.raises(ValueError):audit.bind_rank_prefix(**args)


@pytest.mark.parametrize('slot',[0,2,9,11])
@pytest.mark.parametrize('name',['result.json','request.json','tensor-audit.json','worker-outputs.json',
    'training-outputs.json','resource-audit.json','resource-events.jsonl','process/process.json','process/worker.log'])
def test_every_anchor_and_rank_member_requires_provenance(tmp_path,slot,name):
    args=setup(tmp_path);args['evidence'].pop(f'quality/{slot:02d}/{name}')
    with pytest.raises(ValueError):audit.bind_rank_prefix(**args)


@pytest.mark.parametrize('slot',[1,10])
@pytest.mark.parametrize('bad',['weight','missing_weight','alias','failure','partial','ce','ema','holdout'])
def test_live_anchor_or_rank_corruption_fails_even_when_rehashed(tmp_path,slot,bad):
    args=setup(tmp_path);stem=f'quality/{slot:02d}'
    if bad=='weight':(tmp_path/stem/'checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='missing_weight':(tmp_path/stem/'checkpoints/h200.pt').unlink()
    elif bad=='alias':os.link(tmp_path/stem/'checkpoints/raw.pt',tmp_path/stem/'checkpoints/alias.pt')
    elif bad in ('failure','partial'):write(tmp_path,stem+('/worker-failure.json' if bad=='failure' else '/tensor-audit.json.tmp'),{})
    else:
        name=stem+'/result.json';p=tmp_path/name;r=json.loads(p.read_text())
        if bad=='ce':r['h200']['mean_ce']-=.1
        elif bad=='ema':r['training']['weight_averaging']['snapshots']=40
        else:r['holdout_consumed']=True
        p.write_text(json.dumps(r));args['evidence'][name]=file_digest(p)
    with pytest.raises((ValueError,FileNotFoundError)):audit.bind_rank_prefix(**args)


@pytest.mark.parametrize('bad',['case','trace','source','token_ids','corpus','conflict'])
def test_actual_numerical_input_and_corpus_bindings_remain_required(tmp_path,bad):
    args=setup(tmp_path)
    if bad in ('case','source'):
        p=tmp_path/args['numerical_ref']['path'];r=json.loads(p.read_text())
        if bad=='case':r['cases'].pop()
        else:r['source_sha256']='0'*64
        p.write_text(json.dumps(r));args['numerical_ref']['sha256']=file_digest(p)
        args['evidence'][args['numerical_ref']['path']]=file_digest(p)
    elif bad=='trace':(tmp_path/'numerical/trace.txt').write_bytes(b'changed')
    elif bad=='token_ids':args['training_token_sha256']='0'*64
    elif bad=='corpus':(tmp_path/'corpus.txt').write_bytes(b'changed')
    else:args['evidence'][args['projection_ref']['path']]='0'*64
    with pytest.raises(ValueError):audit.bind_rank_prefix(**args)


def test_saved_request_cannot_replace_independent_configuration(tmp_path):
    args=setup(tmp_path);before=copy.deepcopy(args['rank_requests'])
    name='quality/09/request.json';p=tmp_path/name;r=json.loads(p.read_text())
    r['config']['fixture_only']=False;p.write_text(json.dumps(r));args['evidence'][name]=file_digest(p)
    with pytest.raises(ValueError):audit.bind_rank_prefix(**args)
    assert args['rank_requests']==before

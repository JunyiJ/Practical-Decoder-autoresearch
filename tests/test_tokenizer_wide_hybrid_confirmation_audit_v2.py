"""Disposable byte/provenance fixtures; no actual numerical or quality evidence."""
import copy
import json
import os

import pytest

from src.research import tokenizer_wide_hybrid_confirmation_audit_v2 as audit
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_retention_audit import fixture,write


def inventory(root):
    return {str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}


def setup(root,count=4,pending=False):
    full=fixture(root,'rank',producer_v2=True)
    args=fixture(root,'rank',producer_v2=True,confirmation_prefix_count=count)
    anchors=set(args['requests'])&set(range(15))
    for i in set(full['requests'])-anchors:
        for name in ('raw.pt','h200.pt'):(root/f'quality/{i:02d}/checkpoints'/name).unlink()
    kwargs=dict(root=root,context=args['context'],screen_records=full['records'],rates=args['rates'],
        selection=args['selection'],screen_requests=[],confirmation_requests=[args['requests'][i] for i in range(15,15+count)],
        numerical_ref=args['numerical_ref'],rates_ref=args['selection_refs']['rates'],
        selection_ref=args['selection_refs']['selection'],projection_ref=args['projection_ref'],
        training_token_sha256=digest('synthetic96'))
    # Historical losing LR directories and requests are preserved independently
    # from the rank audit's live candidate set. No historical verification claim.
    for i,row in enumerate(full['records']):
        spec={k:row[k] for k in ('stage','family','seed','lr','divisor','width','steps')}
        request=dict(spec=spec,config={'fixture_only':True,'data':{'path':str(root/'corpus.txt')}},
            result_path=str(root/f'quality/{i:02d}/result.json'))
        kwargs['screen_requests'].append(request)
        if not (root/f'quality/{i:02d}').exists():write(root,f'quality/{i:02d}/request.json',request)
    if pending:
        i=15+count;spec=audit.rules.confirmation_specs(args['selection'])[count]
        request=dict(spec=spec,config={'fixture_only':True,'data':{'path':str(root/'corpus.txt')}},
            result_path=str(root/f'quality/{i:02d}/result.json'))
        write(root,f'quality/{i:02d}/request.json',request);kwargs['pending_request']=request
    kwargs['evidence']={n:sha for n,sha in inventory(root).items() if not n.endswith('.pt')}
    return kwargs


@pytest.mark.parametrize('count',range(10))
def test_every_fresh_prefix_preserves_selection_and_all_pairs(tmp_path,count):
    args=setup(tmp_path,count);before=inventory(tmp_path)
    checked=audit.bind_confirmation_prefix(**args)
    assert len(checked['records'])==15+count and checked['records'][:15]==args['screen_records']
    assert checked['audited_indices']==[9,10,11]+list(range(15,15+count))
    assert len(checked['checkpoint_pairs'])==3+count and len(list(tmp_path.rglob('*.pt')))==2*(3+count)
    assert checked['selection']==args['selection'] and checked['rates']==args['rates']
    assert checked['prefix_audits_bound'] and checked['producer_provenance_bound'] and checked['stage_inventory_complete']
    for k in ('rank_history_verified','stage_audits_bound','inventory_completeness_verified','frozen_plan_verified',
              'storage_verified','adopted','dispatch_authorized','deletion_authorized'):assert checked[k] is False
    assert checked['manifest'] is None and inventory(tmp_path)==before


@pytest.mark.parametrize('count',range(9))
def test_request_only_pending_trial_has_no_completed_or_checkpoint_credit(tmp_path,count):
    args=setup(tmp_path,count,pending=True);before=inventory(tmp_path)
    checked=audit.bind_confirmation_prefix(**args)
    assert len(checked['records'])==15+count and f'quality/{15+count:02d}' not in checked['checkpoint_pairs']
    assert inventory(tmp_path)==before


@pytest.mark.parametrize('bad',['screen_order','screen_count','screen_ce','screen_spec','rates','selection','selection_file',
    'fresh_order','fresh_seed','fresh_lr','fresh_count','gap','future','unknown','pending_extra','pending_spec','pending_config'])
def test_exact_screen_choice_and_frozen_fresh_order_required(tmp_path,bad):
    args=setup(tmp_path,pending=bad.startswith('pending'))
    if bad=='screen_order':args['screen_records'].reverse()
    elif bad=='screen_count':args['screen_records'].pop()
    elif bad=='screen_ce':args['screen_records'][0]['ce']-=.2
    elif bad=='screen_spec':args['screen_requests'][0]['spec']['seed']+=1
    elif bad=='rates':args['rates']['kda']=.0025
    elif bad=='selection':args['selection']['kda']['divisor']=2
    elif bad=='selection_file':
        args['selection_ref']=write(tmp_path,'rank-selection.json',{})
        args['evidence']['rank-selection.json']=args['selection_ref']['sha256']
    elif bad=='fresh_order':args['confirmation_requests'].reverse()
    elif bad=='fresh_seed':args['confirmation_requests'][0]['spec']['seed']+=1
    elif bad=='fresh_lr':args['confirmation_requests'][0]['spec']['lr']=.0025
    elif bad=='fresh_count':args['confirmation_requests']*=3
    elif bad=='gap':(tmp_path/'quality/15').rename(tmp_path/'quality/23')
    elif bad=='future':(tmp_path/'quality/24').mkdir()
    elif bad=='unknown':(tmp_path/'quality/15/unknown').mkdir()
    elif bad=='pending_extra':write(tmp_path,'quality/19/result.json',{})
    elif bad=='pending_spec':args['pending_request']['spec']['divisor']=4
    else:args['pending_request']['config']['fixture_only']=False
    with pytest.raises(ValueError):audit.bind_confirmation_prefix(**args)


@pytest.mark.parametrize('slot',[9,11,15,18])
@pytest.mark.parametrize('name',['result.json','request.json','tensor-audit.json','worker-outputs.json',
    'training-outputs.json','resource-audit.json','resource-events.jsonl','process/process.json','process/worker.log'])
def test_every_selected_anchor_and_fresh_trial_needs_complete_provenance(tmp_path,slot,name):
    args=setup(tmp_path);args['evidence'].pop(f'quality/{slot:02d}/{name}')
    with pytest.raises(ValueError):audit.bind_confirmation_prefix(**args)


@pytest.mark.parametrize('slot',[10,17])
@pytest.mark.parametrize('bad',['weight','missing_weight','alias','failure','partial','ce','ema','holdout','tensor','resource'])
def test_corruption_fails_with_updated_inventory(tmp_path,slot,bad):
    args=setup(tmp_path);stem=f'quality/{slot:02d}'
    if bad=='weight':(tmp_path/stem/'checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='missing_weight':(tmp_path/stem/'checkpoints/h200.pt').unlink()
    elif bad=='alias':os.link(tmp_path/stem/'checkpoints/raw.pt',tmp_path/stem/'checkpoints/alias.pt')
    elif bad in ('failure','partial'):write(tmp_path,stem+('/worker-failure.json' if bad=='failure' else '/tensor-audit.json.tmp'),{})
    else:
        name=stem+'/'+({'tensor':'tensor-audit.json','resource':'resource-audit.json'}.get(bad,'result.json'))
        p=tmp_path/name;r=json.loads(p.read_text())
        if bad=='ce':r['h200']['mean_ce']-=.1
        elif bad=='ema':r['training']['weight_averaging']['snapshots']=40
        elif bad=='tensor':r['values']['exact_export_matches_stored_ema']=False
        elif bad=='resource':r['samples'][-1]['peak_rss_bytes']=4*1024**3
        else:r['holdout_consumed']=True
        p.write_text(json.dumps(r));args['evidence'][name]=file_digest(p)
    with pytest.raises((ValueError,FileNotFoundError)):audit.bind_confirmation_prefix(**args)


@pytest.mark.parametrize('bad',['case','trace','source','token_ids','corpus','conflict'])
def test_actual_numerical_inputs_and_corpus_remain_required(tmp_path,bad):
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
    with pytest.raises(ValueError):audit.bind_confirmation_prefix(**args)

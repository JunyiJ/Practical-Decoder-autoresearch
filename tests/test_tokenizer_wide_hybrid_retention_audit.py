"""Synthetic bound JSON and tiny bytes only; no training or real audit claims."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_retention_audit as audit
from src.research import tokenizer_wide_hybrid_rules as rules
from src.research.protocol import digest

ROOT = Path(__file__).resolve().parents[1]


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes): path.write_bytes(value)
    else: path.write_text(json.dumps(value))
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def score(parameters, path, sha, ce=5.):
    counts = [min(2048, 34278-i) for i in range(0,34278,2048)]
    return dict(context=128,eval_batch_size=8,block_tokens=2048,target_count=34278,
        target_bytes=111536,window_count=math.ceil(34278/128),parameters=parameters,
        block_target_counts=counts,block_loss_sums=[ce*n for n in counts],mean_ce=ce,
        bits_per_byte=ce*34278/111536/math.log(2),perplexity=math.exp(ce),
        checkpoint=str(path),checkpoint_sha256=sha)


def fixture(root, boundary='lr', *, producer_v2=False):
    records=[dict(s,status='completed',ce=5.) for s in rules.lr_specs()]
    rates=rules.choose_rates(records)
    selection=None
    if boundary=='rank':
        records += [dict(s,status='completed',ce=5.) for s in rules.rank_specs(rates)]
        selection=rules.choose_ranks(records[:9],records[9:])
    indices=audit.boundary_candidates(boundary,records,rates,selection)['audit_required_indices']
    context={k:digest(k) for k in ('plan_sha256','source_sha256','runtime_sha256','corpus_sha256')}
    if producer_v2:
        context['corpus_sha256']=write(root,'corpus.txt',b'synthetic corpus')['sha256']
    projection=json.loads((ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json').read_text())
    pref=write(root,'projection.json',projection)
    budgets=audit.allocations(projection)
    trace=write(root,'numerical/trace.txt',b'synthetic numerical trace only')
    numerical=dict(kind='numerical',status='passed',
        **{k:context[k] for k in ('source_sha256','runtime_sha256','corpus_sha256')},
        checks=dict.fromkeys(('all_declared_ranks',)+audit.NUMERICAL_CHECKS,True),
        cases=[dict(width=512,family=f,divisor=d,checks=dict.fromkeys(audit.NUMERICAL_CHECKS,True))
               for f in rules.FAMILIES for d in (8,4,2)],artifacts={trace['path']:trace['sha256']})
    nref=write(root,'numerical/review.json',numerical)
    srefs={'rates':write(root,'lr-selection.json',rates)}
    if selection is not None: srefs['selection']=write(root,'rank-selection.json',selection)
    evidence={}; requests={}; trials={}
    for i in indices:
        record=records[i]; spec={k:record[k] for k in ('stage','family','seed','lr','divisor','width','steps')}
        stem=f'quality/{i:02d}' if producer_v2 else f'trial-{i}'; trials[i]=stem
        request=dict(spec=spec,config={'fixture_only':True},result_path=str(root/stem/'result.json'))
        if producer_v2:request['config']['data']={'path':str(root/'corpus.txt')}
        requests[i]=request
        rq=write(root,stem+'/request.json',request)
        pair={k:write(root,f'{stem}/checkpoints/{k}.pt',f'synthetic {i} {k}'.encode()) for k in ('raw','h200')}
        parameters=next(r['parameters'] for r in projection['records'] if r['width']==512 and r['family']==spec['family'] and r['bottleneck_divisor']==spec['divisor'])
        result=dict(spec,status='completed',holdout_consumed=False,request_sha256=rq['sha256'],
            training=dict(completed_steps=1200,weight_averaging=dict(snapshots=41)),
            **{k:score(parameters,root/pair[k]['path'],pair[k]['sha256']) for k in pair})
        if producer_v2:
            result['training'].update(metrics=[],final_checkpoint=str(root/pair['raw']['path']))
            from src.research.tokenizer_wide_hybrid_pilot_audit_v2 import SAMPLE_SETTINGS
            result['h200'].update(sample='synthetic',sample_settings=SAMPLE_SETTINGS)
        rs=write(root,stem+'/result.json',result)
        files={str(root/rq['path']):rq['sha256'],str(root/rs['path']):rs['sha256'],
               **{str(root/v['path']):v['sha256'] for v in pair.values()}}
        values={**dict.fromkeys(audit.VALUE_CHECKS,True),**dict.fromkeys(audit.NONCLAIMS,False),
            'tensor_counts':dict.fromkeys(('raw_model','stored_ema','optimizer','exported_model'),1),
            'parameters_compared':1,'buffers_compared':0}
        metadata=dict(metadata_reviewed=True,tensor_values_audited=False,adopted=False,
            request_sha256=rq['sha256'],result_sha256=rs['sha256'],selection_row=record,
            checkpoint_hashes={k:v['sha256'] for k,v in pair.items()},
            h200_loss_sums=result['h200']['block_loss_sums'],target_counts=result['h200']['block_target_counts'])
        tensor=dict(kind='quality',adopted=False,status='export_values_reviewed_pending_numerical_and_resource_audits',
                    files=files,metadata=metadata,values=values)
        allowance=budgets[spec['family'],spec['divisor']]
        process=dict(status='completed',returncode=0,error=None,reserve_bytes=audit.RESERVE,
            timeout_seconds=7200,output_limit_bytes=allowance,log_limit_bytes=4*1024**2,
            elapsed_seconds=2.,required_free_bytes=30_631_281_412,
            observation=dict(free_bytes=40*1024**3,output_bytes=5000,log_bytes=10))
        resource=dict(kind='wide_hybrid_trial_resources_v2',status='completed',plan_sha256=context['plan_sha256'],
            request_sha256=rq['sha256'],result_sha256=rs['sha256'],limits=audit.RESOURCE_LIMITS,
            violations=[],samples=[dict(elapsed_seconds=t,peak_rss_bytes=1000,mps_driver_bytes=1000,free_bytes=40*1024**3) for t in (0.,1.)])
        from src.research.tokenizer_wide_hybrid_resources import header
        events=[header(context['plan_sha256'],rq['sha256'])]
        events += [dict(event='sample',**row) for row in resource['samples']]
        events += [dict(event='terminal',status='completed',result_sha256=rs['sha256'],reason=None)]
        event_ref=write(root,stem+'/resource-events.jsonl',('\n'.join(json.dumps(e) for e in events)+'\n').encode())
        resource['events']=dict(path='resource-events.jsonl',sha256=event_ref['sha256'])
        refs=[rq,rs,event_ref,write(root,stem+'/tensor-audit.json',tensor),write(root,stem+'/resource-audit.json',resource),
              write(root,stem+'/process/process.json',process),write(root,stem+'/process/worker.log',b'synthetic log')]
        if producer_v2:
            training_attempt=write(root,stem+'/training-attempt.json',dict(kind='wide_hybrid_training_outputs_v2',
                status='started',config_sha256=digest(request['config']),checkpoint_allowance_bytes=allowance))
            training=write(root,stem+'/training-outputs.json',dict(kind='wide_hybrid_training_outputs_v2',
                status='completed',config_sha256=digest(request['config']),numerical_audits_passed=False,dispatch_authorized=False,
                **{k:dict(path=str(root/v['path']),size=(root/v['path']).stat().st_size,sha256=v['sha256']) for k,v in pair.items()}))
            summary=write(root,stem+'/training-summary.json',result['training'])
            metrics=write(root,stem+'/metrics.jsonl',b'')
            evaluation=write(root,stem+'/evaluation-attempt.json',dict(kind='wide_hybrid_evaluation_v2',status='started',
                request_sha256=rq['sha256'],training_outputs_sha256=training['sha256'],
                training_contract_sha256=digest('synthetic contract'),training_token_sha256=digest('synthetic96'),
                files={str(root/rq['path']):rq['sha256'],str(root/training['path']):training['sha256'],
                    str(root/summary['path']):summary['sha256'],str(root/'corpus.txt'):context['corpus_sha256'],
                    **{str(root/v['path']):v['sha256'] for v in pair.values()}},sample_settings=SAMPLE_SETTINGS))
            attempt=write(root,stem+'/worker-attempt.json',dict(kind='wide_hybrid_quality_execution_v2',status='started',
                slot=i,plan_sha256=context['plan_sha256'],request_sha256=rq['sha256'],projection_ref=pref,
                checkpoint_allowance_bytes=allowance,dispatch_authorized=False))
            from src.research.tokenizer_wide_hybrid_quality_tensors_v2 import attempt_contract,KIND
            tensor_attempt=write(root,stem+'/tensor-attempt.json',attempt_contract(i,context['plan_sha256'],
                rq['sha256'],rs['sha256'],pref,digest('synthetic contract'),digest('synthetic96')))
            tensor['producer']=dict(kind=KIND,attempt=tensor_attempt,numerical_audits_passed=False,
                dispatch_authorized=False,files={str(root/rq['path']):rq['sha256'],str(root/rs['path']):rs['sha256'],
                    str(root/training['path']):training['sha256'],str(root/evaluation['path']):evaluation['sha256'],
                    str(root/pref['path']):pref['sha256'],**{str(root/v['path']):v['sha256'] for v in pair.values()}})
            tensor_ref=write(root,stem+'/tensor-audit.json',tensor)
            refs=[tensor_ref if v['path']==tensor_ref['path'] else v for v in refs]
            worker=write(root,stem+'/worker-outputs.json',dict(kind='wide_hybrid_quality_execution_v2',
                status='quality_outputs_complete_pending_independent_stage_review',slot=i,
                plan_sha256=context['plan_sha256'],request_sha256=rq['sha256'],attempt=attempt,
                artifacts=dict(training_outputs=training,evaluation_attempt=evaluation,result=rs,tensor=tensor_ref,
                    resources=next(v for v in refs if v['path']==stem+'/resource-audit.json')),
                adopted=False,dispatch_authorized=False,stage_audits_bound=False))
            refs += [training_attempt,training,summary,metrics,evaluation,attempt,worker,tensor_attempt]
        evidence.update({ref['path']:ref['sha256'] for ref in refs})
    return dict(root=root,boundary=boundary,records=records,rates=rates,selection=selection,
        context=context,requests=requests,trials=trials,evidence=evidence,numerical_ref=nref,
        selection_refs=srefs,projection_ref=pref)


def mutate(root, name, fn):
    value=json.loads((root/name).read_text());fn(value)
    return write(root,name,value)


@pytest.mark.parametrize('boundary',['lr','rank'])
def test_full_stage_binding_and_limits_of_claim(tmp_path,boundary):
    args=fixture(tmp_path,boundary)
    report=audit.bind_quality_boundary(**args)
    assert report['stage_audits_bound'] is True
    assert len(report['manifest']['delete'])==12
    assert len(report['manifest']['retain'])==6
    for key in ('deletion_authorized','dispatch_authorized','frozen_plan_verified','inventory_completeness_verified'):
        assert report[key] is False
    assert len(list(tmp_path.rglob('*.pt')))==18  # no deletion


@pytest.mark.parametrize('mutation',['missing_case','geometry_only','wrong_source','missing_trace'])
def test_numerical_prerequisites_cannot_be_replaced_by_metadata(tmp_path,mutation):
    args=fixture(tmp_path)
    def change(r):
        if mutation=='missing_case': r['cases'].pop()
        elif mutation=='geometry_only': r['cases'][0]['checks']['resume']=False
        elif mutation=='wrong_source': r['source_sha256']='0'*64
        else: r['artifacts']={}
    args['numerical_ref']=mutate(tmp_path,args['numerical_ref']['path'],change)
    with pytest.raises(ValueError): audit.bind_quality_boundary(**args)


@pytest.mark.parametrize('target,mutation',[
    ('tensor','missing_values'),('tensor','overclaim'),('tensor','empty_optimizer'),
    ('tensor','wrong_row'),('tensor','wrong_hash'),('tensor','loss_sums'),
    ('resource','rss'),('resource','driver'),('resource','disk'),('resource','time'),
    ('resource','missing_sample'),('resource','no_mps'),('resource','limits'),('resource','plan'),
    ('process','failure'),('process','timeout'),('process','output'),
    ('result','holdout'),('result','steps'),('result','snapshots'),('result','ce')])
def test_invalid_selected_winner_also_blocks_cleanup(tmp_path,target,mutation):
    args=fixture(tmp_path)
    # Index0 is selected: even losing-model-only cleanup needs its full review.
    names={'tensor':'tensor-audit.json','resource':'resource-audit.json','process':'process/process.json','result':'result.json'}
    name='trial-0/'+names[target]
    def change(r):
        if mutation=='missing_values': r['values']['exact_export_matches_stored_ema']=False
        elif mutation=='overclaim': r['values']['ema_recurrence_audited']=True
        elif mutation=='empty_optimizer': r['values']['tensor_counts']['optimizer']=0
        elif mutation=='wrong_row': r['metadata']['selection_row']['ce']=4.
        elif mutation=='wrong_hash': r['metadata']['checkpoint_hashes']['raw']='0'*64
        elif mutation=='loss_sums': r['metadata']['h200_loss_sums'][0]+=1
        elif mutation in ('rss','driver','disk','time'):
            key,val={'rss':('peak_rss_bytes',4*1024**3),'driver':('mps_driver_bytes',6*1024**3),'disk':('free_bytes',audit.RESERVE-1),'time':('elapsed_seconds',7200)}[mutation]
            r['samples'][-1][key]=val
        elif mutation=='missing_sample': r['samples'].pop()
        elif mutation=='no_mps':
            for s in r['samples']: s['mps_driver_bytes']=0
        elif mutation=='limits': r['limits']['rss_limit_bytes']+=1
        elif mutation=='plan': r['plan_sha256']='0'*64
        elif mutation=='failure': r['status']='failed'
        elif mutation=='timeout': r['elapsed_seconds']=7200
        elif mutation=='output': r['observation']['output_bytes']=r['output_limit_bytes']+1
        elif mutation=='holdout': r['holdout_consumed']=True
        elif mutation=='steps': r['training']['completed_steps']=1199
        elif mutation=='snapshots': r['training']['weight_averaging']['snapshots']=40
        else: r['h200']['mean_ce']-=.1
    ref=mutate(tmp_path,name,change);args['evidence'][name]=ref['sha256']
    with pytest.raises(ValueError): audit.bind_quality_boundary(**args)


@pytest.mark.parametrize('mutation',['omit_trial','omit_log','changed_trace','missing_weight','changed_selection','request','symlink'])
def test_incomplete_or_changed_evidence(tmp_path,mutation):
    args=fixture(tmp_path)
    if mutation=='omit_trial': args['trials'].pop(8)
    elif mutation=='omit_log': args['evidence'].pop('trial-0/process/worker.log')
    elif mutation=='changed_trace': (tmp_path/'numerical/trace.txt').write_bytes(b'changed')
    elif mutation=='missing_weight': (tmp_path/'trial-0/checkpoints/raw.pt').unlink()
    elif mutation=='changed_selection':
        args['selection_refs']['rates']=mutate(tmp_path,'lr-selection.json',lambda r:r.update(hybrid=.0025))
    elif mutation=='request': args['requests'][0]['config']['fixture_only']=False
    else:
        path=tmp_path/'trial-0/checkpoints/raw.pt';path.unlink();path.symlink_to('h200.pt')
    with pytest.raises(ValueError): audit.bind_quality_boundary(**args)


@pytest.mark.parametrize('name',['unreviewed.pt','extra-config.json','unknown.log'])
def test_unbound_artifact_in_stage_blocks_manifest(tmp_path,name):
    args=fixture(tmp_path)
    write(tmp_path,'trial-0/'+name,b'unlisted artifact')
    with pytest.raises(ValueError,match='Unbound stage artifact'):
        audit.bind_quality_boundary(**args)


def test_extra_noncheckpoint_evidence_is_preserved_when_bound(tmp_path):
    args=fixture(tmp_path)
    ref=write(tmp_path,'trial-0/extra-config.json',dict(extra='preserved'))
    args['evidence'][ref['path']]=ref['sha256']
    report=audit.bind_quality_boundary(**args)
    assert report['stage_inventory_complete'] is True
    assert report['manifest']['evidence'][ref['path']]==ref['sha256']


def test_unknown_symlink_directory_blocks_stage_inventory(tmp_path):
    args=fixture(tmp_path)
    (tmp_path/'trial-0/link').symlink_to(tmp_path/'trial-1',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        audit.bind_quality_boundary(**args)


def test_partial_resource_trace_is_not_a_completed_resource_review(tmp_path):
    args=fixture(tmp_path)
    name='trial-0/resource-events.jsonl'
    lines=(tmp_path/name).read_text().splitlines()
    ref=write(tmp_path,name,('\n'.join(lines[:-1])+'\n').encode())
    args['evidence'][name]=ref['sha256']
    report_name='trial-0/resource-audit.json'
    updated=mutate(tmp_path,report_name,lambda r:r['events'].update(sha256=ref['sha256']))
    args['evidence'][report_name]=updated['sha256']
    with pytest.raises(ValueError,match='trace incomplete'):
        audit.bind_quality_boundary(**args)


@pytest.mark.parametrize('name',audit.OUTPUT_FAILURES)
def test_bound_failed_training_or_evaluation_cannot_pass_quality(tmp_path,name):
    args=fixture(tmp_path)
    ref=write(tmp_path,'trial-0/'+name,{'status':'failed'})
    args['evidence'][ref['path']]=ref['sha256']
    with pytest.raises(ValueError,match='Failed or partial'):audit.bind_quality_boundary(**args)

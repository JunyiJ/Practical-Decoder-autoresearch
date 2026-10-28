"""Tiny synthetic pilot artifacts only; saved flags are not real audit evidence."""
import copy
import json
import math
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_pilot_audit_v2 as audit
from src.research import tokenizer_wide_hybrid_retention_audit as shared
from src.research.tokenizer_wide_hybrid_pilot_config import specs
from src.research.protocol import digest,file_digest


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
    return dict(path=name,sha256=file_digest(path))


def fixture(root):
    projection=json.loads((Path(__file__).resolve().parents[1]/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json').read_text())
    pref=write(root,'meta/projection.json',projection)
    corpus=write(root,'meta/corpus.txt',b'synthetic corpus, no actual training data')
    context={k:digest(k) for k in ('plan_sha256','source_sha256','runtime_sha256')}
    context['corpus_sha256']=corpus['sha256']
    trace=write(root,'meta/numerical.trace',b'synthetic numerical trace')
    numeric=dict(kind='numerical',status='passed',**{k:context[k] for k in ('source_sha256','runtime_sha256','corpus_sha256')},
        checks=dict.fromkeys(('all_declared_ranks',)+shared.NUMERICAL_CHECKS,True),
        cases=[dict(width=512,family=f,divisor=d,checks=dict.fromkeys(shared.NUMERICAL_CHECKS,True))
               for f in ('hybrid','gated_mla','kda') for d in (8,4,2)],artifacts={trace['path']:trace['sha256']})
    nref=write(root,'meta/numerical.json',numeric)
    tokenizer={'synthetic':True};token_sha=digest('96 synthetic training IDs');requests=[];evidence={}
    for i,spec in enumerate(specs()):
        stem=f'pilots/{i:02d}';trial=root/stem
        request=dict(spec=spec,config={'data':{'path':str(root/corpus['path'])},'fixture_only':True},result_path=str(trial/'result.json'))
        requests.append(request);rq=write(root,stem+'/request.json',request)
        pair={k:write(root,f'{stem}/checkpoints/{k}.pt',f'synthetic {i} {k}'.encode()) for k in ('raw','h200')}
        row=next(r for r in projection['records'] if r['width']==512 and r['family']==spec['family'] and r['bottleneck_divisor']==2)
        def score(role):
            counts=[min(2048,34278-j) for j in range(0,34278,2048)]
            return dict(context=128,eval_batch_size=8,block_tokens=2048,target_count=34278,
                target_bytes=111536,window_count=math.ceil(34278/128),parameters=row['parameters'],
                block_target_counts=counts,block_loss_sums=[5.*n for n in counts],mean_ce=5.,
                bits_per_byte=5.*34278/111536/math.log(2),perplexity=math.exp(5),
                checkpoint=str(root/pair[role]['path']),checkpoint_sha256=pair[role]['sha256'])
        result=dict(spec,status='completed',holdout_consumed=False,request_sha256=rq['sha256'],
            training=dict(completed_steps=20,weight_averaging=dict(snapshots=8),final_checkpoint=str(root/pair['raw']['path'])),
            raw=score('raw'),h200=score('h200'))
        rs=write(root,stem+'/result.json',result)
        files={str(root/rq['path']):rq['sha256'],str(root/rs['path']):rs['sha256'],
               **{str(root/ref['path']):ref['sha256'] for ref in pair.values()}}
        metadata=dict(h200_loss_sums=result['h200']['block_loss_sums'],target_counts=result['h200']['block_target_counts'],
            request_sha256=rq['sha256'],result_sha256=rs['sha256'],checkpoint_hashes={k:v['sha256'] for k,v in pair.items()},
            metadata_reviewed=True,tensor_values_audited=False,adopted=False,pilot_spec=spec,selection_eligible=False,
            status='pilot_metadata_reviewed_pending_tensor_cache_resource_audits')
        values={**dict.fromkeys(shared.VALUE_CHECKS,True),**dict.fromkeys(shared.NONCLAIMS,False),
            'tensor_counts':dict.fromkeys(('raw_model','stored_ema','optimizer','exported_model'),1),
            'parameters_compared':1,'buffers_compared':0}
        tensor=dict(kind='pilot',status='export_values_reviewed_pending_numerical_and_resource_audits',
            adopted=False,files=dict(files),metadata=metadata,values=values)
        files[str(root/corpus['path'])]=corpus['sha256'];files[str(root/pref['path'])]=pref['sha256']
        attempt=write(root,stem+'/audit-attempt.json',audit.attempt_contract(i,rq['sha256'],rs['sha256'],pref,tokenizer,token_sha))
        cache=dict(kind=audit.CACHE_KIND,attempt=attempt,status='pilot_cache_checks_complete_pending_resource_review',spec=spec,adopted=False,
            contract=audit.CACHE_CONTRACT,token_sha256=token_sha,tokenizer_sha256=digest(tokenizer),tensor_review=tensor,files=files,
            endpoints={k:dict(checkpoint=str(root/pair[k]['path']),sha256=pair[k]['sha256'],parameters=row['parameters'],
                parameter_storage_bytes=row['parameter_bytes'],checks=[dict(batch=b,passed=True,max_absolute_error=0.,
                    state_bytes=b*row['cache_bytes_batch1'],observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)
                    for b in (1,8)]) for k in pair})
        cache_ref=write(root,stem+'/audit.json',cache)
        resource=dict(cache_audit_sha256=cache_ref['sha256'],kind='wide_hybrid_pilot_resources_v2',status='completed',plan_sha256=context['plan_sha256'],
            request_sha256=rq['sha256'],result_sha256=rs['sha256'],limits=shared.PILOT_RESOURCE_LIMITS,violations=[],
            samples=[dict(elapsed_seconds=t,peak_rss_bytes=1000,mps_driver_bytes=2000,free_bytes=40*1024**3) for t in (0.,1.)])
        events=[dict(event='header',**{k:resource[k] for k in ('kind','plan_sha256','request_sha256','limits')})]
        events += [dict(event='sample',**s) for s in resource['samples']]
        events += [dict(event='terminal',status='completed',result_sha256=rs['sha256'],reason=None,cache_audit_sha256=cache_ref['sha256'])]
        er=write(root,stem+'/resource-events.jsonl',('\n'.join(json.dumps(e) for e in events)+'\n').encode())
        resource['events']=dict(path='resource-events.jsonl',sha256=er['sha256'])
        process=dict(status='completed',returncode=0,error=None,reserve_bytes=8*1024**3,timeout_seconds=600,
            output_limit_bytes=row['per_trajectory_allowance_bytes'],log_limit_bytes=4*1024**2,elapsed_seconds=2.,
            required_free_bytes=30_631_281_412,observation=dict(free_bytes=40*1024**3,output_bytes=5000,log_bytes=30))
        refs=[rq,rs,er,cache_ref,attempt,write(root,stem+'/resource-audit.json',resource),
              write(root,stem+'/process/process.json',process),write(root,stem+'/process/worker.log',b'synthetic process log')]
        evidence.update({ref['path']:ref['sha256'] for ref in refs})
    return dict(root=root,context=context,requests=requests,evidence=evidence,numerical_ref=nref,
                projection_ref=pref,tokenizer=tokenizer,training_token_sha256=token_sha)


def change(args,name,fn):
    value=json.loads((args['root']/name).read_text());fn(value)
    ref=write(args['root'],name,value);args['evidence'][name]=ref['sha256']


def test_all_three_pilots_bind_without_deleting_or_authorizing(tmp_path):
    args=fixture(tmp_path);before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report=audit.bind_pilot_boundary(**args)
    assert report['stage_audits_bound'] and report['measured_pilot_reviews_bound']
    assert len(report['manifest']['delete'])==6 and report['manifest']['retain']==[]
    assert all(not report[k] for k in ('frozen_plan_verified','current_storage_fit_verified',
        'inventory_completeness_verified','deletion_authorized','dispatch_authorized'))
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('which',range(3))
@pytest.mark.parametrize('bad',['cache','tensor','resource','trace'])
def test_every_family_full_audit_required(tmp_path,which,bad):
    args=fixture(tmp_path);stem=f'pilots/{which:02d}'
    if bad=='cache':change(args,stem+'/audit.json',lambda x:x['endpoints']['h200']['checks'].pop())
    elif bad=='tensor':change(args,stem+'/audit.json',lambda x:x['tensor_review']['values'].update(exact_export_matches_stored_ema=False))
    elif bad=='resource':change(args,stem+'/resource-audit.json',lambda x:x['samples'][-1].update(elapsed_seconds=600))
    else:args['evidence'].pop(stem+'/resource-events.jsonl')
    with pytest.raises(ValueError):audit.bind_pilot_boundary(**args)


@pytest.mark.parametrize('bad',['numeric_case','numeric_source','token_ids','tokenizer','order','count',
    'holdout','steps','ema','result_hash','metadata','value_counts','overclaim','geometry','state',
    'driver','cache_contract','cache_files','resource_cache_hash','raw_path','process','extra','symlink','weight_change','corpus_change'])
def test_incomplete_or_changed_pilot_evidence_blocks_boundary(tmp_path,bad):
    args=fixture(tmp_path);name='pilots/00/audit.json'
    if bad.startswith('numeric_'):
        ref=args['numerical_ref'];value=json.loads((tmp_path/ref['path']).read_text())
        if bad=='numeric_case':value['cases'].pop()
        else:value['source_sha256']='0'*64
        args['numerical_ref']=write(tmp_path,ref['path'],value)
    elif bad=='token_ids':args['training_token_sha256']='0'*64
    elif bad=='tokenizer':args['tokenizer']={'different':True}
    elif bad=='order':args['requests'].reverse()
    elif bad=='count':args['requests'].pop()
    elif bad=='holdout':change(args,'pilots/00/result.json',lambda x:x.update(holdout_consumed=True))
    elif bad=='steps':change(args,'pilots/00/result.json',lambda x:x['training'].update(completed_steps=19))
    elif bad=='ema':change(args,'pilots/00/result.json',lambda x:x['training']['weight_averaging'].update(snapshots=7))
    elif bad=='result_hash':change(args,name,lambda x:x['tensor_review']['metadata'].update(result_sha256='0'*64))
    elif bad=='metadata':change(args,name,lambda x:x['tensor_review']['metadata'].update(selection_eligible=True))
    elif bad=='value_counts':change(args,name,lambda x:x['tensor_review']['values']['tensor_counts'].update(optimizer=0))
    elif bad=='overclaim':change(args,name,lambda x:x['tensor_review']['values'].update(optimizer_resume_audited=True))
    elif bad=='geometry':change(args,name,lambda x:x['endpoints']['raw'].update(parameters=1))
    elif bad=='state':change(args,name,lambda x:x['endpoints']['raw']['checks'][0].update(state_bytes=1))
    elif bad=='driver':change(args,name,lambda x:x['endpoints']['raw']['checks'][0].update(observed_mps_driver_bytes=6*1024**3))
    elif bad=='cache_contract':change(args,name,lambda x:x['contract'].update(tokens=95))
    elif bad=='cache_files':change(args,name,lambda x:x.update(files={}))
    elif bad=='resource_cache_hash':change(args,'pilots/00/resource-audit.json',lambda x:x.update(cache_audit_sha256='0'*64))
    elif bad=='raw_path':change(args,'pilots/00/result.json',lambda x:x['training'].update(final_checkpoint='/wrong.pt'))
    elif bad=='process':change(args,'pilots/00/process/process.json',lambda x:x.update(returncode=90))
    elif bad=='extra':write(tmp_path,'pilots/00/unbound.json',{})
    elif bad=='symlink':(tmp_path/'pilots/00/alias').symlink_to(tmp_path/'pilots/00/result.json')
    elif bad=='weight_change':(tmp_path/'pilots/00/checkpoints/raw.pt').write_text('changed')
    elif bad=='corpus_change':(tmp_path/'meta/corpus.txt').write_text('changed')
    with pytest.raises(ValueError):audit.bind_pilot_boundary(**args)


@pytest.mark.parametrize('bad',['kind','attempt','missing_attempt','changed_attempt'])
def test_versioned_cache_attempt_required(tmp_path,bad):
    args=fixture(tmp_path);name='pilots/00/audit.json'
    if bad=='kind':change(args,name,lambda x:x.update(kind='legacy'))
    elif bad=='attempt':change(args,name,lambda x:x['attempt'].update(sha256='0'*64))
    elif bad=='missing_attempt':args['evidence'].pop('pilots/00/audit-attempt.json')
    else:change(args,'pilots/00/audit-attempt.json',lambda x:x.update(slot=1))
    with pytest.raises(ValueError):audit.bind_pilot_boundary(**args)


@pytest.mark.parametrize('name',['audit-failure.json','audit-failure.json.tmp','audit-attempt.json.tmp','audit.json.tmp'])
def test_bound_failure_or_partial_cache_evidence_still_blocks_stage(tmp_path,name):
    args=fixture(tmp_path)
    ref=write(tmp_path,'pilots/00/'+name,{'status':'failed'})
    args['evidence'][ref['path']]=ref['sha256']
    with pytest.raises(ValueError,match='Failed or partial'):audit.bind_pilot_boundary(**args)

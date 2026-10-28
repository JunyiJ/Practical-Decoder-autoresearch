"""Read-only v2 three-pilot review binding; no worker, cleanup, or dispatch API.

The future controller independently supplies frozen requests, context, tokenizer,
training IDs and complete evidence. This checks saved audit provenance; actual
numerical/tensor/cache execution and campaign-wide gates remain indispensable.
"""
import json
from pathlib import Path

from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_pilot_config import specs
from .tokenizer_wide_hybrid_pilot_cache import CACHE_CONTRACT
from .tokenizer_wide_hybrid_pilot_cache_v2 import KIND as CACHE_KIND, attempt_contract
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_audit import (
    _require, _finite, _load, numerical_review, pilot_resource_review,
    VALUE_CHECKS, NONCLAIMS, PILOT_RESOURCE_LIMITS, reject_output_failures,
)
from .tokenizer_wide_hybrid_retention_receipts import (
    _path, _sha, _check_file, _document, planned_manifest,
)
from .tokenizer_wide_hybrid_resources import EVENT_LIMIT, verify_pilot_events
from .tokenizer_wide_hybrid_scores import _score


SAMPLE_SETTINGS=dict(prompt='ROMEO:\n',seed=12345,temperature=.8,top_k=40,new_tokens=96)


def _worker_review(root,stem,index,context,request,result,pair,bound,projection_ref,token_sha,allowance,*,stage='pilots'):
    _require(stage in ('pilots','quality'),'Known producer stage required')
    label='pilot' if stage=='pilots' else 'quality'
    kind=f'wide_hybrid_{label}_execution_v2'
    names={k:stem+'/'+name for k,name in dict(worker='worker-outputs.json',attempt='worker-attempt.json',
        training='training-outputs.json',training_attempt='training-attempt.json',summary='training-summary.json',
        evaluation='evaluation-attempt.json').items()}
    docs={k:_load(root,name,bound) for k,name in names.items()}
    def ref(name):return dict(path=name,sha256=bound[name])
    rqsha=bound[stem+'/request.json'];worker=docs['worker']
    _require(worker.get('kind')==kind
             and worker.get('status')==f'{label}_outputs_complete_pending_independent_stage_review'
             and type(worker.get('slot')) is int and worker['slot']==index
             and worker.get('plan_sha256')==context['plan_sha256'] and worker.get('request_sha256')==rqsha
             and worker.get('adopted') is False and worker.get('dispatch_authorized') is False
             and worker.get('stage_audits_bound') is False,'Completed worker identity/claims differ')
    artifacts=dict(training_outputs=ref(names['training']),evaluation_attempt=ref(names['evaluation']),
                   result=ref(stem+'/result.json'),resources=ref(stem+'/resource-audit.json'))
    if stage=='pilots':artifacts['cache']=ref(stem+'/audit.json')
    _require(digest(worker.get('artifacts'))==digest(artifacts)
             and digest(worker.get('attempt'))==digest(ref(names['attempt'])),'Worker producer references differ')
    expected_attempt=dict(kind=kind,status='started',slot=index,
        plan_sha256=context['plan_sha256'],request_sha256=rqsha,projection_ref=projection_ref,
        checkpoint_allowance_bytes=allowance,dispatch_authorized=False)
    _require(digest(docs['attempt'])==digest(expected_attempt),'Worker attempt binding differs')
    cfgsha=digest(request['config']);training=docs['training']
    _require(digest(docs['training_attempt'])==digest(dict(kind='wide_hybrid_training_outputs_v2',
        status='started',config_sha256=cfgsha,checkpoint_allowance_bytes=allowance)),
        'Training attempt binding differs')
    _require(training.get('kind')=='wide_hybrid_training_outputs_v2' and training.get('status')=='completed'
             and training.get('config_sha256')==cfgsha and training.get('numerical_audits_passed') is False
             and training.get('dispatch_authorized') is False,'Training output identity/claims differ')
    for role,entry in pair.items():
        _require(digest(training.get(role))==digest(dict(entry,path=str(root/entry['path']))),
                 'Training checkpoint receipt differs')
    _require(digest(docs['summary'])==digest(result['training']),'Training summary/result differ')
    metric_name=stem+'/metrics.jsonl'
    _require(metric_name in bound,'Training metrics binding missing')
    metric_path=_check_file(root,metric_name,bound[metric_name])
    _require(metric_path.stat().st_size<=256*1024,'Training metrics exceed group cap')
    metrics=[json.loads(line) for line in metric_path.read_text().splitlines()]
    _require(digest(metrics)==digest(result['training'].get('metrics')),'Training metric trace differs')
    evaluation=docs['evaluation'];_sha(evaluation.get('training_contract_sha256'))
    expected_files={str(root/(stem+'/request.json')):rqsha,
        str(root/names['training']):bound[names['training']],str(root/names['summary']):bound[names['summary']],
        str(Path(request['config']['data']['path']).resolve()):context['corpus_sha256'],
        **{str(root/entry['path']):entry['sha256'] for entry in pair.values()}}
    _require(evaluation.get('kind')=='wide_hybrid_evaluation_v2' and evaluation.get('status')=='started'
             and evaluation.get('request_sha256')==rqsha
             and evaluation.get('training_outputs_sha256')==bound[names['training']]
             and evaluation.get('training_token_sha256')==token_sha
             and digest(evaluation.get('files'))==digest(expected_files)
             and digest(evaluation.get('sample_settings'))==digest(SAMPLE_SETTINGS),
             'Evaluation attempt provenance differs')
    _require(digest(result['h200'].get('sample_settings'))==digest(SAMPLE_SETTINGS)
             and isinstance(result['h200'].get('sample'),str),'Descriptive sample settings differ')


def bind_pilot_boundary(root, **kwargs):
    """Require all three pilots before constructing a candidate cleanup manifest."""
    return _bind_pilots(root,complete=True,**kwargs)


def bind_pilot_prefix(root, **kwargs):
    """Review an ordered nonempty completed prefix; never compile cleanup credit."""
    return _bind_pilots(root,complete=False,**kwargs)


def _bind_pilots(root, *, context, requests, evidence, numerical_ref,
                 projection_ref, tokenizer, training_token_sha256, complete):
    """Bind all three pilot pairs before compiling a candidate cleanup manifest.

    root is the immutable evidence directory. No storage or deletion credit is
    granted here. This cannot substitute for a fresh bind_campaign result, full
    inventory/locks/producer checks or a completed boundary-cleanup receipt.
    """
    root=Path(root).resolve()
    _require(set(context)=={'plan_sha256','source_sha256','runtime_sha256','corpus_sha256'},
             'Independent frozen context required')
    for sha in context.values():_sha(sha)
    _sha(training_token_sha256)
    _require(isinstance(tokenizer,dict) and bool(tokenizer),'Frozen tokenizer required')
    _require(isinstance(requests,list) and (len(requests)==3 if complete else 1<=len(requests)<=3),
             'Exactly three pilots or a nonempty ordered prefix required')
    pilot_root=_path(root,'pilots')
    _require(pilot_root.is_dir() and all(not p.is_symlink() and p.is_dir()
             and p.name in {f'{i:02d}' for i in range(len(requests))} for p in pilot_root.iterdir()),
             'Unreviewed pilot slot outside completed prefix')
    bound=dict(evidence)
    def add(name,sha):
        _require(name not in bound or bound[name]==sha,'Conflicting evidence hashes')
        bound[name]=sha
    for name,sha in numerical_review(root,numerical_ref,context).items():add(name,sha)
    projection=_document(root,projection_ref);budgets=allocations(projection)
    add(projection_ref['path'],projection_ref['sha256'])
    records=[];pairs={};inodes=set();trial_dirs=[];corpora=set()
    for index,(expected,spec) in enumerate(zip(requests,specs())):
        stem=f'pilots/{index:02d}';trial=_path(root,stem);trial_dirs.append(trial)
        reject_output_failures(trial)
        _require(not any((trial/name).exists() or (trial/name).is_symlink() for name in
                 ('audit-failure.json','audit-failure.json.tmp','audit-attempt.json.tmp','audit.json.tmp')),
                 'Failed or partial pilot cache audit cannot pass stage review')
        names={k:stem+'/'+v for k,v in dict(request='request.json',result='result.json',
            cache='audit.json',attempt='audit-attempt.json',resource='resource-audit.json',
            process='process/process.json',log='process/worker.log').items()}
        docs={k:_load(root,name,bound) for k,name in names.items() if k!='log'}
        request,result,cache=docs['request'],docs['result'],docs['cache']
        rqsha,rssha=bound[names['request']],bound[names['result']]
        _require(digest(request)==digest(expected) and digest(request.get('spec'))==digest(spec)
                 and request.get('result_path')==str(trial/'result.json'), 'Pilot request/config/order differs')
        _require(all(digest(result.get(k))==digest(v) for k,v in spec.items())
                 and result.get('status')=='completed' and result.get('holdout_consumed') is False
                 and result.get('request_sha256')==rqsha,'Pilot result identity/status invalid')
        training=result.get('training',{})
        _require(type(training.get('completed_steps')) is int and training['completed_steps']==20
                 and type(training.get('weight_averaging',{}).get('snapshots')) is int
                 and training['weight_averaging']['snapshots']==8,'Pilot training/EMA incomplete')
        row=next(r for r in projection['records'] if r['width']==512 and
                 r['family']==spec['family'] and r['bottleneck_divisor']==2)
        pair={};files={str(root/names['request']):rqsha,str(root/names['result']):rssha}
        for role in ('raw','h200'):
            score=result[role];_score(score,row['parameters'])
            path=Path(score['checkpoint'])
            _require(path.is_absolute() and path.parent==trial/'checkpoints' and path.suffix=='.pt',
                     'Pilot checkpoint outside declared slot')
            name=str(path.relative_to(root));checked=_check_file(root,name,score['checkpoint_sha256'])
            stat=checked.stat();inode=(stat.st_dev,stat.st_ino)
            _require(inode not in inodes and stat.st_nlink==1 and 0<stat.st_size<=2*1024**3,
                     'Pilot checkpoint alias or invalid size')
            inodes.add(inode)
            pair[role]=dict(path=name,size=stat.st_size,sha256=score['checkpoint_sha256'])
            files[str(path)]=score['checkpoint_sha256']
        _require(training.get('final_checkpoint')==str(root/pair['raw']['path']),'Pilot raw endpoint differs')
        metadata=dict(h200_loss_sums=result['h200']['block_loss_sums'],
            target_counts=result['h200']['block_target_counts'],request_sha256=rqsha,result_sha256=rssha,
            checkpoint_hashes={k:pair[k]['sha256'] for k in pair},metadata_reviewed=True,
            tensor_values_audited=False,adopted=False,pilot_spec=spec,selection_eligible=False,
            status='pilot_metadata_reviewed_pending_tensor_cache_resource_audits')
        tensor=cache.get('tensor_review',{})
        _require(tensor.get('kind')=='pilot' and tensor.get('adopted') is False
                 and tensor.get('status')=='export_values_reviewed_pending_numerical_and_resource_audits'
                 and digest(tensor.get('metadata'))==digest(metadata)
                 and digest(tensor.get('files'))==digest(files),'Pilot tensor/metadata binding incomplete')
        values=tensor.get('values',{});counts=values.get('tensor_counts',{})
        _require(all(values.get(k) is True for k in VALUE_CHECKS)
                 and all(values.get(k) is False for k in NONCLAIMS)
                 and set(counts)=={'raw_model','stored_ema','optimizer','exported_model'}
                 and all(type(n) is int and n>0 for n in counts.values())
                 and type(values.get('parameters_compared')) is int and values['parameters_compared']>0
                 and type(values.get('buffers_compared')) is int and values['buffers_compared']>=0,
                 'Pilot tensor value audit incomplete or overstated')
        _require(cache.get('kind')==CACHE_KIND,'Versioned pilot cache producer required')
        attempt_ref=dict(path=names['attempt'],sha256=bound[names['attempt']])
        _require(digest(cache.get('attempt'))==digest(attempt_ref)
                 and digest(docs['attempt'])==digest(attempt_contract(index,rqsha,rssha,
                    projection_ref,tokenizer,training_token_sha256)), 'Pilot cache attempt binding differs')
        _require(cache.get('status')=='pilot_cache_checks_complete_pending_resource_review'
                 and digest(cache.get('spec'))==digest(spec) and cache.get('adopted') is False
                 and digest(cache.get('contract'))==digest(CACHE_CONTRACT)
                 and cache.get('token_sha256')==training_token_sha256
                 and cache.get('tokenizer_sha256')==digest(tokenizer),'Pilot cache input/contract differs')
        endpoints=cache.get('endpoints',{})
        _require(set(endpoints)=={'raw','h200'},'Both pilot cache endpoints required')
        for role,endpoint in endpoints.items():
            _require(endpoint.get('checkpoint')==str(root/pair[role]['path'])
                     and endpoint.get('sha256')==pair[role]['sha256']
                     and type(endpoint.get('parameters')) is int and endpoint['parameters']==row['parameters']
                     and type(endpoint.get('parameter_storage_bytes')) is int
                     and endpoint['parameter_storage_bytes']==row['parameter_bytes'], 'Cache endpoint geometry differs')
            checks=endpoint.get('checks',[])
            _require(isinstance(checks,list) and [c.get('batch') for c in checks]==[1,8],
                     'Both pilot cache batch sizes required')
            for check in checks:
                _require(check.get('passed') is True and _finite(check.get('max_absolute_error'))
                         and type(check.get('batch')) is int and type(check.get('state_bytes')) is int
                         and check['state_bytes']==check['batch']*row['cache_bytes_batch1'],
                         'Pilot cache check/state size invalid')
                for key in ('observed_mps_current_bytes','observed_mps_driver_bytes'):
                    _require(type(check.get(key)) is int and 0<check[key]<PILOT_RESOURCE_LIMITS['driver_limit_bytes'],
                             'Pilot allocator observation invalid')
        corpus=Path(request['config']['data']['path'])
        _require(corpus.is_absolute() and corpus==corpus.resolve(),'Canonical frozen corpus required')
        _require(file_digest(corpus)==context['corpus_sha256'],'Pilot corpus changed')
        corpora.add(corpus)
        files[str(corpus)]=context['corpus_sha256'];files[str(root/projection_ref['path'])]=projection_ref['sha256']
        # The v2 producer binds this immutable campaign projection copy.
        _require(digest(cache.get('files'))==digest(files),'Pilot cache evidence bindings differ')
        allowance=budgets[spec['family'],2]
        pilot_resource_review(docs['resource'],docs['process'],context,rqsha,rssha,allowance)
        _require(docs['resource'].get('cache_audit_sha256')==bound[names['cache']],
                 'Resource completion must bind the completed tensor/cache audit')
        ref=docs['resource'].get('events',{})
        _require(set(ref)=={'path','sha256'} and ref['path']=='resource-events.jsonl', 'Pilot trace binding missing')
        event_name=stem+'/'+ref['path']
        _require(bound.get(event_name)==ref['sha256'],'Pilot trace absent from evidence inventory')
        event_path=_check_file(root,event_name,ref['sha256'])
        _require(event_path.stat().st_size<=EVENT_LIMIT,'Pilot trace exceeds cap')
        verify_pilot_events(docs['resource'],[json.loads(line) for line in event_path.read_text().splitlines()])
        _require(names['log'] in bound,'Pilot process log binding missing')
        _require(_check_file(root,names['log'],bound[names['log']]).stat().st_size<=PILOT_RESOURCE_LIMITS['log_limit_bytes'],
                 'Pilot process log cap exceeded')
        _worker_review(root,stem,index,context,request,result,pair,bound,projection_ref,training_token_sha256,allowance)
        records.append(dict(spec,status='completed',ce=result['h200']['mean_ce']));pairs[index]=pair
    manifest=planned_manifest('pilot',records,None,None,pairs,bound) if complete else None
    weights={e['path']:e for pair in pairs.values() for e in pair.values()};found=set()
    for trial in trial_dirs:
        for path in trial.rglob('*'):
            _require(not path.is_symlink(),'Symlink in pilot evidence')
            if path.is_dir():continue
            _require(path.is_file(),'Nonregular pilot artifact')
            name=str(path.relative_to(root));found.add(name)
            _require(name in bound or name in weights,'Unbound pilot artifact')
    _require(set(weights)<=found,'Pilot checkpoint missing from inventory')
    for name,sha in bound.items():_check_file(root,name,sha)
    for name,entry in weights.items():_check_file(root,name,entry['sha256'],entry['size'])
    for corpus in corpora:
        _require(file_digest(corpus)==context['corpus_sha256'],'Pilot corpus changed during review')
    return dict(manifest=manifest,records=records,stage_audits_bound=complete,stage_inventory_complete=complete,
        prefix_audits_bound=True,completed_pilot_count=len(records),evidence=bound,
        checkpoint_pairs={f'pilot/{i:02d}':pair for i,pair in pairs.items()},
        numerical_review_bound=True,measured_pilot_reviews_bound=True,
        frozen_plan_verified=False,current_storage_fit_verified=False,
        inventory_completeness_verified=False,deletion_authorized=False,dispatch_authorized=False)

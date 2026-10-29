"""Bounded v2 inference body and independent artifact review.

The live entry module supplies source/runtime, retained-history and inherited
process checks at every boundary. The serial controller independently accepts
completed transport and replays full quality history between seeds. This body
itself neither dispatches nor advances the panel.
"""
import os
from pathlib import Path
import shutil

import torch

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_inference_gate import CONTRACT,recheck
from .tokenizer_wide_hybrid_inference import _models
from .tokenizer_wide_hybrid_loading import load_selected_exports
from .tokenizer_wide_hybrid_measurements import mps_callbacks
from .tokenizer_wide_hybrid_timing import measure_triple,token_digest,SEEDS,FAMILIES
from .tokenizer_wide_hybrid_rules import confirmation_specs
from .tokenizer_wide_hybrid_timing_review import review_result
from .memory_guard import start_memory_guard
from .tokenizer_wide_hybrid_inference_preparation_v2 import REQUEST

KIND='wide_hybrid_inference_outputs_v2'
ENTRY_KIND='wide_hybrid_inference_entry_v2'
READY_KIND='wide_hybrid_inference_ready_v2'
LIMIT=CONTRACT['output_limit_bytes']
TERMINAL_RESERVE=16384


class _WorkerGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,seed):
        super().__init__(writer,seed,'inference-worker',CONTRACT['job_seconds'])
        self.name=f'control/inference-worker-{seed}.json'


def _budget(writer):
    _locks(writer.root/'evidence',writer.lock_fds)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents inference outputs')
    with writer._serialized():seen=snapshot(writer.root)
    # Preserve the original whole-panel cap, not three independent 4-MiB caps.
    size=sum(v['size'] for n,v in seen['files'].items() if n.startswith('evidence/inference/'))
    _require(size+TERMINAL_RESERVE<=LIMIT,'Inference panel output/terminal reserve exceeded')
    _require(shutil.disk_usage(writer.root).free>=CONTRACT['reserve_bytes'],'Inference disk reserve crossed')
    return size


def _binding(writer,seed,value):
    _budget(writer)
    _require(type(value) is dict and set(value)=={'kind','seed','parent_pid','request_ref','ready_ref','binding'}
        and value['kind']==ENTRY_KIND and type(value['seed']) is int and value['seed']==seed
        and type(value['parent_pid']) is int and value['parent_pid']>1,
        'Exact independently verified current-parent entry required')
    root=writer.root/'evidence';bound=value['binding']
    _require(set(bound)=={'bindings','selection','tokenizer','token_sha256','files'}
        and set(bound['bindings'])==set(FAMILIES) and bool(bound['files']), 'Complete immutable triple binding required')
    _sha(bound['token_sha256'])
    _require(value['request_ref']['path']==REQUEST
        and value['ready_ref']['path']==f'inference/{seed}/ready.json','Exact shared request and seed readiness required')
    request=_document(root,value['request_ref']);ready=_document(root,value['ready_ref'])
    contract=dict(CONTRACT,version=2,tokenizer_sha256=digest(bound['tokenizer']),token_sha256=bound['token_sha256'])
    _require(request['seed_order']==list(SEEDS) and digest(request['contract'])==digest(contract)
        and digest(request['selection'])==digest(bound['selection'])
        and request['tokenizer_sha256']==digest(bound['tokenizer']) and request['token_sha256']==bound['token_sha256'],
        'Immutable panel workload or selection differs')
    expected_ready=dict(kind=READY_KIND,seed=seed,parent_pid=value['parent_pid'],request_ref=value['request_ref'],
        bindings_sha256=digest(bound['bindings']),protected_files_sha256=digest(bound['files']),contract=contract)
    _require(digest(ready)==digest(expected_ready),'Exact independently sealed seed readiness required')
    _require(bound['files'].get(str(root/value['request_ref']['path']))==value['request_ref']['sha256'],
             'Shared request must be part of protected entry files')
    slots={spec['family']:(slot,spec) for slot,spec in enumerate(confirmation_specs(bound['selection']),15) if spec['seed']==seed}
    for family,binding in bound['bindings'].items():
        _require(set(binding)=={'trial','request','request_sha256','result_sha256','training_contract'},
                 'Exact selected-export binding required')
        slot,spec=slots[family];path=Path(binding['trial'])
        _require(path.is_absolute() and path==path.resolve() and path==root/f'quality/{slot:02d}'
            and digest(binding['request']['spec'])==digest(spec), 'Selected export must be the exact fresh seed/family slot')
        for name,key in [('request.json','request_sha256'),('result.json','result_sha256')]:
            _require(bound['files'].get(str(path/name))==binding[key],'Missing protected selected request/result')
        result=_document(root,dict(path=str((path/'result.json').relative_to(root)),sha256=binding['result_sha256']))
        for role in ('raw','h200'):
            _require(bound['files'].get(result[role]['checkpoint'])==result[role]['checkpoint_sha256'],
                     'Missing protected fresh checkpoint pair')
    recheck(bound['files'])
    _budget(writer)
    return value


def _entry(writer,seed,check,phase,expected=None):
    _budget(writer)
    value=check(phase)
    _require(type(value) is dict and value.get('parent_pid')==os.getppid()>1,'Current parent identity changed')
    _binding(writer,seed,value)
    if expected is not None:_require(digest(value)==digest(expected),'Entry provenance changed during inference')
    return value


def execute_bound_triple(writer,seed,*,entry_check):
    """Execute one triple after a caller-owned live entry verification; no advance.

    entry_check(phase) must independently verify the active v2 process/controls
    and return the same immutable entry binding for entry/loaded/measured/saved.
    It must not freeze mutable supervisor state into protected_files. Parent
    acceptance must re-audit the whole retained quality history after child exit.
    """
    _require(isinstance(writer,MetadataWriter) and type(seed) is int and seed in SEEDS
        and callable(entry_check),'Exact seed and independent live entry verifier required')
    root=writer.root/'evidence';stem=f'inference/{seed}';guard=f'control/inference-worker-{seed}.json'
    names=[stem+'/'+n for n in ('worker-attempt.json','result.json','models.json','worker-outputs.json','worker-failure.json')]
    seen=snapshot(writer.root)['files']
    _require(not any(n in seen or n+'.tmp' in seen for n in ['evidence/'+n for n in names]+[guard]),
             'Existing inference worker attempt cannot be retried')
    try:
        with _WorkerGuard(writer,seed):
            entry=_entry(writer,seed,entry_check,'entry');bound=entry['binding']
            attempt=dict(kind=KIND,status='started',seed=seed,pid=os.getpid(),parent_pid=os.getppid(),
                request_ref=entry['request_ref'],ready_ref=entry['ready_ref'],entry_sha256=digest(entry),
                dispatch_authorized=False,adopted=False)
            path=writer.publish_json('evidence/'+stem+'/worker-attempt.json',attempt)
            attempt_ref=dict(path=stem+'/worker-attempt.json',sha256=file_digest(path))
            start_memory_guard();torch.set_num_threads(4)
            loaded=load_selected_exports(bound['bindings'],seed,bound['selection'],bound['tokenizer'],bound['token_sha256'])
            _models(loaded['metadata'],bound,seed)
            _require(token_digest(loaded['token_ids'])==bound['token_sha256'],'Loaded timing inputs changed')
            _entry(writer,seed,entry_check,'loaded',entry)
            check,measure=mps_callbacks(loaded['models'],loaded['configs'],loaded['token_ids'],bound['token_sha256'])
            result=measure_triple(seed,bound['selection'],loaded['token_ids'],bound['token_sha256'],check=check,measure=measure)
            review_result(result,seed,bound['selection'],bound['token_sha256'])
            _require(all(row['observed_mps_driver_bytes']<CONTRACT['driver_limit_bytes']
                for field in ('records','warmup_records') for row in result[field]), 'Observed MPS driver limit crossed')
            _entry(writer,seed,entry_check,'measured',entry)
            artifacts={}
            for name,value in [('result',result),('models',loaded['metadata'])]:
                path=writer.publish_json(f'evidence/{stem}/{name}.json',value)
                artifacts[name]=dict(path=f'{stem}/{name}.json',sha256=file_digest(path))
                _budget(writer)
            _entry(writer,seed,entry_check,'saved',entry)
            for ref in artifacts.values():_document(root,ref)
            value=dict(kind=KIND,status='inference_outputs_complete_pending_parent_review',seed=seed,
                pid=os.getpid(),parent_pid=os.getppid(),attempt_ref=attempt_ref,
                request_ref=entry['request_ref'],ready_ref=entry['ready_ref'],entry_sha256=digest(entry),
                artifacts=artifacts,adopted=False,dispatch_authorized=False,quality_history_reaudited=False)
            path=writer.publish_json('evidence/'+stem+'/worker-outputs.json',value)
            ref=dict(path=stem+'/worker-outputs.json',sha256=file_digest(path));_budget(writer)
        # Guard finalization may fail; never return a successful worker reference
        # until its finished record and final protected inputs have been checked.
        guard_ref=dict(path=guard,sha256=file_digest(writer.root/guard))
        from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
        _guard(writer,guard_ref,KIND,os.getpid(),600)
        recheck(bound['files']);_document(root,entry['ready_ref']);_budget(writer)
        return ref
    except BaseException as exc:
        try:writer.publish_json('evidence/'+stem+'/worker-failure.json',dict(kind=KIND,status='failed',seed=seed,
            error=repr(exc)[:1000],retry_allowed=False,adopted=False))
        except BaseException as publication:exc.add_note('Failure evidence publication failed: '+repr(publication)[:1000])
        raise



def review_bound_triple(writer,seed,entry,worker_ref):
    """Independent artifact review; transport and full quality replay still required.

    entry must come from independently verified frozen readiness. Historical
    parent/worker identities are returned for the caller's transport comparison;
    this read-only function neither resumes a process nor grants next-seed credit.
    """
    _require(isinstance(writer,MetadataWriter) and type(seed) is int and seed in SEEDS,
             'Exact finite seed and locked writer required')
    root=writer.root/'evidence';stem=f'inference/{seed}'
    _require(worker_ref.get('path')==stem+'/worker-outputs.json','Exact worker receipt required')
    census=snapshot(writer.root)
    _require(not any((n.startswith('evidence/inference/') or n.startswith('control/inference-worker-'))
        and (n.endswith('.tmp') or 'failure' in n) for n in census['files']), 'Failed or partial inference panel')
    _binding(writer,seed,entry);bound=entry['binding'];saved=_document(root,worker_ref)
    pid=saved.get('pid');parent=entry['parent_pid']
    _require(type(pid) is int and pid>1 and pid!=parent,'Distinct historical worker identity required')
    attempt_ref=saved['attempt_ref']
    _require(attempt_ref['path']==stem+'/worker-attempt.json','Exact worker attempt required')
    attempt=dict(kind=KIND,status='started',seed=seed,pid=pid,parent_pid=parent,request_ref=entry['request_ref'],
        ready_ref=entry['ready_ref'],entry_sha256=digest(entry),dispatch_authorized=False,adopted=False)
    _require(digest(_document(root,attempt_ref))==digest(attempt),'Worker attempt provenance differs')
    artifacts=saved['artifacts']
    _require(set(artifacts)=={'result','models'} and all(artifacts[n]['path']==f'{stem}/{n}.json' for n in artifacts),
             'Exact worker result/model artifacts required')
    expected=dict(kind=KIND,status='inference_outputs_complete_pending_parent_review',seed=seed,pid=pid,parent_pid=parent,
        attempt_ref=attempt_ref,request_ref=entry['request_ref'],ready_ref=entry['ready_ref'],entry_sha256=digest(entry),
        artifacts=artifacts,adopted=False,dispatch_authorized=False,quality_history_reaudited=False)
    _require(digest(saved)==digest(expected),'Worker receipt provenance or nonclaims differ')
    guard_ref=dict(path=f'control/inference-worker-{seed}.json')
    guard_ref['sha256']=file_digest(writer.root/guard_ref['path'])
    from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
    guard=_guard(writer,guard_ref,KIND,pid,600)
    _require(set(guard)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
        and digest(guard['limits'])==digest(dict(timeout_seconds=600,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)),
        'Exact worker resource guard required')
    models=_document(root,artifacts['models']);_models(models,bound,seed)
    result=_document(root,artifacts['result'])
    review=review_result(result,seed,bound['selection'],bound['token_sha256'])
    _require(all(row['observed_mps_driver_bytes']<CONTRACT['driver_limit_bytes']
        for field in ('records','warmup_records') for row in result[field]), 'Saved MPS driver limit crossed')
    files=dict(bound['files'])
    for prefix,refs in ((root,[entry['ready_ref'],worker_ref,attempt_ref,*artifacts.values()]),(writer.root,[guard_ref])):
        for ref in refs:
            name=str(_path(prefix,ref['path']))
            _require(name not in files or files[name]==ref['sha256'],'Conflicting worker evidence binding')
            files[name]=ref['sha256']
    recheck(files);_budget(writer)
    _require(snapshot(writer.root)==census,'Inference census changed during completed review')
    return dict(kind='wide_hybrid_inference_output_review_v2',status='artifacts_reviewed_pending_transport_and_quality_history',
        seed=seed,worker_pid=pid,parent_pid=parent,worker_ref=worker_ref,guard_ref=guard_ref,review=review,files=files,
        models_sha256=artifacts['models']['sha256'],adopted=False,dispatch_authorized=False,
        transport_verified=False,quality_history_reaudited=False,next_seed_authorized=False)

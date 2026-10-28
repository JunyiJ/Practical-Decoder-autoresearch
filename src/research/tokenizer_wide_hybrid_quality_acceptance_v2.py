"""Read-only parent acceptance after one completed LR child; no dispatcher.

Requires owned worker locks, an outer resource guard and a reaped child. Saved
process success is insufficient: every completed LR producer/tensor/resource
record, pilot cleanup history and remaining campaign storage is audited again.
"""
import math
import os
import shutil
import time

from .protocol import digest,file_digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_retention_receipts import _document,_path
from .tokenizer_wide_hybrid_cleanup import _locks,_check,TIMEOUT
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_quality_startup_v2 import _history
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_lr_prefix
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_metadata import snapshot
from . import tokenizer_wide_hybrid_quality_readiness_v2 as readiness
from . import tokenizer_wide_hybrid_quality_entry_v2 as entry

KIND='wide_hybrid_quality_acceptance_v2'


def _guard(writer,ref,kind,pid,seconds):
    value=_document(writer.root,ref);sample=value.get('observation',{});elapsed=value.get('elapsed_seconds')
    limits=dict(timeout_seconds=seconds,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(value.get('kind')==kind and value.get('status')=='finished' and value.get('reason') is None
        and type(value.get('pid')) is int and value['pid']==pid and value.get('limits')==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<seconds
        and set(sample)=={'peak_rss_bytes','free_bytes'} and all(type(v) is int and v>0 for v in sample.values())
        and violation(sample,elapsed,**limits) is None,'Completed bounded child/readiness guard required')
    return value


def review_completed_lr(writer,slot,startup_ref,readiness_ref,process):
    """Parent call only, under its outer guard; never grants stage continuation."""
    argv=entry.worker_argv(writer,slot,startup_ref,readiness_ref)
    root=writer.root/'evidence';stem=f'quality/{slot:02d}';_locks(root,writer.lock_fds)
    seal=_document(root,readiness_ref);startup=_document(root,startup_ref)
    _require(seal.get('kind')==readiness.KIND and seal.get('status')=='idle_review_sealed_pending_child_integration'
        and type(seal.get('slot')) is int and seal['slot']==slot
        and type(seal.get('parent_pid')) is int and seal['parent_pid']==os.getpid()
        and seal.get('startup_ref')==startup_ref and seal.get('dispatch_authorized') is False
        and seal.get('deletion_authorized') is False,'Exact same-parent readiness required')
    for key in ('request','plan_sha256','projection_ref','context'):
        _require(digest(seal[key])==digest(startup[key]),'Readiness/startup identity differs')
    bound,request,allowance=readiness._bound(writer,slot,startup)
    _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
             and type(seal.get('output_limit_bytes')) is int and seal['output_limit_bytes']==allowance,
             'Frozen postprocess provenance differs')
    deadline=time.monotonic()+TIMEOUT
    for name,binding in seal['inventory'].items():_check(root,name,binding,deadline)
    for name,binding in seal['control_inventory'].items():
        _require(name.startswith('control/') and name.endswith('.json'),'Sealed control path differs')
        _check(writer.root,name,binding,deadline)
    _require(seal['guard']['path']==f'control/quality-readiness-{slot:02d}.json'
        and seal['control_inventory'][seal['guard']['path']]['sha256']==seal['guard']['sha256'],
        'Readiness guard binding differs')
    _guard(writer,seal['guard'],readiness.KIND,os.getpid(),1800)
    refs={name:dict(path=stem+'/'+tail,sha256=file_digest(root/stem/tail)) for name,tail in
          [('launch','process/launch.json'),('process','process/process.json'),('worker','worker-outputs.json')]}
    terminal=_document(root,refs['process']);launch=_document(root,refs['launch'])
    _require(digest(terminal)==digest(process),'Returned process differs from immutable terminal')
    expected=dict(status='preflight',argv=argv,parent_pid=os.getpid(),pid=None,returncode=None,error=None,
        required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,output_limit_bytes=allowance,
        log_limit_bytes=4*1024**2,timeout_seconds=7200,observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    _require(digest(launch)==digest(expected),'Exact original sealed launch required')
    for key in ('argv','parent_pid','required_free_bytes','reserve_bytes','output_limit_bytes','log_limit_bytes',
                'timeout_seconds','capture_kind','log_truncated'):
        _require(type(terminal.get(key)) is type(expected[key]) and terminal[key]==expected[key],
                 'Terminal process provenance differs')
    _require(terminal.get('status')=='completed' and type(terminal.get('returncode')) is int
        and terminal['returncode']==0 and terminal.get('error') is None and type(terminal.get('pid')) is int
        and terminal['pid']>1 and terminal['pid']!=os.getpid(),'Successful distinct child process required')
    elapsed=terminal.get('elapsed_seconds');obs=terminal.get('observation',{})
    census=snapshot(writer.root)
    total=sum(v['size'] for n,v in census['files'].items() if n.startswith('evidence/'+stem+'/'))
    log=(root/stem/'process/worker.log').stat().st_size
    _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<7200
        and set(obs)=={'free_bytes','output_bytes','log_bytes'} and all(type(v) is int for v in obs.values())
        and obs['free_bytes']>=8*1024**3 and obs['output_bytes']==total<=allowance
        and obs['log_bytes']==log<=4*1024**2,'Terminal resource census differs')
    guard_name=f'control/quality-entry-{slot:02d}.json'
    guard_ref=dict(path=guard_name,sha256=file_digest(writer.root/guard_name))
    guard=_guard(writer,guard_ref,entry.KIND,terminal['pid'],7200)
    inventory=_inventory(root)
    plan=_document(root,dict(path='meta/plan.json',sha256=seal['plan_sha256']))
    requests=[bound['requests']['quality'][str(i)][0] for i in range(slot+1)]
    _require(all(len(bound['requests']['quality'][str(i)])==1 for i in range(slot+1)),
             'Independent exact LR requests required')
    history=_history(writer,seal['plan_sha256'],startup['pilot_completion_ref'],inventory)
    prefix=bind_lr_prefix(root,context=bound['context'],requests=requests,
        evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
        numerical_ref=plan['numerical_ref'],projection_ref=seal['projection_ref'],
        training_token_sha256=plan['training_token_sha256'])
    projection=_document(root,seal['projection_ref'])
    storage=bind_storage(writer.root,projection,pilots=history['records'],records=prefix['records'],
        rates=None,selection=None,stages=[history['chain_stage']],current_inventory=inventory,
        checkpoint_pairs={**history['checkpoint_pairs'],**prefix['checkpoint_pairs']},lock_fds=writer.lock_fds)
    again,actual,actual_allowance=readiness._bound(writer,slot,startup)
    _require(digest(again)==digest(bound) and actual==request and actual_allowance==allowance,
             'Frozen postprocess inputs changed')
    _check_inventory(root,inventory)
    _guard(writer,guard_ref,entry.KIND,terminal['pid'],7200)
    for name,binding in seal['control_inventory'].items():_check(writer.root,name,binding,time.monotonic()+TIMEOUT)
    _locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Full remaining storage changed')
    return dict(kind=KIND,status='lr_child_reviewed_pending_parent_receipt',slot=slot,plan_sha256=seal['plan_sha256'],
        startup=startup_ref,readiness=readiness_ref,child=refs,entry_guard=dict(**guard_ref,record=guard),
        completed_records=prefix['records'],checkpoint_pairs=prefix['checkpoint_pairs'],storage=storage,
        evidence_inventory_sha256=digest(inventory),adopted=False,dispatch_authorized=False,
        deletion_authorized=False,stage_cleanup_complete=False,next_stage_authorized=False,selection=None)

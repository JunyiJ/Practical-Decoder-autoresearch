"""Read-only parent acceptance after one completed LR/rank/first-fresh child.

Requires owned worker locks, an outer resource guard and a reaped child. Saved
process success is insufficient: every completed LR producer/tensor/resource
record, cleanup history and remaining campaign storage is audited again.
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
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_lr_prefix,bind_rank_prefix
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_metadata import snapshot
from . import tokenizer_wide_hybrid_quality_readiness_v2 as readiness
from . import tokenizer_wide_hybrid_quality_entry_v2 as entry

KIND='wide_hybrid_quality_acceptance_v2'
RANK_KIND='wide_hybrid_rank_acceptance_v2'
CONFIRM_KIND='wide_hybrid_first_confirmation_acceptance_v2'


def _guard(writer,ref,kind,pid,seconds):
    value=_document(writer.root,ref);sample=value.get('observation',{});elapsed=value.get('elapsed_seconds')
    limits=dict(timeout_seconds=seconds,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(value.get('kind')==kind and value.get('status')=='finished' and value.get('reason') is None
        and type(value.get('pid')) is int and value['pid']==pid and value.get('limits')==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<seconds
        and set(sample)=={'peak_rss_bytes','free_bytes'} and all(type(v) is int and v>0 for v in sample.values())
        and violation(sample,elapsed,**limits) is None,'Completed bounded child/readiness guard required')
    return value


def _review_completed(writer,slot,startup_ref,readiness_ref,process):
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
    expected_review=entry._expected_review(slot,startup_ref,seal)
    _require(digest(seal['predispatch_review'])==digest(expected_review)
        and type(seal['required_free_bytes']) is int
        and seal['required_free_bytes']==startup['storage']['required_free_bytes'],
        'Exact complete sealed predispatch review required')
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
    extra={};selected_choice=None
    if slot<9:
        requests=[bound['requests']['quality'][str(i)][0] for i in range(slot+1)]
        _require(all(len(bound['requests']['quality'][str(i)])==1 for i in range(slot+1)),
                 'Independent exact LR requests required')
        history=_history(writer,seal['plan_sha256'],startup['pilot_completion_ref'],inventory)
        prefix=bind_lr_prefix(root,context=bound['context'],requests=requests,
            evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
            numerical_ref=plan['numerical_ref'],projection_ref=seal['projection_ref'],
            training_token_sha256=plan['training_token_sha256'])
        rates=None;stages=[history['chain_stage']]
        pairs={**history['checkpoint_pairs'],**prefix['checkpoint_pairs']}
    elif slot<15:
        from .tokenizer_wide_hybrid_lr_boundary_v2 import verify_lr_history
        from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
        from .tokenizer_wide_hybrid_quality_rank_startup_v2 import _frozen
        lr=verify_lr_history(writer,seal['plan_sha256'],startup['lr_completion_ref'],current_inventory=inventory)
        _require(all(digest(startup[k])==digest(lr[k]) for k in
            ('rates','stages','selection_ref','pilot_completion_ref','projection_ref')),
            'Original rank startup history differs after child')
        exact,requests=_frozen(writer,slot,seal['plan_sha256'],seal['projection_ref'],lr['rates'])
        _require(digest(exact)==digest(bound) and digest(requests[-1])==digest(request),
                 'Independent frozen rank requests differ after child')
        prefix=bind_rank_prefix(root,context=bound['context'],lr_records=lr['records'],rates=lr['rates'],
            lr_requests=requests[:9],rank_requests=requests[9:],
            evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
            numerical_ref=plan['numerical_ref'],rates_ref=lr['selection_ref'],projection_ref=seal['projection_ref'],
            training_token_sha256=plan['training_token_sha256'])
        _require(digest(prefix['records'][:-1])==digest(startup['completed_records']),
                 'Completed rank prefix changed from sealed history')
        for name,pair in prefix['checkpoint_pairs'].items():
            if int(name.split('/')[1])<9:
                _require(digest(pair)==digest(lr['checkpoint_pairs'][name]),'Live LR anchor changed after rank child')
        history=verify_pilot_history(writer,seal['plan_sha256'],lr['pilot_completion_ref'],
            current_inventory=inventory,stages=lr['stages'])
        rates=lr['rates'];stages=lr['stages']
        pairs={**history['checkpoint_pairs'],**lr['checkpoint_pairs'],**prefix['checkpoint_pairs']}
        extra=dict(rates=rates,stages=stages,lr_completion_ref=startup['lr_completion_ref'],
                   selection_ref=lr['selection_ref'],audited_indices=prefix['audited_indices'])
    else:
        from .tokenizer_wide_hybrid_rank_history_v2 import verify_rank_history
        from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
        from .tokenizer_wide_hybrid_first_confirmation_startup_v2 import _frozen
        from .tokenizer_wide_hybrid_confirmation_audit_v2 import bind_confirmation_prefix
        rank=verify_rank_history(writer,seal['plan_sha256'],startup['rank_completion_ref'],current_inventory=inventory)
        records,selected_pairs,rates,selected_choice=entry._first_confirmation_history(writer,seal,startup)
        _require(records==rank['records'] and rates==rank['rates'] and selected_choice==rank['selection'],
                 'Original first-confirmation history differs after child')
        exact,requests=_frozen(writer,slot,seal['plan_sha256'],seal['projection_ref'],rates,selected_choice)
        _require(digest(exact)==digest(bound) and requests[-1]==request,
                 'Independent frozen confirmation requests differ after child')
        prefix=bind_confirmation_prefix(root,context=bound['context'],screen_records=rank['records'],
            rates=rates,selection=selected_choice,screen_requests=requests[:15],confirmation_requests=requests[15:],
            evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
            numerical_ref=plan['numerical_ref'],rates_ref=rank['rates_ref'],selection_ref=rank['selection_ref'],
            projection_ref=seal['projection_ref'],training_token_sha256=plan['training_token_sha256'])
        _require(prefix['records'][:-1]==records and all(prefix['checkpoint_pairs'][n]==v for n,v in selected_pairs.items()),
                 'Selected anchors or screening history changed after first fresh child')
        history=verify_pilot_history(writer,seal['plan_sha256'],rank['pilot_completion_ref'],
            current_inventory=inventory,stages=rank['stages'])
        stages=rank['stages']
        pairs={**history['checkpoint_pairs'],**rank['checkpoint_pairs'],**prefix['checkpoint_pairs']}
        extra=dict(rates=rates,stages=stages,rank_completion_ref=startup['rank_completion_ref'],
            selection_ref=rank['selection_ref'],rates_ref=rank['rates_ref'],audited_indices=prefix['audited_indices'])
    projection=_document(root,seal['projection_ref'])
    storage=bind_storage(writer.root,projection,pilots=history['records'],records=prefix['records'],
        rates=rates,selection=selected_choice,stages=stages,current_inventory=inventory,
        checkpoint_pairs=pairs,lock_fds=writer.lock_fds)
    again,actual,actual_allowance=readiness._bound(writer,slot,startup)
    _require(digest(again)==digest(bound) and actual==request and actual_allowance==allowance,
             'Frozen postprocess inputs changed')
    _check_inventory(root,inventory)
    _guard(writer,guard_ref,entry.KIND,terminal['pid'],7200)
    for name,binding in seal['control_inventory'].items():_check(writer.root,name,binding,time.monotonic()+TIMEOUT)
    _locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Full remaining storage changed')
    return dict(kind=KIND if slot<9 else (RANK_KIND if slot<15 else CONFIRM_KIND),
        status=('lr' if slot<9 else ('rank' if slot<15 else 'first_confirmation'))+'_child_reviewed_pending_parent_receipt',slot=slot,plan_sha256=seal['plan_sha256'],
        startup=startup_ref,readiness=readiness_ref,child=refs,entry_guard=dict(**guard_ref,record=guard),
        completed_records=prefix['records'],checkpoint_pairs=prefix['checkpoint_pairs'],storage=storage,
        evidence_inventory_sha256=digest(inventory),adopted=False,dispatch_authorized=False,
        deletion_authorized=False,stage_cleanup_complete=False,next_stage_authorized=False,selection=selected_choice,**extra)



def review_completed_lr(writer,slot,startup_ref,readiness_ref,process):
    """Review only declared LR slots; caller has reaped the child and owns guards."""
    _require(type(slot) is int and 0<=slot<9,'Exact LR acceptance slot required')
    return _review_completed(writer,slot,startup_ref,readiness_ref,process)


def review_completed_rank(writer,slot,startup_ref,readiness_ref,process):
    """Review one rank result; never select, delete, launch or permit continuation."""
    _require(type(slot) is int and 9<=slot<15,'Exact rank acceptance slot required')
    return _review_completed(writer,slot,startup_ref,readiness_ref,process)


def review_completed_first_confirmation(writer,slot,startup_ref,readiness_ref,process):
    """Review slot15 against saved selections, all selected anchors and original history."""
    _require(type(slot) is int and slot==15,'Exact first-confirmation acceptance slot15 required')
    return _review_completed(writer,slot,startup_ref,readiness_ref,process)

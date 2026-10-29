"""Readiness-bound LR/rank/first-fresh child CLI and composition.

An exact parent-supplied seal hash transfers the parent's full idle audit. The
child verifies every sealed byte, control receipt and frozen numerical gate, and
reconstructs current storage before invoking the separately bounded producer.
"""
import math
import os
import shutil
import time

from .protocol import digest,file_digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_retention_receipts import _document,_path
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks,_check,_scan,TIMEOUT
from .tokenizer_wide_hybrid_pilot_boundary_v2 import _receipt
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_storage_v2 import remaining_projection,required_free
from . import tokenizer_wide_hybrid_quality_readiness_v2 as readiness
from . import tokenizer_wide_hybrid_quality_handoff_v2 as transport

KIND='wide_hybrid_quality_entry_v2'


def worker_argv(writer,slot,startup_ref,readiness_ref):
    return transport.worker_argv(writer,slot,startup_ref,readiness_ref=readiness_ref)


def _controls(writer,seal,slot):
    controls=seal['control_inventory'];name=f'control/quality-readiness-{slot:02d}.json'
    _require(name in controls and seal['guard']==dict(path=name,sha256=controls[name]['sha256']),
             'Sealed readiness guard reference differs')
    deadline=time.monotonic()+TIMEOUT
    for path,binding in controls.items():
        _require(path.startswith('control/') and path.endswith('.json'),'Exact sealed control file required')
        _check(writer.root,path,binding,deadline)
    guard=_document(writer.root,seal['guard']);sample=guard.get('observation',{});elapsed=guard.get('elapsed_seconds')
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(guard.get('kind')==readiness.KIND and guard.get('status')=='finished' and guard.get('reason') is None
        and type(guard.get('pid')) is int and guard['pid']==os.getppid() and guard.get('limits')==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<1800
        and set(sample)=={'peak_rss_bytes','free_bytes'} and all(type(v) is int and v>0 for v in sample.values())
        and violation(sample,elapsed,**limits) is None,'Successful same-parent readiness guard required')


def _stamps(root,inventory):
    keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_blocks','st_nlink')
    return {n:tuple(getattr(_path(root,n).stat(),k) for k in keys) for n in inventory}


def _storage(writer,slot,seal,startup,inventory):
    root=writer.root/'evidence'
    pilot,_,stage,_=_receipt(writer,seal['plan_sha256'],startup['pilot_completion_ref'])
    selection=None
    if slot==15:
        records,pairs,rates,selection=_first_confirmation_history(writer,seal,startup)
        boundaries=3
    else:
        records=[] if slot==0 else startup['completed_records']
        pairs={} if slot==0 else startup['checkpoint_pairs']
        rates=None;boundaries=1
    _require(isinstance(records,list) and len(records)==slot,'Exact completed quality prefix required')
    if slot<9:
        _require(set(pairs)=={f'quality/{i:02d}' for i in range(slot)},'Exact completed LR prefix required')
    elif slot<15:
        # The live parent's immutable seal transfers full idle audits. Rebind
        # original cleanup identities and selection before counting live bytes;
        # deliberately deleted pairs are historical evidence, never capacity.
        from .tokenizer_wide_hybrid_lr_boundary_v2 import _receipt as lr_receipt
        complete,review,_=lr_receipt(writer,seal['plan_sha256'],startup['lr_completion_ref'])
        rates=startup['rates'];boundaries=2
        _require(digest(startup['stages'])==digest([pilot['chain_stage'],complete['chain_stage']])
            and digest(records[:9])==digest(review['records']) and digest(rates)==digest(review['rates'])
            and startup['selection_ref']==review['selection_ref']
            and startup['projection_ref']==review['projection_ref']
            and startup['pilot_completion_ref']==review['pilot_completion_ref'],
            'Sealed original rank history or LR selection differs')
        original={**stage['checkpoint_pairs'],**review['checkpoint_pairs']}
        _require(set(pairs)==set(original)|{f'quality/{i:02d}' for i in range(9,slot)}
            and all(digest(pairs[n])==digest(v) for n,v in original.items()),
            'Original deleted and retained rank history pairs required')
        anchors=[i for i,r in enumerate(records[:9]) if r['lr']==rates[r['family']]]
        indices=anchors+list(range(9,slot))
        _require(len(anchors)==3 and digest(startup['audited_indices'])==digest(indices),
                 'Exact live LR anchors and completed rank members required')
        pairs={f'quality/{i:02d}':pairs[f'quality/{i:02d}'] for i in indices}
    projection=_document(root,seal['projection_ref'])
    state=remaining_projection(projection,stage['records'],records,rates,selection,boundaries)
    census=snapshot(writer.root);weights=set();inodes=set();logical=occupied=0
    for key,pair in pairs.items():
        _require(set(pair)=={'raw','h200'},'Complete retained checkpoint pairs required')
        for entry in pair.values():
            name=entry['path'];path=_path(root,name);st=path.stat();inode=(st.st_dev,st.st_ino)
            _require(path.parent==root/key/'checkpoints' and path.suffix=='.pt'
                and st.st_nlink==1 and inode not in inodes and name not in weights
                and inventory[name]==dict(size=entry['size'],sha256=entry['sha256'])
                and st.st_size==entry['size'],'Retained checkpoint capacity identity differs')
            inodes.add(inode);weights.add(name);logical+=st.st_size;occupied+=min(st.st_size,st.st_blocks*512)
    actual={n.removeprefix('evidence/') for n,v in census['files'].items() if v['group'].endswith('/weights')}
    _require(actual==weights and census['checkpoint_logical_bytes']==logical,'Unreviewed checkpoint in live handoff')
    needed=required_free(state,occupied);saved=startup['storage']
    expected=dict(required_free_bytes=needed,occupied_checkpoint_capacity_bytes=occupied,
                  checkpoint_logical_bytes=logical,cleanup_boundaries_verified=boundaries)
    _require(all(type(saved.get(k)) is int and saved[k]==v for k,v in expected.items())
        and digest(saved['projection'])==digest(state) and type(seal.get('required_free_bytes')) is int
        and seal['required_free_bytes']==needed,'Sealed remaining storage or actual allocation changed')
    _require(shutil.disk_usage(writer.root).free>=needed,'Complete remaining panel no longer fits')
    return needed


def _first_confirmation_history(writer,seal,startup):
    """Expand compact original receipts; the same-parent seal binds their full audit.

    Reconstruct all three cleanup proofs and original pairs, then count only the
    three selected live screens. Deleted weights never receive capacity credit.
    """
    from .tokenizer_wide_hybrid_rank_boundary_v2 import _receipt as rank_receipt
    saved,_,_,view=rank_receipt(writer,seal['plan_sha256'],startup['rank_completion_ref'])
    summary=view['summary'];selected=view['selected'];history=view['history']
    stages=view['stages']+[dict(manifest=view['manifest'],execution_ref=saved['execution_ref'],
                              context_sha256=saved['context_sha256'])]
    expected=dict(projection_ref=selected['projection_ref'],pilot_completion_ref=history['pilot_completion_ref'],
        rates_ref=history['selection_ref'],selection_ref=selected['selection_ref'],
        rates=summary['rates'],selection=summary['selection'],
        completed_records_sha256=digest(summary['records']),checkpoint_pairs_sha256=digest(summary['checkpoint_pairs']),
        stages_sha256=digest(stages),frozen_files_sha256=digest(seal['frozen_files']))
    _require(all(digest(startup[k])==digest(v) for k,v in expected.items()),
             'Compact first-confirmation original history differs')
    indices=[i for i,r in enumerate(summary['records'])
             if dict(lr=r['lr'],divisor=r['divisor'])==summary['selection'][r['family']]]
    _require(len(indices)==3 and digest(indices)==digest(startup['audited_indices']),
             'Exact three selected screen anchors required')
    pairs={f'quality/{i:02d}':summary['checkpoint_pairs'][f'quality/{i:02d}'] for i in indices}
    _require({e['path'] for pair in pairs.values() for e in pair.values()}==
             {e['path'] for e in view['manifest']['retain']},'Selected pairs differ from rank retention')
    return summary['records'],pairs,summary['rates'],summary['selection']


def _expected_review(slot,startup_ref,seal):
    expected=dict(required_free_bytes=seal['required_free_bytes'],request_sha256=seal['request']['sha256'],
                  pilot_history_verified=True,dispatch_authorized=False)
    if slot==0:expected['first_quality_request_verified']=True
    elif slot<9:expected.update(startup_ref=startup_ref,lr_prefix_verified=True,lr_request_verified=True)
    elif slot<15:expected.update(startup_ref=startup_ref,lr_history_verified=True,rank_prefix_verified=True,rank_request_verified=True)
    else:
        _require(slot==15,'Only first fresh slot15 supported')
        expected.update(startup_ref=startup_ref,rank_history_verified=True,
                        selected_screen_prefix_verified=True,first_confirmation_request_verified=True)
    return expected


def verify_ready_worker(writer,slot,startup_ref,readiness_ref,actual_argv):
    """Caller owns outer resource guard; exact argv anchors trusted parent seal.

    Historical full audits are inherited only through the same live parent's
    immutable seal and unchanged evidence, never from unanchored boolean claims.
    """
    _require(actual_argv==worker_argv(writer,slot,startup_ref,readiness_ref),'Actual sealed child argv differs')
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    seal=_document(root,readiness_ref);startup=_document(root,startup_ref)
    _require(seal.get('kind')==readiness.KIND and seal.get('status')=='idle_review_sealed_pending_child_integration'
        and type(seal.get('slot')) is int and seal['slot']==slot
        and type(seal.get('parent_pid')) is int and seal['parent_pid']==os.getppid()>1
        and seal.get('startup_ref')==startup_ref and seal.get('dispatch_authorized') is False
        and seal.get('deletion_authorized') is False,'Same-parent immutable readiness seal required')
    for key in ('request','plan_sha256','projection_ref','context'):
        _require(digest(seal[key])==digest(startup[key]),'Seal/startup identity differs')
    expected=_expected_review(slot,startup_ref,seal)
    _require(digest(seal['predispatch_review'])==digest(expected),'Sealed full idle review differs')
    inventory=dict(startup['inventory'])
    for ref in (startup['request'],startup_ref):
        _require(ref['path'] not in inventory,'Startup evidence overlap')
        p=_path(root,ref['path']);inventory[ref['path']]=dict(size=p.stat().st_size,sha256=ref['sha256'])
    _require(inventory==seal['inventory'] and readiness_ref['path'] not in inventory,'Complete sealed inventory differs')
    p=_path(root,readiness_ref['path']);inventory[readiness_ref['path']]=dict(size=p.stat().st_size,sha256=readiness_ref['sha256'])
    initial=_stamps(root,inventory);_controls(writer,seal,slot)
    bound,request,allowance=readiness._bound(writer,slot,startup)
    _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
        and type(seal.get('output_limit_bytes')) is int and seal['output_limit_bytes']==allowance,
        'Readiness frozen files or output allowance differs')
    live=transport.verify_live_handoff(writer,slot,startup_ref,actual_argv,inventory=inventory,
        required_free_bytes=seal['required_free_bytes'],output_limit_bytes=allowance,readiness_ref=readiness_ref)
    log=_path(root,f'quality/{slot:02d}/process/worker.log');log_before=log.stat()
    log_prefix=transport._log_prefix(log,log_before.st_size)
    needed=_storage(writer,slot,seal,startup,inventory)
    again,actual,actual_allowance=readiness._bound(writer,slot,startup)
    _require(digest(again)==digest(bound) and actual==request and actual_allowance==allowance,
             'Frozen provenance changed during sealed handoff')
    _controls(writer,seal,slot)
    names=set(inventory)|{f'quality/{slot:02d}/process/'+n for n in ('launch.json','worker.log')}
    _require(_scan(root,None)==names and _stamps(root,inventory)==initial,'Sealed evidence changed during handoff')
    log_after=log.stat()
    _require((log_after.st_dev,log_after.st_ino)==(log_before.st_dev,log_before.st_ino)
        and log_after.st_nlink==1 and log_before.st_size<=log_after.st_size<=transport.LOG_LIMIT
        and transport._log_prefix(log,log_before.st_size)==log_prefix,'Worker log changed before quality entry')
    snapshot(writer.root);_locks(root,writer.lock_fds)
    _require(os.getppid()==seal['parent_pid'] and not (writer.root/'STOP').exists()
        and not (writer.root/'STOP').is_symlink(),'Parent or STOP changed before quality entry')
    _require(shutil.disk_usage(writer.root).free>=needed,'Remaining storage changed before quality entry')
    return dict(kind=KIND,slot=slot,plan_sha256=seal['plan_sha256'],projection_ref=seal['projection_ref'],
        request_sha256=seal['request']['sha256'],required_free_bytes=needed,readiness_ref=readiness_ref,
        launch_ref=live['launch_ref'],parent_idle_audit_bound=True,frozen_plan_verified=True,
        live_transport_verified=True,storage_verified=True,dispatch_authorized=False)


class _EntryGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot):
        super().__init__(writer,slot,'entry',7200);self.name=f'control/quality-entry-{slot:02d}.json'


def run_worker(writer,slot,startup_ref,readiness_ref,argv):
    """Finite child body only; inherited locks and validated handoff precede model import."""
    worker_argv(writer,slot,startup_ref,readiness_ref)
    with writer._serialized():
        _require(not (writer.root/f'control/quality-entry-{slot:02d}.json').exists()
            and not (writer.root/f'control/quality-entry-{slot:02d}.json.tmp').exists(),'Entry cannot be retried')
    with _EntryGuard(writer,slot):
        checked=verify_ready_worker(writer,slot,startup_ref,readiness_ref,argv)
        from .tokenizer_wide_hybrid_quality_execution_v2 import execute_quality_outputs
        return execute_quality_outputs(writer,slot,checked['plan_sha256'],checked['request_sha256'],
                                       projection_ref=checked['projection_ref'])


def main(argv=None):
    """Child-only CLI: never freeze, select, clean up, retry or launch a parent."""
    import argparse
    import sys
    from pathlib import Path
    from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=['worker'])
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--slot',required=True,type=int,choices=range(16))
    parser.add_argument('--startup-sha256',required=True)
    parser.add_argument('--global-fd',required=True,type=int)
    parser.add_argument('--campaign-fd',required=True,type=int)
    parser.add_argument('--readiness-sha256',required=True)
    args=parser.parse_args(argv)
    from .tokenizer_wide_hybrid_retention_receipts import _sha
    _sha(args.startup_sha256);_sha(args.readiness_sha256)
    writer=MetadataWriter(args.campaign,(args.global_fd,args.campaign_fd))
    phase='lr' if args.slot<9 else ('rank' if args.slot<15 else 'confirm')
    startup=dict(path='meta/quality-first-startup.json' if args.slot==0
                 else f'meta/quality-{phase}-startup-{args.slot:02d}.json',sha256=args.startup_sha256)
    ready=dict(path=f'meta/quality-readiness-{args.slot:02d}.json',sha256=args.readiness_sha256)
    actual=[sys.executable,'-m','src.research.tokenizer_wide_hybrid_quality_entry_v2',
            *(sys.argv[1:] if argv is None else argv)]
    _require(actual==worker_argv(writer,args.slot,startup,ready),'Noncanonical child command refused')
    return run_worker(writer,args.slot,startup,ready,actual)


if __name__=='__main__':main()

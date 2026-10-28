"""Bounded rank request preparation; no child, retry, selection or dispatch.

Caller owns both locks and an outer resource guard, and inspects actual workers
first. Complete cleanup history, live prefix and future storage are verified
before creating a request. The outer guard also covers final receipt checks.
The eventual parent/child dispatcher must independently recheck this receipt.
"""
import os
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_lr_boundary_v2 import verify_lr_history
from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_rank_prefix
from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign,PROJECTION
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_storage_v2 import bind_storage,remaining_projection,required_free
from .tokenizer_wide_hybrid_rules import lr_specs,rank_specs
from .tokenizer_wide_hybrid_scores import _require

KIND='wide_hybrid_rank_request_startup_v2'


class _PreparationGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot):
        super().__init__(writer,slot,'startup',1800)
        self.name=f'control/quality-rank-startup-{slot:02d}.json'


def _frozen(writer,slot,plan_sha,projection_ref,rates):
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
             'STOP prevents rank request preparation')
    root=writer.root/'evidence';bound=bind_campaign(root/'meta/plan.json',plan_sha)
    _require(projection_ref['path'].startswith('meta/')
        and bound['files'].get(str(_path(root,projection_ref['path'])))==projection_ref['sha256']
        and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
        'Rank startup projection differs from frozen copy')
    requests=[]
    for i,spec in enumerate(lr_specs()+rank_specs(rates)[:slot-8]):
        options=bound['requests']['quality'][str(i)]
        matches=[r for r in options if digest(r.get('spec'))==digest(spec)]
        _require(len(matches)==1 and (i>=9 or len(options)==1)
            and matches[0].get('result_path')==str(root/f'quality/{i:02d}/result.json'),
            'Unique independently frozen conditional request required')
        requests.append(matches[0])
    return bound,requests


def _audit(writer,slot,plan_sha,projection_ref,lr_completion_ref):
    """Idle-only full preparation gate, before the next request exists."""
    root=writer.root/'evidence';_locks(root,writer.lock_fds);snapshot(writer.root)
    inventory=_inventory(root)
    history=verify_lr_history(writer,plan_sha,lr_completion_ref,current_inventory=inventory)
    _require(history['projection_ref']==projection_ref,'Historical projection reference differs')
    bound,requests=_frozen(writer,slot,plan_sha,projection_ref,history['rates'])
    plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha))
    prefix=bind_rank_prefix(root,context=bound['context'],lr_records=history['records'],rates=history['rates'],
        lr_requests=requests[:9],rank_requests=requests[9:-1],
        evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
        numerical_ref=plan['numerical_ref'],rates_ref=history['selection_ref'],projection_ref=projection_ref,
        training_token_sha256=plan['training_token_sha256'])
    _require(prefix['records'][:9]==history['records'] and prefix['rates']==history['rates'],
             'Live prefix differs from verified LR history')
    for name,pair in prefix['checkpoint_pairs'].items():
        if int(name.split('/')[1])<9:
            _require(pair==history['checkpoint_pairs'][name],'Live LR anchor differs from original pair')
    pilot=verify_pilot_history(writer,plan_sha,history['pilot_completion_ref'],
        current_inventory=inventory,stages=history['stages'])
    pairs={**pilot['checkpoint_pairs'],**history['checkpoint_pairs'],**prefix['checkpoint_pairs']}
    storage=bind_storage(writer.root,_document(root,projection_ref),pilots=pilot['records'],
        records=prefix['records'],rates=history['rates'],selection=None,stages=history['stages'],
        current_inventory=inventory,checkpoint_pairs=pairs,lock_fds=writer.lock_fds)
    again,again_requests=_frozen(writer,slot,plan_sha,projection_ref,history['rates'])
    _require(digest(again)==digest(bound) and again_requests==requests,'Frozen rank inputs changed during audits')
    _check_inventory(root,inventory);_locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],
             'Full remaining storage changed during rank preparation')
    return dict(context=bound['context'],frozen_files=bound['files'],requests=requests,
        inventory=inventory,history=history,prefix=prefix,pilot=pilot,checkpoint_pairs=pairs,storage=storage)


def prepare_rank_request(writer,slot,plan_sha256,*,projection_ref,lr_completion_ref):
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 9<=slot<=14,
             'Rank preparation supports exact slots9 through14 only')
    _sha(plan_sha256);root=writer.root/'evidence';_locks(root,writer.lock_fds)
    base=f'meta/quality-rank-startup-{slot:02d}';guard_name=f'control/quality-rank-startup-{slot:02d}.json'
    seen=snapshot(writer.root)['files']
    _require(not (root/f'quality/{slot:02d}').exists(),'Next rank slot must be absent before idle review')
    _require(not any(n.startswith('evidence/'+base) or n in (guard_name,guard_name+'.tmp') for n in seen),
             'Existing rank preparation cannot be retried')
    try:
        with _PreparationGuard(writer,slot):
            attempt=dict(kind=KIND,status='started',slot=slot,plan_sha256=plan_sha256,
                projection_ref=projection_ref,lr_completion_ref=lr_completion_ref,dispatch_authorized=False)
            path=writer.publish_json('evidence/'+base+'-attempt.json',attempt)
            attempt_ref=dict(path=base+'-attempt.json',sha256=file_digest(path))
            checked=_audit(writer,slot,plan_sha256,projection_ref,lr_completion_ref)
            request=checked['requests'][-1]
            path=writer.publish_json(f'evidence/quality/{slot:02d}/request.json',request)
            request_ref=dict(path=f'quality/{slot:02d}/request.json',sha256=file_digest(path))
            expected=dict(checked['inventory']);expected[request_ref['path']]=dict(size=path.stat().st_size,sha256=request_ref['sha256'])
            report=dict(kind=KIND,status='rank_request_pending_predispatch_review',parent_pid=os.getpid(),slot=slot,
                plan_sha256=plan_sha256,projection_ref=projection_ref,lr_completion_ref=lr_completion_ref,
                pilot_completion_ref=checked['history']['pilot_completion_ref'],selection_ref=checked['history']['selection_ref'],
                attempt=attempt_ref,request=request_ref,context=checked['context'],frozen_files=checked['frozen_files'],
                inventory=checked['inventory'],completed_records=checked['prefix']['records'],rates=checked['history']['rates'],
                checkpoint_pairs=checked['checkpoint_pairs'],stages=checked['history']['stages'],
                audited_indices=checked['prefix']['audited_indices'],storage=checked['storage'],
                adopted=False,dispatch_authorized=False,deletion_authorized=False,next_stage_authorized=False,selection=None)
        guard=dict(path=guard_name,sha256=file_digest(writer.root/guard_name))
        _guard(writer,guard,KIND,os.getpid(),1800);report['guard']=guard
        again,requests=_frozen(writer,slot,plan_sha256,projection_ref,report['rates'])
        _require(again['context']==checked['context'] and again['files']==checked['frozen_files']
            and requests==checked['requests'],'Frozen rank inputs changed before publication')
        history=verify_lr_history(writer,plan_sha256,lr_completion_ref,current_inventory=expected)
        # The chain's current-inventory digest/count legitimately changes when
        # the pending request is appended. Its entire new inventory was verified
        # above; only the historical fields must remain byte-identical.
        _require({k:v for k,v in history.items() if k!='chain'}==
                 {k:v for k,v in checked['history'].items() if k!='chain'},
                 'LR history changed before rank completion')
        _check_inventory(root,expected);_locks(root,writer.lock_fds)
        _require(shutil.disk_usage(writer.root).free>=checked['storage']['required_free_bytes'],
                 'Full remaining storage changed before rank completion')
        path=writer.publish_json('evidence/'+base+'.json',report)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Rank preparation failure publication failed: '+repr(error)[:1000])
        raise


def verify_rank_request(writer,slot,reference):
    """Read-only idle predispatch recheck, under both locks and an outer guard.

    Reconstruct current capacity without giving the pending request completed
    trajectory credit. No process artifacts may yet exist in that slot. A later
    readiness/child/parent transport layer is still required for dispatch.
    """
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 9<=slot<=14,
             'Exact rank slot9 through14 required')
    root=writer.root/'evidence';base=f'meta/quality-rank-startup-{slot:02d}'
    _locks(root,writer.lock_fds)
    _require(reference['path']==base+'.json','Exact rank startup reference required')
    report=_document(root,reference);pid=report.get('parent_pid')
    _require(report.get('kind')==KIND and report.get('status')=='rank_request_pending_predispatch_review'
        and type(report.get('slot')) is int and report['slot']==slot and type(pid) is int and pid>1
        and all(report.get(k) is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
        and 'selection' in report and report['selection'] is None,'Rank startup identity/claims differ')
    guard=report['guard'];_require(guard['path']==f'control/quality-rank-startup-{slot:02d}.json',
                                 'Exact rank preparation guard required')
    _guard(writer,guard,KIND,pid,1800)
    attempt=dict(kind=KIND,status='started',slot=slot,plan_sha256=report['plan_sha256'],
        projection_ref=report['projection_ref'],lr_completion_ref=report['lr_completion_ref'],dispatch_authorized=False)
    _require(report['attempt']['path']==base+'-attempt.json'
        and _document(root,report['attempt'])==attempt,'Rank preparation attempt differs')
    inventory=dict(report['inventory']);rq=report['request']
    _require(inventory[report['attempt']['path']]['sha256']==report['attempt']['sha256'],'Attempt inventory differs')
    _require(rq['path']==f'quality/{slot:02d}/request.json','Exact pending rank request required')
    for ref in (rq,reference):
        path=_path(root,ref['path']);_document(root,ref)
        _require(ref['path'] not in inventory,'Rank startup output overlaps old inventory')
        inventory[ref['path']]=dict(size=path.stat().st_size,sha256=ref['sha256'])
    _check_inventory(root,inventory)
    history=verify_lr_history(writer,report['plan_sha256'],report['lr_completion_ref'],current_inventory=inventory)
    _require(all(digest(report[k])==digest(history[k]) for k in ('projection_ref','pilot_completion_ref','selection_ref','rates','stages')),
             'Rank startup differs from verified original LR history')
    bound,requests=_frozen(writer,slot,report['plan_sha256'],report['projection_ref'],history['rates'])
    _require(report['context']==bound['context'] and report['frozen_files']==bound['files']
        and digest(_document(root,rq))==digest(requests[-1]),'Frozen rank provenance or pending request changed')
    plan=_document(root,dict(path='meta/plan.json',sha256=report['plan_sha256']))
    prefix=bind_rank_prefix(root,context=bound['context'],lr_records=history['records'],rates=history['rates'],
        lr_requests=requests[:9],rank_requests=requests[9:-1],pending_request=requests[-1],
        evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
        numerical_ref=plan['numerical_ref'],rates_ref=history['selection_ref'],projection_ref=report['projection_ref'],
        training_token_sha256=plan['training_token_sha256'])
    for name,pair in prefix['checkpoint_pairs'].items():
        if int(name.split('/')[1])<9:
            _require(pair==history['checkpoint_pairs'][name],'Live LR anchor differs from original pair')
    pilot=verify_pilot_history(writer,report['plan_sha256'],history['pilot_completion_ref'],
        current_inventory=inventory,stages=history['stages'])
    pairs={**pilot['checkpoint_pairs'],**history['checkpoint_pairs'],**prefix['checkpoint_pairs']}
    _require(digest(report['completed_records'])==digest(prefix['records']) and digest(report['checkpoint_pairs'])==digest(pairs)
        and digest(report['audited_indices'])==digest(prefix['audited_indices']),'Saved rank prefix or original pairs differ')
    state=remaining_projection(_document(root,report['projection_ref']),pilot['records'],
        prefix['records'],history['rates'],None,2)
    census=snapshot(writer.root);weights=set();inodes=set();occupied=0;logical=0;stamps={}
    for pair in prefix['checkpoint_pairs'].values():
        for entry in pair.values():
            path=_path(root,entry['path']);st=path.stat();inode=(st.st_dev,st.st_ino)
            _require(st.st_nlink==1 and inode not in inodes
                and inventory[entry['path']]==dict(size=entry['size'],sha256=entry['sha256']),
                'Rank checkpoint census alias or binding differs')
            inodes.add(inode);weights.add(entry['path']);logical+=st.st_size
            occupied+=min(st.st_size,st.st_blocks*512)
            stamps[entry['path']]=(st.st_dev,st.st_ino,st.st_size,st.st_blocks,st.st_mtime_ns,st.st_ctime_ns)
    actual={n.removeprefix('evidence/') for n,v in census['files'].items() if v['group'].endswith('/weights')}
    _require(actual==weights and census['checkpoint_logical_bytes']==logical,'Unexpected checkpoint before rank dispatch')
    needed=required_free(state,occupied);storage=report['storage']
    expected=dict(required_free_bytes=needed,occupied_checkpoint_capacity_bytes=occupied,
                  checkpoint_logical_bytes=logical,cleanup_boundaries_verified=2)
    _require(all(type(storage.get(k)) is int and storage[k]==v for k,v in expected.items())
        and storage.get('status')=='remaining_campaign_storage_verified' and digest(storage.get('projection'))==digest(state)
        and storage.get('current_inventory_sha256')==digest(report['inventory'])
        and type(storage.get('free_bytes')) is int and storage['free_bytes']>=needed
        and type(storage.get('metadata_bytes')) is int and 0<=storage['metadata_bytes']<=state['metadata_allowance_bytes']
        and all(storage.get(k) is True for k in ('storage_projection_fits','current_phase_space_verified'))
        and all(storage.get(k) is False for k in ('dispatch_authorized','frozen_plan_verified','audit_semantics_verified')),
        'Saved full-panel rank storage evidence differs')
    free=shutil.disk_usage(writer.root).free
    _require(type(free) is int and free>=needed,'Complete remaining panel no longer fits')
    again,again_requests=_frozen(writer,slot,report['plan_sha256'],report['projection_ref'],history['rates'])
    _require(digest(again)==digest(bound) and again_requests==requests,'Frozen rank inputs changed during predispatch')
    _require(verify_lr_history(writer,report['plan_sha256'],report['lr_completion_ref'],current_inventory=inventory)==history,
             'LR history changed during rank predispatch')
    _guard(writer,guard,KIND,pid,1800);_check_inventory(root,inventory);_locks(root,writer.lock_fds)
    for name,stamp in stamps.items():
        st=_path(root,name).stat()
        _require(st.st_nlink==1 and (st.st_dev,st.st_ino,st.st_size,st.st_blocks,st.st_mtime_ns,st.st_ctime_ns)==stamp,
                 'Live rank checkpoint changed during predispatch')
    _require(snapshot(writer.root)==census,'Rank predispatch census changed')
    free=shutil.disk_usage(writer.root).free
    _require(type(free) is int and free>=needed,'Remaining storage changed before rank handoff')
    return dict(required_free_bytes=needed,request_sha256=rq['sha256'],startup_ref=reference,
        pilot_history_verified=True,lr_history_verified=True,rank_prefix_verified=True,
        rank_request_verified=True,dispatch_authorized=False)

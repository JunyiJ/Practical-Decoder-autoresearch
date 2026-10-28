"""Prepare one later LR request after complete prefix and storage review.

Only slots1..8 are supported. There is no dispatch, selection, cleanup, retry,
child entry or rank/fresh transition. The caller owns both worker locks and
inspects actual workers first. Real frozen numerical prerequisites are required.
"""
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_quality_startup_v2 import _history
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_lr_prefix
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign,PROJECTION
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_storage_v2 import bind_storage,remaining_projection,required_free
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_rules import lr_specs
from .cpu_prerequisite_guard import violation

KIND='wide_hybrid_lr_request_startup_v2'


class _PreparationGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot):
        super().__init__(writer,slot,'startup',1800)
        self.name=f'control/quality-lr-startup-{slot:02d}.json'


def _frozen(writer,slot,plan_sha,projection_ref):
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
             'STOP prevents LR request preparation')
    bound=bind_campaign(writer.root/'evidence/meta/plan.json',plan_sha)
    _require(projection_ref['path'].startswith('meta/')
             and bound['files'].get(str(_path(writer.root/'evidence',projection_ref['path'])))==projection_ref['sha256']
             and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
             'LR startup projection differs from frozen copy')
    requests=[]
    for i,spec in enumerate(lr_specs()[:slot+1]):
        options=bound['requests']['quality'][str(i)]
        _require(len(options)==1 and digest(options[0]['spec'])==digest(spec)
                 and options[0]['result_path']==str(writer.root/f'evidence/quality/{i:02d}/result.json'),
                 'Exact independently frozen LR requests required')
        requests.append(options[0])
    return bound,requests


def prepare_lr_request(writer,slot,plan_sha256,*,projection_ref,pilot_completion_ref):
    """Audit the complete live LR prefix before publishing its next request.

    Successful preparation is not process authorization. A later predispatch
    and child-entry verifier must recheck this immutable receipt and all guards.
    """
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 1<=slot<=8,
             'Later LR preparation supports exact slots1 through8 only')
    _sha(plan_sha256);root=writer.root/'evidence';base=f'meta/quality-lr-startup-{slot:02d}'
    guard_name=f'control/quality-lr-startup-{slot:02d}.json'
    with writer._serialized():
        _require(not (root/f'quality/{slot:02d}').exists(),'Next LR slot must be absent before idle review')
        _require(not any((root/(base+s)).exists() for s in ('-attempt.json','-attempt.json.tmp',
            '.json','.json.tmp','-failure.json','-failure.json.tmp'))
            and not (writer.root/guard_name).exists(),'Existing LR preparation cannot be retried')
    try:
        with _PreparationGuard(writer,slot):
            attempt=dict(kind=KIND,status='started',slot=slot,plan_sha256=plan_sha256,
                projection_ref=projection_ref,pilot_completion_ref=pilot_completion_ref,dispatch_authorized=False)
            marker=writer.publish_json('evidence/'+base+'-attempt.json',attempt)
            marker_ref=dict(path=base+'-attempt.json',sha256=file_digest(marker))
            bound,requests=_frozen(writer,slot,plan_sha256,projection_ref)
            plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha256))
            projection=_document(root,projection_ref)
            inventory=_inventory(root)
            history=_history(writer,plan_sha256,pilot_completion_ref,inventory)
            prefix=bind_lr_prefix(root,context=bound['context'],requests=requests[:-1],
                evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
                numerical_ref=plan['numerical_ref'],projection_ref=projection_ref,
                training_token_sha256=plan['training_token_sha256'])
            storage=bind_storage(writer.root,projection,pilots=history['records'],records=prefix['records'],
                rates=None,selection=None,stages=[history['chain_stage']],current_inventory=inventory,
                checkpoint_pairs={**history['checkpoint_pairs'],**prefix['checkpoint_pairs']},lock_fds=writer.lock_fds)
            _check_inventory(root,inventory)
            again,actual=_frozen(writer,slot,plan_sha256,projection_ref)
            _require(digest(again['files'])==digest(bound['files']) and digest(actual)==digest(requests),
                     'Frozen LR request provenance changed during preparation')
            _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],
                     'Complete remaining panel no longer fits')
            path=writer.publish_json(f'evidence/quality/{slot:02d}/request.json',requests[-1])
            ref=dict(path=f'quality/{slot:02d}/request.json',sha256=file_digest(path))
            expected=dict(inventory,**{ref['path']:dict(size=path.stat().st_size,sha256=ref['sha256'])})
            _check_inventory(root,expected)
            report=dict(kind=KIND,status='lr_request_pending_predispatch_review',slot=slot,
                plan_sha256=plan_sha256,context=bound['context'],projection_ref=projection_ref,
                pilot_completion_ref=pilot_completion_ref,attempt=marker_ref,request=ref,
                inventory=inventory,completed_records=prefix['records'],checkpoint_pairs=prefix['checkpoint_pairs'],
                storage=storage,dispatch_authorized=False,deletion_authorized=False,selection=None)
        guard_path=writer.root/guard_name
        report['guard']=dict(path=str(guard_path),sha256=file_digest(guard_path))
        _verify_guard(writer,slot,report['guard'])
        _frozen(writer,slot,plan_sha256,projection_ref);_check_inventory(root,expected)
        _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],
                 'Complete remaining panel storage changed after preparation')
        path=writer.publish_json('evidence/'+base+'.json',report)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('LR preparation failure publication failed: '+repr(error)[:1000])
        raise


def _verify_guard(writer,slot,ref):
    name=f'control/quality-lr-startup-{slot:02d}.json'
    _require(ref['path']==str(writer.root/name),'LR preparation guard path differs')
    guard=_document(writer.root,dict(path=name,sha256=ref['sha256']))
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(guard.get('kind')==KIND and guard.get('status')=='finished' and guard.get('reason') is None
             and guard.get('limits')==limits and type(guard.get('pid')) is int and guard['pid']>0,
             'Completed LR preparation guard required')
    sample=guard['observation'];elapsed=guard['elapsed_seconds']
    _require(type(elapsed) in (int,float) and 0<=elapsed<1800
             and set(sample)=={'peak_rss_bytes','free_bytes'}
             and all(type(v) is int and v>0 for v in sample.values())
             and violation(sample,elapsed,**limits) is None,'LR preparation guard observation invalid')


def verify_lr_request(writer,slot,reference):
    """Read-only idle predispatch recheck, under caller-owned locks/outer guard.

    Re-audits every prior member and the pilot receipt chain, then independently
    reconstructs occupied capacity and the complete remaining panel. No process
    artifacts may exist for the pending slot. This is not a child-entry gate.
    """
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 1<=slot<=8,
             'Exact later LR slot required')
    root=writer.root/'evidence';base=f'meta/quality-lr-startup-{slot:02d}'
    _locks(root,writer.lock_fds)
    _require(reference['path']==base+'.json','Exact LR startup receipt required')
    for suffix in ('-failure.json','-failure.json.tmp','.json.tmp','-attempt.json.tmp'):
        _require(not (root/(base+suffix)).exists(),'Failed or partial LR preparation cannot dispatch')
    report=_document(root,reference)
    _require(report.get('kind')==KIND and report.get('status')=='lr_request_pending_predispatch_review'
             and type(report.get('slot')) is int and report['slot']==slot
             and report.get('dispatch_authorized') is False and report.get('deletion_authorized') is False
             and 'selection' in report and report['selection'] is None,'LR startup identity/claims differ')
    _verify_guard(writer,slot,report['guard'])
    bound,requests=_frozen(writer,slot,report['plan_sha256'],report['projection_ref'])
    _require(digest(report['context'])==digest(bound['context']),'LR startup context changed')
    rq=report['request']
    _require(rq['path']==f'quality/{slot:02d}/request.json'
             and digest(_document(root,rq))==digest(requests[-1]),'Pending frozen request changed')
    attempt=dict(kind=KIND,status='started',slot=slot,plan_sha256=report['plan_sha256'],
        projection_ref=report['projection_ref'],pilot_completion_ref=report['pilot_completion_ref'],dispatch_authorized=False)
    _require(report['attempt']['path']==base+'-attempt.json'
             and digest(_document(root,report['attempt']))==digest(attempt),'LR startup attempt differs')
    inventory=dict(report['inventory'])
    _require(inventory[report['attempt']['path']]['sha256']==report['attempt']['sha256'],'Attempt inventory differs')
    for ref in (rq,reference):
        path=_path(root,ref['path']);_require(ref['path'] not in inventory,'Startup output overlaps prior inventory')
        inventory[ref['path']]=dict(size=path.stat().st_size,sha256=ref['sha256'])
    _check_inventory(root,inventory)
    history=_history(writer,report['plan_sha256'],report['pilot_completion_ref'],inventory)
    plan=_document(root,dict(path='meta/plan.json',sha256=report['plan_sha256']))
    prefix=bind_lr_prefix(root,context=bound['context'],requests=requests[:-1],pending_request=requests[-1],
        evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
        numerical_ref=plan['numerical_ref'],projection_ref=report['projection_ref'],
        training_token_sha256=plan['training_token_sha256'])
    _require(digest(prefix['records'])==digest(report['completed_records'])
             and digest(prefix['checkpoint_pairs'])==digest(report['checkpoint_pairs']),
             'Saved LR prefix or checkpoint bindings differ')
    projection=_document(root,report['projection_ref'])
    state=remaining_projection(projection,history['records'],prefix['records'],None,None,1)
    census=snapshot(writer.root);weights=set();inodes=set();occupied=0;logical=0
    for pair in prefix['checkpoint_pairs'].values():
        for entry in pair.values():
            path=_path(root,entry['path']);st=path.stat();inode=(st.st_dev,st.st_ino)
            _require(st.st_nlink==1 and inode not in inodes
                     and inventory[entry['path']]==dict(size=entry['size'],sha256=entry['sha256']),
                     'LR checkpoint census alias or binding differs')
            inodes.add(inode);weights.add(entry['path']);logical+=st.st_size
            occupied+=min(st.st_size,st.st_blocks*512)
    actual={n.removeprefix('evidence/') for n,v in census['files'].items() if v['group'].endswith('/weights')}
    _require(actual==weights and census['checkpoint_logical_bytes']==logical,'Unexpected checkpoint before LR dispatch')
    needed=required_free(state,occupied);storage=report['storage']
    expected=dict(required_free_bytes=needed,occupied_checkpoint_capacity_bytes=occupied,
                  checkpoint_logical_bytes=logical,cleanup_boundaries_verified=1)
    _require(all(type(storage.get(k)) is int and storage[k]==v for k,v in expected.items())
             and storage.get('status')=='remaining_campaign_storage_verified'
             and digest(storage['projection'])==digest(state)
             and storage.get('current_inventory_sha256')==digest(report['inventory'])
             and all(storage.get(k) is True for k in ('storage_projection_fits','current_phase_space_verified'))
             and all(storage.get(k) is False for k in ('dispatch_authorized','frozen_plan_verified','audit_semantics_verified')),
             'Saved full-panel storage evidence differs')
    _require(shutil.disk_usage(writer.root).free>=needed,'Complete remaining panel no longer fits')
    again,actual_requests=_frozen(writer,slot,report['plan_sha256'],report['projection_ref'])
    _require(digest(again['files'])==digest(bound['files']) and digest(actual_requests)==digest(requests),
             'Frozen LR provenance changed during predispatch')
    _verify_guard(writer,slot,report['guard']);_check_inventory(root,inventory)
    _locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=needed,'Remaining storage changed before handoff')
    return dict(required_free_bytes=needed,request_sha256=rq['sha256'],startup_ref=reference,
                pilot_history_verified=True,lr_prefix_verified=True,lr_request_verified=True,dispatch_authorized=False)

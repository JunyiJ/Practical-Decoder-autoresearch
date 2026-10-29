"""Seal a fully audited idle LR, rank or first fresh request for a future live handoff.

No child entry, process launch, selection or cleanup. The seal is immutable input
for future integration, not permission to dispatch. All existing idle scientific,
receipt, prefix and complete remaining-storage checks run before publication.
"""
import os
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_receipts import _document
from .tokenizer_wide_hybrid_scores import _require
from . import tokenizer_wide_hybrid_quality_startup_v2 as first
from . import tokenizer_wide_hybrid_quality_lr_startup_v2 as later
from .cpu_prerequisite_guard import violation

KIND='wide_hybrid_quality_readiness_v2'


class _ReadinessGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot):
        super().__init__(writer,slot,'readiness',1800)
        self.name=f'control/quality-readiness-{slot:02d}.json'


def _controls(writer,excluded):
    """Bind existing control bytes; only this seal's own running guard is exempt.

    Call outside a running parent dispatcher: every other existing control file
    must stay unchanged. A future parent may create its separate control files.
    """
    snapshot(writer.root)
    result={}
    for path in sorted((writer.root/'control').rglob('*')):
        if path.is_dir():continue
        name=str(path.relative_to(writer.root))
        if name in {excluded,'control/.metadata.lock'}:continue
        before=path.stat();sha=file_digest(path);after=path.stat()
        _require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')) and after.st_nlink==1,'Control changed during readiness binding')
        result[name]=dict(size=after.st_size,sha256=sha)
    return result


def _bound(writer,slot,report):
    if slot==0:
        bound,request=first._frozen(writer,report['plan_sha256'],report['projection_ref'])
    elif slot<9:
        bound,requests=later._frozen(writer,slot,report['plan_sha256'],report['projection_ref'])
        request=requests[-1]
    elif slot<15:
        # Rank history depends on readiness helpers; load it after module setup.
        from . import tokenizer_wide_hybrid_quality_rank_startup_v2 as rank
        bound,requests=rank._frozen(writer,slot,report['plan_sha256'],report['projection_ref'],report['rates'])
        request=requests[-1]
    else:
        from . import tokenizer_wide_hybrid_first_confirmation_startup_v2 as fresh
        _require(slot in (15,16,17,18,19,20,21),'Only fresh slots15..21 have startup integration')
        bound,requests=fresh._frozen(writer,slot,report['plan_sha256'],report['projection_ref'],report['rates'],report['selection'])
        request=requests[-1]
    _require(digest(_document(writer.root/'evidence',report['request']))==digest(request),
             'Readiness request differs from independently frozen option')
    projection=_document(writer.root/'evidence',report['projection_ref'])
    allowance=allocations(projection)[request['spec']['family'],request['spec']['divisor']]
    return bound,request,allowance


def seal_quality_readiness(writer,slot,startup_ref):
    """Own shared locks and inspect workers first; a failed seal is never retried."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<=21,'Exact supported quality writer/slot required')
    phase='lr' if slot<9 else ('rank' if slot<15 else 'confirm')
    expected='meta/quality-first-startup.json' if slot==0 else f'meta/quality-{phase}-startup-{slot:02d}.json'
    _require(startup_ref['path']==expected,'Exact startup reference required')
    evidence=writer.root/'evidence';base=f'meta/quality-readiness-{slot:02d}'
    guard_name=f'control/quality-readiness-{slot:02d}.json'
    with writer._serialized():
        _locks(evidence,writer.lock_fds)
        _require(not (writer.root/guard_name).exists() and not (writer.root/(guard_name+'.tmp')).exists()
            and not any((evidence/(base+s)).exists() for s in ('.json','.json.tmp','-failure.json','-failure.json.tmp')),
            'Existing readiness attempt cannot be retried')
    try:
        with _ReadinessGuard(writer,slot):
            report=_document(evidence,startup_ref)
            inventory=_inventory(evidence);controls=_controls(writer,guard_name)
            bound,request,allowance=_bound(writer,slot,report)
            if slot==0:
                checked=first.verify_first_quality_request(writer,startup_ref)
                required=('pilot_history_verified','first_quality_request_verified')
            elif slot<9:
                checked=later.verify_lr_request(writer,slot,startup_ref)
                required=('pilot_history_verified','lr_request_verified','lr_prefix_verified')
            elif slot<15:
                from . import tokenizer_wide_hybrid_quality_rank_startup_v2 as rank
                checked=rank.verify_rank_request(writer,slot,startup_ref)
                required=('pilot_history_verified','lr_history_verified','rank_prefix_verified','rank_request_verified')
            elif slot==15:
                from . import tokenizer_wide_hybrid_first_confirmation_startup_v2 as fresh
                checked=fresh.verify_first_confirmation_request(writer,slot,startup_ref)
                required=('pilot_history_verified','rank_history_verified','selected_screen_prefix_verified','first_confirmation_request_verified')
            else:
                if slot==16:
                    from . import tokenizer_wide_hybrid_confirmation_startup_v2 as fresh
                elif slot==17:
                    from . import tokenizer_wide_hybrid_later_confirmation_startup_v2 as fresh
                else:
                    from . import tokenizer_wide_hybrid_remaining_confirmation_startup_v2 as fresh
                guard_ref=dict(path=guard_name,sha256=file_digest(writer.root/guard_name))
                checked=fresh.verify_confirmation_for_readiness(writer,slot,startup_ref,guard_ref)
                required=('prior_transport_verified','selected_and_fresh_prefix_verified','confirmation_request_verified')
            _require(all(checked.get(k) is True for k in required)
                and checked.get('dispatch_authorized') is False
                and checked.get('request_sha256')==report['request']['sha256'],
                'Full idle predispatch review required')
            needed=checked['required_free_bytes']
            _require(type(needed) is int and needed==report['storage']['required_free_bytes'],
                     'Independently reconstructed storage requirement differs')
            _check_inventory(evidence,inventory)
            again,actual,actual_allowance=_bound(writer,slot,report)
            _require(digest(again)==digest(bound) and digest(actual)==digest(request) and actual_allowance==allowance,
                     'Frozen readiness inputs changed during review')
            _require(_controls(writer,guard_name)==controls,'Audited controls changed during review')
            _locks(evidence,writer.lock_fds)
            _require(shutil.disk_usage(writer.root).free>=needed,'Full remaining panel no longer fits')
        guard_path=writer.root/guard_name;guard_ref=dict(path=guard_name,sha256=file_digest(guard_path))
        guard=_document(writer.root,guard_ref)
        limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
        _require(guard.get('kind')==KIND and guard.get('status')=='finished' and guard.get('reason') is None
            and guard.get('pid')==os.getpid() and guard.get('limits')==limits
            and violation(guard['observation'],guard['elapsed_seconds'],**limits) is None,
            'Finished readiness guard required')
        _check_inventory(evidence,inventory)
        final,actual,actual_allowance=_bound(writer,slot,report)
        _require(digest(final)==digest(bound) and digest(actual)==digest(request) and actual_allowance==allowance
            and _controls(writer,guard_name)==controls,'Readiness inputs changed after guard completion')
        controls[guard_name]=dict(size=guard_path.stat().st_size,sha256=guard_ref['sha256'])
        _locks(evidence,writer.lock_fds)
        _require(shutil.disk_usage(writer.root).free>=needed,'Full remaining storage changed before sealing')
        payload=dict(kind=KIND,status='idle_review_sealed_pending_child_integration',slot=slot,
            parent_pid=os.getpid(),startup_ref=startup_ref,request=report['request'],
            plan_sha256=report['plan_sha256'],projection_ref=report['projection_ref'],context=bound['context'],
            frozen_files=bound['files'],inventory=inventory,control_inventory=controls,guard=guard_ref,
            predispatch_review=checked,required_free_bytes=needed,output_limit_bytes=allowance,
            dispatch_authorized=False,deletion_authorized=False)
        from .tokenizer_wide_hybrid_inventory_delta import encode_inventory
        anchor=startup_ref
        if slot in (16,17):
            prior=_document(evidence,report['prior_dispatch_ref'])
            anchor=prior['review']['readiness']
            _require(anchor['path']==f'meta/quality-readiness-{slot-1:02d}.json','Exact previous readiness inventory anchor required')
        elif slot in (18,19,20,21):
            # Reuse a stable earlier seal; one link per fresh slot exceeds MAX_DEPTH.
            prior=_document(evidence,report['prior_dispatch_ref'])
            anchors=[ref for ref in prior['prior_dispatches'] if ref['path']=='meta/quality-dispatch-16.json']
            _require(len(anchors)==1,'Exact stable earlier dispatch required')
            previous=anchors[0]
            original=_document(evidence,previous)
            anchor=original['review']['readiness']
            _require(anchor['path']=='meta/quality-readiness-16.json','Exact stable slot16 inventory anchor required')
        wire=encode_inventory(evidence,payload,anchor)
        path=writer.publish_json('evidence/'+base+'.json',wire)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            startup_ref=startup_ref,error=repr(exc)[:1000],retry_allowed=False,dispatch_authorized=False))
        except BaseException as error:exc.add_note('Readiness failure publication failed: '+repr(error)[:1000])
        raise

"""Exclusive slot16 startup receipt and independent idle predispatch review.

Caller owns both locks, inspects workers and holds an outer resource guard over
the entire call, including final checks. No readiness, child, retry or later slot.
"""
import math
import os
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_rank_selection_v2 import _storage
from .cpu_prerequisite_guard import violation
from . import tokenizer_wide_hybrid_confirmation_append_v2 as append

KIND=append.KIND
BASE='meta/quality-confirm-startup-16'
RECEIPT=BASE+'.json'
NONCLAIMS=('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized')


class _PreparationGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,16,'startup',1800);self.name=append.GUARD


def _guard_ref(writer):
    return dict(path=append.GUARD,sha256=file_digest(writer.root/append.GUARD))


def _payload(saved,preparation,inventory,controls,seal,storage):
    review=saved['review']
    return dict(kind=KIND,status='confirmation_pending_predispatch_review',slot=16,parent_pid=os.getpid(),
        plan_sha256=saved['plan_sha256'],projection_ref=saved['projection_ref'],rank_completion_ref=saved['rank_completion_ref'],
        prior_dispatch_ref=preparation['prior_dispatch_ref'],attempt=preparation['attempt_ref'],
        guard=preparation['guard_ref'],request=preparation['pending_request_ref'],
        context=seal['context'],frozen_files_sha256=digest(seal['frozen_files']),
        completed_records_sha256=review['completed_records_sha256'],checkpoint_pairs_sha256=review['checkpoint_pairs_sha256'],
        stages_sha256=review['stages_sha256'],rates=review['rates'],selection=review['selection'],
        audited_indices=review['audited_indices'],inventory_sha256=digest(inventory),controls_sha256=digest(controls),
        storage=storage,**dict.fromkeys(NONCLAIMS,False))


def _receipt_binding(writer,reference,saved,preparation,inventory,controls):
    """Exact receipt append; the caller must still complete full history replay."""
    _require(reference['path']==RECEIPT and preparation['pending_request_ref'] is not None,
             'Exact slot16 startup receipt and pending request required')
    root=writer.root/'evidence';report=_document(root,reference)
    guard=preparation['guard_ref']
    append.history.acceptance._guard(writer,guard,KIND,os.getpid(),1800)
    before={n:v for n,v in inventory.items() if n!=RECEIPT}
    seal=_document(root,saved['review']['readiness'])
    prior=dict(path='meta/quality-dispatch-15.json',sha256=file_digest(root/'meta/quality-dispatch-15.json'))
    expected=_payload(saved,dict(preparation,prior_dispatch_ref=prior),before,controls,seal,report['storage'])
    _require(digest(report)==digest(expected),'Complete slot16 startup claims or bindings differ')
    _storage(report['storage'],saved['review']['storage'],before)
    return report,dict(size=_path(root,RECEIPT).stat().st_size,sha256=reference['sha256'])


def prepare_confirmation_request(writer,slot,plan_sha256,*,projection_ref,rank_completion_ref,prior_dispatch_ref):
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and slot==16,'Exact slot16 startup required')
    _sha(plan_sha256);root=writer.root/'evidence';_locks(root,writer.lock_fds)
    seen=snapshot(writer.root)['files']
    _require(not (root/'quality/16').exists() and not (root/'quality/16').is_symlink()
        and not any(n.startswith('evidence/'+BASE) or n in (append.GUARD,append.GUARD+'.tmp') for n in seen),
        'Existing slot16 preparation cannot be retried')
    args=dict(writer=writer,plan_sha256=plan_sha256,completion_ref=prior_dispatch_ref,
              projection_ref=projection_ref,rank_completion_ref=rank_completion_ref)
    try:
        with _PreparationGuard(writer):
            attempt=dict(kind=KIND,status='started',slot=16,parent_pid=os.getpid(),plan_sha256=plan_sha256,
                projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,prior_dispatch_ref=prior_dispatch_ref,
                dispatch_authorized=False,deletion_authorized=False)
            path=writer.publish_json('evidence/'+append.ATTEMPT,attempt)
            args.update(attempt_ref=dict(path=append.ATTEMPT,sha256=file_digest(path)),guard_ref=_guard_ref(writer))
            before=append.verify_first_confirmation_preparation(**args)
            bound,requests=append.startup._frozen(writer,16,plan_sha256,projection_ref,before['rates'],before['selection'])
            path=writer.publish_json('evidence/'+append.REQUEST,requests[-1])
            args['pending_request_ref']=dict(path=append.REQUEST,sha256=file_digest(path))
        args['guard_ref']=_guard_ref(writer)
        controls=append.history.acceptance.readiness._controls(writer,'not-a-control')
        checked=append.verify_first_confirmation_preparation(**args)
        _require(append.history.acceptance.readiness._controls(writer,'not-a-control')==controls,
                 'Slot16 controls changed during final preparation replay')
        _require(all(checked[k]==before[k] for k in ('records','checkpoint_pairs','stages','rates','selection','audited_indices')),
                 'Completed history changed during slot16 preparation')
        saved=_document(root,prior_dispatch_ref);seal=_document(root,saved['review']['readiness'])
        inventory=append.history.acceptance._inventory(root)
        report=_payload(saved,dict(args,prior_dispatch_ref=prior_dispatch_ref),inventory,controls,seal,checked['storage'])
        _require(report['storage']['current_inventory_sha256']==digest(inventory),
                 'Slot16 preparation inventory changed after replay')
        again,actual=append.startup._frozen(writer,16,plan_sha256,projection_ref,checked['rates'],checked['selection'])
        _require(digest(again)==digest(bound) and actual==requests,'Frozen slot16 request changed before receipt')
        append.history.acceptance._check_inventory(root,inventory)
        _require(append.history.acceptance.readiness._controls(writer,'not-a-control')==controls,
                 'Slot16 controls changed before receipt')
        _locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
                 'STOP prevents slot16 receipt publication')
        _require(shutil.disk_usage(writer.root).free>=checked['storage']['required_free_bytes'],
                 'Full remaining capacity changed before slot16 receipt')
        path=writer.publish_json('evidence/'+RECEIPT,report)
        return dict(path=RECEIPT,sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(kind=KIND,status='failed',slot=16,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Slot16 failure publication failed: '+repr(error)[:1000])
        raise


def verify_confirmation_request(writer,slot,reference):
    """Reconstruct the complete same-parent slot16 receipt; never launch a child."""
    return _verify_request(writer,slot,reference)


def _readiness_control(writer,reference):
    from .tokenizer_wide_hybrid_quality_readiness_v2 import KIND as readiness_kind
    _require(reference['path']=='control/quality-readiness-16.json','Exact slot16 readiness guard required')
    guard=_document(writer.root,reference);elapsed=guard.get('elapsed_seconds');sample=guard.get('observation',{})
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(set(guard)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
        and guard['kind']==readiness_kind and guard['status'] in ('starting','finished') and guard['reason'] is None
        and type(guard['pid']) is int and guard['pid']==os.getpid() and guard['limits']==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<1800
        and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
        and all(type(v) is int and v>0 for v in sample.values()) and violation(sample,elapsed,**limits) is None,
        'Bounded current-parent readiness guard required')
    return reference['path'],dict(size=_path(writer.root,reference['path']).stat().st_size,sha256=reference['sha256'])


def verify_confirmation_for_readiness(writer,slot,reference,guard_ref):
    """Explicit single-guard append; the ordinary idle reader stays strict."""
    _require(guard_ref is not None,'Explicit readiness guard required')
    return _verify_request(writer,slot,reference,readiness_guard_ref=guard_ref)


def _verify_request(writer,slot,reference,*,readiness_guard_ref=None):
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and slot==16,'Exact slot16 predispatch required')
    _require(reference['path']==RECEIPT,'Exact slot16 startup reference required')
    report=_document(writer.root/'evidence',reference)
    preparation=dict(attempt_ref=report['attempt'],guard_ref=report['guard'],pending_request_ref=report['request'])
    checked=append.history._verify(writer,report['plan_sha256'],report['prior_dispatch_ref'],
        projection_ref=report['projection_ref'],rank_completion_ref=report['rank_completion_ref'],
        preparation=preparation,startup_ref=reference,readiness_guard_ref=readiness_guard_ref)
    return dict(startup_ref=reference,request_sha256=report['request']['sha256'],
        required_free_bytes=checked['storage']['required_free_bytes'],prior_transport_verified=True,
        selected_and_fresh_prefix_verified=True,confirmation_request_verified=True,dispatch_authorized=False)

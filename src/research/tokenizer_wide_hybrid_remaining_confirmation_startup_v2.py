"""Exclusive slot18/19 startup receipts and independent idle predispatch review.

Caller owns both locks, inspects workers and holds an outer resource guard over
the entire call, including final checks. No readiness, child, retry or later slot.
"""
import math
import os
import shutil

from .cpu_prerequisite_guard import violation
from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_rank_selection_v2 import _storage
from . import tokenizer_wide_hybrid_remaining_confirmation_append_v2 as append

KIND=append.KIND
PATHS=append.paths(18)
BASE=PATHS['attempt'].removesuffix('-attempt.json')
RECEIPT=BASE+'.json'
NONCLAIMS=('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized')


def _names(slot):
    _require(type(slot) is int and slot in (18,19),'Exact slot18 or slot19 startup required')
    names=append.paths(slot);base=names['attempt'].removesuffix('-attempt.json')
    return dict(names,base=base,receipt=base+'.json')


def _next_slot(saved):
    _require(type(saved.get('slot')) is int and saved['slot'] in (17,18),'Exact integrated predecessor required')
    return saved['slot']+1


class _PreparationGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot=18):
        super().__init__(writer,slot,'startup',1800);self.name=_names(slot)['guard']


def _guard_ref(writer,slot=18):
    name=_names(slot)['guard']
    return dict(path=name,sha256=file_digest(writer.root/name))


def _payload(saved,preparation,inventory,controls,seal,storage):
    review=saved['review'];slot=_next_slot(saved)
    return dict(kind=KIND,status='confirmation_pending_predispatch_review',slot=slot,parent_pid=os.getpid(),
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
    slot=_next_slot(saved);names=_names(slot);receipt=names['receipt']
    _require(reference['path']==receipt and preparation['pending_request_ref'] is not None,
             'Exact supported startup receipt and pending request required')
    root=writer.root/'evidence';report=_document(root,reference)
    guard=preparation['guard_ref']
    append.history.acceptance._guard(writer,guard,KIND,os.getpid(),1800)
    before={n:v for n,v in inventory.items() if n!=receipt}
    seal=_document(root,saved['review']['readiness'])
    prior=dict(path=names['prior'],sha256=file_digest(root/names['prior']))
    expected=_payload(saved,dict(preparation,prior_dispatch_ref=prior),before,controls,seal,report['storage'])
    _require(digest(report)==digest(expected),'Complete startup claims or bindings differ')
    _storage(report['storage'],saved['review']['storage'],before)
    return report,dict(size=_path(root,receipt).stat().st_size,sha256=reference['sha256'])


def prepare_confirmation_request(writer,slot,plan_sha256,*,projection_ref,rank_completion_ref,prior_dispatch_ref):
    _require(isinstance(writer,MetadataWriter),'Metadata writer required')
    names=_names(slot);base=names['base'];receipt=names['receipt']
    _sha(plan_sha256);root=writer.root/'evidence';_locks(root,writer.lock_fds)
    seen=snapshot(writer.root)['files']
    _require(not (root/f'quality/{slot:02d}').exists() and not (root/f'quality/{slot:02d}').is_symlink()
        and not any(n.startswith('evidence/'+base) or n in (names['guard'],names['guard']+'.tmp') for n in seen),
        'Existing startup preparation cannot be retried')
    args=dict(writer=writer,slot=slot,plan_sha256=plan_sha256,completion_ref=prior_dispatch_ref,
              projection_ref=projection_ref,rank_completion_ref=rank_completion_ref)
    try:
        with _PreparationGuard(writer,slot):
            attempt=dict(kind=KIND,status='started',slot=slot,parent_pid=os.getpid(),plan_sha256=plan_sha256,
                projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,prior_dispatch_ref=prior_dispatch_ref,
                dispatch_authorized=False,deletion_authorized=False)
            path=writer.publish_json('evidence/'+names['attempt'],attempt)
            args.update(attempt_ref=dict(path=names['attempt'],sha256=file_digest(path)),guard_ref=_guard_ref(writer,slot))
            before=append.verify_confirmation_preparation(**args)
            bound,requests=append._frozen(writer,slot,plan_sha256,projection_ref,before['rates'],before['selection'])
            path=writer.publish_json('evidence/'+names['request'],requests[-1])
            args['pending_request_ref']=dict(path=names['request'],sha256=file_digest(path))
        args['guard_ref']=_guard_ref(writer,slot)
        controls=append.history.acceptance.readiness._controls(writer,'not-a-control')
        checked=append.verify_confirmation_preparation(**args)
        _require(append.history.acceptance.readiness._controls(writer,'not-a-control')==controls,
                 'Startup controls changed during final preparation replay')
        _require(all(checked[k]==before[k] for k in ('records','checkpoint_pairs','stages','rates','selection','audited_indices')),
                 'Completed history changed during startup preparation')
        saved=_document(root,prior_dispatch_ref);seal=_document(root,saved['review']['readiness'])
        inventory=append.history.acceptance._inventory(root)
        report=_payload(saved,dict(args,prior_dispatch_ref=prior_dispatch_ref),inventory,controls,seal,checked['storage'])
        _require(report['storage']['current_inventory_sha256']==digest(inventory),
                 'Startup preparation inventory changed after replay')
        again,actual=append._frozen(writer,slot,plan_sha256,projection_ref,checked['rates'],checked['selection'])
        _require(digest(again)==digest(bound) and actual==requests,'Frozen startup request changed before receipt')
        append.history.acceptance._check_inventory(root,inventory)
        _require(append.history.acceptance.readiness._controls(writer,'not-a-control')==controls,
                 'Startup controls changed before receipt')
        _locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
                 'STOP prevents startup receipt publication')
        _require(shutil.disk_usage(writer.root).free>=checked['storage']['required_free_bytes'],
                 'Full remaining capacity changed before startup receipt')
        path=writer.publish_json('evidence/'+receipt,report)
        return dict(path=receipt,sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Startup failure publication failed: '+repr(error)[:1000])
        raise


def verify_confirmation_request(writer,slot,reference):
    return _verify_request(writer,slot,reference)


def _readiness_control(writer,reference,slot=18):
    from .tokenizer_wide_hybrid_quality_readiness_v2 import KIND as readiness_kind
    _names(slot)
    _require(reference['path']==f'control/quality-readiness-{slot:02d}.json','Exact supported readiness guard required')
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
    _names(slot)
    _require(guard_ref is not None,'Explicit readiness guard required')
    return _verify_request(writer,slot,reference,readiness_guard_ref=guard_ref)


def _verify_request(writer,slot,reference,*,readiness_guard_ref=None):
    """Independently reconstruct the same-parent supported receipt; no launch grant."""
    _require(isinstance(writer,MetadataWriter),'Metadata writer required')
    names=_names(slot)
    _require(reference['path']==names['receipt'],'Exact supported startup reference required')
    report=_document(writer.root/'evidence',reference)
    preparation=dict(attempt_ref=report['attempt'],guard_ref=report['guard'],pending_request_ref=report['request'])
    replay=append.history;options=dict(preparation=preparation,startup_ref=reference,readiness_guard_ref=readiness_guard_ref)
    if slot==19:
        from . import tokenizer_wide_hybrid_remaining_confirmation_history_v2 as replay
    checked=replay._verify(writer,slot-1,report['plan_sha256'],report['prior_dispatch_ref'],
        projection_ref=report['projection_ref'],rank_completion_ref=report['rank_completion_ref'],
        **options)
    return dict(startup_ref=reference,request_sha256=report['request']['sha256'],
        required_free_bytes=checked['storage']['required_free_bytes'],prior_transport_verified=True,
        selected_and_fresh_prefix_verified=True,confirmation_request_verified=True,dispatch_authorized=False)

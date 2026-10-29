"""One immutable request for the finite three-seed inference panel.

Preparation only: no child process, inference measurement, retry or adoption.
Caller owns both locks, inspects workers and holds an outer resource guard for
idle binding and final checks. The internal guard covers the append transition.
The ordinary slot23 history reader remains strict; only this exact prepared
state is admitted through its private scoped replay.
"""
import math
import os
import shutil

from .protocol import digest,file_digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_retention_receipts import _document,_path
from .tokenizer_wide_hybrid_scores import _require
from . import tokenizer_wide_hybrid_inference_gate_v2 as gate

KIND='wide_hybrid_inference_preparation_v2'
REQUEST='inference/3511/panel-request.json'
GUARD='control/inference-preparation.json'
FAILURE='inference/3511/preparation-failure.json'
LIMITS=dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=1800)


class _PreparationGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,3511,'inference-preparation',1800)
        self.name=GUARD


def _payload(bound,projection_ref,rank_completion_ref,parent_pid):
    return dict(kind=KIND,status='prepared_pending_serial_integration',parent_pid=parent_pid,
        plan_sha256=bound['plan_sha256'],completion_ref=bound['completion_ref'],
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,
        seed_order=list(gate.SEEDS),contract=bound['contract'],selection=bound['selection'],
        tokenizer_sha256=digest(bound['tokenizer']),token_sha256=bound['token_sha256'],
        records_sha256=bound['records_sha256'],checkpoint_pairs_sha256=bound['checkpoint_pairs_sha256'],
        retention_sha256=bound['retention_sha256'],quality_inventory_sha256=bound['inventory_sha256'],
        quality_controls_sha256=bound['controls_sha256'],triples_sha256=digest(bound['triples']),
        adopted=False,dispatch_authorized=False,deletion_authorized=False,next_stage_authorized=False)


def _guard(writer,parent_pid,*,finished=False):
    value=_document(writer.root,dict(path=GUARD,sha256=file_digest(_path(writer.root,GUARD))))
    elapsed=value.get('elapsed_seconds');sample=value.get('observation',{})
    _require(set(value)=={'kind','status','pid','limits','elapsed_seconds','observation','reason'}
        and value['kind']==KIND and value['status'] in (('finished',) if finished else ('starting','finished'))
        and value['reason'] is None and type(value['pid']) is int and value['pid']==parent_pid
        and digest(value['limits'])==digest(LIMITS)
        and type(elapsed) in (float,int) and math.isfinite(elapsed) and 0<=elapsed<1800
        and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
        and all(type(v) is int and v>0 for v in sample.values())
        and violation(sample,elapsed,**LIMITS) is None,'Exact bounded inference preparation guard required')
    return value


def _request(writer,request_ref,plan_sha256,completion_ref,projection_ref,rank_completion_ref):
    """Validate immutable request semantics; namespace scope is checked separately."""
    _require(isinstance(writer,MetadataWriter) and set(request_ref)=={'path','sha256'}
        and request_ref['path']==REQUEST,'Exact finite inference request reference required')
    root=writer.root/'evidence';_locks(root,writer.lock_fds);snapshot(writer.root)
    request=_document(root,request_ref)
    saved=_document(root,completion_ref);review=saved['review']
    plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha256))
    expected=dict(kind=KIND,status='prepared_pending_serial_integration',plan_sha256=plan_sha256,
        completion_ref=completion_ref,projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,
        seed_order=list(gate.SEEDS),contract=dict(gate.CONTRACT,version=2,
            tokenizer_sha256=digest(plan['tokenizer']),token_sha256=plan['training_token_sha256']),
        selection=review['selection'],tokenizer_sha256=digest(plan['tokenizer']),token_sha256=plan['training_token_sha256'],
        records_sha256=review['completed_records_sha256'],checkpoint_pairs_sha256=review['checkpoint_pairs_sha256'],
        retention_sha256=review['stages_sha256'],adopted=False,dispatch_authorized=False,
        deletion_authorized=False,next_stage_authorized=False)
    extra={'parent_pid','quality_inventory_sha256','quality_controls_sha256','triples_sha256'}
    _require(set(request)==set(expected)|extra and digest({k:request[k] for k in expected})==digest(expected)
        and type(request['parent_pid']) is int and request['parent_pid']>1,'Exact unpromoted finite inference request required')
    for key in extra-{'parent_pid'}:gate._sha(request[key])
    guard=_guard(writer,request['parent_pid'])
    return request,guard


def _scope(writer,request_ref,plan_sha256,completion_ref,projection_ref,rank_completion_ref):
    """Validate only one exact request/guard append, never a caller allowlist."""
    request,guard=_request(writer,request_ref,plan_sha256,completion_ref,projection_ref,rank_completion_ref)
    root=writer.root/'evidence';review=_document(root,completion_ref)['review']
    inventory=gate.history.acceptance._inventory(root)
    controls=gate.history.acceptance.readiness._controls(writer,'not-a-control')
    evidence={REQUEST:inventory[REQUEST]};added_controls={GUARD:controls[GUARD]}
    _require({n for n in inventory if n.startswith('inference/')}=={REQUEST},
             'Only the shared inference panel request may append; no worker outputs or later seed')
    original={n:v for n,v in inventory.items() if n!=REQUEST}
    original_controls={n:v for n,v in controls.items() if n!=GUARD}
    _require(digest(original)==request['quality_inventory_sha256']
        and digest(original_controls)==request['quality_controls_sha256'],
        'Original quality evidence or controls changed during inference preparation')
    _require(digest({n:v for n,v in original.items() if n!=completion_ref['path']})==review['evidence_inventory_sha256'],
             'Original completed slot23 inventory differs')
    return dict(request=request,guard=guard,evidence=evidence,controls=added_controls)


def _history_scope(writer,request_ref,plan_sha256):
    """Explicit lower-history binding, independently validated at each consumer."""
    _require(request_ref.get('path')==REQUEST,'Exact inference history scope required')
    request=_document(writer.root/'evidence',request_ref)
    _require(request['plan_sha256']==plan_sha256,'Inference history plan differs')
    return _scope(writer,request_ref,plan_sha256,request['completion_ref'],request['projection_ref'],request['rank_completion_ref'])


def verify_prepared_inference(writer,request_ref,*,finished=True):
    """Reaudit all retained science/transport and all three triples, read-only."""
    _require(request_ref.get('path')==REQUEST,'Exact finite inference request required')
    request=_document(writer.root/'evidence',request_ref)
    args={k:request[k] for k in ('plan_sha256','completion_ref','projection_ref','rank_completion_ref')}
    _scope(writer,request_ref,**args)
    bound=gate._bind(writer,**args,inference_request_ref=request_ref)
    _require(digest(request)==digest(_payload(bound,args['projection_ref'],args['rank_completion_ref'],request['parent_pid'])),
             'Prepared inference triple/input bindings differ from full independent replay')
    _guard(writer,request['parent_pid'],finished=finished)
    return dict(bound,request_ref=request_ref,preparation_verified=True)


def prepare_inference(writer,*,plan_sha256,completion_ref,projection_ref,rank_completion_ref):
    """Append one shared panel request; never prepare a second attempt."""
    _require(isinstance(writer,MetadataWriter),'Locked metadata writer required')
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    seen=snapshot(writer.root)['files']
    _require(not any(n.startswith('evidence/inference/') or n.startswith('control/inference-')
        or n.startswith('control/process-inference-') for n in seen),'Existing inference attempt cannot be retried')
    args=dict(plan_sha256=plan_sha256,completion_ref=completion_ref,projection_ref=projection_ref,rank_completion_ref=rank_completion_ref)
    # The caller's outer guard covers this immutable idle replay before the new
    # internal control exists; no old audit is weakened to admit that control.
    bound=gate.bind_inference(writer,**args)
    try:
        with _PreparationGuard(writer):
            path=writer.publish_json('evidence/'+REQUEST,_payload(bound,projection_ref,rank_completion_ref,os.getpid()))
            ref=dict(path=REQUEST,sha256=file_digest(path))
            # Full scoped replay, not a success flag copied from idle binding.
            checked=verify_prepared_inference(writer,ref,finished=False)
        _guard(writer,os.getpid(),finished=True)
        gate.recheck(bound['files']);_scope(writer,ref,**args);_locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink()
            and shutil.disk_usage(writer.root).free>=checked['history']['storage']['required_free_bytes'],
            'STOP or capacity changed before preparation completion')
        return ref
    except BaseException as exc:
        try:writer.publish_json('evidence/'+FAILURE,dict(kind=KIND,status='failed',error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as publication:exc.add_note('Failure evidence publication failed: '+repr(publication)[:1000])
        raise

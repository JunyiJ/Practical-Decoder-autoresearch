"""Explicit slot17 preparation append contract around compact slot16 replay.

Caller owns both locks, inspects workers and holds an outer resource guard. Only
one current-parent bounded startup guard, its attempt and optional independently
frozen request may append. No startup receipt, readiness, child or later slot is
allowed, and a pending request earns no completed-record/checkpoint credit.
"""
import math
import os

from .protocol import digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_retention_receipts import _document,_path
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_first_confirmation_startup_v2 import _frozen
from . import tokenizer_wide_hybrid_confirmation_history_v2 as history

KIND=history.startup.KIND
APPEND_KIND='wide_hybrid_later_confirmation_append_v2'
ATTEMPT='meta/quality-confirm-startup-17-attempt.json'
GUARD='control/quality-confirm-startup-17.json'
REQUEST='quality/17/request.json'


def _append(writer,plan_sha,completion_ref,projection_ref,rank_completion_ref,saved,preparation):
    _require(completion_ref['path']=='meta/quality-dispatch-16.json','Exact completed slot16 reference required')
    _require(type(preparation) is dict and set(preparation)=={'attempt_ref','guard_ref','pending_request_ref'},
             'Exact slot17 preparation references required')
    root=writer.root/'evidence';attempt_ref=preparation['attempt_ref'];guard_ref=preparation['guard_ref']
    _require(attempt_ref['path']==ATTEMPT and guard_ref['path']==GUARD,'Exact slot17 preparation paths required')
    attempt=_document(root,attempt_ref);guard=_document(writer.root,guard_ref);pid=os.getpid()
    expected=dict(kind=KIND,status='started',slot=17,parent_pid=pid,plan_sha256=plan_sha,
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,prior_dispatch_ref=completion_ref,
        dispatch_authorized=False,deletion_authorized=False)
    _require(digest(attempt)==digest(expected),'Exact current-parent slot17 attempt required')
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    elapsed=guard.get('elapsed_seconds');sample=guard.get('observation',{})
    _require(set(guard)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
        and guard['kind']==KIND and guard['status'] in ('starting','finished') and guard['reason'] is None
        and type(guard['pid']) is int and guard['pid']==pid and guard['limits']==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<1800
        and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
        and all(type(v) is int and v>0 for v in sample.values()) and violation(sample,elapsed,**limits) is None,
        'Bounded current-parent slot17 preparation guard required')
    appended={ATTEMPT:dict(size=_path(root,ATTEMPT).stat().st_size,sha256=attempt_ref['sha256'])}
    pending=None;request_ref=preparation['pending_request_ref']
    if request_ref is not None:
        _require(request_ref['path']==REQUEST,'Only exact pending slot17 request may append')
        pending=_document(root,request_ref);review=saved['review']
        bound,requests=_frozen(writer,17,plan_sha,projection_ref,review['rates'],review['selection'])
        seal=_document(root,review['readiness'])
        _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
            and digest(pending)==digest(requests[-1]),'Pending request differs from independent frozen slot17 choice')
        appended[REQUEST]=dict(size=_path(root,REQUEST).stat().st_size,sha256=request_ref['sha256'])
    control=(GUARD,dict(size=_path(writer.root,GUARD).stat().st_size,sha256=guard_ref['sha256']))
    return appended,pending,control


def verify_confirmation_preparation(writer,slot,plan_sha256,completion_ref,*,projection_ref,rank_completion_ref,
        attempt_ref,guard_ref,pending_request_ref=None):
    """Replay completed slot16 with exactly the declared slot17 preparation."""
    _require(type(slot) is int and slot==17,'Exact preparation slot17 required')
    preparation=dict(attempt_ref=attempt_ref,guard_ref=guard_ref,pending_request_ref=pending_request_ref)
    checked=history._verify(writer,16,plan_sha256,completion_ref,projection_ref=projection_ref,
        rank_completion_ref=rank_completion_ref,preparation=preparation)
    return dict(checked,kind=APPEND_KIND,preparation_slot=17,attempt_ref=attempt_ref,guard_ref=guard_ref,
        pending_request_ref=pending_request_ref,preparation_append_verified=True,pending_request_verified=pending_request_ref is not None)

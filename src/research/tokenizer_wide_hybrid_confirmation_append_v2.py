"""Explicit read-only slot16 preparation append contract around slot15 replay.

Own both locks, inspect workers and hold an outer resource guard. This accepts
only one bounded startup guard, its exact attempt and optionally its frozen
pending request. It creates nothing and authorizes no child, retry or next stage.
Startup receipt publication and slots17..23 require separate integration.
"""
import math
import os

from .protocol import digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_retention_receipts import _document,_path
from .tokenizer_wide_hybrid_scores import _require
from . import tokenizer_wide_hybrid_first_confirmation_startup_v2 as startup
from . import tokenizer_wide_hybrid_first_confirmation_history_v2 as history

KIND='wide_hybrid_confirmation_startup_v2'
APPEND_KIND='wide_hybrid_confirmation_append_v2'
ATTEMPT='meta/quality-confirm-startup-16-attempt.json'
GUARD='control/quality-confirm-startup-16.json'
REQUEST='quality/16/request.json'


def _append(writer,plan_sha,completion_ref,projection_ref,rank_completion_ref,saved,preparation):
    _require(type(preparation) is dict and set(preparation)=={'attempt_ref','guard_ref','pending_request_ref'},
             'Exact slot16 preparation references required')
    root=writer.root/'evidence';attempt_ref=preparation['attempt_ref'];guard_ref=preparation['guard_ref']
    _require(attempt_ref['path']==ATTEMPT and guard_ref['path']==GUARD,'Exact slot16 preparation paths required')
    attempt=_document(root,attempt_ref);guard=_document(writer.root,guard_ref);pid=os.getpid()
    expected=dict(kind=KIND,status='started',slot=16,parent_pid=pid,plan_sha256=plan_sha,
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,prior_dispatch_ref=completion_ref,
        dispatch_authorized=False,deletion_authorized=False)
    _require(digest(attempt)==digest(expected),'Exact current-parent slot16 attempt required')
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    elapsed=guard.get('elapsed_seconds');sample=guard.get('observation',{})
    _require(set(guard)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
        and guard['kind']==KIND and guard['status'] in ('starting','finished') and guard['reason'] is None
        and type(guard['pid']) is int and guard['pid']==pid and guard['limits']==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<1800
        and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
        and all(type(v) is int and v>0 for v in sample.values())
        and violation(sample,elapsed,**limits) is None,'Bounded current-parent preparation guard required')
    appended={ATTEMPT:dict(size=_path(root,ATTEMPT).stat().st_size,sha256=attempt_ref['sha256'])}
    pending=None;request_ref=preparation['pending_request_ref']
    if request_ref is not None:
        _require(request_ref['path']==REQUEST,'Only exact pending slot16 request may append')
        pending=_document(root,request_ref)
        review=saved['review']
        bound,requests=startup._frozen(writer,16,plan_sha,projection_ref,review['rates'],review['selection'])
        seal=_document(root,review['readiness'])
        _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
            and digest(pending)==digest(requests[-1]),'Pending request differs from independent frozen slot16 choice')
        appended[REQUEST]=dict(size=_path(root,REQUEST).stat().st_size,sha256=request_ref['sha256'])
    control=(GUARD,dict(size=_path(writer.root,GUARD).stat().st_size,sha256=guard_ref['sha256']))
    return appended,pending,control


def verify_first_confirmation_preparation(writer,plan_sha256,completion_ref,*,projection_ref,rank_completion_ref,
        attempt_ref,guard_ref,pending_request_ref=None):
    """Replay slot15 before/after only the frozen slot16 request is appended.

    The pending request receives no completed-record, checkpoint or capacity
    credit. No startup/readiness/launch/result/weight or future control is allowed.
    """
    preparation=dict(attempt_ref=attempt_ref,guard_ref=guard_ref,pending_request_ref=pending_request_ref)
    checked=history._verify(writer,plan_sha256,completion_ref,projection_ref=projection_ref,
                           rank_completion_ref=rank_completion_ref,preparation=preparation)
    return dict(checked,kind=APPEND_KIND,attempt_ref=attempt_ref,guard_ref=guard_ref,
        pending_request_ref=pending_request_ref,preparation_append_verified=True,
        pending_request_verified=pending_request_ref is not None)

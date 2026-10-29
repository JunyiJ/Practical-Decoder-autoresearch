"""Bounded shared remaining-fresh append contract; only slot18 replay integrated.

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
from . import tokenizer_wide_hybrid_later_confirmation_history_v2 as history

KIND=history.startup.KIND
APPEND_KIND='wide_hybrid_remaining_confirmation_append_v2'
FIRST_SLOT=18
LAST_SLOT=23


def paths(slot):
    """Finite shared naming contract, without granting replay or dispatch support."""
    _require(type(slot) is int and FIRST_SLOT<=slot<=LAST_SLOT,'Exact remaining fresh slot18..23 required')
    return dict(attempt=f'meta/quality-confirm-startup-{slot:02d}-attempt.json',
        guard=f'control/quality-confirm-startup-{slot:02d}.json',request=f'quality/{slot:02d}/request.json',
        prior=f'meta/quality-dispatch-{slot-1:02d}.json')


def _append(writer,slot,plan_sha,completion_ref,projection_ref,rank_completion_ref,saved,preparation):
    names=paths(slot)
    _require(completion_ref['path']==names['prior'] and type(saved.get('slot')) is int
        and saved['slot']==slot-1,'Exact immediately preceding completed slot required')
    _require(type(preparation) is dict and set(preparation)=={'attempt_ref','guard_ref','pending_request_ref'},
             'Exact remaining-fresh preparation references required')
    root=writer.root/'evidence';attempt_ref=preparation['attempt_ref'];guard_ref=preparation['guard_ref']
    _require(attempt_ref['path']==names['attempt'] and guard_ref['path']==names['guard'],'Exact remaining-fresh preparation paths required')
    attempt=_document(root,attempt_ref);guard=_document(writer.root,guard_ref);pid=os.getpid()
    expected=dict(kind=KIND,status='started',slot=slot,parent_pid=pid,plan_sha256=plan_sha,
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,prior_dispatch_ref=completion_ref,
        dispatch_authorized=False,deletion_authorized=False)
    _require(digest(attempt)==digest(expected),'Exact current-parent remaining-fresh attempt required')
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    elapsed=guard.get('elapsed_seconds');sample=guard.get('observation',{})
    _require(set(guard)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
        and guard['kind']==KIND and guard['status'] in ('starting','finished') and guard['reason'] is None
        and type(guard['pid']) is int and guard['pid']==pid and guard['limits']==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<1800
        and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
        and all(type(v) is int and v>0 for v in sample.values()) and violation(sample,elapsed,**limits) is None,
        'Bounded current-parent remaining-fresh preparation guard required')
    appended={names['attempt']:dict(size=_path(root,names['attempt']).stat().st_size,sha256=attempt_ref['sha256'])}
    pending=None;request_ref=preparation['pending_request_ref']
    if request_ref is not None:
        _require(request_ref['path']==names['request'],'Only exact next-fresh request may append')
        pending=_document(root,request_ref);review=saved['review']
        bound,requests=_frozen(writer,slot,plan_sha,projection_ref,review['rates'],review['selection'])
        seal=_document(root,review['readiness'])
        _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
            and digest(pending)==digest(requests[-1]),'Pending request differs from independent frozen next-fresh choice')
        appended[names['request']]=dict(size=_path(root,names['request']).stat().st_size,sha256=request_ref['sha256'])
    control=(names['guard'],dict(size=_path(writer.root,names['guard']).stat().st_size,sha256=guard_ref['sha256']))
    return appended,pending,control


def verify_confirmation_preparation(writer,slot,plan_sha256,completion_ref,*,projection_ref,rank_completion_ref,
        attempt_ref,guard_ref,pending_request_ref=None):
    """Replay completed slot17 with exactly the declared slot18 preparation."""
    _require(type(slot) is int and slot==18,'Only preparation slot18 has integrated history replay')
    preparation=dict(attempt_ref=attempt_ref,guard_ref=guard_ref,pending_request_ref=pending_request_ref)
    checked=history._verify(writer,17,plan_sha256,completion_ref,projection_ref=projection_ref,
        rank_completion_ref=rank_completion_ref,preparation=preparation)
    return dict(checked,kind=APPEND_KIND,preparation_slot=18,attempt_ref=attempt_ref,guard_ref=guard_ref,
        pending_request_ref=pending_request_ref,preparation_append_verified=True,pending_request_verified=pending_request_ref is not None)

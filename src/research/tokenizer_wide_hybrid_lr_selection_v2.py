"""Persist the prospectively specified LR choice after all nine full audits.

Caller owns both worker locks and inspects actual processes. This component does
not delete, launch, retry, choose ranks or adopt a model. The read-only verifier
is for the idle boundary before cleanup and requires an outer resource guard.
"""
import os
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_retention_receipts import _document,_sha
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_quality_startup_v2 import _history
from .tokenizer_wide_hybrid_quality_lr_startup_v2 import _frozen
from .tokenizer_wide_hybrid_quality_dispatch_v2 import _prior
from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_lr_prefix,bind_quality_stage
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_rules import choose_rates

KIND='wide_hybrid_lr_selection_v2'
BASE='meta/lr-boundary-selection'
CHOICE='meta/lr-selection.json'
GUARD='control/quality-lr-selection.json'


class _SelectionGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,8,'selection',1800);self.name=GUARD


def _audit(writer,plan_sha256,projection_ref,pilot_completion_ref,selection_ref=None):
    """Re-derive choices from all independently audited final H200 scores."""
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    bound,requests=_frozen(writer,8,plan_sha256,projection_ref)
    dispatch=_prior(writer,9,plan_sha256,projection_ref,pilot_completion_ref)
    inventory=_inventory(root);history=_history(writer,plan_sha256,pilot_completion_ref,inventory)
    plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha256))
    projection=_document(root,projection_ref)
    evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')}
    args=dict(context=bound['context'],requests=requests,evidence=evidence,
        numerical_ref=plan['numerical_ref'],projection_ref=projection_ref,
        training_token_sha256=plan['training_token_sha256'])
    prefix=bind_lr_prefix(root,**args);rates=choose_rates(prefix['records']);stage=None
    if selection_ref is not None:
        _require(selection_ref['path']==CHOICE and digest(_document(root,selection_ref))==digest(rates),
                 'Persisted LR selection differs from full audited panel')
        stage=bind_quality_stage(root,**dict(args,requests=dict(enumerate(requests))),boundary='lr',
            records=prefix['records'],rates=rates,selection=None,trials={i:f'quality/{i:02d}' for i in range(9)},
            selection_refs=dict(rates=selection_ref))
    storage=bind_storage(writer.root,projection,pilots=history['records'],records=prefix['records'],
        rates=None if selection_ref is None else rates,selection=None,stages=[history['chain_stage']],
        current_inventory=inventory,checkpoint_pairs={**history['checkpoint_pairs'],**prefix['checkpoint_pairs']},
        lock_fds=writer.lock_fds)
    again,actual=_frozen(writer,8,plan_sha256,projection_ref)
    _require(digest(again)==digest(bound) and actual==requests,'Frozen selection inputs changed')
    _require(_prior(writer,9,plan_sha256,projection_ref,pilot_completion_ref)==dispatch,'Dispatch chain changed')
    _check_inventory(root,inventory);_locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Complete remaining storage changed')
    return dict(context=bound['context'],frozen_files=bound['files'],dispatch=dispatch,inventory=inventory,
        prefix=prefix,pilot_history=history,rates=rates,stage=stage,storage=storage)


def _add(root,inventory,ref):
    return dict(inventory,**{ref['path']:dict(size=(root/ref['path']).stat().st_size,sha256=ref['sha256'])})


def _fresh(writer):
    _locks(writer.root/'evidence',writer.lock_fds);seen=snapshot(writer.root)['files']
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents selection')
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in seen),'Failed/partial campaign blocks selection')
    _require(not any(n.startswith('evidence/'+BASE) or n in ('evidence/'+CHOICE,GUARD,GUARD+'.tmp') for n in seen),
             'Existing LR selection attempt cannot be retried')


def save_lr_selection(writer,plan_sha256,*,projection_ref,pilot_completion_ref):
    """Full audits precede immutable choice; partial publication cannot authorize cleanup."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256);_fresh(writer)
    root=writer.root/'evidence'
    try:
        with _SelectionGuard(writer):
            attempt=dict(kind=KIND,status='started',plan_sha256=plan_sha256,projection_ref=projection_ref,
                pilot_completion_ref=pilot_completion_ref,retry_allowed=False)
            path=writer.publish_json('evidence/'+BASE+'-attempt.json',attempt)
            attempt_ref=dict(path=BASE+'-attempt.json',sha256=file_digest(path))
            before=_audit(writer,plan_sha256,projection_ref,pilot_completion_ref)
            prereview=dict(kind=KIND,status='all_nine_lr_screens_reviewed_before_selection',plan_sha256=plan_sha256,
                projection_ref=projection_ref,pilot_completion_ref=pilot_completion_ref,attempt=attempt_ref,audit=before)
            path=writer.publish_json('evidence/'+BASE+'-review.json',prereview)
            review_ref=dict(path=BASE+'-review.json',sha256=file_digest(path))
            path=writer.publish_json('evidence/'+CHOICE,before['rates'])
            selection_ref=dict(path=CHOICE,sha256=file_digest(path))
            after=_audit(writer,plan_sha256,projection_ref,pilot_completion_ref,selection_ref)
            expected=_add(root,_add(root,before['inventory'],review_ref),selection_ref)
            _require(after['inventory']==expected and after['rates']==before['rates']
                and after['prefix']['records']==before['prefix']['records']
                and after['prefix']['checkpoint_pairs']==before['prefix']['checkpoint_pairs']
                and after['dispatch']==before['dispatch'] and after['context']==before['context']
                and after['frozen_files']==before['frozen_files'],'Selection publication changed reviewed inputs')
            result=dict(kind=KIND,status='lr_selection_reviewed_pending_cleanup',parent_pid=os.getpid(),
                plan_sha256=plan_sha256,projection_ref=projection_ref,pilot_completion_ref=pilot_completion_ref,
                review_ref=review_ref,selection_ref=selection_ref,audit=after,adopted=False,
                dispatch_authorized=False,deletion_authorized=False,stage_cleanup_complete=False,next_stage_authorized=False)
        guard_ref=dict(path=GUARD,sha256=file_digest(writer.root/GUARD))
        _guard(writer,guard_ref,KIND,os.getpid(),1800);result['guard']=guard_ref
        _locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents selection receipt')
        _require(shutil.disk_usage(writer.root).free>=after['storage']['required_free_bytes'],
                 'Full storage changed before selection receipt')
        path=writer.publish_json('evidence/'+BASE+'-complete.json',result)
        return dict(path=BASE+'-complete.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(kind=KIND,status='failed',
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Selection failure publication failed: '+repr(error)[:1000])
        raise


def verify_lr_selection(writer,plan_sha256,completion_ref):
    """Read-only, pre-cleanup verifier. Full stage/receipt/storage audit, no credit for deletion."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    _require(completion_ref['path']==BASE+'-complete.json','Exact selection completion reference required')
    complete=_document(root,completion_ref);pid=complete.get('parent_pid')
    _require(complete.get('kind')==KIND and complete.get('status')=='lr_selection_reviewed_pending_cleanup'
        and complete.get('plan_sha256')==plan_sha256 and type(pid) is int and pid>1
        and all(complete.get(k) is False for k in ('adopted','dispatch_authorized','deletion_authorized',
            'stage_cleanup_complete','next_stage_authorized')),'Complete unpromoted selection review required')
    _require(complete['guard']['path']==GUARD,'Exact selection guard required')
    _guard(writer,complete['guard'],KIND,pid,1800)
    _require(complete['review_ref']['path']==BASE+'-review.json','Exact preselection review required')
    prereview=_document(root,complete['review_ref'])
    _require(prereview['kind']==KIND and prereview['status']=='all_nine_lr_screens_reviewed_before_selection'
        and prereview['plan_sha256']==plan_sha256 and prereview['projection_ref']==complete['projection_ref']
        and prereview['pilot_completion_ref']==complete['pilot_completion_ref'],'Preselection provenance differs')
    _require(prereview['attempt']['path']==BASE+'-attempt.json','Exact selection attempt required')
    attempt=_document(root,prereview['attempt'])
    _require(attempt==dict(kind=KIND,status='started',plan_sha256=plan_sha256,
        projection_ref=complete['projection_ref'],pilot_completion_ref=complete['pilot_completion_ref'],retry_allowed=False),
        'Selection attempt provenance differs')
    current=_audit(writer,plan_sha256,complete['projection_ref'],complete['pilot_completion_ref'],complete['selection_ref'])
    saved=complete['audit'];before=prereview['audit']
    expected=_add(root,_add(root,before['inventory'],complete['review_ref']),complete['selection_ref'])
    _require(saved['inventory']==expected and current['inventory']==_add(root,expected,completion_ref),
             'Selection boundary inventory changed')
    for key in ('context','frozen_files','dispatch','rates'):
        _require(current[key]==saved[key]==before[key],'Selection stage identity differs')
    for key in ('records','checkpoint_pairs'):
        _require(current['prefix'][key]==saved['prefix'][key]==before['prefix'][key],
                 'Selection scientific panel differs')
    # The saved stage excludes this completion document; current replay binds it
    # too. Recheck the saved full stage against precisely its original inventory.
    bound,requests=_frozen(writer,8,plan_sha256,complete['projection_ref'])
    plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha256))
    stage=bind_quality_stage(root,boundary='lr',records=current['prefix']['records'],rates=current['rates'],selection=None,
        context=bound['context'],requests=dict(enumerate(requests)),trials={i:f'quality/{i:02d}' for i in range(9)},
        evidence={n:v['sha256'] for n,v in saved['inventory'].items() if not n.endswith('.pt')},
        numerical_ref=plan['numerical_ref'],projection_ref=complete['projection_ref'],
        selection_refs=dict(rates=complete['selection_ref']),training_token_sha256=plan['training_token_sha256'])
    _require(digest(stage)==digest(saved['stage']),'Saved complete stage audit differs')
    _check_inventory(root,current['inventory']);_guard(writer,complete['guard'],KIND,pid,1800)
    _locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=current['storage']['required_free_bytes'],'Remaining storage changed')
    return dict(kind=KIND,status='lr_selection_verified_pending_cleanup',completion_ref=completion_ref,
        selection_ref=complete['selection_ref'],rates=current['rates'],audit=current,
        adopted=False,dispatch_authorized=False,deletion_authorized=False,stage_cleanup_complete=False,next_stage_authorized=False)

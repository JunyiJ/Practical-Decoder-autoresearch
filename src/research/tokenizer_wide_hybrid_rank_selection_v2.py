"""Full final-rank selection audits with compact immutable receipts.

No child, deletion, fresh confirmation, retry or adoption. Caller owns both
worker locks, inspects processes, and guards read-only verification externally.
Digest summaries do not replace audits: verification reconstructs every field
from retained evidence, original cleanup history and independently frozen inputs.
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
from .tokenizer_wide_hybrid_quality_rank_startup_v2 import _frozen
from .tokenizer_wide_hybrid_lr_boundary_v2 import verify_lr_history
from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
from .tokenizer_wide_hybrid_quality_dispatch_v2 import _prior
from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_rank_prefix,bind_quality_stage
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_inventory_delta import encode_inventory
from .tokenizer_wide_hybrid_rules import choose_ranks
from .tokenizer_wide_hybrid_lr_selection_v2 import _add

KIND='wide_hybrid_rank_selection_v2'
BASE='meta/rank-boundary-selection'
CHOICE='meta/rank-selection.json'
GUARD='control/quality-rank-selection.json'
NONCLAIMS=('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized')


class _SelectionGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,14,'selection',1800);self.name=GUARD


def _audit(writer,plan_sha,projection_ref,lr_completion_ref,selection_ref=None,stage_inventory=None):
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    seen=snapshot(writer.root)['files']
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in seen),'Failed/partial campaign blocks selection')
    inventory=_inventory(root)
    history=verify_lr_history(writer,plan_sha,lr_completion_ref,current_inventory=inventory)
    _require(history['projection_ref']==projection_ref,'Historical projection differs')
    bound,requests=_frozen(writer,14,plan_sha,projection_ref,history['rates'])
    dispatch=_prior(writer,15,plan_sha,projection_ref,history['pilot_completion_ref'],lr_completion_ref=lr_completion_ref)
    plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha))
    args=dict(context=bound['context'],numerical_ref=plan['numerical_ref'],projection_ref=projection_ref,
        training_token_sha256=plan['training_token_sha256'])
    evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')}
    prefix=bind_rank_prefix(root,**args,lr_records=history['records'],rates=history['rates'],
        lr_requests=requests[:9],rank_requests=requests[9:],evidence=evidence,rates_ref=history['selection_ref'])
    for name,pair in prefix['checkpoint_pairs'].items():
        if int(name.split('/')[1])<9:_require(pair==history['checkpoint_pairs'][name],'Live LR anchor changed')
    pilot=verify_pilot_history(writer,plan_sha,history['pilot_completion_ref'],current_inventory=inventory,stages=history['stages'])
    pairs={**pilot['checkpoint_pairs'],**history['checkpoint_pairs'],**prefix['checkpoint_pairs']}
    choice=choose_ranks(prefix['records'][:9],prefix['records'][9:]);stage=None
    if selection_ref is not None:
        _require(selection_ref['path']==CHOICE and digest(_document(root,selection_ref))==digest(choice),
                 'Persisted final rank selection differs')
        stage_evidence=inventory if stage_inventory is None else stage_inventory
        _require(all(inventory.get(n)==v for n,v in stage_evidence.items()),'Saved stage evidence changed')
        indices=prefix['audited_indices']
        stage=bind_quality_stage(root,**args,boundary='rank',records=prefix['records'],rates=history['rates'],selection=choice,
            requests={i:requests[i] for i in indices},trials={i:f'quality/{i:02d}' for i in indices},
            evidence={n:v['sha256'] for n,v in stage_evidence.items() if not n.endswith('.pt')},
            selection_refs=dict(rates=history['selection_ref'],selection=selection_ref))
    storage=bind_storage(writer.root,_document(root,projection_ref),pilots=pilot['records'],records=prefix['records'],
        rates=history['rates'],selection=None if selection_ref is None else choice,stages=history['stages'],
        current_inventory=inventory,checkpoint_pairs=pairs,lock_fds=writer.lock_fds)
    again,actual=_frozen(writer,14,plan_sha,projection_ref,history['rates'])
    _require(digest(again)==digest(bound) and actual==requests,'Frozen rank selection inputs changed')
    _require(_prior(writer,15,plan_sha,projection_ref,history['pilot_completion_ref'],lr_completion_ref=lr_completion_ref)==dispatch,
             'Rank dispatch chain changed')
    _check_inventory(root,inventory);_locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Complete remaining storage changed')
    summary=dict(frozen_files_sha256=digest(bound['files']),dispatch=dispatch,records=prefix['records'],
        checkpoint_pairs=pairs,rates=history['rates'],selection=choice,audited_indices=prefix['audited_indices'],
        history_sha256=digest({k:v for k,v in history.items() if k!='chain'}))
    return dict(context=bound['context'],inventory=inventory,summary=summary,stage=stage,storage=storage,
        history=history,pilot=pilot,prefix=prefix)


def _fresh(writer):
    _locks(writer.root/'evidence',writer.lock_fds);seen=snapshot(writer.root)['files']
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents selection')
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in seen),'Failed/partial campaign blocks selection')
    _require(not any(n.startswith('evidence/'+BASE) or n in ('evidence/'+CHOICE,GUARD,GUARD+'.tmp') for n in seen),
             'Existing rank selection attempt cannot be retried')


def _publish(writer,name,value,base):
    path=writer.publish_json('evidence/'+name,encode_inventory(writer.root/'evidence',value,base))
    return dict(path=name,sha256=file_digest(path))


def save_rank_selection(writer,plan_sha256,*,projection_ref,lr_completion_ref):
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256);_fresh(writer)
    root=writer.root/'evidence'
    identity=dict(kind=KIND,plan_sha256=plan_sha256,projection_ref=projection_ref,lr_completion_ref=lr_completion_ref)
    try:
        with _SelectionGuard(writer):
            path=writer.publish_json('evidence/'+BASE+'-attempt.json',dict(identity,status='started',retry_allowed=False))
            attempt=dict(path=BASE+'-attempt.json',sha256=file_digest(path))
            before=_audit(writer,plan_sha256,projection_ref,lr_completion_ref)
            anchor=dict(path='meta/quality-readiness-14.json',sha256=file_digest(root/'meta/quality-readiness-14.json'))
            review_ref=_publish(writer,BASE+'-review.json',dict(identity,status='all_rank_candidates_reviewed_before_selection',
                attempt=attempt,context=before['context'],inventory=before['inventory'],summary=before['summary'],storage=before['storage']),anchor)
            path=writer.publish_json('evidence/'+CHOICE,before['summary']['selection'])
            selection_ref=dict(path=CHOICE,sha256=file_digest(path))
            after=_audit(writer,plan_sha256,projection_ref,lr_completion_ref,selection_ref)
            expected=_add(root,_add(root,before['inventory'],review_ref),selection_ref)
            _require(after['inventory']==expected and after['summary']==before['summary'] and after['context']==before['context'],
                     'Rank selection publication changed reviewed evidence')
            result=dict(identity,status='rank_selection_reviewed_pending_cleanup',parent_pid=os.getpid(),
                context=after['context'],inventory=after['inventory'],summary_sha256=digest(after['summary']),
                stage_sha256=digest(after['stage']),manifest_sha256=digest(after['stage']['manifest']),
                storage=after['storage'],review_ref=review_ref,selection_ref=selection_ref,**{k:False for k in NONCLAIMS})
        guard=dict(path=GUARD,sha256=file_digest(writer.root/GUARD));_guard(writer,guard,KIND,os.getpid(),1800)
        result['guard']=guard
        final,_=_frozen(writer,14,plan_sha256,projection_ref,after['summary']['rates'])
        _require(final['context']==after['context'] and digest(final['files'])==after['summary']['frozen_files_sha256'],
                 'Frozen inputs changed before rank selection receipt')
        _require(_prior(writer,15,plan_sha256,projection_ref,after['history']['pilot_completion_ref'],
            lr_completion_ref=lr_completion_ref)==after['summary']['dispatch'],'Rank controls changed before receipt')
        _locks(root,writer.lock_fds);_check_inventory(root,expected)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents selection receipt')
        _require(shutil.disk_usage(writer.root).free>=after['storage']['required_free_bytes'],'Full storage changed before receipt')
        return _publish(writer,BASE+'-complete.json',result,review_ref)
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(identity,status='failed',error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Rank selection failure publication failed: '+repr(error)[:1000])
        raise


def _storage(saved,current,inventory):
    # Live observations change across immutable publications. All capacity,
    # projection, reserve and non-authorization fields must match a full replay.
    dynamic={'free_bytes','metadata_bytes','current_inventory_sha256'}
    _require(set(saved)==set(current) and digest({k:v for k,v in saved.items() if k not in dynamic})==
             digest({k:v for k,v in current.items() if k not in dynamic})
        and type(saved['free_bytes']) is int and saved['free_bytes']>=current['required_free_bytes']
        and type(saved['metadata_bytes']) is int and 0<=saved['metadata_bytes']<=current['projection']['metadata_allowance_bytes']
        and saved['current_inventory_sha256']==digest(inventory),'Saved complete storage evidence differs')


def verify_rank_selection(writer,plan_sha256,completion_ref):
    """Read-only idle boundary before cleanup; full audits reconstruct summaries."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    _require(completion_ref['path']==BASE+'-complete.json','Exact rank selection receipt required')
    saved=_document(root,completion_ref);pid=saved.get('parent_pid')
    _require(saved.get('kind')==KIND and saved.get('status')=='rank_selection_reviewed_pending_cleanup'
        and saved.get('plan_sha256')==plan_sha256 and type(pid) is int and pid>1
        and all(saved.get(k) is False for k in NONCLAIMS),'Unpromoted rank selection receipt required')
    _require(saved['guard']['path']==GUARD,'Exact rank selection guard required');_guard(writer,saved['guard'],KIND,pid,1800)
    _require(saved['review_ref']['path']==BASE+'-review.json','Exact rank preselection review required')
    review=_document(root,saved['review_ref'])
    identity={k:saved[k] for k in ('kind','plan_sha256','projection_ref','lr_completion_ref')}
    _require(all(review.get(k)==v for k,v in identity.items())
        and review.get('status')=='all_rank_candidates_reviewed_before_selection'
        and review['context']==saved['context'],'Rank review identity differs')
    _require(review['attempt']['path']==BASE+'-attempt.json' and
        _document(root,review['attempt'])==dict(identity,status='started',retry_allowed=False),'Rank attempt differs')
    expected=_add(root,_add(root,review['inventory'],saved['review_ref']),saved['selection_ref'])
    _require(saved['inventory']==expected,'Saved selection inventory differs')
    current=_audit(writer,plan_sha256,saved['projection_ref'],saved['lr_completion_ref'],saved['selection_ref'],expected)
    _require(current['inventory']==_add(root,expected,completion_ref) and current['context']==saved['context']
        and current['summary']==review['summary'] and digest(current['summary'])==saved['summary_sha256']
        and digest(current['stage'])==saved['stage_sha256'] and digest(current['stage']['manifest'])==saved['manifest_sha256'],
        'Full rank selection replay differs')
    _storage(saved['storage'],current['storage'],expected)
    # Before publication the same full panel used the worst of all27 choices.
    before=_audit(writer,plan_sha256,saved['projection_ref'],saved['lr_completion_ref'])
    _storage(review['storage'],before['storage'],review['inventory'])
    _require(before['inventory']==current['inventory'] and before['summary']==current['summary'],'Selection changed during replay')
    _check_inventory(root,current['inventory']);_guard(writer,saved['guard'],KIND,pid,1800);_locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=current['storage']['required_free_bytes'],'Full remaining storage changed')
    return dict(kind=KIND,status='rank_selection_verified_pending_cleanup',completion_ref=completion_ref,
        selection_ref=saved['selection_ref'],selection=current['summary']['selection'],audit=current,**{k:False for k in NONCLAIMS})

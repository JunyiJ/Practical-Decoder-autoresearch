"""Audited rank cleanup and strict idle cumulative receipt verification.

No CLI, training, fresh dispatch or retry. Caller owns both locks and inspects
workers. Only reviewed losing screen weights are eligible. Post-cleanup replay
uses original full audit receipts and journals, never recreating deleted weights.
Read-only verification requires an outer resource guard.
"""
import os
import shutil
import time

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _document,_sha,planned_manifest
from .tokenizer_wide_hybrid_cleanup import execute_boundary,_locks,_scan,_check,TIMEOUT
from .tokenizer_wide_hybrid_retention_chain import recheck_chain
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
from .tokenizer_wide_hybrid_quality_readiness_v2 import _controls
from .tokenizer_wide_hybrid_scores import _require
from . import tokenizer_wide_hybrid_rank_selection_v2 as selection
from . import tokenizer_wide_hybrid_lr_boundary_v2 as lr

KIND='wide_hybrid_rank_boundary_v2'
BASE='meta/rank-boundary-cleanup'
GUARD='control/quality-rank-boundary.json'


class _BoundaryGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,14,'boundary',1800);self.name=GUARD


def _inputs(summary,inventory):
    return dict(boundary='rank',records=summary['records'],rates=summary['rates'],selection=summary['selection'],
        pairs={i:summary['checkpoint_pairs'][f'quality/{i:02d}'] for i in summary['audited_indices']},
        evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')})


def _view(writer,review):
    """Expand immutable pre-deletion scientific receipts without loading tensors."""
    root=writer.root/'evidence';ref=review['selection_completion_ref'];selected=_document(root,ref)
    _require(ref['path']==selection.BASE+'-complete.json' and selected.get('kind')==selection.KIND
        and selected.get('status')=='rank_selection_reviewed_pending_cleanup'
        and selected.get('plan_sha256')==review['plan_sha256']
        and all(selected.get(k) is False for k in selection.NONCLAIMS),'Original full rank selection required')
    _require(selected['guard']['path']==selection.GUARD,'Exact original selection guard required')
    _guard(writer,selected['guard'],selection.KIND,selected['parent_pid'],1800)
    _require(selected['review_ref']['path']==selection.BASE+'-review.json','Exact original selection review required')
    original=_document(root,selected['review_ref']);summary=original['summary']
    identity={k:selected[k] for k in ('kind','plan_sha256','projection_ref','lr_completion_ref')}
    _require(all(original.get(k)==v for k,v in identity.items())
        and original.get('status')=='all_rank_candidates_reviewed_before_selection'
        and original['context']==selected['context']==review['context'],'Original selection review differs')
    _require(original['attempt']['path']==selection.BASE+'-attempt.json' and
        _document(root,original['attempt'])==dict(identity,status='started',retry_allowed=False),'Original selection attempt differs')
    choice=selection.choose_ranks(summary['records'][:9],summary['records'][9:])
    _require(selected['selection_ref']['path']==selection.CHOICE and
        _document(root,selected['selection_ref'])==summary['selection']==choice
        and digest(summary)==selected['summary_sha256']==review['summary_sha256'],'Original rank choice or summary changed')
    inventory=selection._add(root,selection._add(root,original['inventory'],selected['review_ref']),selected['selection_ref'])
    _require(selected['inventory']==inventory,'Original scientific stage inventory differs')
    expected=selection._add(root,selection._add(root,inventory,ref),review['attempt'])
    _require(review['inventory']==expected,'Rank boundary inventory differs from complete selection')
    manifest=planned_manifest(**_inputs(summary,inventory))
    stage=dict(review['stage_flags'],manifest=manifest)
    _require(digest(stage)==selected['stage_sha256'] and digest(manifest)==selected['manifest_sha256']==review['manifest_sha256'],
             'Original full stage/manifest differs')
    complete,old,old_ref=lr._receipt(writer,review['plan_sha256'],selected['lr_completion_ref'])
    stages=[old['pilot_stage'],complete['chain_stage']]
    history=lr._history_fields(old,stages)
    _require(digest(history)==summary['history_sha256'] and history['records']==summary['records'][:9]
        and history['rates']==summary['rates'] and history['projection_ref']==selected['projection_ref'],
        'Original LR history differs from rank selection')
    for name,pair in history['checkpoint_pairs'].items():
        _require(summary['checkpoint_pairs'][name]==pair,'Original LR pair identity changed')
    # Original control bindings and frozen context remain protected even when
    # rank selection legitimately retires an earlier selected LR anchor.
    _require(old['context']==review['context'] and digest(old['frozen_files'])==summary['frozen_files_sha256'],
             'Original LR frozen identity differs')
    _require(all(review['controls'].get(n)==v for n,v in old['controls'].items()),'Original LR controls omitted')
    old_identity=dict(kind=lr.KIND,plan_sha256=review['plan_sha256'],context=old['context'],
        frozen_files_sha256=digest(old['frozen_files']),review_ref=old_ref,
        selection_completion_ref=old['selection_completion_ref'],manifest_sha256=digest(old['stage']['manifest']),
        controls_sha256=digest(old['controls']))
    _require(digest(old_identity)==complete['chain_stage']['context_sha256'],'Original LR execution identity differs')
    return dict(selected=selected,summary=summary,manifest=manifest,stage=stage,history=history,
        stage_inventory=inventory,stages=stages)


def _context(writer,review,ref,view,completion_ref=None,*,appended_inventory=None,appended_controls=None):
    root=writer.root/'evidence';_locks(root,writer.lock_fds);snapshot(writer.root)
    selected=view['selected'];summary=view['summary']
    bound,_=selection._frozen(writer,14,review['plan_sha256'],selected['projection_ref'],summary['rates'])
    _require(bound['context']==review['context'] and digest(bound['files'])==summary['frozen_files_sha256'],
             'Frozen rank boundary identity changed')
    _document(root,ref);inventory=selection._add(root,review['inventory'],ref)
    if completion_ref is not None:
        _document(root,completion_ref);inventory=selection._add(root,inventory,completion_ref)
    if appended_inventory is not None:
        _require(not set(appended_inventory)&set(inventory),'Fresh append overlaps protected rank history')
        inventory.update(appended_inventory)
    candidates={e['path'] for e in view['manifest']['delete']}
    actual=_scan(root,root/'.retention/rank')
    _require(actual<=set(inventory) and set(inventory)-candidates<=actual,'Rank protected inventory changed')
    controls=dict(review['controls'])
    if appended_controls is not None:
        _require(not set(appended_controls)&set(controls),'Fresh controls overlap protected rank history')
        controls.update(appended_controls)
    _require(_controls(writer,GUARD)==controls,'Protected control census changed')
    deadline=time.monotonic()+TIMEOUT
    for name,binding in inventory.items():
        if name not in candidates:_check(root,name,binding,deadline)
    for name,binding in controls.items():
        _require(name.startswith('control/') and name not in (GUARD,'control/.metadata.lock'),'Unexpected protected control')
        _check(writer.root,name,binding,deadline)
    _locks(root,writer.lock_fds)
    return dict(kind=KIND,plan_sha256=review['plan_sha256'],context=bound['context'],
        frozen_files_sha256=digest(bound['files']),review_ref=ref,selection_completion_ref=review['selection_completion_ref'],
        manifest_sha256=digest(view['manifest']),controls_sha256=digest(review['controls']))


def _storage(writer,review,view,stages,inventory):
    history=verify_pilot_history(writer,review['plan_sha256'],view['history']['pilot_completion_ref'],
        current_inventory=inventory,stages=stages)
    summary=view['summary']
    for name,pair in history['checkpoint_pairs'].items():
        _require(summary['checkpoint_pairs'][name]==pair,'Original pilot pair identity differs')
    return bind_storage(writer.root,_document(writer.root/'evidence',view['selected']['projection_ref']),
        pilots=history['records'],records=summary['records'],rates=summary['rates'],selection=summary['selection'],
        stages=stages,current_inventory=inventory,checkpoint_pairs=summary['checkpoint_pairs'],lock_fds=writer.lock_fds)


def cleanup_rank_boundary(writer,plan_sha256,selection_completion_ref):
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    root=writer.root/'evidence';_locks(root,writer.lock_fds);seen=snapshot(writer.root)['files']
    _require(not any(n.startswith('evidence/'+BASE) or n in (GUARD,GUARD+'.tmp') for n in seen)
        and not (root/'.retention/rank').exists(),'Existing rank cleanup attempt cannot be retried')
    try:
        with _BoundaryGuard(writer):
            checked=selection.verify_rank_selection(writer,plan_sha256,selection_completion_ref);audit=checked['audit']
            selected=_document(root,selection_completion_ref)
            attempt=dict(kind=KIND,status='started',plan_sha256=plan_sha256,
                selection_completion_ref=selection_completion_ref,retry_allowed=False)
            path=writer.publish_json('evidence/'+BASE+'-attempt.json',attempt)
            attempt_ref=dict(path=BASE+'-attempt.json',sha256=file_digest(path));inventory=_inventory(root)
            _require(inventory==selection._add(root,audit['inventory'],attempt_ref),'Pre-cleanup inventory changed')
            # Reuse the original full stage evidence, not the later verifier's
            # expanded inventory. The executor protects all additional receipts.
            manifest=planned_manifest(**_inputs(audit['summary'],selected['inventory']))
            _require(digest(manifest)==selected['manifest_sha256'],'Original rank manifest changed')
            review=dict(kind=KIND,status='all_rank_candidates_reviewed_before_cleanup',plan_sha256=plan_sha256,
                selection_completion_ref=selection_completion_ref,attempt=attempt_ref,context=audit['context'],inventory=inventory,
                summary_sha256=digest(audit['summary']),manifest_sha256=digest(manifest),
                stage_flags={k:v for k,v in audit['stage'].items() if k!='manifest'},
                controls=_controls(writer,GUARD),adopted=False,next_stage_authorized=False)
            ref=selection._publish(writer,BASE+'-review.json',review,selection_completion_ref)
            view=_view(writer,review);expected=selection._add(root,inventory,ref)
            def revalidate():return _context(writer,review,ref,view)
            def authorize():
                fresh=selection._audit(writer,plan_sha256,selected['projection_ref'],selected['lr_completion_ref'],
                    selected['selection_ref'],selected['inventory'])
                _require(fresh['inventory']==expected and fresh['summary']==audit['summary']
                    and fresh['context']==review['context'] and digest(fresh['stage'])==selected['stage_sha256'],
                    'Full rank audit changed before deletion')
                return dict(inputs=_inputs(fresh['summary'],selected['inventory']),inventory=fresh['inventory'])
            identity=revalidate();execute_boundary(root,'rank',writer.lock_fds,authorize,revalidate)
            execution=dict(path='.retention/rank/execution-review.json',sha256=file_digest(root/'.retention/rank/execution-review.json'))
            stage=dict(manifest=manifest,execution_ref=execution,context_sha256=digest(identity))
            stages=view['stages']+[stage];current=_inventory(root)
            chain=recheck_chain(root,stages,current);storage=_storage(writer,review,view,stages,current)
            _require(revalidate()==identity,'Rank context changed after deletion')
            result=dict(kind=KIND,status='rank_cleanup_verified',parent_pid=os.getpid(),plan_sha256=plan_sha256,
                review_ref=ref,execution_ref=execution,context_sha256=digest(identity),manifest_sha256=digest(manifest),
                chain_sha256=digest(chain),storage=storage,adopted=False,dispatch_authorized=False,next_stage_authorized=False,
                limitation='Deleted losing-screen weights cannot be reloaded; original full audits remain retained.')
        guard=dict(path=GUARD,sha256=file_digest(writer.root/GUARD));_guard(writer,guard,KIND,os.getpid(),1800)
        result['guard']=guard
        _require(_context(writer,review,ref,view)==identity,'Rank context changed before completion')
        _check_inventory(root,current);_locks(root,writer.lock_fds)
        _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Full storage changed before completion')
        path=writer.publish_json('evidence/'+BASE+'-complete.json',result)
        return dict(path=BASE+'-complete.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(kind=KIND,status='failed',
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Rank cleanup failure publication failed: '+repr(error)[:1000])
        raise


def _receipt(writer,plan_sha256,completion_ref):
    """Bind original complete rank receipt, without granting continuation credit."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    seen=snapshot(writer.root)['files']
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in seen),'Failed/partial cleanup cannot be accepted')
    _require(completion_ref['path']==BASE+'-complete.json','Exact rank cleanup receipt required')
    saved=_document(root,completion_ref);pid=saved.get('parent_pid')
    _require(saved.get('kind')==KIND and saved.get('status')=='rank_cleanup_verified'
        and saved.get('plan_sha256')==plan_sha256 and type(pid) is int and pid>1
        and all(saved.get(k) is False for k in ('adopted','dispatch_authorized','next_stage_authorized')),
        'Unpromoted rank cleanup receipt required')
    _require(saved['guard']['path']==GUARD,'Exact rank cleanup guard required');_guard(writer,saved['guard'],KIND,pid,1800)
    ref=saved['review_ref'];_require(ref['path']==BASE+'-review.json','Exact rank cleanup review required')
    review=_document(root,ref)
    _require(review.get('kind')==KIND and review.get('status')=='all_rank_candidates_reviewed_before_cleanup'
        and review.get('plan_sha256')==plan_sha256 and review.get('adopted') is False
        and review.get('next_stage_authorized') is False,'Original full cleanup review required')
    _require(review['attempt']['path']==BASE+'-attempt.json' and _document(root,review['attempt'])==dict(kind=KIND,
        status='started',plan_sha256=plan_sha256,selection_completion_ref=review['selection_completion_ref'],retry_allowed=False),
        'Original cleanup attempt differs')
    view=_view(writer,review)
    return saved,review,ref,view


def verify_rank_boundary(writer,plan_sha256,completion_ref):
    """Strict idle gate before any fresh request; no tensors or storage speculation."""
    saved,review,ref,view=_receipt(writer,plan_sha256,completion_ref)
    root=writer.root/'evidence';pid=saved['parent_pid']
    identity=_context(writer,review,ref,view,completion_ref)
    _require(digest(identity)==saved['context_sha256'] and digest(view['manifest'])==saved['manifest_sha256']
        and saved['execution_ref']['path']=='.retention/rank/execution-review.json','Rank execution identity differs')
    stage=dict(manifest=view['manifest'],execution_ref=saved['execution_ref'],context_sha256=saved['context_sha256'])
    stages=view['stages']+[stage];current=_inventory(root);chain=recheck_chain(root,stages,current)
    storage=_storage(writer,review,view,stages,current)
    # Completion is appended after the saved post-deletion chain observation.
    original={n:v for n,v in current.items() if n!=completion_ref['path']}
    old_chain=dict(chain,current_inventory_sha256=digest(original),protected_files_verified=len(original))
    _require(digest(old_chain)==saved['chain_sha256'],'Saved cumulative cleanup observation differs')
    selection._storage(saved['storage'],storage,original)
    _require(_context(writer,review,ref,view,completion_ref)==identity,'Rank boundary changed during verification')
    _guard(writer,saved['guard'],KIND,pid,1800);_locks(root,writer.lock_fds)
    return dict(kind=KIND,status='rank_boundary_verified_pending_fresh',records=view['summary']['records'],
        rates=view['summary']['rates'],selection=view['summary']['selection'],checkpoint_pairs=view['summary']['checkpoint_pairs'],
        stages=stages,chain=chain,storage=storage,selection_ref=view['selected']['selection_ref'],
        projection_ref=view['selected']['projection_ref'],adopted=False,dispatch_authorized=False,next_stage_authorized=False)

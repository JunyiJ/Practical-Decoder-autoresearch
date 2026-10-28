"""Audited LR selection -> explicit losing-screen cleanup -> verified chain.

No CLI, training, retry or rank dispatch. Caller owns shared locks and inspects
actual workers. Post-cleanup verification binds retained original audits and
journals; it never reloads or recreates intentionally deleted losing weights.
"""
import os
import re
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
from .tokenizer_wide_hybrid_quality_audit_v2 import QUALITY_ARTIFACTS
from .tokenizer_wide_hybrid_scores import _require
from . import tokenizer_wide_hybrid_lr_selection_v2 as selection

KIND='wide_hybrid_lr_boundary_v2'
BASE='meta/lr-boundary-cleanup'
GUARD='control/quality-lr-boundary.json'


class _BoundaryGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,8,'boundary',1800);self.name=GUARD


def _inputs(review):
    return dict(boundary='lr',records=review['records'],rates=review['rates'],selection=None,
        pairs={i:review['checkpoint_pairs'][f'quality/{i:02d}'] for i in range(9)},
        evidence=review['stage']['manifest']['evidence'])


def _context(writer,review,review_ref,completion_ref=None,*,appended_inventory=None):
    """Revalidate frozen identity and all protected bytes throughout deletion."""
    root=writer.root/'evidence';_locks(root,writer.lock_fds);snapshot(writer.root)
    bound,_=selection._frozen(writer,8,review['plan_sha256'],review['projection_ref'])
    _require(bound['context']==review['context'] and bound['files']==review['frozen_files'],
             'Frozen LR cleanup identity changed')
    _document(root,review_ref)
    inventory=selection._add(root,review['inventory'],review_ref)
    if completion_ref is not None:
        _document(root,completion_ref);inventory=selection._add(root,inventory,completion_ref)
    if appended_inventory is not None:
        _require(not set(appended_inventory)&set(inventory),'Appended rank evidence overlaps LR history')
        inventory.update(appended_inventory)
    candidates={e['path'] for e in review['stage']['manifest']['delete']}
    actual=_scan(root,root/'.retention/lr')
    _require(actual<=set(inventory) and set(inventory)-candidates<=actual,'LR protected inventory changed')
    deadline=time.monotonic()+TIMEOUT
    for name,binding in inventory.items():
        if name not in candidates:_check(root,name,binding,deadline)
    for name,binding in review['controls'].items():
        _require(name.startswith('control/') and name!=GUARD,'Unexpected protected control path')
        _check(writer.root,name,binding,deadline)
    _locks(root,writer.lock_fds)
    return dict(kind=KIND,plan_sha256=review['plan_sha256'],context=bound['context'],
        frozen_files_sha256=digest(bound['files']),review_ref=review_ref,
        selection_completion_ref=review['selection_completion_ref'],
        manifest_sha256=digest(review['stage']['manifest']),controls_sha256=digest(review['controls']))


def _storage(writer,review,stages,inventory):
    history=verify_pilot_history(writer,review['plan_sha256'],review['pilot_completion_ref'],
        current_inventory=inventory,stages=stages)
    return bind_storage(writer.root,_document(writer.root/'evidence',review['projection_ref']),
        pilots=history['records'],records=review['records'],rates=review['rates'],selection=None,
        stages=stages,current_inventory=inventory,
        checkpoint_pairs={**history['checkpoint_pairs'],**review['checkpoint_pairs']},lock_fds=writer.lock_fds)


def cleanup_lr_boundary(writer,plan_sha256,selection_completion_ref):
    """A complete LR selection is necessary; every audit is repeated before unlink."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    root=writer.root/'evidence';_locks(root,writer.lock_fds);seen=snapshot(writer.root)['files']
    _require(not any(n.startswith('evidence/'+BASE) or n in (GUARD,GUARD+'.tmp') for n in seen)
        and not (root/'.retention/lr').exists(),'Existing LR cleanup attempt cannot be retried')
    try:
        with _BoundaryGuard(writer):
            # No new evidence file precedes this exact idle selection verifier.
            checked=selection.verify_lr_selection(writer,plan_sha256,selection_completion_ref)
            complete=_document(root,selection_completion_ref);audit=checked['audit']
            attempt=dict(kind=KIND,status='started',plan_sha256=plan_sha256,
                selection_completion_ref=selection_completion_ref,retry_allowed=False)
            path=writer.publish_json('evidence/'+BASE+'-attempt.json',attempt)
            attempt_ref=dict(path=BASE+'-attempt.json',sha256=file_digest(path))
            inventory=_inventory(root)
            _require(inventory==selection._add(root,audit['inventory'],attempt_ref),'Pre-cleanup inventory changed')
            review=dict(kind=KIND,status='all_lr_screens_reviewed_before_cleanup',plan_sha256=plan_sha256,
                projection_ref=complete['projection_ref'],pilot_completion_ref=complete['pilot_completion_ref'],
                selection_completion_ref=selection_completion_ref,selection_ref=checked['selection_ref'],attempt=attempt_ref,
                context=audit['context'],frozen_files=audit['frozen_files'],inventory=inventory,
                records=audit['prefix']['records'],checkpoint_pairs=audit['prefix']['checkpoint_pairs'],
                rates=checked['rates'],stage=audit['stage'],pilot_stage=audit['pilot_history']['chain_stage'],
                controls=_controls(writer,GUARD),storage=audit['storage'],adopted=False,next_stage_authorized=False)
            _require(digest(planned_manifest(**_inputs(review)))==digest(review['stage']['manifest']),
                     'LR cleanup manifest differs from complete stage audit')
            path=writer.publish_json('evidence/'+BASE+'-review.json',review)
            review_ref=dict(path=BASE+'-review.json',sha256=file_digest(path))
            expected=selection._add(root,inventory,review_ref)
            def revalidate():return _context(writer,review,review_ref)
            def authorize():
                # The selection verifier has an exact pre-boundary inventory;
                # replay its underlying full audit now that boundary metadata exists.
                fresh=selection._audit(writer,plan_sha256,review['projection_ref'],review['pilot_completion_ref'],review['selection_ref'])
                _require(fresh['inventory']==expected and fresh['prefix']['records']==review['records']
                    and fresh['prefix']['checkpoint_pairs']==review['checkpoint_pairs'] and fresh['rates']==review['rates']
                    and fresh['context']==review['context'] and fresh['frozen_files']==review['frozen_files'],
                    'Full LR audit changed before cleanup')
                # A fresh audit includes our own review/attempt in its evidence.
                # Keep the originally reviewed manifest identity; all additions
                # are nevertheless bound in the executor's protected inventory.
                old_evidence=review['stage']['manifest']['evidence']
                _require(all(fresh['stage']['manifest']['evidence'].get(n)==sha for n,sha in old_evidence.items()),
                         'Original stage evidence changed')
                _require(fresh['stage']['manifest']['delete']==review['stage']['manifest']['delete']
                    and fresh['stage']['manifest']['retain']==review['stage']['manifest']['retain'],
                    'Cleanup candidates changed')
                return dict(inputs=_inputs(review),inventory=fresh['inventory'])
            identity=revalidate()
            execute_boundary(root,'lr',writer.lock_fds,authorize,revalidate)
            execution=dict(path='.retention/lr/execution-review.json',sha256=file_digest(root/'.retention/lr/execution-review.json'))
            stage=dict(manifest=review['stage']['manifest'],execution_ref=execution,context_sha256=digest(identity))
            stages=[review['pilot_stage'],stage];current=_inventory(root)
            chain=recheck_chain(root,stages,current);storage=_storage(writer,review,stages,current)
            _require(revalidate()==identity,'LR cleanup context changed after deletion')
            result=dict(kind=KIND,status='lr_cleanup_verified',parent_pid=os.getpid(),plan_sha256=plan_sha256,
                review_ref=review_ref,chain_stage=stage,chain=chain,storage=storage,adopted=False,next_stage_authorized=False,
                limitation='Deleted losing-screen weights cannot be reloaded; original full audits remain retained.')
        guard=dict(path=GUARD,sha256=file_digest(writer.root/GUARD));_guard(writer,guard,KIND,os.getpid(),1800)
        result['guard']=guard;_locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents cleanup completion')
        _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Full storage changed before cleanup completion')
        path=writer.publish_json('evidence/'+BASE+'-complete.json',result)
        return dict(path=BASE+'-complete.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(kind=KIND,status='failed',
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('LR cleanup failure publication failed: '+repr(error)[:1000])
        raise


def _receipt(writer,plan_sha256,completion_ref):
    root=writer.root/'evidence';_sha(plan_sha256)
    _require(completion_ref['path']==BASE+'-complete.json','Exact LR cleanup completion reference required')
    seen=snapshot(writer.root)['files']
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in seen),'Failed/partial cleanup cannot be accepted')
    complete=_document(root,completion_ref);pid=complete.get('parent_pid')
    _require(complete.get('kind')==KIND and complete.get('status')=='lr_cleanup_verified'
        and complete.get('plan_sha256')==plan_sha256 and type(pid) is int and pid>1
        and complete.get('adopted') is False and complete.get('next_stage_authorized') is False,'Complete LR cleanup identity required')
    _require(complete['guard']['path']==GUARD,'Exact cleanup guard required');_guard(writer,complete['guard'],KIND,pid,1800)
    ref=complete['review_ref'];_require(ref['path']==BASE+'-review.json','Exact pre-cleanup review required')
    review=_document(root,ref)
    _require(review['kind']==KIND and review['status']=='all_lr_screens_reviewed_before_cleanup'
        and review['plan_sha256']==plan_sha256 and review['adopted'] is False
        and review['next_stage_authorized'] is False,'Original complete LR audit required')
    selected=_document(root,review['selection_completion_ref']);choice=_document(root,review['selection_ref'])
    _require(review['selection_completion_ref']['path']==selection.BASE+'-complete.json'
        and review['selection_ref']['path']==selection.CHOICE and selected['kind']==selection.KIND
        and selected['status']=='lr_selection_reviewed_pending_cleanup' and selected['plan_sha256']==plan_sha256
        and selected['selection_ref']==review['selection_ref'] and selected['projection_ref']==review['projection_ref']
        and selected['pilot_completion_ref']==review['pilot_completion_ref']
        and choice==review['rates']==selection.choose_rates(review['records'])
        and selected['audit']['prefix']['records']==review['records']
        and selected['audit']['prefix']['checkpoint_pairs']==review['checkpoint_pairs'],
        'Original saved selection/pairs changed')
    _require(review['attempt']['path']==BASE+'-attempt.json','Exact LR cleanup attempt required')
    _require(_document(root,review['attempt'])==dict(kind=KIND,status='started',plan_sha256=plan_sha256,
        selection_completion_ref=review['selection_completion_ref'],retry_allowed=False),'LR cleanup attempt differs')
    manifest=planned_manifest(**_inputs(review))
    _require(digest(manifest)==digest(review['stage']['manifest'])==digest(complete['chain_stage']['manifest']),
             'LR manifest reconstruction differs')
    return complete,review,ref


def verify_lr_boundary(writer,plan_sha256,completion_ref):
    """Read-only idle post-cleanup gate, before any rank request; requires outer guard."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');root=writer.root/'evidence'
    _locks(root,writer.lock_fds)
    complete,review,ref=_receipt(writer,plan_sha256,completion_ref)
    identity=_context(writer,review,ref,completion_ref)
    _require(complete['chain_stage']['context_sha256']==digest(identity),'LR execution context differs')
    stages=[review['pilot_stage'],complete['chain_stage']];current=_inventory(root)
    chain=recheck_chain(root,stages,current);storage=_storage(writer,review,stages,current)
    _context(writer,review,ref,completion_ref);_locks(root,writer.lock_fds)
    return dict(kind=KIND,status='lr_boundary_verified_pending_rank',records=review['records'],rates=review['rates'],
        checkpoint_pairs=review['checkpoint_pairs'],stages=stages,chain=chain,storage=storage,
        selection_ref=review['selection_ref'],pilot_completion_ref=review['pilot_completion_ref'],
        projection_ref=review['projection_ref'],adopted=False,dispatch_authorized=False,next_stage_authorized=False)


def _rank_append(name):
    """Declared pre-cleanup rank output names, never a content/lineage audit."""
    slots={f'{i:02d}' for i in range(9,15)}
    parts=name.split('/')
    if len(parts)>=3 and parts[0]=='quality' and parts[1] in slots:
        relative='/'.join(parts[2:])
        return relative in QUALITY_ARTIFACTS or relative=='checkpoints/h200.pt' or bool(
            re.fullmatch(r'checkpoints/final_001200_[0-9]+\.pt',relative))
    if len(parts)==2 and parts[0]=='meta':
        allowed={'rank-selection.json'}
        stems={'rank-boundary-selection'}|{f'quality-{phase}-{slot}'
            for slot in slots for phase in ('rank-startup','readiness','dispatch')}
        for stem in stems:
            allowed.update(stem+suffix+'.json' for suffix in ('','-attempt','-review','-complete'))
        return parts[1] in allowed
    return False


def verify_lr_history(writer,plan_sha256,completion_ref,*,current_inventory):
    """Read-only LR history with declared rank appends, before rank cleanup.

    Caller owns both locks, has inspected workers and supplies an outer guard
    and exact immutable current inventory. New rank bytes are hash-checked, but
    receive no scientific, storage, selection or dispatch credit. All selected
    LR anchor weights must still exist. A later rank-cleanup consumer must bind
    its own complete scientific review and cumulative deletion chain.
    """
    _require(isinstance(writer,MetadataWriter),'Bounded writer required')
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    complete,review,ref=_receipt(writer,plan_sha256,completion_ref)
    _check_inventory(root,current_inventory)
    baseline=set(review['inventory'])|{ref['path'],completion_ref['path']}
    appended={}
    for name,entry in current_inventory.items():
        if name in baseline or name.startswith('.retention/lr/'):continue
        _require(_rank_append(name),'Unreviewed append outside declared rank continuation namespace')
        appended[name]=entry
    identity=_context(writer,review,ref,completion_ref,appended_inventory=appended)
    _require(complete['chain_stage']['context_sha256']==digest(identity),'LR historical context differs')
    stages=[review['pilot_stage'],complete['chain_stage']]
    history=verify_pilot_history(writer,plan_sha256,review['pilot_completion_ref'],
        current_inventory=current_inventory,stages=stages)
    chain=recheck_chain(root,stages,current_inventory)
    _require(history['chain']==chain,'Pilot/LR historical chain differs')
    _require(_context(writer,review,ref,completion_ref,appended_inventory=appended)==identity,
             'LR history changed during verification')
    _check_inventory(root,current_inventory);_locks(root,writer.lock_fds);snapshot(writer.root)
    return dict(kind='wide_hybrid_lr_history_v2',records=review['records'],rates=review['rates'],
        checkpoint_pairs=review['checkpoint_pairs'],stages=stages,chain=chain,
        selection_ref=review['selection_ref'],pilot_completion_ref=review['pilot_completion_ref'],
        projection_ref=review['projection_ref'],lr_history_verified=True,rank_audits_bound=False,
        storage_verified=False,adopted=False,dispatch_authorized=False,next_stage_authorized=False)

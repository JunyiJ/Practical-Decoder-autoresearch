"""Strict idle replay of compact slot17 dispatch; no slot18 append or launch.

Caller owns both worker locks, inspects processes and holds an outer resource
guard. Historical PID/descriptors identify evidence, never a process to resume.
All original transport, selected/fresh science and remaining storage are audited.
"""
import shutil

from .protocol import digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_rank_selection_v2 import _storage
from . import tokenizer_wide_hybrid_quality_acceptance_v2 as acceptance
from . import tokenizer_wide_hybrid_later_confirmation_dispatch_v2 as dispatch
from . import tokenizer_wide_hybrid_later_confirmation_startup_v2 as startup
from . import tokenizer_wide_hybrid_first_confirmation_history_v2 as first
from . import tokenizer_wide_hybrid_confirmation_history_v2 as previous

KIND='wide_hybrid_later_confirmation_history_v2'


def _binding(root,ref):
    _document(root,dict(path=ref['path'],sha256=ref['sha256']))
    return dict(size=_path(root,ref['path']).stat().st_size,sha256=ref['sha256'])


def _transport(writer,plan_sha,reference,projection_ref,rank_completion_ref):
    root=writer.root/'evidence'
    _require(reference['path']=='meta/quality-dispatch-17.json','Exact slot17 dispatch reference required')
    saved=_document(root,reference);pid=saved.get('parent_pid');review=saved.get('review',{})
    _require(set(saved)=={'kind','status','slot','parent_pid','plan_sha256','projection_ref',
        'pilot_completion_ref','rank_completion_ref','prior_dispatches','guards','review'}
        and saved['kind']==dispatch.KIND and saved['status']=='confirmation_reviewed'
        and type(saved['slot']) is int and saved['slot']==17 and type(pid) is int and pid>1
        and saved['plan_sha256']==plan_sha and saved['projection_ref']==projection_ref
        and saved['rank_completion_ref']==rank_completion_ref,'Exact completed slot17 dispatch required')
    _require(review.get('representation')==dispatch.REPRESENTATION and review.get('kind')==acceptance.LATER_CONFIRM_KIND
        and review.get('status')=='confirmation_child_reviewed_pending_parent_receipt'
        and type(review.get('slot')) is int and review['slot']==17 and review.get('plan_sha256')==plan_sha
        and review.get('prior_transport_verified') is True and all(review.get(k) is False for k in first.NONCLAIMS)
        and not set(review)&{'completed_records','checkpoint_pairs','stages'},'Exact unpromoted compact slot17 acceptance required')
    _require(review['startup']['path']==startup.RECEIPT
        and review['readiness']['path']=='meta/quality-readiness-17.json','Exact original startup/readiness paths required')
    prepared=_document(root,review['startup']);seal=_document(root,review['readiness'])
    prior_ref=prepared['prior_dispatch_ref']
    _require(prior_ref['path']=='meta/quality-dispatch-16.json'
        and review['prior_dispatch_ref']==prior_ref,'Exact second-fresh lineage required')
    prior=_document(root,prior_ref)
    prior_controls=dict(_document(root,prior['review']['readiness'])['control_inventory'])
    for ref in [*prior['guards'].values(),prior['review']['entry_guard'],prior['review']['supervisor_ref']]:
        binding=_binding(writer.root,ref)
        _require(ref['path'] not in prior_controls or prior_controls[ref['path']]==binding,
                 'Conflicting original slot16 control reference')
        prior_controls[ref['path']]=binding
    prior,_,prior_context=previous._transport(writer,plan_sha,prior_ref,projection_ref,rank_completion_ref,
        sealed_controls=prior_controls)
    _require(saved['prior_dispatches']==[*prior['prior_dispatches'],prior_ref],
             'Exact two-prior-fresh lineage required')
    _require(saved['pilot_completion_ref']==prior['pilot_completion_ref'],'Original pilot lineage differs')
    _require(prepared['attempt']['path']==startup.append.ATTEMPT and prepared['guard']['path']==startup.append.GUARD
        and prepared['request']['path']==startup.append.REQUEST,'Exact original slot17 preparation references required')
    attempt=dict(kind=startup.KIND,status='started',slot=17,parent_pid=pid,plan_sha256=plan_sha,
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,prior_dispatch_ref=prior_ref,
        dispatch_authorized=False,deletion_authorized=False)
    _require(digest(_document(root,prepared['attempt']))==digest(attempt),'Original slot17 attempt differs')
    acceptance._guard(writer,prepared['guard'],startup.KIND,pid,1800)
    acceptance.entry._slot17_inventory(seal,prepared,review['startup'])
    before={n:v for n,v in seal['inventory'].items() if n!=review['startup']['path']}
    old_controls=dict(_document(root,prior['review']['readiness'])['control_inventory'])
    for ref in [*prior['guards'].values(),prior['review']['entry_guard'],prior['review']['supervisor_ref'],prepared['guard']]:
        binding=_binding(writer.root,ref)
        _require(ref['path'] not in old_controls or old_controls[ref['path']]==binding,'Conflicting original control reference')
        old_controls[ref['path']]=binding
    preparation=dict(prior_dispatch_ref=prior_ref,attempt_ref=prepared['attempt'],guard_ref=prepared['guard'],
        pending_request_ref=prepared['request'])
    expected=startup._payload(prior,preparation,before,old_controls,
        _document(root,prior['review']['readiness']),prepared['storage'])
    expected['parent_pid']=pid
    _require(digest(prepared)==digest(expected),'Complete original slot17 startup claims differ')
    _require(seal['guard']['path']=='control/quality-readiness-17.json','Exact original readiness guard required')
    acceptance._guard(writer,seal['guard'],acceptance.readiness.KIND,pid,1800)
    expected_controls=dict(old_controls,**{seal['guard']['path']:_binding(writer.root,seal['guard'])})
    _require(seal['control_inventory']==expected_controls,'Exact original sealed control inventory required')
    old_inventory=dict(before)
    for ref in (prior_ref,prepared['attempt'],prepared['request']):
        _require(old_inventory.get(ref['path'])==_binding(root,ref),'Original preparation inventory differs')
        old_inventory.pop(ref['path'])
    _require(digest(old_inventory)==prior['review']['evidence_inventory_sha256'],
             'Exact original slot16 post-child inventory required')
    _require(set(saved['guards'])=={'prepare','child','review'},'Complete slot17 parent guards required')
    for phase,seconds in [('prepare',1800),('child',7200),('review',1800)]:
        ref=saved['guards'][phase]
        _require(ref['path']==f'control/quality-controller-{phase}-17.json','Exact original parent guard path required')
        acceptance._guard(writer,ref,dispatch.KIND,pid,seconds)
    _require(set(review['child'])=={'launch','process','worker'},'Complete slot17 child references required')
    for key,tail in [('launch','process/launch.json'),('process','process/process.json'),('worker','worker-outputs.json')]:
        _require(review['child'][key]['path']=='quality/17/'+tail,'Exact original child reference path required')
        _document(root,review['child'][key])
    _require(review['entry_guard']['path']=='control/quality-entry-17.json'
        and review['supervisor_ref']['path']=='control/process-quality-17.json','Exact child control references required')
    for ref in [*saved['guards'].values(),review['entry_guard'],review['supervisor_ref']]:
        _require(ref['path'] not in expected_controls,'Current child controls overlap sealed history')
        expected_controls[ref['path']]=_binding(writer.root,ref)
    _require(acceptance.readiness._controls(writer,'not-a-control')==expected_controls,
             'Exact original controls required; no later controls allowed')
    launch=_document(root,review['child']['launch']);argv=launch.get('argv')
    _require(type(argv) is list and len(argv)==16 and all(type(v) is str for v in argv)
        and argv[11].isdigit() and argv[13].isdigit(),'Historical canonical argv required')
    identity=dict(parent_pid=pid,lock_fds=(int(argv[11]),int(argv[13])))
    return saved,identity,dict(prepared=prepared,seal=seal,prior=prior,before_startup=before,prior_inventory=old_inventory,prior_context=prior_context)


def _prechild_storage(writer,context,current):
    """Reconstruct pre-slot17 capacity, without credit for its later weights."""
    entry=acceptance.entry;root=writer.root/'evidence';seal=context['seal'];prepared=context['prepared']
    records,pairs,rates,selection,pilot_ref=entry._slot17_history(writer,seal,prepared)
    _,_,pilot_stage,_=entry._receipt(writer,seal['plan_sha256'],pilot_ref)
    state=entry.remaining_projection(_document(root,seal['projection_ref']),pilot_stage['records'],records,rates,selection,3)
    logical=occupied=0;inodes=set()
    for pair in pairs.values():
        for record in pair.values():
            path=_path(root,record['path']);stat=path.stat();inode=(stat.st_dev,stat.st_ino)
            _require(stat.st_nlink==1 and inode not in inodes and stat.st_size==record['size']
                and seal['inventory'][record['path']]==dict(size=record['size'],sha256=record['sha256']),
                'Original selected/fresh checkpoint capacity changed')
            inodes.add(inode);logical+=stat.st_size;occupied+=min(stat.st_size,stat.st_blocks*512)
    expected=dict(current,projection=state,required_free_bytes=entry.required_free(state,occupied),
        occupied_checkpoint_capacity_bytes=occupied,checkpoint_logical_bytes=logical,cleanup_boundaries_verified=3)
    _storage(prepared['storage'],expected,context['before_startup'])
    _storage(context['prior']['review']['storage'],expected,context['prior_inventory'])
    _require(context['seal']['required_free_bytes']==expected['required_free_bytes'],'Historical readiness capacity differs')
    previous._prechild_storage(writer,context['prior_context'],current)


def verify_confirmation_dispatch(writer,slot,plan_sha256,completion_ref,*,projection_ref,rank_completion_ref):
    """Strict idle slot17 replay: no append, process resumption or launch grant."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and slot==17,'Exact idle slot17 replay required')
    _sha(plan_sha256);root=writer.root/'evidence';_locks(root,writer.lock_fds);census=snapshot(writer.root)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents replay')
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in census['files']),'Failed/partial history cannot be accepted')
    inventory=acceptance._inventory(root);controls=acceptance.readiness._controls(writer,'not-a-control')
    saved,identity,context=_transport(writer,plan_sha256,completion_ref,projection_ref,rank_completion_ref)
    historical={n:v for n,v in inventory.items() if n!=completion_ref['path']};review=saved['review']
    _require(review['evidence_inventory_sha256']==digest(historical),'Exact slot17 post-child inventory required; no append')
    actual=acceptance._review_completed(writer,17,review['startup'],review['readiness'],
        _document(root,review['child']['process']),historical_identity=identity)
    compact=dict(actual,representation=dispatch.REPRESENTATION)
    for key in ('completed_records','checkpoint_pairs','stages'):compact[key+'_sha256']=digest(compact.pop(key))
    dynamic={'storage','evidence_inventory_sha256'}
    _require(set(review)==set(compact) and digest({k:v for k,v in review.items() if k not in dynamic})==
        digest({k:v for k,v in compact.items() if k not in dynamic}),'Reconstructed full slot17 audit differs from compact receipt')
    _storage(review['storage'],actual['storage'],historical)
    _prechild_storage(writer,context,actual['storage'])
    _require(actual['evidence_inventory_sha256']==digest(inventory),'Current evidence changed during replay')
    again,_,_=_transport(writer,plan_sha256,completion_ref,projection_ref,rank_completion_ref)
    _require(again==saved and acceptance.readiness._controls(writer,'not-a-control')==controls,'Transport/control changed during replay')
    acceptance._check_inventory(root,inventory);_locks(root,writer.lock_fds)
    _require(snapshot(writer.root)==census and not (writer.root/'STOP').exists()
        and not (writer.root/'STOP').is_symlink(),'History census or STOP changed')
    _require(shutil.disk_usage(writer.root).free>=actual['storage']['required_free_bytes'],'Complete remaining capacity changed')
    return dict(kind=KIND,slot=17,dispatch_ref=completion_ref,plan_sha256=plan_sha256,projection_ref=projection_ref,
        rank_completion_ref=rank_completion_ref,records=actual['completed_records'],checkpoint_pairs=actual['checkpoint_pairs'],
        stages=actual['stages'],rates=actual['rates'],selection=actual['selection'],audited_indices=actual['audited_indices'],
        storage=actual['storage'],historical_transport_verified=True,confirmation_audits_bound=True,
        adopted=False,dispatch_authorized=False,deletion_authorized=False,next_stage_authorized=False)

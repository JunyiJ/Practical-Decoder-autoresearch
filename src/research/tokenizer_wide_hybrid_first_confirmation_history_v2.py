"""Read-only idle replay of the first compact fresh parent receipt.

Own both worker locks, inspect processes and hold an outer resource guard.
Reconstruct the complete scientific/transport/storage audit under the historical
parent identity. No child, retry, later-slot append, selection or dispatch.
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
from . import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch
from . import tokenizer_wide_hybrid_first_confirmation_startup_v2 as startup

KIND='wide_hybrid_first_confirmation_history_v2'
REPRESENTATION='wide_hybrid_first_confirmation_acceptance_digest_v1'
NONCLAIMS=('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized')


def _transport(writer,plan_sha,reference,projection_ref,rank_completion_ref):
    root=writer.root/'evidence'
    _require(reference['path']=='meta/quality-dispatch-15.json','Exact first-fresh dispatch reference required')
    saved=_document(root,reference);pid=saved.get('parent_pid');review=saved.get('review',{})
    _require(set(saved)=={'kind','status','slot','parent_pid','plan_sha256','projection_ref',
        'pilot_completion_ref','rank_completion_ref','prior_dispatches','guards','review'}
        and saved.get('kind')==dispatch.KIND and saved.get('status')=='first_confirmation_reviewed'
        and type(saved.get('slot')) is int and saved['slot']==15 and type(pid) is int and pid>1
        and saved['plan_sha256']==plan_sha and saved['projection_ref']==projection_ref
        and saved['rank_completion_ref']==rank_completion_ref and saved['prior_dispatches']==[],
        'Exact completed first-fresh dispatch required')
    _require(review.get('representation')==REPRESENTATION and review.get('kind')==acceptance.CONFIRM_KIND
        and review.get('status')=='first_confirmation_child_reviewed_pending_parent_receipt'
        and type(review.get('slot')) is int and review['slot']==15 and review.get('plan_sha256')==plan_sha
        and all(review.get(k) is False for k in NONCLAIMS)
        and not set(review)&{'completed_records','checkpoint_pairs','stages'},
        'Exact unpromoted compact first-fresh acceptance required')
    _require(set(saved['guards'])=={'prepare','child','review'},'Complete historical parent guards required')
    for phase,seconds in [('prepare',1800),('child',7200),('review',1800)]:
        ref=saved['guards'][phase]
        _require(ref['path']==f'control/quality-controller-{phase}-15.json','Exact historical parent guard path required')
        acceptance._guard(writer,ref,dispatch.KIND,pid,seconds)
    _require(review['startup']['path']=='meta/quality-confirm-startup-15.json'
        and review['readiness']['path']=='meta/quality-readiness-15.json','Exact historical startup/readiness required')
    prepared=_document(root,review['startup']);seal=_document(root,review['readiness'])
    _require(prepared.get('kind')==startup.KIND and prepared.get('status')=='first_confirmation_pending_predispatch_review'
        and type(prepared.get('slot')) is int and prepared['slot']==15
        and type(prepared.get('parent_pid')) is int and prepared['parent_pid']==pid
        and prepared.get('plan_sha256')==plan_sha and prepared.get('projection_ref')==projection_ref
        and prepared.get('rank_completion_ref')==rank_completion_ref
        and prepared.get('pilot_completion_ref')==saved['pilot_completion_ref']
        and all(prepared.get(k) is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized')),
        'Original first-fresh preparation identity differs')
    _require(prepared['guard']['path']=='control/quality-confirm-startup-15.json','Exact startup guard required')
    acceptance._guard(writer,prepared['guard'],startup.KIND,pid,1800)
    attempt=dict(kind=startup.KIND,status='started',slot=15,plan_sha256=plan_sha,
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref,dispatch_authorized=False)
    _require(prepared['attempt']['path']=='meta/quality-confirm-startup-15-attempt.json'
        and _document(root,prepared['attempt'])==attempt,'Original startup attempt differs')
    original=dict(prepared['inventory'])
    for ref in (prepared['request'],review['startup']):
        _require(ref['path'] not in original,'Startup inventory overlap')
        _document(root,ref);original[ref['path']]=dict(size=_path(root,ref['path']).stat().st_size,sha256=ref['sha256'])
    _require(original==seal['inventory'],'Complete original sealed inventory differs')
    _require(set(review['child'])=={'launch','process','worker'},'Complete historical child references required')
    for name,tail in [('launch','process/launch.json'),('process','process/process.json'),('worker','worker-outputs.json')]:
        _require(review['child'][name]['path']=='quality/15/'+tail,'Exact historical child reference required')
        _document(root,review['child'][name])
    launch=_document(root,review['child']['launch']);argv=launch.get('argv')
    _require(type(argv) is list and len(argv)==16 and all(type(v) is str for v in argv)
        and argv[11].isdigit() and argv[13].isdigit(),'Historical canonical argv required')
    identity=dict(parent_pid=pid,lock_fds=(int(argv[11]),int(argv[13])))
    # Full canonical argv, successful exit, self-inclusive byte census, sealed
    # controls and child/readiness guards are independently checked by replay.
    return saved,identity


def verify_first_confirmation_dispatch(writer,plan_sha256,completion_ref,*,projection_ref,rank_completion_ref):
    """Strict idle verification before any later fresh request/attempt exists."""
    return _verify(writer,plan_sha256,completion_ref,projection_ref=projection_ref,
                   rank_completion_ref=rank_completion_ref)


def _verify(writer,plan_sha256,completion_ref,*,projection_ref,rank_completion_ref,preparation=None,startup_ref=None):
    """Shared replay; only the separately checked slot16 preparation may append."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    root=writer.root/'evidence';_locks(root,writer.lock_fds)
    census=snapshot(writer.root)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents history review')
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in census['files']),
             'Failed/partial history cannot be accepted')
    inventory=acceptance._inventory(root);controls=acceptance.readiness._controls(writer,'not-a-control')
    saved,identity=_transport(writer,plan_sha256,completion_ref,projection_ref,rank_completion_ref)
    _require(startup_ref is None or preparation is not None,'Startup receipt requires explicit preparation')
    appended={};pending=None;extra_control=None;startup_report=None
    if preparation is not None:
        from .tokenizer_wide_hybrid_confirmation_append_v2 import _append
        appended,pending,extra_control=_append(writer,plan_sha256,completion_ref,projection_ref,
                                              rank_completion_ref,saved,preparation)
    if startup_ref is not None:
        from .tokenizer_wide_hybrid_confirmation_startup_v2 import _receipt_binding
        startup_report,binding=_receipt_binding(writer,startup_ref,saved,preparation,inventory,controls)
        appended[startup_ref['path']]=binding
    review=saved['review'];historical=dict(inventory);historical.pop(completion_ref['path'])
    for name,binding in appended.items():
        _require(historical.get(name)==binding,'Declared preparation append binding differs')
        historical.pop(name)
    _require(review.get('evidence_inventory_sha256')==digest(historical),
             'Exact original post-child inventory required; later evidence needs separate integration')
    # The original seal contains all controls preceding the first child. Only
    # that child's finished controls and the three finished parent phases follow.
    expected_controls=dict(_document(root,review['readiness'])['control_inventory'])
    for ref in [*saved['guards'].values(),review['entry_guard'],review['supervisor_ref']]:
        binding=dict(size=_path(writer.root,ref['path']).stat().st_size,sha256=ref['sha256'])
        _require(ref['path'] not in expected_controls or expected_controls[ref['path']]==binding,
                 'Historical control reference conflict')
        expected_controls[ref['path']]=binding
    if extra_control is not None:
        name,binding=extra_control
        _require(name not in expected_controls,'Preparation control overlaps original history')
        expected_controls[name]=binding
    _require(controls==expected_controls,'Exact original controls plus declared preparation guard required')
    process=_document(root,review['child']['process'])
    actual=acceptance._review_completed(writer,15,review['startup'],review['readiness'],process,
                                         historical_identity=identity,pending_confirmation_request=pending)
    compact=dict(actual,representation=REPRESENTATION)
    for key in ('completed_records','checkpoint_pairs','stages'):
        compact[key+'_sha256']=digest(compact.pop(key))
    dynamic={'storage','evidence_inventory_sha256'}
    _require(set(review)==set(compact) and digest({k:v for k,v in review.items() if k not in dynamic})==
        digest({k:v for k,v in compact.items() if k not in dynamic}),
        'Reconstructed complete fresh audit differs from compact receipt')
    _storage(review['storage'],actual['storage'],historical)
    if startup_report is not None:
        before_receipt={n:v for n,v in inventory.items() if n!=startup_ref['path']}
        _storage(startup_report['storage'],actual['storage'],before_receipt)
    _require(actual['evidence_inventory_sha256']==digest(inventory),'Current evidence changed during history replay')
    again,_=_transport(writer,plan_sha256,completion_ref,projection_ref,rank_completion_ref)
    _require(again==saved and acceptance.readiness._controls(writer,'not-a-control')==controls,
             'Historical transport/control changed during replay')
    acceptance._check_inventory(root,inventory);_locks(root,writer.lock_fds)
    _require(snapshot(writer.root)==census and not (writer.root/'STOP').exists()
        and not (writer.root/'STOP').is_symlink(),'History census or STOP changed')
    _require(shutil.disk_usage(writer.root).free>=actual['storage']['required_free_bytes'],
             'Complete remaining capacity changed during history replay')
    return dict(kind=KIND,dispatch_ref=completion_ref,plan_sha256=plan_sha256,projection_ref=projection_ref,
        rank_completion_ref=rank_completion_ref,records=actual['completed_records'],
        checkpoint_pairs=actual['checkpoint_pairs'],stages=actual['stages'],rates=actual['rates'],
        selection=actual['selection'],audited_indices=actual['audited_indices'],storage=actual['storage'],
        historical_transport_verified=True,first_confirmation_audits_bound=True,
        adopted=False,dispatch_authorized=False,next_stage_authorized=False)

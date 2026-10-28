"""Bounded rank request preparation; no child, retry, selection or dispatch.

Caller owns both locks and an outer resource guard, and inspects actual workers
first. Complete cleanup history, live prefix and future storage are verified
before creating a request. The outer guard also covers final receipt checks.
The eventual parent/child dispatcher must independently recheck this receipt.
"""
import os
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_lr_boundary_v2 import verify_lr_history
from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
from .tokenizer_wide_hybrid_quality_audit_v2 import bind_rank_prefix
from .tokenizer_wide_hybrid_quality_acceptance_v2 import _guard
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign,PROJECTION
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_rules import lr_specs,rank_specs
from .tokenizer_wide_hybrid_scores import _require

KIND='wide_hybrid_rank_request_startup_v2'


class _PreparationGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot):
        super().__init__(writer,slot,'startup',1800)
        self.name=f'control/quality-rank-startup-{slot:02d}.json'


def _frozen(writer,slot,plan_sha,projection_ref,rates):
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
             'STOP prevents rank request preparation')
    root=writer.root/'evidence';bound=bind_campaign(root/'meta/plan.json',plan_sha)
    _require(projection_ref['path'].startswith('meta/')
        and bound['files'].get(str(_path(root,projection_ref['path'])))==projection_ref['sha256']
        and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
        'Rank startup projection differs from frozen copy')
    requests=[]
    for i,spec in enumerate(lr_specs()+rank_specs(rates)[:slot-8]):
        options=bound['requests']['quality'][str(i)]
        matches=[r for r in options if digest(r.get('spec'))==digest(spec)]
        _require(len(matches)==1 and (i>=9 or len(options)==1)
            and matches[0].get('result_path')==str(root/f'quality/{i:02d}/result.json'),
            'Unique independently frozen conditional request required')
        requests.append(matches[0])
    return bound,requests


def _audit(writer,slot,plan_sha,projection_ref,lr_completion_ref):
    """Idle-only full preparation gate, before the next request exists."""
    root=writer.root/'evidence';_locks(root,writer.lock_fds);snapshot(writer.root)
    inventory=_inventory(root)
    history=verify_lr_history(writer,plan_sha,lr_completion_ref,current_inventory=inventory)
    _require(history['projection_ref']==projection_ref,'Historical projection reference differs')
    bound,requests=_frozen(writer,slot,plan_sha,projection_ref,history['rates'])
    plan=_document(root,dict(path='meta/plan.json',sha256=plan_sha))
    prefix=bind_rank_prefix(root,context=bound['context'],lr_records=history['records'],rates=history['rates'],
        lr_requests=requests[:9],rank_requests=requests[9:-1],
        evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
        numerical_ref=plan['numerical_ref'],rates_ref=history['selection_ref'],projection_ref=projection_ref,
        training_token_sha256=plan['training_token_sha256'])
    _require(prefix['records'][:9]==history['records'] and prefix['rates']==history['rates'],
             'Live prefix differs from verified LR history')
    for name,pair in prefix['checkpoint_pairs'].items():
        if int(name.split('/')[1])<9:
            _require(pair==history['checkpoint_pairs'][name],'Live LR anchor differs from original pair')
    pilot=verify_pilot_history(writer,plan_sha,history['pilot_completion_ref'],
        current_inventory=inventory,stages=history['stages'])
    pairs={**pilot['checkpoint_pairs'],**history['checkpoint_pairs'],**prefix['checkpoint_pairs']}
    storage=bind_storage(writer.root,_document(root,projection_ref),pilots=pilot['records'],
        records=prefix['records'],rates=history['rates'],selection=None,stages=history['stages'],
        current_inventory=inventory,checkpoint_pairs=pairs,lock_fds=writer.lock_fds)
    again,again_requests=_frozen(writer,slot,plan_sha,projection_ref,history['rates'])
    _require(digest(again)==digest(bound) and again_requests==requests,'Frozen rank inputs changed during audits')
    _check_inventory(root,inventory);_locks(root,writer.lock_fds)
    _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],
             'Full remaining storage changed during rank preparation')
    return dict(context=bound['context'],frozen_files=bound['files'],requests=requests,
        inventory=inventory,history=history,prefix=prefix,pilot=pilot,checkpoint_pairs=pairs,storage=storage)


def prepare_rank_request(writer,slot,plan_sha256,*,projection_ref,lr_completion_ref):
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 9<=slot<=14,
             'Rank preparation supports exact slots9 through14 only')
    _sha(plan_sha256);root=writer.root/'evidence';_locks(root,writer.lock_fds)
    base=f'meta/quality-rank-startup-{slot:02d}';guard_name=f'control/quality-rank-startup-{slot:02d}.json'
    seen=snapshot(writer.root)['files']
    _require(not (root/f'quality/{slot:02d}').exists(),'Next rank slot must be absent before idle review')
    _require(not any(n.startswith('evidence/'+base) or n in (guard_name,guard_name+'.tmp') for n in seen),
             'Existing rank preparation cannot be retried')
    try:
        with _PreparationGuard(writer,slot):
            attempt=dict(kind=KIND,status='started',slot=slot,plan_sha256=plan_sha256,
                projection_ref=projection_ref,lr_completion_ref=lr_completion_ref,dispatch_authorized=False)
            path=writer.publish_json('evidence/'+base+'-attempt.json',attempt)
            attempt_ref=dict(path=base+'-attempt.json',sha256=file_digest(path))
            checked=_audit(writer,slot,plan_sha256,projection_ref,lr_completion_ref)
            request=checked['requests'][-1]
            path=writer.publish_json(f'evidence/quality/{slot:02d}/request.json',request)
            request_ref=dict(path=f'quality/{slot:02d}/request.json',sha256=file_digest(path))
            expected=dict(checked['inventory']);expected[request_ref['path']]=dict(size=path.stat().st_size,sha256=request_ref['sha256'])
            report=dict(kind=KIND,status='rank_request_pending_predispatch_review',parent_pid=os.getpid(),slot=slot,
                plan_sha256=plan_sha256,projection_ref=projection_ref,lr_completion_ref=lr_completion_ref,
                pilot_completion_ref=checked['history']['pilot_completion_ref'],selection_ref=checked['history']['selection_ref'],
                attempt=attempt_ref,request=request_ref,context=checked['context'],frozen_files=checked['frozen_files'],
                inventory=checked['inventory'],completed_records=checked['prefix']['records'],rates=checked['history']['rates'],
                checkpoint_pairs=checked['checkpoint_pairs'],stages=checked['history']['stages'],
                audited_indices=checked['prefix']['audited_indices'],storage=checked['storage'],
                adopted=False,dispatch_authorized=False,deletion_authorized=False,next_stage_authorized=False,selection=None)
        guard=dict(path=guard_name,sha256=file_digest(writer.root/guard_name))
        _guard(writer,guard,KIND,os.getpid(),1800);report['guard']=guard
        again,requests=_frozen(writer,slot,plan_sha256,projection_ref,report['rates'])
        _require(again['context']==checked['context'] and again['files']==checked['frozen_files']
            and requests==checked['requests'],'Frozen rank inputs changed before publication')
        history=verify_lr_history(writer,plan_sha256,lr_completion_ref,current_inventory=expected)
        # The chain's current-inventory digest/count legitimately changes when
        # the pending request is appended. Its entire new inventory was verified
        # above; only the historical fields must remain byte-identical.
        _require({k:v for k,v in history.items() if k!='chain'}==
                 {k:v for k,v in checked['history'].items() if k!='chain'},
                 'LR history changed before rank completion')
        _check_inventory(root,expected);_locks(root,writer.lock_fds)
        _require(shutil.disk_usage(writer.root).free>=checked['storage']['required_free_bytes'],
                 'Full remaining storage changed before rank completion')
        path=writer.publish_json('evidence/'+base+'.json',report)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Rank preparation failure publication failed: '+repr(error)[:1000])
        raise

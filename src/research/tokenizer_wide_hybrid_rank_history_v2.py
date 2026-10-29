"""Retained rank history with declared fresh appends; no fresh audit or launch.

Caller owns both locks, inspects workers and holds an outer resource guard.
Names and bytes of fresh files are bound here, never accepted as scientific or
transport evidence. Fresh prefix, capacity and process gates remain separate.
"""
import re

from .protocol import digest
from .tokenizer_wide_hybrid_rank_boundary_v2 import (_receipt,_context,_controls,_inventory,_check_inventory,
    _guard,_locks,recheck_chain,verify_pilot_history,selection,GUARD,KIND,_document,_require)
from .tokenizer_wide_hybrid_storage_v2 import remaining_projection,required_free


def fresh_artifact(name):
    from .tokenizer_wide_hybrid_quality_audit_v2 import QUALITY_ARTIFACTS
    parts=name.split('/');slots={f'{i:02d}' for i in range(15,24)}
    if len(parts)>=3 and parts[0]=='quality' and parts[1] in slots:
        relative='/'.join(parts[2:])
        return relative in QUALITY_ARTIFACTS or relative=='checkpoints/h200.pt' or bool(
            re.fullmatch(r'checkpoints/final_001200_[0-9]+\.pt',relative))
    if len(parts)==2 and parts[0]=='meta':
        return any(parts[1]==f'quality-{phase}-{slot}{suffix}.json'
            for phase in ('confirm-startup','readiness','dispatch') for slot in slots
            for suffix in ('','-attempt','-review','-complete'))
    return False


def fresh_control(name):
    return bool(re.fullmatch(r'control/(?:quality-(?:confirm-startup|readiness|entry|controller-(?:prepare|child|review))|process-quality)-(?:1[5-9]|2[0-3])\.json',name))


def verify_rank_history(writer,plan_sha256,completion_ref,*,current_inventory,inference_request_ref=None):
    """Verify original selection/deletion proof; never authorize appended trials."""
    saved,review,ref,view=_receipt(writer,plan_sha256,completion_ref);root=writer.root/'evidence'
    _check_inventory(root,current_inventory)
    baseline=set(review['inventory'])|{ref['path'],completion_ref['path']}
    appended={n:v for n,v in current_inventory.items() if n not in baseline and not n.startswith('.retention/rank/')}
    scope={'evidence':{},'controls':{}}
    if inference_request_ref is not None:
        from .tokenizer_wide_hybrid_inference_preparation_v2 import _history_scope
        scope=_history_scope(writer,inference_request_ref,plan_sha256)
    _require(all(fresh_artifact(n) or scope['evidence'].get(n)==v for n,v in appended.items()),'Undeclared fresh evidence append')
    controls=_controls(writer,GUARD)
    added_controls={n:v for n,v in controls.items() if n not in review['controls']}
    _require(all(fresh_control(n) or scope['controls'].get(n)==v for n,v in added_controls.items()),'Undeclared fresh control append')
    identity=_context(writer,review,ref,view,completion_ref,
        appended_inventory=appended,appended_controls=added_controls)
    _require(digest(identity)==saved['context_sha256'] and digest(view['manifest'])==saved['manifest_sha256']
        and saved['execution_ref']['path']=='.retention/rank/execution-review.json','Original rank execution differs')
    stage=dict(manifest=view['manifest'],execution_ref=saved['execution_ref'],context_sha256=saved['context_sha256'])
    stages=view['stages']+[stage];chain=recheck_chain(root,stages,current_inventory)
    pilot=verify_pilot_history(writer,plan_sha256,view['history']['pilot_completion_ref'],current_inventory=current_inventory,stages=stages,inference_request_ref=inference_request_ref)
    _require(pilot['chain']==chain,'Pilot/rank cumulative history differs')
    summary=view['summary']
    for n,pair in pilot['checkpoint_pairs'].items():_require(summary['checkpoint_pairs'][n]==pair,'Original pilot pair changed')
    # Reconstruct the historical observation before completion/fresh appends.
    original={n:v for n,v in current_inventory.items() if n not in appended and n!=completion_ref['path']}
    old_chain=dict(chain,current_inventory_sha256=digest(original),protected_files_verified=len(original))
    _require(digest(old_chain)==saved['chain_sha256'],'Historical rank chain observation differs')
    state=remaining_projection(_document(root,view['selected']['projection_ref']),pilot['records'],
        summary['records'],summary['rates'],summary['selection'],3)
    occupied=0;logical=0;inodes=set()
    for entry in view['manifest']['retain']:
        stat=(root/entry['path']).stat();inode=(stat.st_dev,stat.st_ino)
        _require(stat.st_nlink==1 and inode not in inodes,'Retained selected checkpoint alias')
        inodes.add(inode);logical+=stat.st_size;occupied+=min(stat.st_size,stat.st_blocks*512)
    historical=dict(status='remaining_campaign_storage_verified',projection=state,free_bytes=0,
        required_free_bytes=required_free(state,occupied),occupied_checkpoint_capacity_bytes=occupied,
        checkpoint_logical_bytes=logical,metadata_bytes=0,cleanup_boundaries_verified=3,
        current_inventory_sha256=digest(original),storage_projection_fits=True,current_phase_space_verified=True,
        frozen_plan_verified=False,audit_semantics_verified=False,dispatch_authorized=False)
    selection._storage(saved['storage'],historical,original)
    _require(_context(writer,review,ref,view,completion_ref,appended_inventory=appended,
        appended_controls=added_controls)==identity,'Rank history changed during verification')
    _check_inventory(root,current_inventory);_guard(writer,saved['guard'],KIND,saved['parent_pid'],1800);_locks(root,writer.lock_fds)
    return dict(kind='wide_hybrid_rank_history_v2',records=summary['records'],rates=summary['rates'],selection=summary['selection'],
        checkpoint_pairs=summary['checkpoint_pairs'],stages=stages,chain=chain,
        selection_ref=view['selected']['selection_ref'],rates_ref=view['history']['selection_ref'],
        projection_ref=view['selected']['projection_ref'],pilot_completion_ref=view['history']['pilot_completion_ref'],
        rank_history_verified=True,fresh_audits_bound=False,storage_verified=False,
        adopted=False,dispatch_authorized=False,next_stage_authorized=False)

"""Full pilot audit -> bounded cleanup -> independently verified receipt chain.

No CLI, training, automatic continuation or retry. Caller inspects actual workers
and owns both shared locks throughout. Only a complete frozen v2 campaign and
three reviewed pilot dispatches can reach the deletion executor.
"""
from pathlib import Path
import sys
import time

from .protocol import digest,file_digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign,PROJECTION
from .tokenizer_wide_hybrid_pilot_audit_v2 import bind_pilot_boundary
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard,KIND as DISPATCH_KIND,MODULE
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha,planned_manifest
from .tokenizer_wide_hybrid_cleanup import execute_boundary,_scan,_check,TIMEOUT
from .tokenizer_wide_hybrid_retention_chain import recheck_chain
from .tokenizer_wide_hybrid_storage_v2 import bind_storage
from .tokenizer_wide_hybrid_scores import _require,_json

KIND='wide_hybrid_pilot_boundary_v2'
BASE='meta/pilot-boundary'


def _guard(value,seconds,pid=None):
    limits=dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=seconds)
    sample=value.get('observation',{});elapsed=value.get('elapsed_seconds')
    _require(value.get('kind')==DISPATCH_KIND and value.get('status')=='finished' and value.get('reason') is None
             and type(value.get('pid')) is int and value['pid']>1 and (pid is None or value['pid']==pid)
             and value.get('limits')==limits and set(sample)=={'peak_rss_bytes','free_bytes'}
             and all(type(v) is int and v>0 for v in sample.values())
             and type(elapsed) in (int,float) and 0<=elapsed<seconds
             and violation(sample,elapsed,**limits) is None,'Successful bounded dispatch guard required')


def _dispatches(writer,plan_sha,bound,stage):
    evidence=writer.root/'evidence';refs={};controls={}
    for i in range(3):
        stem=f'pilots/{i:02d}';name=f'meta/pilot-dispatch-{i:02d}.json'
        for suffix in ('-failure.json','-failure.json.tmp','.json.tmp'):
            _require(not (evidence/(f'meta/pilot-dispatch-{i:02d}'+suffix)).exists(),'Failed/partial pilot dispatch')
        sha=file_digest(evidence/name);value=_json(evidence/name,sha);refs[name]=sha
        _require(value.get('kind')==DISPATCH_KIND and value.get('status')=='pilot_reviewed'
                 and type(value.get('slot')) is int and value['slot']==i and value.get('plan_sha256')==plan_sha
                 and all(value.get(k) is False for k in ('adopted','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
                 and digest(value.get('completed_pilots'))==digest(stage['records'][:i+1])
                 and digest(value.get('checkpoint_pairs'))==digest({f'pilot/{j:02d}':stage['checkpoint_pairs'][f'pilot/{j:02d}'] for j in range(i+1)}),
                 'Pilot dispatch identity/prefix differs')
        expected={k:dict(path=stem+'/'+tail,sha256=file_digest(evidence/stem/tail))
                  for k,tail in [('worker','worker-outputs.json'),('process','process/process.json')]}
        _require(digest(value.get('child'))==digest(expected),'Pilot child provenance differs')
        process=_document(evidence,expected['process'])
        startup=value['startup'];_require(startup.get('path')==f'meta/pilot-startup-{i:02d}.json','Startup slot differs')
        start=_document(evidence,startup)
        _require(start.get('kind')=='wide_hybrid_pilot_startup_v2'
                 and start.get('status')=='request_materialized_pending_process_dispatch'
                 and start.get('dispatch_authorized') is False and start.get('deletion_authorized') is False
                 and type(start.get('slot')) is int and start['slot']==i
                 and start.get('plan_sha256')==plan_sha and digest(start.get('context'))==digest(bound['context'])
                 and start.get('request')==dict(path=stem+'/request.json',sha256=file_digest(evidence/stem/'request.json')),
                 'Dispatch startup binding differs')
        launch=_json(evidence/stem/'process/launch.json',file_digest(evidence/stem/'process/launch.json'))
        argv=launch.get('argv',[])
        _require(isinstance(argv,list) and len(argv)==14 and argv[:11]==[
            sys.executable,'-m',MODULE,'worker','--campaign',str(writer.root),'--slot',str(i),
            '--startup-sha256',startup['sha256'],'--global-fd'] and argv[12]=='--campaign-fd'
            and all(type(argv[j]) is str and argv[j].isdigit() and int(argv[j])>2 for j in (11,13))
            and argv[11]!=argv[13] and process.get('argv')==argv
            and type(launch.get('parent_pid')) is int and launch['parent_pid']>1
            and process.get('parent_pid')==launch['parent_pid'] and launch.get('status')=='preflight'
            and launch.get('pid') is None and launch.get('returncode') is None and launch.get('error') is None,
            'Original child launch provenance differs')
        for key in ('required_free_bytes','reserve_bytes','output_limit_bytes','log_limit_bytes','timeout_seconds'):
            _require(type(launch.get(key)) is int and launch[key]==process.get(key),'Launch/process limits differ')
        _require(launch['required_free_bytes']==start['storage']['required_free_bytes'],'Startup launch storage differs')
        entry=writer.root/f'control/pilot-entry-{i:02d}.json';entrysha=file_digest(entry);guard=_json(entry,entrysha)
        _guard(guard,600,process.get('pid'))
        _require(value.get('entry_guard')==dict(path=str(entry),sha256=entrysha,record=guard),'Entry guard receipt differs')
        controller=writer.root/f'control/pilot-controller-{i:02d}.json';controllersha=file_digest(controller)
        _guard(_json(controller,controllersha),1800,launch['parent_pid'])
        controls[str(entry)]=entrysha;controls[str(controller)]=controllersha
    return dict(dispatch_refs=refs,control_refs=controls)


def _inputs(stage):
    return dict(boundary='pilot',records=stage['records'],rates=None,selection=None,
        pairs={i:stage['checkpoint_pairs'][f'pilot/{i:02d}'] for i in range(3)},evidence=stage['manifest']['evidence'])


def _frozen(writer,plan_sha,projection_ref):
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents pilot boundary')
    bound=bind_campaign(writer.root/'evidence/meta/plan.json',plan_sha)
    _require(projection_ref['path'].startswith('meta/')
             and bound['files'].get(str(_path(writer.root/'evidence',projection_ref['path'])))==projection_ref['sha256']
             and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
             'Boundary projection differs from frozen copy')
    return bound


def _context(writer,review,review_ref,*,completion_ref=None,appended_inventory=None):
    """Frozen identity recheck that never requires intentionally removed weights."""
    evidence=writer.root/'evidence';bound=_frozen(writer,review['plan_sha256'],review['projection_ref'])
    _require(digest(bound['files'])==digest(review['frozen_files'])
             and digest(bound['context'])==digest(review['context']),'Frozen boundary identity changed')
    _document(evidence,review_ref)
    inventory=dict(review['inventory']);path=_path(evidence,review_ref['path'])
    inventory[review_ref['path']]=dict(size=path.stat().st_size,sha256=review_ref['sha256'])
    if completion_ref is not None:
        path=_path(evidence,completion_ref['path']);_document(evidence,completion_ref)
        inventory[completion_ref['path']]=dict(size=path.stat().st_size,sha256=completion_ref['sha256'])
    if appended_inventory is not None:
        _require(not set(appended_inventory)&set(inventory),'Appended evidence overlaps protected history')
        inventory.update(appended_inventory)
    candidates={e['path'] for e in review['stage']['manifest']['delete']}
    actual=_scan(evidence,evidence/'.retention/pilot')
    _require(actual<=set(inventory) and set(inventory)-candidates<=actual,'Boundary protected inventory changed')
    deadline=time.monotonic()+TIMEOUT
    for name,entry in inventory.items():
        if name not in candidates:_check(evidence,name,entry,deadline)
    for name,sha in review['dispatch']['control_refs'].items():
        _require(file_digest(Path(name))==sha,'Dispatch guard changed during cleanup')
    return dict(kind=KIND,plan_sha256=review['plan_sha256'],context=bound['context'],
        frozen_files_sha256=digest(bound['files']),review_ref=review_ref,
        manifest_sha256=digest(review['stage']['manifest']),dispatch=review['dispatch'])


def cleanup_pilot_boundary(writer,plan_sha256,*,projection_ref):
    """Fresh complete boundary only; preserve all failed attempts and metadata."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    evidence=writer.root/'evidence'
    with writer._serialized():
        _require(not any((evidence/(BASE+s)).exists() for s in (
            '-attempt.json','-attempt.json.tmp','-review.json','-review.json.tmp',
            '-complete.json','-complete.json.tmp','-failure.json','-failure.json.tmp')),
            'Existing pilot boundary cannot be retried')
        _require(not (evidence/'.retention').exists(),'Existing retention attempt blocks fresh pilot cleanup')
    writer.publish_json('evidence/'+BASE+'-attempt.json',dict(kind=KIND,status='started',plan_sha256=plan_sha256,
        projection_ref=projection_ref,retry_allowed=False))
    try:
        with _Guard(writer,0,'boundary',1800):
            bound=_frozen(writer,plan_sha256,projection_ref)
            plan=_json(evidence/'meta/plan.json',plan_sha256);projection=_document(evidence,projection_ref)
            inventory=_inventory(evidence)
            args=dict(context=bound['context'],requests=bound['requests']['pilots'],
                evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
                numerical_ref=plan['numerical_ref'],projection_ref=projection_ref,
                tokenizer=plan['tokenizer'],training_token_sha256=plan['training_token_sha256'])
            stage=bind_pilot_boundary(evidence,**args);dispatch=_dispatches(writer,plan_sha256,bound,stage)
            storage=bind_storage(writer.root,projection,pilots=stage['records'],records=[],rates=None,selection=None,
                stages=[],current_inventory=inventory,checkpoint_pairs=stage['checkpoint_pairs'],lock_fds=writer.lock_fds)
            review=dict(kind=KIND,status='all_pilots_reviewed_before_cleanup',plan_sha256=plan_sha256,
                projection_ref=projection_ref,context=bound['context'],frozen_files=bound['files'],
                inventory=inventory,stage=stage,dispatch=dispatch,storage=storage,
                adopted=False,next_stage_authorized=False)
            path=writer.publish_json('evidence/'+BASE+'-review.json',review)
            ref=dict(path=BASE+'-review.json',sha256=file_digest(path))
            def revalidate():return _context(writer,review,ref)
            def authorize():
                fresh=bind_pilot_boundary(evidence,**args)
                _require(digest(fresh)==digest(stage),'Pilot audits changed before cleanup')
                current=_inventory(evidence)
                expected=dict(inventory);expected[ref['path']]=dict(size=path.stat().st_size,sha256=ref['sha256'])
                _require(current==expected,'Cleanup authorization inventory changed')
                return dict(inputs=_inputs(fresh),inventory=current)
            identity=revalidate()
            execute_boundary(evidence,'pilot',writer.lock_fds,authorize,revalidate)
            execution=dict(path='.retention/pilot/execution-review.json',
                           sha256=file_digest(evidence/'.retention/pilot/execution-review.json'))
            chain_stage=dict(manifest=stage['manifest'],execution_ref=execution,context_sha256=digest(identity))
            current=_inventory(evidence);chain=recheck_chain(evidence,[chain_stage],current)
            after=bind_storage(writer.root,projection,pilots=stage['records'],records=[],rates=None,selection=None,
                stages=[chain_stage],current_inventory=current,checkpoint_pairs=stage['checkpoint_pairs'],lock_fds=writer.lock_fds)
            _require(digest(revalidate())==digest(identity),'Boundary identity changed after cleanup')
            result=dict(kind=KIND,status='pilot_cleanup_verified',plan_sha256=plan_sha256,review_ref=ref,
                chain_stage=chain_stage,chain=chain,storage=after,adopted=False,next_stage_authorized=False,
                limitation='Deleted pilot weights cannot be reloaded; original full audit evidence is retained.')
        guard_path=writer.root/'control/pilot-boundary-00.json';guard_sha=file_digest(guard_path)
        guard=_json(guard_path,guard_sha);_guard(guard,1800)
        result['boundary_guard']=dict(path=str(guard_path),sha256=guard_sha,record=guard)
        path=writer.publish_json('evidence/'+BASE+'-complete.json',result)
        return dict(path=BASE+'-complete.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(kind=KIND,status='failed',
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Boundary failure publication failed: '+repr(error)[:1000])
        raise


def _receipt(writer,plan_sha256,completion_ref):
    """Validate the original complete pilot boundary identity, without tensors."""
    evidence=writer.root/'evidence';_sha(plan_sha256)
    _require(completion_ref['path']==BASE+'-complete.json','Exact completion reference required')
    for suffix in ('-failure.json','-failure.json.tmp','-review.json.tmp','-complete.json.tmp','-attempt.json.tmp'):
        _require(not (evidence/(BASE+suffix)).exists(),'Failed/partial boundary cannot be accepted')
    complete=_document(evidence,completion_ref)
    _require(complete.get('kind')==KIND and complete.get('status')=='pilot_cleanup_verified'
             and complete.get('plan_sha256')==plan_sha256 and complete.get('adopted') is False
             and complete.get('next_stage_authorized') is False,'Boundary completion identity differs')
    guard_path=writer.root/'control/pilot-boundary-00.json';guard_sha=file_digest(guard_path)
    guard=_json(guard_path,guard_sha);_guard(guard,1800)
    _require(complete.get('boundary_guard')==dict(path=str(guard_path),sha256=guard_sha,record=guard),
             'Boundary controller guard changed')
    ref=complete['review_ref'];_require(ref['path']==BASE+'-review.json','Boundary review location differs')
    review=_document(evidence,ref);stage=review['stage']
    _require(review.get('kind')==KIND and review.get('status')=='all_pilots_reviewed_before_cleanup'
             and review.get('plan_sha256')==plan_sha256 and review.get('adopted') is False
             and review.get('next_stage_authorized') is False,'Pre-deletion review identity differs')
    manifest=planned_manifest(**_inputs(stage))
    _require(digest(manifest)==digest(stage['manifest'])
             and digest(complete['chain_stage']['manifest'])==digest(manifest),'Pilot manifest reconstruction differs')
    return complete,review,stage,ref


def verify_pilot_boundary(writer,plan_sha256,completion_ref):
    """Read-only first-boundary verifier, before any quality request is created.

    Requires an outer time/memory guard and both locks. Reconstructs the manifest
    from protected pre-deletion records, then checks the complete deletion chain.
    It neither reloads deleted weights nor authorizes a subsequent worker.
    """
    evidence=writer.root/'evidence'
    complete,review,stage,ref=_receipt(writer,plan_sha256,completion_ref)
    identity=_context(writer,review,ref,completion_ref=completion_ref)
    _require(complete['chain_stage']['context_sha256']==digest(identity),'Cleanup execution context differs')
    bound=_frozen(writer,plan_sha256,review['projection_ref'])
    _require(_dispatches(writer,plan_sha256,bound,stage)==review['dispatch'],'Protected dispatch provenance differs')
    current=_inventory(evidence);chain=recheck_chain(evidence,[complete['chain_stage']],current)
    projection=_document(evidence,review['projection_ref'])
    storage=bind_storage(writer.root,projection,pilots=stage['records'],records=[],rates=None,selection=None,
        stages=[complete['chain_stage']],current_inventory=current,checkpoint_pairs=stage['checkpoint_pairs'],lock_fds=writer.lock_fds)
    snapshot(writer.root)
    return dict(chain_stage=complete['chain_stage'],records=stage['records'],checkpoint_pairs=stage['checkpoint_pairs'],
        chain=chain,storage=storage,pilot_boundary_verified=True,dispatch_authorized=False)



def verify_pilot_history(writer,plan_sha256,completion_ref,*,current_inventory,stages):
    """Verify retained pilot evidence after quality files have been appended.

    Caller owns both locks and an outer resource guard. Unlike the strict idle
    first-boundary verifier this grants no current storage or quality-stage
    acceptance. All later cleanup receipts must be supplied and pass the chain.
    The original pilot context remains byte-identical; deleted tensors stay gone.
    """
    _require(isinstance(writer,MetadataWriter),'Bounded writer required')
    evidence=writer.root/'evidence';snapshot(writer.root)
    complete,review,stage,ref=_receipt(writer,plan_sha256,completion_ref)
    _require(isinstance(stages,list) and bool(stages)
             and digest(stages[0])==digest(complete['chain_stage']),'Original pilot chain entry required')
    baseline=set(review['inventory'])|{ref['path'],completion_ref['path']}
    appended={}
    for name,entry in current_inventory.items():
        if name in baseline or name.startswith('.retention/pilot/'):continue
        parts=Path(name).parts
        allowed=(len(parts)>=3 and parts[0]=='quality' and parts[1] in {f'{i:02d}' for i in range(24)})
        allowed=allowed or (len(parts)==2 and parts[0]=='meta' and name.endswith('.json') and
            (parts[1].startswith(('quality-','lr-boundary-','rank-boundary-')) or
             parts[1] in ('lr-selection.json','rank-selection.json')))
        allowed=allowed or (len(parts)==3 and parts[0]=='.retention' and parts[1] in ('lr','rank'))
        _require(allowed and not name.endswith('.tmp') and 'failure' not in parts[-1],
                 'Unreviewed or failed append outside quality continuation namespace')
        appended[name]=entry
    identity=_context(writer,review,ref,completion_ref=completion_ref,appended_inventory=appended)
    _require(complete['chain_stage']['context_sha256']==digest(identity),'Pilot historical context differs')
    bound=_frozen(writer,plan_sha256,review['projection_ref'])
    _require(_dispatches(writer,plan_sha256,bound,stage)==review['dispatch'],'Protected pilot dispatch changed')
    chain=recheck_chain(evidence,stages,current_inventory)
    snapshot(writer.root)
    return dict(chain_stage=complete['chain_stage'],records=stage['records'],checkpoint_pairs=stage['checkpoint_pairs'],
                chain=chain,pilot_history_verified=True,quality_audits_bound=False,storage_verified=False,
                dispatch_authorized=False)

"""Idle pilot preflight -> immutable request transition; no process launcher.

The caller owns both worker locks, has inspected actual processes, and runs this
bounded preparation under an outer time/memory guard. Prefix audits and the full
remaining storage gate precede the new trial directory. Failed attempts persist.
"""
from pathlib import Path
import shutil
import time

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign,PROJECTION
from .tokenizer_wide_hybrid_pilot_audit_v2 import bind_pilot_prefix
from .tokenizer_wide_hybrid_storage_v2 import bind_storage,remaining_projection,required_free
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_cleanup import _scan,_check,TIMEOUT
from .tokenizer_wide_hybrid_scores import _json,_require

KIND='wide_hybrid_pilot_startup_v2'


def _inventory(root):
    result={};deadline=time.monotonic()+TIMEOUT
    for name in sorted(_scan(root,None)):
        _require(time.monotonic()<deadline,'Startup inventory deadline exceeded')
        path=_path(root,name);before=path.stat()
        result[name]=dict(size=before.st_size,sha256=file_digest(path))
        after=path.stat()
        _require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                 'Startup inventory changed while hashing')
    return result


def _check_inventory(root,inventory):
    _require(_scan(root,None)==set(inventory),'Startup evidence inventory changed')
    deadline=time.monotonic()+TIMEOUT
    for name,binding in inventory.items():_check(root,name,binding,deadline)


def _slot(writer,slot):
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<3,'Exact pilot startup slot required')
    return f'meta/pilot-startup-{slot:02d}'


def prepare_pilot_request(writer,slot,plan_sha256,*,projection_ref):
    """Complete idle gates before creating request; never launch or grant cleanup."""
    base=_slot(writer,slot);_sha(plan_sha256)
    evidence=writer.root/'evidence';trial=evidence/f'pilots/{slot:02d}'
    names=[base+suffix for suffix in ('-attempt.json','.json','-failure.json')]
    with writer._serialized():
        _require(not any(_path(evidence,n+s).exists() for n in names for s in ('','.tmp')),
                 'Existing startup attempt cannot be retried')
        _require(not trial.exists(),'Next pilot slot must not exist before idle storage review')
        _require(not (writer.root/'STOP').exists(),'STOP prevents pilot startup')
    marker=writer.publish_json('evidence/'+names[0],dict(kind=KIND,status='started',slot=slot,
        plan_sha256=plan_sha256,projection_ref=projection_ref,dispatch_authorized=False))
    marker_ref=dict(path=names[0],sha256=file_digest(marker))
    try:
        bound=bind_campaign(evidence/'meta/plan.json',plan_sha256)
        plan=_json(evidence/'meta/plan.json',plan_sha256)
        _require(isinstance(projection_ref,dict) and set(projection_ref)=={'path','sha256'}
                 and type(projection_ref['path']) is str and projection_ref['path'].startswith('meta/'),
                 'Immutable projection copy required')
        projection=_document(evidence,projection_ref)
        _require(bound['files'].get(str(_path(evidence,projection_ref['path'])))==projection_ref['sha256']
                 and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
                 'Startup projection differs from frozen copy')
        # No live worker exists in this transition. Namespace locking also
        # rejects accidental competing producer activity rather than skipping it.
        with writer._serialized():
            pilot_root=evidence/'pilots'
            expected={f'{i:02d}' for i in range(slot)}
            _require((not pilot_root.exists() and not expected) or
                     (pilot_root.is_dir() and {p.name for p in pilot_root.iterdir()}==expected
                      and all(p.is_dir() and not p.is_symlink() for p in pilot_root.iterdir())),
                     'Unexpected pilot directory before idle startup')
            census=snapshot(writer.root);inventory=_inventory(evidence)
            metadata={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')}
            if slot:
                prefix=bind_pilot_prefix(evidence,context=bound['context'],requests=bound['requests']['pilots'][:slot],
                    evidence=metadata,numerical_ref=plan['numerical_ref'],projection_ref=projection_ref,
                    tokenizer=plan['tokenizer'],training_token_sha256=plan['training_token_sha256'])
                pilots,pairs=prefix['records'],prefix['checkpoint_pairs']
            else:
                prefix=None;pilots=[];pairs={}
            storage=bind_storage(writer.root,projection,pilots=pilots,records=[],rates=None,selection=None,
                stages=[],current_inventory=inventory,checkpoint_pairs=pairs,lock_fds=writer.lock_fds)
            _require(snapshot(writer.root)==census,'Startup census changed during gates')
        again=bind_campaign(evidence/'meta/plan.json',plan_sha256)
        _require(digest(again['files'])==digest(bound['files']),'Frozen startup provenance changed')
        _check_inventory(evidence,inventory)
        _require(file_digest(marker)==marker_ref['sha256'] and not (writer.root/'STOP').exists(),
                 'Startup attempt or STOP changed')
        _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],
                 'Full remaining storage changed before request publication')
        request=bound['requests']['pilots'][slot]
        path=writer.publish_json(f'evidence/pilots/{slot:02d}/request.json',request)
        request_ref=dict(path=f'pilots/{slot:02d}/request.json',sha256=file_digest(path))
        # The request is the only allowed new evidence since the idle review.
        expanded=dict(inventory);expanded[request_ref['path']]=dict(size=path.stat().st_size,sha256=request_ref['sha256'])
        _check_inventory(evidence,expanded)
        report=dict(kind=KIND,status='request_materialized_pending_process_dispatch',slot=slot,
            plan_sha256=plan_sha256,context=bound['context'],projection_ref=projection_ref,attempt=marker_ref,
            request=request_ref,inventory=inventory,storage=storage,
            completed_pilots=pilots,checkpoint_pairs=pairs,prefix_audits_bound=slot>0,
            dispatch_authorized=False,deletion_authorized=False)
        out=writer.publish_json('evidence/'+names[1],report)
        _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes']
                 and not (writer.root/'STOP').exists(),'Storage/STOP changed after startup publication')
        return dict(path=names[1],sha256=file_digest(out))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+names[2],dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Startup failure publication failed: '+repr(error)[:1000])
        raise


def _verify_before_process(writer,slot,startup_ref):
    """Recheck immutable startup transition before the process helper is invoked.

    A process attempt already present is rejected. The existing process helper
    must still check disk immediately before Popen, enforce exact argv/limits and
    carry inherited locks. This read-only function itself grants no dispatch.
    """
    base=_slot(writer,slot);evidence=writer.root/'evidence'
    _require(startup_ref.get('path')==base+'.json','Startup reference slot differs')
    for suffix in ('-failure.json','-failure.json.tmp','-attempt.json.tmp','.json.tmp'):
        _require(not _path(evidence,base+suffix).exists(),'Failed or partial startup cannot dispatch')
    report=_document(evidence,startup_ref)
    _require(report.get('kind')==KIND and report.get('status')=='request_materialized_pending_process_dispatch'
             and type(report.get('slot')) is int and report['slot']==slot
             and report.get('dispatch_authorized') is False and report.get('deletion_authorized') is False,
             'Startup identity/claims differ')
    bound=bind_campaign(evidence/'meta/plan.json',report['plan_sha256'])
    _require(digest(report.get('context'))==digest(bound['context']),'Startup frozen context changed')
    projection=_document(evidence,report['projection_ref'])
    _require(bound['files'].get(str(_path(evidence,report['projection_ref']['path'])))==report['projection_ref']['sha256']
             and bound['files'].get(str(PROJECTION.resolve()))==report['projection_ref']['sha256'],
             'Predispatch projection differs from frozen copy')
    pilots=report['completed_pilots'];pairs=report['checkpoint_pairs']
    _require(isinstance(pilots,list) and len(pilots)==slot
             and report.get('prefix_audits_bound') is (slot>0)
             and set(pairs)=={f'pilot/{i:02d}' for i in range(slot)},'Completed prefix differs')
    state=remaining_projection(projection,pilots,[],None,None,0)
    inventory=dict(report['inventory'])
    _require(report['attempt']['path']==base+'-attempt.json'
             and inventory[report['attempt']['path']]['sha256']==report['attempt']['sha256'],
             'Startup attempt inventory binding differs')
    _require(_document(evidence,report['attempt'])==dict(kind=KIND,status='started',slot=slot,
        plan_sha256=report['plan_sha256'],projection_ref=report['projection_ref'],dispatch_authorized=False),
        'Startup attempt differs')
    rq=report['request'];_require(rq.get('path')==f'pilots/{slot:02d}/request.json','Startup request location differs')
    _require(digest(_document(evidence,rq))==digest(bound['requests']['pilots'][slot]),'Startup request changed')
    for ref in (rq,startup_ref):
        path=_path(evidence,ref['path'])
        _require(ref['path'] not in inventory,'Startup transition aliases earlier evidence')
        inventory[ref['path']]=dict(size=path.stat().st_size,sha256=ref['sha256'])
    with writer._serialized():
        _check_inventory(evidence,inventory);census=snapshot(writer.root)
        pilot_root=evidence/'pilots';trial=pilot_root/f'{slot:02d}'
        _require({p.name for p in pilot_root.iterdir()}=={f'{i:02d}' for i in range(slot+1)}
                 and all(p.is_dir() and not p.is_symlink() for p in pilot_root.iterdir())
                 and {p.name for p in trial.iterdir()}=={'request.json'},
                 'Unexpected pilot/process directory before dispatch')
        # The supplied startup hash anchors the original full prefix audit. Now
        # reconstruct its checkpoint accounting from the unchanged result files,
        # not from claimed occupied bytes or the saved required-free total.
        occupied=0;logical=0;weights=set();inodes=set()
        for i,row in enumerate(pilots):
            stem=f'pilots/{i:02d}'
            result=_json(evidence/(stem+'/result.json'),inventory[stem+'/result.json']['sha256'])
            _require(row['ce']==result['h200']['mean_ce'],'Completed prefix score differs')
            pair=pairs[f'pilot/{i:02d}']
            _require(set(pair)=={'raw','h200'},'Complete checkpoint pair required')
            for role,entry in pair.items():
                name=entry['path'];path=_path(evidence,name);st=path.stat();inode=(st.st_dev,st.st_ino)
                _require(set(entry)=={'path','size','sha256'} and type(entry['size']) is int and entry['size']>0
                         and path.parent==evidence/stem/'checkpoints' and path.suffix=='.pt'
                         and st.st_nlink==1 and inode not in inodes
                         and inventory[name]==dict(size=entry['size'],sha256=entry['sha256'])
                         and result[role]['checkpoint']==str(path)
                         and result[role]['checkpoint_sha256']==entry['sha256'], 'Startup checkpoint binding differs')
                weights.add(name);inodes.add(inode);logical+=st.st_size
                occupied+=min(st.st_size,st.st_blocks*512)
        actual={n.removeprefix('evidence/') for n,v in census['files'].items() if v['group'].endswith('/weights')}
        _require(actual==weights and census['checkpoint_logical_bytes']==logical,'Startup checkpoint census differs')
        storage=report['storage'];needed=required_free(state,occupied)
        _require(digest(storage.get('projection'))==digest(state)
                 and type(storage.get('required_free_bytes')) is int and storage['required_free_bytes']==needed
                 and type(storage.get('occupied_checkpoint_capacity_bytes')) is int
                 and storage['occupied_checkpoint_capacity_bytes']==occupied
                 and type(storage.get('checkpoint_logical_bytes')) is int and storage['checkpoint_logical_bytes']==logical
                 and type(storage.get('cleanup_boundaries_verified')) is int and storage['cleanup_boundaries_verified']==0,
                 'Startup storage arithmetic differs')
        _require(storage.get('status')=='remaining_campaign_storage_verified'
                 and storage.get('current_inventory_sha256')==digest(report['inventory'])
                 and storage.get('dispatch_authorized') is False
                 and storage.get('frozen_plan_verified') is False
                 and storage.get('audit_semantics_verified') is False
                 and storage.get('storage_projection_fits') is True
                 and storage.get('current_phase_space_verified') is True
                 and type(needed) is int and needed>=8*1024**3
                 and shutil.disk_usage(writer.root).free>=needed and not (writer.root/'STOP').exists(),
                 'Startup storage evidence or actual space differs')
    return dict(required_free_bytes=needed,request_sha256=rq['sha256'],startup_ref=startup_ref,
                startup_transition_verified=True,dispatch_authorized=False)


def verify_before_process(writer,slot,startup_ref):
    """Check the hash-anchored startup receipt; persist any failed predispatch gate.

    No process is launched. Caller must carry the returned required-free value
    into the bounded process helper and keep both worker locks throughout.
    """
    base=_slot(writer,slot)
    try:return _verify_before_process(writer,slot,startup_ref)
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            phase='predispatch',startup_ref=startup_ref,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Startup failure publication failed: '+repr(error)[:1000])
        raise

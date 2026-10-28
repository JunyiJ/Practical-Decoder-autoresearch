"""First quality request after verified pilot cleanup; deliberately no launch.

This finite transition supports only quality slot0, before LR selection. Later
quality prefixes, selection-aware handoff and child entry remain separate work.
The full panel must already be frozen and all real numerical gates must pass.
"""
from pathlib import Path
import shutil

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
from .tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign,PROJECTION
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_storage_v2 import bind_storage,remaining_projection,required_free
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_rules import lr_specs
from .cpu_prerequisite_guard import violation

KIND='wide_hybrid_first_quality_startup_v2'
BASE='meta/quality-first-startup'


class _StartupGuard(_Guard):
    KIND=KIND
    def __init__(self,writer):
        super().__init__(writer,0,'startup',1800)
        self.name='control/quality-first-startup.json'


def _frozen(writer,plan_sha,projection_ref):
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents quality startup')
    bound=bind_campaign(writer.root/'evidence/meta/plan.json',plan_sha)
    _require(projection_ref['path'].startswith('meta/')
             and bound['files'].get(str(_path(writer.root/'evidence',projection_ref['path'])))==projection_ref['sha256']
             and bound['files'].get(str(PROJECTION.resolve()))==projection_ref['sha256'],
             'Quality startup projection differs from frozen copy')
    options=bound['requests']['quality']['0']
    _require(len(options)==1 and digest(options[0]['spec'])==digest(lr_specs()[0]),'Exact first LR request required')
    return bound,options[0]


def _history(writer,plan_sha,completion_ref,inventory):
    complete=_document(writer.root/'evidence',completion_ref)
    return verify_pilot_history(writer,plan_sha,completion_ref,current_inventory=inventory,
                                stages=[complete['chain_stage']])


def _only_first_request(writer,present):
    root=writer.root/'evidence/quality'
    if not present:
        _require(not root.exists(),'Quality must not exist before idle storage gate');return
    _require(root.is_dir() and {p.name for p in root.iterdir()}=={'00'}
             and (root/'00').is_dir() and {p.name for p in (root/'00').iterdir()}=={'request.json'},
             'Only the first quality request may exist before dispatch')


def prepare_first_quality_request(writer,plan_sha256,*,projection_ref,pilot_completion_ref):
    """Own both locks and inspect workers first. Never launch or retry."""
    _require(isinstance(writer,MetadataWriter),'Bounded writer required');_sha(plan_sha256)
    evidence=writer.root/'evidence'
    with writer._serialized():
        _only_first_request(writer,False)
        _require(not any((evidence/(BASE+s)).exists() for s in ('-attempt.json','-attempt.json.tmp',
            '.json','.json.tmp','-failure.json','-failure.json.tmp'))
            and not (writer.root/'control/quality-first-startup.json').exists(),'Existing quality startup cannot be retried')
    attempt=dict(kind=KIND,status='started',plan_sha256=plan_sha256,projection_ref=projection_ref,
                 pilot_completion_ref=pilot_completion_ref,slot=0,dispatch_authorized=False)
    try:
        with _StartupGuard(writer):
            marker=writer.publish_json('evidence/'+BASE+'-attempt.json',attempt)
            marker_ref=dict(path=BASE+'-attempt.json',sha256=file_digest(marker))
            bound,request=_frozen(writer,plan_sha256,projection_ref)
            projection=_document(evidence,projection_ref)
            inventory=_inventory(evidence);history=_history(writer,plan_sha256,pilot_completion_ref,inventory)
            storage=bind_storage(writer.root,projection,pilots=history['records'],records=[],rates=None,selection=None,
                stages=[history['chain_stage']],current_inventory=inventory,
                checkpoint_pairs=history['checkpoint_pairs'],lock_fds=writer.lock_fds)
            _only_first_request(writer,False);_check_inventory(evidence,inventory)
            again,_=_frozen(writer,plan_sha256,projection_ref)
            _require(digest(again['files'])==digest(bound['files']),'Frozen quality startup changed')
            _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Full panel storage changed')
            path=writer.publish_json('evidence/quality/00/request.json',request)
            request_ref=dict(path='quality/00/request.json',sha256=file_digest(path))
            expected=dict(inventory,**{request_ref['path']:dict(size=path.stat().st_size,sha256=request_ref['sha256'])})
            _check_inventory(evidence,expected);_only_first_request(writer,True)
            _frozen(writer,plan_sha256,projection_ref)
            report=dict(kind=KIND,status='first_quality_request_pending_dispatch',slot=0,
                plan_sha256=plan_sha256,context=bound['context'],projection_ref=projection_ref,
                pilot_completion_ref=pilot_completion_ref,attempt=marker_ref,request=request_ref,
                inventory=inventory,storage=storage,dispatch_authorized=False,deletion_authorized=False)
        guard=writer.root/'control/quality-first-startup.json'
        report['guard']=dict(path=str(guard),sha256=file_digest(guard))
        _verify_guard(writer,report['guard'])
        _frozen(writer,plan_sha256,projection_ref)
        _require(shutil.disk_usage(writer.root).free>=storage['required_free_bytes'],'Storage changed after preparation')
        path=writer.publish_json('evidence/'+BASE+'.json',report)
        return dict(path=BASE+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+BASE+'-failure.json',dict(kind=KIND,status='failed',
            slot=0,plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Quality startup failure publication failed: '+repr(error)[:1000])
        raise


def _verify_guard(writer,ref):
    path=writer.root/'control/quality-first-startup.json'
    _require(ref['path']==str(path),'Quality startup guard path differs')
    guard=_document(writer.root,dict(path='control/quality-first-startup.json',sha256=ref['sha256']))
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(guard.get('kind')==KIND and guard.get('status')=='finished' and guard.get('reason') is None
             and guard.get('limits')==limits and type(guard.get('pid')) is int and guard['pid']>0,
             'Completed bounded quality startup guard required')
    sample=guard['observation'];elapsed=guard['elapsed_seconds']
    _require(type(elapsed) in (int,float) and 0<=elapsed<1800
             and set(sample)=={'peak_rss_bytes','free_bytes'}
             and all(type(v) is int and v>0 for v in sample.values())
             and violation(sample,elapsed,**limits) is None,'Quality startup guard observation invalid')


def verify_first_quality_request(writer,reference):
    """Read-only idle predispatch recheck; caller owns outer guard and locks.

    Requires the exact request-only state. This is not a child entry gate and
    cannot accept process files or authorize a worker by itself.
    """
    evidence=writer.root/'evidence'
    _require(reference['path']==BASE+'.json','First quality startup reference required')
    for suffix in ('-failure.json','-failure.json.tmp','.json.tmp','-attempt.json.tmp'):
        _require(not (evidence/(BASE+suffix)).exists(),'Failed or partial quality startup cannot dispatch')
    report=_document(evidence,reference)
    _require(report.get('kind')==KIND and report.get('status')=='first_quality_request_pending_dispatch'
             and type(report.get('slot')) is int and report['slot']==0
             and report.get('dispatch_authorized') is False and report.get('deletion_authorized') is False,
             'First quality startup identity differs')
    _verify_guard(writer,report['guard'])
    bound,request=_frozen(writer,report['plan_sha256'],report['projection_ref'])
    _require(digest(report['context'])==digest(bound['context']),'Quality startup context changed')
    _require(report['request']['path']=='quality/00/request.json'
             and digest(_document(evidence,report['request']))==digest(request),'First quality request changed')
    attempt=dict(kind=KIND,status='started',plan_sha256=report['plan_sha256'],projection_ref=report['projection_ref'],
                 pilot_completion_ref=report['pilot_completion_ref'],slot=0,dispatch_authorized=False)
    _require(report['attempt']['path']==BASE+'-attempt.json'
             and digest(_document(evidence,report['attempt']))==digest(attempt),'Quality startup attempt changed')
    inventory=dict(report['inventory'])
    _require(inventory[report['attempt']['path']]['sha256']==report['attempt']['sha256'],'Attempt inventory differs')
    for ref in (report['request'],reference):
        p=_path(evidence,ref['path']);_require(ref['path'] not in inventory,'Startup output overlaps prior inventory')
        inventory[ref['path']]=dict(size=p.stat().st_size,sha256=ref['sha256'])
    _check_inventory(evidence,inventory);_only_first_request(writer,True)
    history=_history(writer,report['plan_sha256'],report['pilot_completion_ref'],inventory)
    projection=_document(evidence,report['projection_ref'])
    state=remaining_projection(projection,history['records'],[],None,None,1)
    needed=required_free(state,0);census=snapshot(writer.root)
    _require(census['checkpoint_logical_bytes']==0 and not any(v['group'].endswith('/weights') for v in census['files'].values()),
             'Unexpected checkpoint before first quality dispatch')
    storage=report['storage']
    _require(storage.get('status')=='remaining_campaign_storage_verified' and digest(storage['projection'])==digest(state)
             and storage.get('required_free_bytes')==needed and storage.get('occupied_checkpoint_capacity_bytes')==0
             and storage.get('checkpoint_logical_bytes')==0 and storage.get('cleanup_boundaries_verified')==1
             and storage.get('current_inventory_sha256')==digest(report['inventory'])
             and storage.get('storage_projection_fits') is True and storage.get('current_phase_space_verified') is True
             and storage.get('dispatch_authorized') is False,'Original full storage binding differs')
    _require(shutil.disk_usage(writer.root).free>=needed,'Complete remaining panel no longer fits')
    _frozen(writer,report['plan_sha256'],report['projection_ref']);_check_inventory(evidence,inventory)
    return dict(required_free_bytes=needed,request_sha256=report['request']['sha256'],
                pilot_history_verified=True,first_quality_request_verified=True,dispatch_authorized=False)

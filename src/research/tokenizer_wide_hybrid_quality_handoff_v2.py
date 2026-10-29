"""Read-only quality launch transport checks; no worker entry or launcher.

The future dispatcher must supply an independently audited immutable inventory,
full remaining-storage requirement and exact frozen trajectory allowance. This
component cannot establish those scientific/storage prerequisites by itself.
"""
import os
import hashlib
import sys
import time

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention import RESERVE,METADATA
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_cleanup import _locks,_scan,_check,TIMEOUT
from .tokenizer_wide_hybrid_scores import _require

# Reserved child module contract; this file deliberately has no runnable entry.
MODULE='src.research.tokenizer_wide_hybrid_quality_handoff_v2'
KIND='wide_hybrid_quality_live_handoff_v2'
LOG_LIMIT=4*1024**2


def _log_prefix(path,size):
    digest=hashlib.sha256();remaining=size
    with path.open('rb') as stream:
        while remaining:
            chunk=stream.read(min(65536,remaining))
            _require(bool(chunk),'Worker log truncated while hashing')
            digest.update(chunk);remaining-=len(chunk)
    return digest.hexdigest()


def worker_argv(writer,slot,startup_ref,*,readiness_ref=None):
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<=15,
             'Exact LR/rank/first-fresh writer/slot required')
    phase='lr' if slot<9 else ('rank' if slot<15 else 'confirm')
    path='meta/quality-first-startup.json' if slot==0 else f'meta/quality-{phase}-startup-{slot:02d}.json'
    _require(slot<9 or readiness_ref is not None,'Rank/fresh transport requires an immutable readiness seal')
    _require(startup_ref['path']==path,'Exact quality startup reference required')
    module=MODULE
    extra=[]
    if readiness_ref is not None:
        _require(readiness_ref['path']==f'meta/quality-readiness-{slot:02d}.json','Exact readiness reference required')
        module='src.research.tokenizer_wide_hybrid_quality_entry_v2'
        extra=['--readiness-sha256',_sha(readiness_ref['sha256'])]
    return [sys.executable,'-m',module,'worker','--campaign',str(writer.root),'--slot',str(slot),
        '--startup-sha256',_sha(startup_ref['sha256']),'--global-fd',str(writer.lock_fds[0]),
        '--campaign-fd',str(writer.lock_fds[1])]+extra


def verify_live_handoff(writer,slot,startup_ref,actual_argv,*,inventory,required_free_bytes,output_limit_bytes,readiness_ref=None):
    """Bind the exact live parent launch and immutable prelaunch inventory.

    Caller owns an outer guard and must already establish frozen numerical,
    pilot/prefix, startup and full-storage semantics. Here only transport and
    unchanged bytes are checked. No scientific, storage or dispatch credit.
    """
    expected_argv=worker_argv(writer,slot,startup_ref,readiness_ref=readiness_ref)
    _require(actual_argv==expected_argv,'Actual quality child argv differs')
    _require(type(required_free_bytes) is int and required_free_bytes>=RESERVE+METADATA
             and type(output_limit_bytes) is int and output_limit_bytes>LOG_LIMIT,
             'Independent full-storage and trajectory byte limits required')
    root=writer.root/'evidence';stem=f'quality/{slot:02d}';process=stem+'/process'
    _locks(root,writer.lock_fds)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents handoff')
    snapshot(writer.root)
    _require(isinstance(inventory,dict) and inventory,'Independent complete prelaunch inventory required')
    _require(startup_ref['path'] in inventory and inventory[startup_ref['path']]['sha256']==startup_ref['sha256'],
             'Startup reference not anchored in prelaunch inventory')
    report=_document(root,startup_ref)
    rq=report.get('request',{})
    _require(rq.get('path')==stem+'/request.json' and rq['path'] in inventory
             and inventory[rq['path']]['sha256']==rq.get('sha256'),'Pending request binding differs')
    quality=_path(root,'quality');trial=_path(root,stem);directory=_path(root,process)
    _require({p.name for p in quality.iterdir()}=={f'{i:02d}' for i in range(slot+1)}
             and all(p.is_dir() and not p.is_symlink() for p in quality.iterdir())
             and {p.name for p in trial.iterdir()}=={'request.json','process'}
             and directory.is_dir() and {p.name for p in directory.iterdir()}=={'launch.json','worker.log'},
             'Exact fresh quality process files required')
    launch_name=process+'/launch.json';log_name=process+'/worker.log'
    _require(not {launch_name,log_name}&set(inventory),'Launch files overlap prelaunch evidence')
    launch_ref=dict(path=launch_name,sha256=file_digest(root/launch_name))
    launch=_document(root,launch_ref)
    parent=os.getppid();_require(parent>1,'Live parent required')
    expected=dict(status='preflight',argv=expected_argv,parent_pid=parent,pid=None,returncode=None,error=None,
        required_free_bytes=required_free_bytes,reserve_bytes=RESERVE,output_limit_bytes=output_limit_bytes,
        log_limit_bytes=LOG_LIMIT,timeout_seconds=7200,observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    _require(digest(launch)==digest(expected),'Quality parent launch identity or frozen limits differ')
    expected_names=set(inventory)|{launch_name,log_name};log=_path(root,log_name)
    first=log.stat();_require(first.st_nlink==1 and first.st_size<=LOG_LIMIT,'Bounded unique worker log required')
    prefix_sha=_log_prefix(log,first.st_size)
    _require(_scan(root,None)==expected_names,'Quality handoff evidence set differs')
    deadline=time.monotonic()+TIMEOUT
    # Never hold the metadata namespace lock while hashing retained weights.
    # The parent's capped pipe writer may append only to its one log inode.
    for name,binding in inventory.items():_check(root,name,binding,deadline)
    _check(root,launch_name,dict(size=(root/launch_name).stat().st_size,sha256=launch_ref['sha256']),deadline)
    _require(_scan(root,None)==expected_names,'Quality handoff evidence changed while hashing')
    last=log.stat()
    _require((last.st_dev,last.st_ino)==(first.st_dev,first.st_ino) and last.st_nlink==1
             and first.st_size<=last.st_size<=LOG_LIMIT,'Worker log replaced, truncated or over cap')
    _require(_log_prefix(log,first.st_size)==prefix_sha,'Existing worker log bytes changed')
    snapshot(writer.root);_locks(root,writer.lock_fds)
    _require(os.getppid()==parent and not (writer.root/'STOP').exists()
             and not (writer.root/'STOP').is_symlink(),'Parent or STOP changed during handoff')
    return dict(kind=KIND,slot=slot,startup_ref=startup_ref,launch_ref=launch_ref,
        parent_pid=parent,request_sha256=rq['sha256'],prelaunch_inventory_sha256=digest(inventory),
        live_transport_verified=True,quality_prerequisites_verified=False,storage_verified=False,
        dispatch_authorized=False)

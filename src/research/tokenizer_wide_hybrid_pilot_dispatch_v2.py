"""One v2 pilot transition and a gated child entry point; no campaign run CLI.

Call run_pilot_once only with owned shared locks after inspecting actual workers.
It never freezes plans, retries attempts, cleans weights or advances stages. Real
launches still require the complete frozen nine-case numerical/input evidence.
"""
import argparse
import os
from pathlib import Path
import sys
import threading
import time

from .protocol import ROOT,digest,file_digest
from .cpu_prerequisite_guard import observation,violation
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_pilot_startup_v2 import (
    prepare_pilot_request,verify_before_process,verify_worker_startup,_inventory)
from .tokenizer_wide_hybrid_process_v2 import run_process
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_receipts import _document,_sha
from .tokenizer_wide_hybrid_scores import _require,_json
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign
from .tokenizer_wide_hybrid_pilot_audit_v2 import bind_pilot_prefix
from .tokenizer_wide_hybrid_storage_v2 import bind_storage

MODULE='src.research.tokenizer_wide_hybrid_pilot_dispatch_v2'
KIND='wide_hybrid_pilot_dispatch_v2'


class _Guard:
    """Bounded sampled CPU guard around preparation/entry, before model work.

    The existing worker recorder additionally enforces MPS driver memory; the
    parent process cap includes child import/preflight/training/evaluation time.
    """
    KIND=KIND

    def __init__(self,writer,slot,phase,seconds):
        self.writer=writer;self.name=f'control/pilot-{phase}-{slot:02d}.json'
        self.limits=dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=seconds)
        self.stop=threading.Event();self.started=None;self.created=False

    def _write(self,status,reason=None):
        elapsed=time.monotonic()-self.started;sample=observation(self.writer.root)
        why=violation(sample,elapsed,**self.limits)
        record=dict(kind=self.KIND,status='failed' if why else status,pid=os.getpid(),limits=self.limits,
                    elapsed_seconds=elapsed,observation=sample,reason=why or reason)
        self.writer.publish_json(self.name,record,replace=self.created);self.created=True
        return why

    def _watch(self):
        try:
            while not self.stop.wait(1):
                # Keep immutable/idle inventories stable while audit code owns
                # the metadata lock. Sampling continues; only failures publish.
                why=violation(observation(self.writer.root),time.monotonic()-self.started,**self.limits)
                if why:
                    try:self._write('failed',why)
                    finally:os._exit(90)
        except BaseException as exc:
            try:self._write('guard_error',repr(exc)[:1000])
            finally:os._exit(91)

    def __enter__(self):
        _require(self.started is None,'Guard cannot be reused')
        self.started=time.monotonic()
        if self._write('starting'):raise RuntimeError('Preparation resource limit')
        self.thread=threading.Thread(target=self._watch,daemon=True);self.thread.start()
        return self

    def __exit__(self,kind,exc,tb):
        self.stop.set();self.thread.join()
        why=self._write('finished' if exc is None else 'failed',None if exc is None else repr(exc)[:1000])
        if why:raise RuntimeError('Preparation resource limit: '+why)
        return False


def worker_argv(writer,slot,startup_ref):
    _require(type(slot) is int and 0<=slot<3 and startup_ref['path']==f'meta/pilot-startup-{slot:02d}.json',
             'Exact startup reference slot required')
    return [sys.executable,'-m',MODULE,'worker','--campaign',str(writer.root),'--slot',str(slot),
            '--startup-sha256',_sha(startup_ref['sha256']),'--global-fd',str(writer.lock_fds[0]),
            '--campaign-fd',str(writer.lock_fds[1])]


def run_worker(writer,slot,startup_ref,argv):
    """Child entry only: inherited locks, exact bounded parent, then real gates."""
    with _Guard(writer,slot,'entry',600):
        _require(argv==worker_argv(writer,slot,startup_ref),'Worker entry argv differs')
        checked=verify_worker_startup(writer,slot,startup_ref,argv)
        report=_document(writer.root/'evidence',startup_ref)
        # Import the training implementation only after the handoff passes.
        from .tokenizer_wide_hybrid_pilot_execution_v2 import execute_pilot_outputs
        return execute_pilot_outputs(writer,slot,report['plan_sha256'],checked['request_sha256'],
                                     projection_ref=report['projection_ref'])


def run_pilot_once(writer,slot,plan_sha256,*,projection_ref):
    """Prepare, dispatch and independently review exactly one fresh pilot.

    The caller owns both locks for the entire call and must inspect actual
    processes first. No completed/failed slot can be dispatched twice. Later
    pilot calls require all earlier successful dispatch receipts, then the
    startup component independently re-audits their full immutable evidence.
    """
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<3,'Exact pilot writer/slot required')
    evidence=writer.root/'evidence';base=f'meta/pilot-dispatch-{slot:02d}'
    _sha(plan_sha256)
    with writer._serialized():
        for i in range(3):
            stem=evidence/f'meta/pilot-dispatch-{i:02d}'
            _require(not any(Path(str(stem)+s).exists() for s in ('-failure.json','-failure.json.tmp','.json.tmp')),
                     'Failed dispatch blocks continuation')
            report=Path(str(stem)+'.json')
            if i<slot:
                previous=_json(report,file_digest(report))
                _require(previous.get('kind')==KIND and previous.get('status')=='pilot_reviewed'
                         and previous.get('slot')==i and previous.get('plan_sha256')==plan_sha256,
                         'Prior successful dispatch receipt required')
            else:_require(not report.exists(),'Existing dispatch receipt cannot be rerun')
    try:
        with _Guard(writer,slot,'controller',1800):
            startup=prepare_pilot_request(writer,slot,plan_sha256,projection_ref=projection_ref)
            checked=verify_before_process(writer,slot,startup)
            projection=_document(evidence,projection_ref)
            bound=bind_campaign(evidence/'meta/plan.json',plan_sha256)
            request=bound['requests']['pilots'][slot]
            allowance=allocations(projection)[request['spec']['family'],2]
            process=run_process(writer,'pilots',slot,worker_argv(writer,slot,startup),ROOT,
                required_free_bytes=checked['required_free_bytes'],output_limit_bytes=allowance,
                log_limit_bytes=4*1024**2,timeout_seconds=600)
            _require(process.get('status')=='completed' and process.get('returncode')==0,'Pilot process failed')
            entry=writer.root/f'control/pilot-entry-{slot:02d}.json';entry_sha=file_digest(entry)
            guard=_json(entry,entry_sha)
            _require(guard.get('kind')==KIND and guard.get('status')=='finished' and guard.get('reason') is None
                     and guard.get('pid')==process['pid'] and guard.get('limits')==dict(
                         rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=600),
                     'Complete child entry guard required')
            sample=guard.get('observation',{});elapsed=guard.get('elapsed_seconds')
            _require(set(sample)=={'peak_rss_bytes','free_bytes'}
                     and all(type(v) is int and v>0 for v in sample.values())
                     and type(elapsed) in (float,int) and 0<=elapsed<600
                     and violation(sample,elapsed,**guard['limits']) is None,
                     'Child entry resource observation invalid')
            plan=_json(evidence/'meta/plan.json',plan_sha256)
            bound=bind_campaign(evidence/'meta/plan.json',plan_sha256)
            with writer._serialized():
                inventory=_inventory(evidence)
                prefix=bind_pilot_prefix(evidence,context=bound['context'],requests=bound['requests']['pilots'][:slot+1],
                    evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
                    numerical_ref=plan['numerical_ref'],projection_ref=projection_ref,
                    tokenizer=plan['tokenizer'],training_token_sha256=plan['training_token_sha256'])
                storage=bind_storage(writer.root,projection,pilots=prefix['records'],records=[],rates=None,selection=None,
                    stages=[],current_inventory=inventory,checkpoint_pairs=prefix['checkpoint_pairs'],lock_fds=writer.lock_fds)
            again=bind_campaign(evidence/'meta/plan.json',plan_sha256)
            _require(digest(again['files'])==digest(bound['files']),'Frozen dispatch provenance changed during review')
            _require(file_digest(entry)==entry_sha,'Child entry guard changed during review')
            # Preserve identities of both the actual child and its final guard.
            refs={name:dict(path=f'pilots/{slot:02d}/'+tail,sha256=file_digest(evidence/f'pilots/{slot:02d}'/tail))
                  for name,tail in [('worker','worker-outputs.json'),('process','process/process.json')]}
            result=dict(kind=KIND,status='pilot_reviewed',slot=slot,plan_sha256=plan_sha256,startup=startup,
                child=refs,entry_guard=dict(path=str(entry),sha256=entry_sha,record=guard),
                completed_pilots=prefix['records'],checkpoint_pairs=prefix['checkpoint_pairs'],storage=storage,
                evidence_inventory_sha256=digest(inventory),adopted=False,deletion_authorized=False,
                stage_cleanup_complete=False,next_stage_authorized=False)
        # Controller guard must have finished before publishing success.
        path=writer.publish_json('evidence/'+base+'.json',result)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Dispatch failure publication failed: '+repr(error)[:1000])
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['worker'])
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--slot',required=True,type=int,choices=range(3))
    parser.add_argument('--startup-sha256',required=True)
    parser.add_argument('--global-fd',required=True,type=int)
    parser.add_argument('--campaign-fd',required=True,type=int)
    args=parser.parse_args(argv)
    writer=MetadataWriter(args.campaign,(args.global_fd,args.campaign_fd))
    ref=dict(path=f'meta/pilot-startup-{args.slot:02d}.json',sha256=args.startup_sha256)
    actual=[sys.executable,'-m',MODULE,*(sys.argv[1:] if argv is None else argv)]
    return run_worker(writer,args.slot,ref,actual)


if __name__=='__main__':main()

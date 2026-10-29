"""Finite changed-source nontraining handoff check: one unpaced checkpoint producer and supervisor.

Never reads campaign checkpoints/corpus, trains, retries, or edits production.
Instrumentation wraps the original namespace context manager without changing it.
"""
import argparse,contextlib,hashlib,json,os,signal,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard,observation,violation
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-metadata-contention-handoff-001'
SOURCES=['src/research/tokenizer_wide_hybrid_metadata_writer.py','src/research/tokenizer_wide_hybrid_checkpoint_writer.py',
 'src/research/tokenizer_wide_hybrid_process_v2.py','src/research/tokenizer_wide_hybrid_metadata.py',
 'src/research/cpu_prerequisite_guard.py','src/research/tokenizer_wide_hybrid_cleanup.py']
PLAN=HERE/'metadata-contention-handoff-v1-plan.json'
MiB=1024**2

def save(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    assert len(data)<=MiB,'Diagnostic byte cap'
    with path.open('xb') as f:f.write(data)

def verify():
    plan=json.loads(PLAN.read_text())
    assert plan['script_sha256']==file_digest(Path(__file__))
    assert all(file_digest(ROOT/n)==h for n,h in plan['sources'].items())
    return plan

class Trace:
    def __init__(self,writer):
        self.records=[];self.original=writer._serialized;self.origin=time.monotonic()
        writer._serialized=self.context
    @contextlib.contextmanager
    def context(self):
        begin=time.monotonic();acquired=None;end=None;error=None
        try:
            with self.original():
                acquired=time.monotonic()
                try:yield
                finally:end=time.monotonic()
        except BaseException as exc:
            error=repr(exc)[:200];raise
        finally:
            if len(self.records)>=5000:raise RuntimeError('Trace count cap')
            now=time.monotonic()
            self.records.append(dict(start=begin-self.origin,wait=(acquired or now)-begin,
                hold=None if acquired is None else (end or now)-acquired,
                thread=threading.current_thread().name,error=error))
    def report(self):
        return dict(records=self.records,max_wait=max((x['wait'] for x in self.records),default=0),
          max_hold=max((x['hold'] or 0 for x in self.records),default=0),
          failures=[x for x in self.records if x['error']])

def worker(args):
    plan=verify();writer=MetadataWriter(RUN/'campaign',(args.global_fd,args.campaign_fd));trace=Trace(writer)
    stop=threading.Event();errors=[];ticks=[]
    def signal_stop(signum,frame):raise KeyboardInterrupt('Fixture signal '+str(signum))
    signal.signal(signal.SIGTERM,signal_stop)
    with CpuPrerequisiteGuard(RUN/'child-guard.json',RUN,timeout_seconds=150):
        stream=writer.open_append('evidence/quality/23/resource-events.jsonl')
        def sample():
            try:
                while not stop.is_set():
                    value=dict(event='cpu_sample',elapsed=time.monotonic()-trace.origin,**observation(RUN))
                    if violation(value,value['elapsed'],4*1024**3,8*1024**3,150):raise RuntimeError('Resource bound')
                    stream.write((json.dumps(value)+'\n').encode());ticks.append(value)
                    stop.wait(1)
            except BaseException as exc:errors.append(repr(exc))
        thread=threading.Thread(target=sample,name='resource');thread.start()
        try:
            payload=b'x'*plan['checkpoint_bytes']
            with CheckpointWriter(writer,'quality',23,plan['checkpoint_bytes']).open('fixture.pt') as output:
                output.write(payload);receipt=output.publish()
            assert not errors,errors
            writer.publish_json('evidence/quality/23/result.json',dict(synthetic_only=True,receipt=receipt))
        finally:
            stop.set();thread.join(timeout=5)
            assert not thread.is_alive(),'Resource recorder did not finish'
            stream.close();save(RUN/'child-trace.json',dict(**trace.report(),resource_errors=errors,samples=ticks))

def parent():
    plan=verify();RUN.mkdir(exist_ok=False);campaign=RUN/'campaign';campaign.mkdir()
    with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(campaign) as c:
        with CpuPrerequisiteGuard(RUN/'parent-guard.json',RUN,timeout_seconds=180):
            control=campaign/'control';control.mkdir()
            for i in range(plan['census_files']):
                (control/f'fixture-{i:04d}.json').write_text('{"synthetic_only":true}\n')
            writer=MetadataWriter(campaign,(g,c));trace=Trace(writer)
            started=time.monotonic();result=None;error=None
            def check():
                reason=violation(observation(RUN),time.monotonic()-started,4*1024**3,8*1024**3,170)
                if reason:raise RuntimeError(reason)
            try:
                result=run_process(writer,'quality',23,[sys.executable,str(Path(__file__).resolve()),'worker',
                    '--global-fd',str(g),'--campaign-fd',str(c)],ROOT,
                    required_free_bytes=8*1024**3+64*MiB,output_limit_bytes=58*MiB,
                    log_limit_bytes=65536,timeout_seconds=150,parent_check=check)
            except BaseException as exc:error=repr(exc)
            save(RUN/'parent-trace.json',trace.report())
            files={str(p.relative_to(RUN)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in RUN.rglob('*') if p.is_file() and p.name!='parent-guard.json'}
            assert sum(x['size'] for x in files.values())<64*MiB
            if result:
                expected=hashlib.sha256(b'x'*plan['checkpoint_bytes']).hexdigest()
                assert file_digest(campaign/'evidence/quality/23/checkpoints/fixture.pt')==expected
            save(RUN/'result.json',dict(status='completed' if result else 'failed_closed',error=error,
                process=result,plan_sha256=file_digest(PLAN),files=files,
                training=False,retry_allowed=False,source_unchanged=verify()==plan))
            print(json.dumps(dict(status='completed' if result else 'failed_closed',error=error,
                parent_max_wait=trace.report()['max_wait'],parent_max_hold=trace.report()['max_hold'])),flush=True)
            if error:raise RuntimeError(error)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parent','worker']);p.add_argument('--global-fd',type=int);p.add_argument('--campaign-fd',type=int);a=p.parse_args()
    worker(a) if a.mode=='worker' else parent()

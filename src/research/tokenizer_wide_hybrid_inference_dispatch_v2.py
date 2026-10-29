"""Fresh-only serial inference controller, requiring a prepared frozen panel.

Caller owns global/campaign locks and inspects actual workers before entry.
Idle phases have hard resource guards; live-child failures propagate through
the owned process supervisor. No training, retries, cleanup or adoption here.
"""
from contextlib import contextmanager
import os
import signal
import threading
import time

from .protocol import ROOT,digest,file_digest
from .cpu_prerequisite_guard import observation,violation
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _document
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_process_v2 import run_process
from . import tokenizer_wide_hybrid_inference_acceptance_v2 as acceptance
from . import tokenizer_wide_hybrid_inference_entry_v2 as entry
from . import tokenizer_wide_hybrid_inference_outputs_v2 as outputs

KIND='wide_hybrid_inference_controller_v2'


class _Monitor:
    def __init__(self,writer,seed,phase):
        self.writer,self.seed,self.phase=writer,seed,phase
        self.started=time.monotonic()
        self.limits=dict(timeout_seconds=600 if phase=='child' else 1800,
            rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
        self.record=None

    def sample(self,error=None):
        elapsed=time.monotonic()-self.started;sample=observation(self.writer.root)
        why=violation(sample,elapsed,**self.limits)
        return dict(kind=KIND,status='failed' if error or why else 'finished',pid=os.getpid(),
            limits=self.limits,elapsed_seconds=elapsed,observation=sample,
            reason=why or (repr(error)[:1000] if error else None))

    def check(self):
        value=self.sample()
        _require(value['reason'] is None,'Inference parent resource limit: '+str(value['reason']))
        outputs._budget(self.writer)

    def finish(self,error=None):
        self.record=self.sample(error)
        _require(self.record['reason'] is None,'Inference parent resource limit: '+str(self.record['reason']))
        return self.record


class _IdleMonitor(_Monitor):
    """No child may be alive in this context; never wrap Popen in this guard."""
    def __enter__(self):
        self.check();self.stop=threading.Event()
        def watch():
            try:
                while not self.stop.wait(1):
                    value=self.sample()
                    if value['reason'] is not None:
                        try:_failure(self.writer,self.seed,RuntimeError(value['reason']),{self.phase:value})
                        finally:os._exit(90)
            except BaseException as exc:
                try:_failure(self.writer,self.seed,exc,{})
                finally:os._exit(91)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
        return self

    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join()
        self.record=self.sample(error)
        if error is None:self.finish()
        return False


@contextmanager
def _signals():
    _require(threading.current_thread() is threading.main_thread(),'Serial controller requires the main thread')
    def stop(signum,frame):raise KeyboardInterrupt('Inference controller signal '+str(signum))
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for sig in previous:signal.signal(sig,stop)
        yield
    finally:
        for sig,value in previous.items():signal.signal(sig,value)


def _failure(writer,seed,error,guards):
    writer.publish_json(f'emergency/inference-controller-failure-{seed}.json',
        dict(kind=KIND,status='failed',seed=seed,parent_pid=os.getpid(),error=repr(error)[:1000],
            guards=guards,retry_allowed=False,adopted=False))


def verify_controller(writer,seed,ready_ref,review_ref,controller_ref):
    root=writer.root/'evidence'
    _require(type(seed) is int and seed in outputs.SEEDS
        and controller_ref.get('path')==f'inference/{seed}/controller.json','Exact controller completion required')
    ready=_document(root,ready_ref);review=_document(root,review_ref);value=_document(root,controller_ref)
    expected=dict(kind=KIND,status='completed',seed=seed,parent_pid=ready['parent_pid'],
        ready_ref=ready_ref,review_ref=review_ref,adopted=False,dispatch_authorized=False)
    _require(set(value)==set(expected)|{'guards'} and digest({k:value[k] for k in expected})==digest(expected)
        and type(ready['parent_pid']) is int and ready['parent_pid']>1
        and ready_ref['path']==f'inference/{seed}/ready.json' and review_ref['path']==f'inference/{seed}/review.json'
        and review['ready_ref']==ready_ref and review['kind']==acceptance.KIND
        and review['status']=='inference_seed_reviewed' and review['seed']==seed,
        'Controller/readiness/acceptance lineage differs')
    _require(type(value['guards']) is dict and set(value['guards'])=={'prepare','child','review'},'All parent phase guards required')
    import math
    for phase,guard in value['guards'].items():
        limits=dict(timeout_seconds=600 if phase=='child' else 1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
        elapsed=guard.get('elapsed_seconds');sample=guard.get('observation')
        _require(set(guard)=={'kind','status','pid','limits','elapsed_seconds','observation','reason'}
            and guard['kind']==KIND and guard['status']=='finished' and guard['reason'] is None
            and type(guard['pid']) is int and guard['pid']==ready['parent_pid']
            and digest(guard['limits'])==digest(limits)
            and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<limits['timeout_seconds']
            and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
            and all(type(v) is int and v>0 for v in sample.values())
            and violation(sample,elapsed,**limits) is None,'Completed parent resource guard differs')
    return value


def run_seed(writer,seed,previous_ref):
    """Prepare, supervise and accept exactly one fresh seed; no resume/retry."""
    entry._seed(seed);outputs._budget(writer)
    seen=snapshot(writer.root)['files']
    _require(not any(n.startswith('emergency/') or n.endswith('.tmp') or 'failure' in n for n in seen),
             'Failed/partial campaign cannot dispatch')
    _require(not any(n.startswith(f'evidence/inference/{seed}/') and n!='evidence/'+entry.prep.REQUEST for n in seen),
             'Existing inference seed cannot be retried')
    guards={};child=None
    try:
        with _signals():
            with _IdleMonitor(writer,seed,'prepare') as prepare:
                if seed==outputs.SEEDS[0]:
                    _require(previous_ref.get('path')==entry.prep.REQUEST,'Prepared panel request required')
                    ready=entry.prepare_first_readiness(writer,previous_ref)
                else:
                    prior=outputs.SEEDS[outputs.SEEDS.index(seed)-1]
                    _require(previous_ref.get('path')==f'inference/{prior}/review.json','Immediately preceding accepted seed required')
                    # Legacy artifact-only acceptance is readable, but this
                    # controller advances only after its prior guarded child.
                    old_ready=acceptance._ref(writer.root/'evidence',f'inference/{prior}/ready.json')
                    verify_controller(writer,prior,old_ready,previous_ref,
                        acceptance._ref(writer.root/'evidence',f'inference/{prior}/controller.json'))
                    ready=acceptance.prepare_next_readiness(writer,previous_ref)
                acceptance._scope(writer,ready)
                _require(_document(writer.root/'evidence',ready)['parent_pid']==os.getpid(),'Same-parent readiness required')
            guards['prepare']=prepare.record
            child=_Monitor(writer,seed,'child')
            process=run_process(writer,'inference',seed,entry.worker_argv(writer,seed,ready),ROOT,
                required_free_bytes=outputs.CONTRACT['reserve_bytes'],output_limit_bytes=outputs.LIMIT,
                log_limit_bytes=outputs.LIMIT,timeout_seconds=600,parent_check=child.check)
            guards['child']=child.finish()
            with _IdleMonitor(writer,seed,'review') as review:
                worker=acceptance._ref(writer.root/'evidence',f'inference/{seed}/worker-outputs.json')
                accepted=acceptance.accept_completed(writer,worker,process)
            guards['review']=review.record
            value=dict(kind=KIND,status='completed',seed=seed,parent_pid=os.getpid(),ready_ref=ready,
                review_ref=accepted,guards=guards,adopted=False,dispatch_authorized=False)
            path=writer.publish_json(f'evidence/inference/{seed}/controller.json',value)
            ref=dict(path=f'inference/{seed}/controller.json',sha256=file_digest(path))
            verify_controller(writer,seed,ready,accepted,ref);outputs._budget(writer)
            return accepted
    except BaseException as exc:
        if child is not None and 'child' not in guards:guards['child']=child.sample(exc)
        try:
            if isinstance(exc,KeyboardInterrupt) and not (writer.root/'STOP').exists():
                writer.publish_json('STOP',dict(reason=str(exc)[:1000]))
            _failure(writer,seed,exc,guards)
        except BaseException as publication:exc.add_note('Controller failure publication failed: '+repr(publication)[:1000])
        raise


def run_panel(writer,request_ref):
    """Run the fixed three fresh seeds serially; stop on any failure."""
    previous=request_ref;reviews=[]
    for seed in outputs.SEEDS:
        previous=run_seed(writer,seed,previous);reviews.append(previous)
    return dict(status='inference_panel_reviewed_pending_quality_cost_decision',reviews=reviews,
        adopted=False,dispatch_authorized=False)

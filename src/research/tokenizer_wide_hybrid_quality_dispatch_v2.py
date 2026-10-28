"""One serial LR or rank transition with immutable parent acceptance; no campaign CLI.

Own both worker locks and inspect actual processes before calling. Preparation
and post-child review have separate hard guards. During the child, sampled parent
failures raise through the process supervisor so it terminates and reaps its
owned group. There is no retry, selection, cleanup or fresh-confirmation transition.
"""
import math
import os
import sys
import shutil
import time

from .protocol import ROOT,digest,file_digest
from .cpu_prerequisite_guard import observation,violation
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_retention_receipts import _document,_sha
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .tokenizer_wide_hybrid_process_v2 import run_process
from . import tokenizer_wide_hybrid_quality_acceptance_v2 as acceptance
from . import tokenizer_wide_hybrid_quality_readiness_v2 as readiness
from .tokenizer_wide_hybrid_quality_entry_v2 import worker_argv

KIND='wide_hybrid_quality_dispatch_v2'


class _PhaseGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,slot,phase):
        super().__init__(writer,slot,phase,1800)
        self.name=f'control/quality-controller-{phase}-{slot:02d}.json'


class _ChildMonitor:
    """Cooperative parent sampling; run_process owns every live-child failure."""
    def __init__(self,writer,slot):
        self.writer=writer;self.name=f'control/quality-controller-child-{slot:02d}.json'
        self.started=time.monotonic()
        self.limits=dict(timeout_seconds=7200,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)

    def check(self):
        why=violation(observation(self.writer.root),time.monotonic()-self.started,**self.limits)
        if why:raise RuntimeError('Parent child-supervision limit: '+why)

    def finish(self,error=None):
        elapsed=time.monotonic()-self.started;sample=observation(self.writer.root)
        why=violation(sample,elapsed,**self.limits)
        self.writer.publish_json(self.name,dict(kind=KIND,status='failed' if error or why else 'finished',
            reason=why or (repr(error)[:1000] if error else None),pid=os.getpid(),elapsed_seconds=elapsed,
            observation=sample,limits=self.limits))
        if why:raise RuntimeError('Parent child-supervision limit: '+why)


def _guard_ref(writer,slot,phase):
    name=f'control/quality-controller-{phase}-{slot:02d}.json'
    return dict(path=name,sha256=file_digest(writer.root/name))


def _prior(writer,slot,plan_sha256,projection_ref,pilot_completion_ref,*,lr_completion_ref=None):
    """Historical parents need not equal this PID or use these inherited FDs.

    These transport receipts supplement the startup's independent full scientific
    prefix audit. They do not substitute for it or authorize any later stage.
    """
    rank=lr_completion_ref is not None
    _require(type(slot) is int and (9<=slot<=15 if rank else 0<=slot<=9),'Exact dispatch prefix range required')
    root=writer.root/'evidence';refs=[]
    for i in range(9 if rank else 0,slot):
        ref=dict(path=f'meta/quality-dispatch-{i:02d}.json')
        ref['sha256']=file_digest(root/ref['path']);saved=_document(root,ref)
        pid=saved.get('parent_pid');review=saved['review']
        _require(saved.get('kind')==KIND and saved.get('status')==('rank_reviewed' if rank else 'lr_reviewed')
            and type(saved.get('slot')) is int and saved['slot']==i
            and saved.get('plan_sha256')==plan_sha256 and saved.get('projection_ref')==projection_ref
            and saved.get('pilot_completion_ref')==pilot_completion_ref
            and saved.get('prior_dispatches')==refs and type(pid) is int and pid>1,
            'Exact successful earlier dispatch required')
        if rank:
            _require(saved.get('lr_completion_ref')==lr_completion_ref
                and review.get('lr_completion_ref')==lr_completion_ref,'Earlier LR cleanup lineage differs')
        _require(review.get('kind')==(acceptance.RANK_KIND if rank else acceptance.KIND)
            and review.get('status')==('rank_child_reviewed_pending_parent_receipt' if rank else 'lr_child_reviewed_pending_parent_receipt')
            and type(review.get('slot')) is int and review['slot']==i and review.get('plan_sha256')==plan_sha256
            and review.get('selection') is None and all(review.get(k) is False for k in
                ('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized')),
            'Earlier acceptance claims differ')
        for phase,seconds in [('prepare',1800),('child',7200),('review',1800)]:
            guard=saved['guards'][phase]
            _require(guard['path']==f'control/quality-controller-{phase}-{i:02d}.json','Parent guard path differs')
            acceptance._guard(writer,guard,KIND,pid,seconds)
        startup=_document(root,review['startup']);seal=_document(root,review['readiness'])
        _require(review['startup']['path']==('meta/quality-first-startup.json' if i==0 else f'meta/quality-{"rank" if rank else "lr"}-startup-{i:02d}.json')
            and review['readiness']['path']==f'meta/quality-readiness-{i:02d}.json'
            and seal['parent_pid']==pid and seal['startup_ref']==review['startup']
            and startup['projection_ref']==projection_ref and startup['pilot_completion_ref']==pilot_completion_ref
            and startup['plan_sha256']==seal['plan_sha256']==plan_sha256,'Earlier startup/readiness differs')
        if rank:_require(startup.get('lr_completion_ref')==lr_completion_ref,'Earlier startup LR cleanup differs')
        # These controls were immutable before the earlier child started. This
        # includes startup guards and controls from still earlier parents, not
        # merely the three controller guards checked above.
        deadline=time.monotonic()+1800
        for name,binding in seal['control_inventory'].items():
            _require(name.startswith('control/') and name.endswith('.json'),'Earlier sealed control path differs')
            acceptance._check(writer.root,name,binding,deadline)
        child=review['child']
        for key,tail in [('launch','process/launch.json'),('process','process/process.json'),('worker','worker-outputs.json')]:
            _require(child[key]['path']==f'quality/{i:02d}/{tail}','Earlier child reference path differs')
            _document(root,child[key])
        launch=_document(root,child['launch']);terminal=_document(root,child['process'])
        argv=launch['argv']
        _require(len(argv)==16 and argv[0]==sys.executable and argv[1:4]==['-m','src.research.tokenizer_wide_hybrid_quality_entry_v2','worker']
            and argv[4:10]==['--campaign',str(writer.root),'--slot',str(i),'--startup-sha256',review['startup']['sha256']]
            and argv[10]=='--global-fd' and argv[12]=='--campaign-fd'
            and argv[14:]==['--readiness-sha256',review['readiness']['sha256']], 'Earlier canonical child argv differs')
        _require(argv[11].isdigit() and argv[13].isdigit() and int(argv[11])>2 and int(argv[13])>2
            and argv[11]!=argv[13],'Earlier inherited lock descriptors differ')
        expected=dict(status='preflight',argv=argv,parent_pid=pid,pid=None,returncode=None,error=None,
            required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,
            output_limit_bytes=seal['output_limit_bytes'],log_limit_bytes=4*1024**2,timeout_seconds=7200,
            observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
        _require(digest(launch)==digest(expected) and terminal['argv']==argv and terminal['parent_pid']==pid
            and terminal['status']=='completed' and type(terminal['returncode']) is int and terminal['returncode']==0
            and terminal['error'] is None and type(terminal['pid']) is int and terminal['pid']>1
            and terminal['pid']!=pid,'Earlier child completion differs')
        for key in ('argv','parent_pid','required_free_bytes','reserve_bytes','output_limit_bytes',
                    'log_limit_bytes','timeout_seconds','capture_kind','log_truncated'):
            _require(type(terminal.get(key)) is type(expected[key]) and terminal[key]==expected[key],
                     'Earlier terminal process limits differ')
        elapsed=terminal.get('elapsed_seconds');sample=terminal.get('observation',{})
        census=snapshot(writer.root)
        total=sum(v['size'] for n,v in census['files'].items() if n.startswith(f'evidence/quality/{i:02d}/'))
        log=(root/f'quality/{i:02d}/process/worker.log').stat().st_size
        _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<7200
            and set(sample)=={'free_bytes','output_bytes','log_bytes'} and all(type(v) is int for v in sample.values())
            and sample['free_bytes']>=8*1024**3 and sample['output_bytes']==total<=seal['output_limit_bytes']
            and sample['log_bytes']==log<=4*1024**2,'Earlier terminal resource census differs')
        _require(seal['guard']['path']==f'control/quality-readiness-{i:02d}.json','Earlier readiness guard path differs')
        acceptance._guard(writer,seal['guard'],readiness.KIND,pid,1800)
        entry=review['entry_guard']
        _require(entry['path']==f'control/quality-entry-{i:02d}.json','Earlier child guard path differs')
        value=acceptance._guard(writer,dict(path=entry['path'],sha256=entry['sha256']),acceptance.entry.KIND,terminal['pid'],7200)
        _require(value==entry['record'],'Earlier child guard record differs')
        refs.append(ref)
    return refs


def _fresh(writer,slot,stop=9):
    _locks(writer.root/'evidence',writer.lock_fds)
    files=snapshot(writer.root)['files']
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents dispatch')
    _require(not any(n.endswith('.tmp') or '-failure' in n for n in files),'Partial/failed campaign cannot dispatch')
    for i in range(slot,stop):
        _require(not any(n.startswith(f'control/quality-controller-') and n.endswith(f'-{i:02d}.json')
            or n.startswith(f'evidence/meta/quality-dispatch-{i:02d}') for n in files),
            'Existing controller attempt cannot be retried')


def run_lr_once(writer,slot,plan_sha256,*,projection_ref,pilot_completion_ref):
    """Dispatch exactly one fresh LR slot, then stop at an immutable receipt."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<9,'Exact LR writer/slot required')
    return _run_once(writer,slot,plan_sha256,projection_ref,pilot_completion_ref)


def run_rank_once(writer,slot,plan_sha256,*,projection_ref,lr_completion_ref):
    """Dispatch one rank screen after verified LR cleanup; no fresh or selection."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 9<=slot<=14,'Exact rank writer/slot required')
    _require(isinstance(lr_completion_ref,dict),'Exact LR completion reference required')
    return _run_once(writer,slot,plan_sha256,projection_ref,None,lr_completion_ref=lr_completion_ref)


def _run_once(writer,slot,plan_sha256,projection_ref,pilot_completion_ref,*,lr_completion_ref=None):
    rank=lr_completion_ref is not None
    _sha(plan_sha256);_fresh(writer,slot,15 if rank else 9)
    root=writer.root/'evidence';base=f'meta/quality-dispatch-{slot:02d}'
    try:
        with _PhaseGuard(writer,slot,'prepare'):
            if rank:
                # Lazy import: rank startup itself uses acceptance/readiness.
                from . import tokenizer_wide_hybrid_quality_rank_startup_v2 as ranks
                history=ranks.verify_lr_history(writer,plan_sha256,lr_completion_ref,
                    current_inventory=acceptance._inventory(root))
                _require(history['projection_ref']==projection_ref,'Dispatch LR projection differs')
                pilot_completion_ref=history['pilot_completion_ref']
            prior_args=dict(lr_completion_ref=lr_completion_ref)
            previous=_prior(writer,slot,plan_sha256,projection_ref,pilot_completion_ref,**prior_args)
            if rank:
                startup=ranks.prepare_rank_request(writer,slot,plan_sha256,
                    projection_ref=projection_ref,lr_completion_ref=lr_completion_ref)
            else:
                args=dict(projection_ref=projection_ref,pilot_completion_ref=pilot_completion_ref)
                startup=(readiness.first.prepare_first_quality_request(writer,plan_sha256,**args) if slot==0 else
                         readiness.later.prepare_lr_request(writer,slot,plan_sha256,**args))
        # No running preparation control may be captured by the readiness seal.
        ready=readiness.seal_quality_readiness(writer,slot,startup)
        seal=_document(root,ready)
        monitor=_ChildMonitor(writer,slot)
        try:
            process=run_process(writer,'quality',slot,worker_argv(writer,slot,startup,ready),ROOT,
                required_free_bytes=seal['required_free_bytes'],output_limit_bytes=seal['output_limit_bytes'],
                log_limit_bytes=4*1024**2,timeout_seconds=7200,parent_check=monitor.check)
        except BaseException as exc:
            try:monitor.finish(exc)
            except BaseException as error:exc.add_note('Parent monitor publication failed: '+repr(error)[:1000])
            raise
        monitor.finish()
        with _PhaseGuard(writer,slot,'review'):
            review=(acceptance.review_completed_rank if rank else acceptance.review_completed_lr)(writer,slot,startup,ready,process)
            _require(_prior(writer,slot,plan_sha256,projection_ref,pilot_completion_ref,**prior_args)==previous,
                     'Earlier dispatch receipts changed during child/review')
        guards={phase:_guard_ref(writer,slot,phase) for phase in ('prepare','child','review')}
        for phase,ref in guards.items():acceptance._guard(writer,ref,KIND,os.getpid(),7200 if phase=='child' else 1800)
        _locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents success publication')
        _require(shutil.disk_usage(writer.root).free>=review['storage']['required_free_bytes'],
                 'Remaining full campaign storage changed before receipt')
        result=dict(kind=KIND,status='rank_reviewed' if rank else 'lr_reviewed',slot=slot,parent_pid=os.getpid(),plan_sha256=plan_sha256,
            projection_ref=projection_ref,pilot_completion_ref=pilot_completion_ref,prior_dispatches=previous,
            guards=guards,review=review)
        if rank:result['lr_completion_ref']=lr_completion_ref
        path=writer.publish_json('evidence/'+base+'.json',result)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            plan_sha256=plan_sha256,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Dispatch failure publication failed: '+repr(error)[:1000])
        raise

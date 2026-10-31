"""One bounded activation transition. An unfrozen or incomplete plan cannot run.

No freeze builder or implicit resume: interrupted reservations, partial cleanup,
missing dispatch completion and prior failures require review, never a retry.
"""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .protocol import ROOT, digest, file_digest
from .runner import source_fingerprint
from .tokenizer_study import runtime_identity
from .tokenizer_activation_config import LIMITS, BASELINE, BASELINE_SHA256
from .tokenizer_activation_rules import pilot_specs, screen_specs, confirmation_specs, FAMILIES, RATES
from .tokenizer_activation_session import Session, reference, require
from .tokenizer_activation_worker import produce, accept_outputs, ALLOWANCE
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention import RESERVE, METADATA
from .tokenizer_wide_hybrid_cleanup import _locks, GLOBAL_LOCK
from .tokenizer_wide_hybrid_driver import session_lock, CORPUS_SHA256
from .tokenizer_wide_hybrid_process_v2 import run_process
from .tokenizer_serial_process import _group_exists
from .tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from .cpu_prerequisite_guard import observation, violation

KIND = 'activation_dispatch_v1'
HIGH_WATER = 8*ALLOWANCE
FULL_FREE = RESERVE + METADATA + HIGH_WATER
PREREQUISITES = dict(
    model='passed', state='activation_serialized_state_checks_passed_transport_integration_pending',
    rules='passed', session='session_core_and_synthetic_filesystem_transport_passed',
    worker='worker_body_and_independent_acceptance_prerequisites_passed',
    dispatch='passed', retention='passed', inference='passed', capacity='passed')


def design():
    return dict(pilots=pilot_specs(), screen=screen_specs(), confirmations=[
        confirmation_specs(dict(zip(FAMILIES, (a,b)))) for a in RATES for b in RATES])


def idle(allowed=()):
    """Read-only process check; only this controller/explicit parent may match."""
    output = subprocess.check_output(['ps','-axo','pid=,ppid=,command='], text=True)
    unexpected = []
    for line in output.splitlines():
        parts = line.strip().split(None,2)
        if len(parts) != 3: continue
        pid, _, command = parts; executable = Path(command.split()[0]).name.lower()
        research = (executable.startswith('python') or executable.startswith('pytest')) and any(
            marker in command for marker in ('src.research', 'src/research', 'tokenizer_', 'pytest', str(ROOT)))
        if research and int(pid) not in {os.getpid(), *allowed}: unexpected.append(dict(pid=int(pid),command=command))
    require(not unexpected, 'Other research/test workers are live: '+repr(unexpected))
    return dict(checked=True, allowed_pids=sorted({os.getpid(),*allowed}))


def bind(ledger, *, capacity=False):
    """Verify immutable freeze, code/runtime/prerequisites; never infer readiness."""
    require(not (ledger.root/'STOP').exists() and not (ledger.root/'STOP').is_symlink(), 'STOP blocks execution')
    _locks(ledger.root/'evidence', ledger.writer.lock_fds)
    require(ledger.root.parent == ROOT/'research_runs'
            and re.fullmatch(r'tokenizer-activation-[0-9]{3}',ledger.root.name) is not None,
            'Only a dedicated new activation campaign root is executable')
    plan = ledger.document(ledger.plan)
    require(set(plan) == {'kind','status','root','source','runtime','design','limits','inputs','prerequisites','storage'}
            and plan['kind'] == 'activation_campaign_v1' and plan['status'] == 'frozen'
            and plan['root'] == str(ledger.root), 'Complete frozen activation plan required')
    require(digest(plan['source']) == digest(source_fingerprint()) and plan['runtime'] == runtime_identity(),
            'Frozen source/runtime changed')
    require(digest(plan['design']) == digest(design()) and plan['limits'] == LIMITS,
            'Finite design or resource limits changed')
    require(plan['storage'] == dict(trajectory_allowance_bytes=ALLOWANCE, maximum_weight_bytes=HIGH_WATER,
        metadata_bytes=METADATA, reserve_bytes=RESERVE, full_required_free_bytes=FULL_FREE), 'Storage freeze differs')
    require(file_digest(BASELINE) == BASELINE_SHA256, 'Incumbent recipe changed')
    inputs = plan['inputs']
    require(type(inputs) is dict and set(inputs) == {'tokenizer','training96_sha256','validation_sha256','validation_targets','validation_target_bytes'}
            and type(inputs['tokenizer']) is dict and inputs['tokenizer']
            and type(inputs['validation_targets']) is int and inputs['validation_targets'] == 34278
            and type(inputs['validation_target_bytes']) is int and inputs['validation_target_bytes'] == 111536,
            'Exact frozen input identities required')
    from .tokenizer_wide_hybrid_retention_receipts import _sha
    _sha(inputs['training96_sha256']); _sha(inputs['validation_sha256'])
    require(set(plan['prerequisites']) == set(PREREQUISITES), 'All launch prerequisites required')
    docs = {}
    for name, expected_status in PREREQUISITES.items():
        ref = plan['prerequisites'][name]
        require(ref['path'].startswith('research/tokenizer-evidence/activation-preparation-001/')
                and ref['path'].endswith('.json') and ref['size'] <= 2*1024**2,
                'Bound activation prerequisite required')
        require(reference(ROOT,ref['path']) == ref, 'Prerequisite changed')
        docs[name] = json.loads((ROOT/ref['path']).read_text())
        require(docs[name].get('status') == expected_status, 'Prerequisite incomplete: '+name)
    cap = docs['capacity']
    require(cap.get('source_sha256') == digest(plan['source']) and cap.get('runtime') == plan['runtime']
            and cap.get('prerequisites') == {k:v for k,v in plan['prerequisites'].items() if k != 'capacity'}
            and cap.get('storage') == plan['storage'] and cap.get('inputs') == inputs
            and cap.get('complete_executable_review') is True, 'Final capacity/applicability review incomplete')
    if capacity: require_capacity(ledger)
    return plan


def require_capacity(ledger):
    seen = snapshot(ledger.root)
    state = ledger.inspect()
    stems = {str(Path(ref['path']).parent) for ref in state['pins'].values() if ref['path'].endswith('/request.json')}
    require(all(any(name.startswith(stem+'/checkpoints/') for stem in stems)
                for name, entry in seen['files'].items() if entry['group'].endswith('/weights')),
            'Unknown weights cannot count as allocated campaign capacity')
    used = seen['checkpoint_logical_bytes']; metadata = seen['metadata_bytes']
    require(used <= HIGH_WATER and metadata <= METADATA, 'Campaign high-water allocation exceeded')
    # Existing actual bytes are already reflected in filesystem free space.
    # Only completed boundary receipts permit removal; no deletion is forecast.
    required = RESERVE + (HIGH_WATER-used) + (METADATA-metadata)
    free = shutil.disk_usage(ledger.root).free
    require(free >= required, f'Full remaining panel needs {required} free bytes; observed {free}')
    return dict(required_free_bytes=required, observed_free_bytes=free, weight_bytes=used, metadata_bytes=metadata)


class PhaseGuard(_Guard):
    KIND = KIND
    def __init__(self, writer, stage, slot, phase):
        super().__init__(writer, slot, phase, 600)
        self.name = f'control/activation-{phase}-{stage}-{slot:02d}.json'


def guard_ref(ledger, stage, slot, phase):
    return reference(ledger.root, f'control/activation-{phase}-{stage}-{slot:02d}.json')


def verify_guard(ledger, ref, parent_pid):
    value = ledger.document(ref)
    limits = dict(rss_limit_bytes=4*1024**3, reserve_bytes=RESERVE, timeout_seconds=600)
    require(value.get('kind') == KIND and value.get('status') == 'finished' and value.get('reason') is None
            and value.get('pid') == parent_pid and value.get('limits') == limits
            and violation(value['observation'],value['elapsed_seconds'],**limits) is None, 'Parent audit guard incomplete')


def argv_for(ledger, request_ref, seal_ref, parent_pid, fds):
    return [sys.executable,'-m','src.research.tokenizer_activation_entry','worker',
            '--root',str(ledger.root),'--plan-sha256',ledger.plan['sha256'],
            '--request',request_ref['path'],'--request-sha256',request_ref['sha256'],
            '--seal',seal_ref['path'],'--seal-sha256',seal_ref['sha256'],
            '--parent-pid',str(parent_pid),'--global-fd',str(fds[0]),'--campaign-fd',str(fds[1])]


def transport(ledger, stage, slot, seal_ref):
    seal = ledger.document(seal_ref); parent = seal['parent_pid']
    require(type(parent) is int and parent > 1, 'Exact parent PID required')
    process_ref = reference(ledger.root, f'evidence/{stage}/{slot:02d}/process/process.json')
    process = ledger.document(process_ref)
    expected_argv = argv_for(ledger, seal['request'], seal_ref, parent, seal['lock_fds'])
    require(process.get('status') == 'completed' and process.get('returncode') == 0 and process.get('error') is None
            and process.get('parent_pid') == parent and type(process.get('pid')) is int and process['pid'] > 1
            and process.get('argv') == expected_argv and process.get('capture_kind') == 'bounded_pipe_v2'
            and process.get('log_truncated') is False and process.get('reserve_bytes') == RESERVE
            and process.get('timeout_seconds') == (600 if stage == 'pilots' else 7200)
            and process.get('required_free_bytes') == seal['capacity']['required_free_bytes']
            and process.get('output_limit_bytes') == ALLOWANCE+9*1024**2
            and process.get('log_limit_bytes') == 4*1024**2, 'Completed canonical child transport required')
    require(not any(k in process for k in ('cleanup_error','log_close_error','failure_observation_error')),
            'Uncertain child cleanup is not acceptance')
    sample=process.get('observation',{})
    require(set(sample)=={'free_bytes','output_bytes','log_bytes'}
            and all(type(v) is int and v>=0 for v in sample.values())
            and sample['free_bytes']>=RESERVE and sample['output_bytes']<=ALLOWANCE+9*1024**2
            and sample['log_bytes']<=4*1024**2
            and type(process.get('elapsed_seconds')) in (int,float)
            and 0<=process['elapsed_seconds']<process['timeout_seconds'], 'Child resource bounds differ')
    return process_ref, process


def prefix(ledger, state):
    refs = []
    for i, acceptance in enumerate(state['reviews']):
        stage, slot = ('pilots',i) if i < 2 else ('quality',i-2)
        name = f'evidence/meta/dispatch-complete-{stage}-{slot:02d}.json'
        ref = reference(ledger.root,name); done = ledger.document(ref)
        seal_ref = done['seal']; seal = ledger.document(seal_ref)
        require(done == dict(kind=KIND,status='accepted',plan=ledger.plan,request=seal['request'],
                    acceptance=acceptance,seal=seal_ref,process=done['process'],review_guard=done['review_guard'],entry_guard=done['entry_guard'])
                and seal['kind'] == KIND and seal['plan'] == ledger.plan and seal['prior_dispatches'] == refs
                and seal['stage'] == stage and seal['slot'] == slot,
                'Complete ordered dispatch receipts required')
        require(seal_ref['path']==f'evidence/meta/dispatch-seal-{stage}-{slot:02d}.json'
                and seal['request']['path']==f'evidence/{stage}/{slot:02d}/request.json'
                and ledger.document(acceptance)['request']==seal['request']
                and seal['prepare_guard']['path']==f'control/activation-prepare-{stage}-{slot:02d}.json'
                and done['review_guard']['path']==f'control/activation-review-{stage}-{slot:02d}.json'
                and done['entry_guard']['path']==f'control/activation-entry-{stage}-{slot:02d}.json',
                'Dispatch stage paths differ')
        actual, process = transport(ledger,stage,slot,seal_ref)
        verify_guard(ledger,done['entry_guard'],process['pid'])
        require(done['process'] == actual, 'Saved process receipt changed')
        verify_guard(ledger,seal['prepare_guard'],seal['parent_pid'])
        verify_guard(ledger,done['review_guard'],seal['parent_pid'])
        refs.append(ref)
    return refs


def worker(ledger, request_ref, seal_ref, parent_pid):
    stage, slot = Path(request_ref['path']).parts[1:3]; slot = int(slot)
    require(stage in ('pilots','quality') and 0 <= slot < (2 if stage == 'pilots' else 12),
            'Finite child entry slot required')
    with PhaseGuard(ledger.writer,stage,slot,'entry'):
        plan = bind(ledger,capacity=True); seal = ledger.document(seal_ref)
        require(os.getppid() == parent_pid and seal['parent_pid'] == parent_pid
                and seal['lock_fds'] == list(ledger.writer.lock_fds) and seal['plan'] == ledger.plan
                and seal['kind'] == KIND and seal['request'] == request_ref,
                'Child parent/locks/request seal differs')
        state = ledger.inspect()
        require(state['pending'] is not None and state['pending']['ref'] == request_ref
                and prefix(ledger,state) == seal['prior_dispatches'], 'Child accepted-prefix seal differs')
        require(seal_ref['path'] == f'evidence/meta/dispatch-seal-{stage}-{slot:02d}.json'
                and seal['stage'] == stage and seal['slot'] == slot, 'Child slot differs')
        verify_guard(ledger,seal['prepare_guard'],parent_pid); idle((parent_pid,))
    def recheck():
        bind(ledger)
        require(ledger.document(seal_ref) == seal and ledger.verify(request_ref).is_file(), 'Active worker seal changed')
    return produce(ledger,request_ref,plan['inputs'],recheck)


def run_once(ledger):
    """Exactly one trial, saved selection, or reviewed retention transition."""
    bind(ledger); idle(); state = ledger.inspect(); prior = prefix(ledger,state)
    from .tokenizer_activation_retention import verify_boundaries, execute_boundary
    verify_boundaries(ledger,state)
    action = state['next']['kind']
    if action == 'save_selection': return ledger.save_selection()
    if action in ('review_pilot_boundary','review_screen_boundary'):
        return execute_boundary(ledger)
    require(action in ('pilot','quality'), 'No runnable transition (pending/failed/final states never retry)')
    stage,slot = ('pilots',len(state['pilots'])) if action == 'pilot' else ('quality',len(state['records']))
    request_ref = None
    try:
        with PhaseGuard(ledger.writer,stage,slot,'prepare'):
            bind(ledger,capacity=True); idle()
            request_ref = ledger.reserve()
        cap = require_capacity(ledger)
        seal_name = f'evidence/meta/dispatch-seal-{stage}-{slot:02d}.json'
        ledger.writer.publish_json(seal_name,dict(kind=KIND,plan=ledger.plan,request=request_ref,stage=stage,slot=slot,
            parent_pid=os.getpid(),lock_fds=list(ledger.writer.lock_fds),prior_dispatches=prior,capacity=cap,
            prepare_guard=guard_ref(ledger,stage,slot,'prepare')))
        seal_ref = reference(ledger.root,seal_name)
        argv = argv_for(ledger,request_ref,seal_ref,os.getpid(),ledger.writer.lock_fds)
        started = time.monotonic(); seconds = 600 if stage == 'pilots' else 7200
        def parent_check():
            why=violation(observation(ledger.root),time.monotonic()-started,
                          rss_limit_bytes=4*1024**3,reserve_bytes=RESERVE,timeout_seconds=seconds)
            require(why is None,'Parent supervision limit: '+str(why))
        run_process(ledger.writer,stage,slot,argv,ROOT,required_free_bytes=cap['required_free_bytes'],
                    output_limit_bytes=ALLOWANCE+9*1024**2,timeout_seconds=seconds,parent_check=parent_check)
        with PhaseGuard(ledger.writer,stage,slot,'review'):
            bind(ledger); idle(); process_ref, process = transport(ledger,stage,slot,seal_ref)
            require(not _group_exists(process['pid']), 'Child left a live process group')
            verify_guard(ledger,guard_ref(ledger,stage,slot,'entry'),process['pid'])
            accepted = accept_outputs(ledger,request_ref,bind(ledger)['inputs'],lambda:bind(ledger))
        name=f'evidence/meta/dispatch-complete-{stage}-{slot:02d}.json'
        ledger.writer.publish_json(name,dict(kind=KIND,status='accepted',plan=ledger.plan,request=request_ref,
            acceptance=accepted,seal=seal_ref,process=process_ref,review_guard=guard_ref(ledger,stage,slot,'review'),
            entry_guard=guard_ref(ledger,stage,slot,'entry')))
        return reference(ledger.root,name)
    except BaseException as exc:
        if request_ref is not None:
            try:
                name=f'evidence/{stage}/{slot:02d}/session-failure.json'
                ledger.writer.publish_json(name,dict(request=request_ref,retry_allowed=False,error=repr(exc)[:1000]))
                # STOP and incomplete output may block the ledger event; the
                # permanent request/process/failure files still forbid retry.
                ledger.record('failed',reference(ledger.root,name))
            except BaseException as error: exc.add_note('Failure latch: '+repr(error)[:1000])
        raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('step','worker'))
    parser.add_argument('--root',required=True);parser.add_argument('--plan-sha256',required=True)
    for name in ('request','request-sha256','seal','seal-sha256'):parser.add_argument('--'+name)
    for name in ('parent-pid','global-fd','campaign-fd'):parser.add_argument('--'+name,type=int)
    args=parser.parse_args(argv);root=Path(args.root).absolute()
    require(root == root.resolve() and root.is_dir(),'Canonical existing campaign root required')
    def run(fds):
        writer=MetadataWriter(root,fds);plan_ref=reference(root,'evidence/meta/activation-plan.json')
        require(plan_ref['sha256']==args.plan_sha256,'Plan CLI hash differs');ledger=Session(writer,plan_ref)
        if args.mode=='step':return run_once(ledger)
        rq=reference(root,args.request);seal=reference(root,args.seal)
        require(rq['sha256']==args.request_sha256 and seal['sha256']==args.seal_sha256,'Child CLI hashes differ')
        return worker(ledger,rq,seal,args.parent_pid)
    if args.mode=='step':
        require(all(getattr(args,name) is None for name in ('request','request_sha256','seal','seal_sha256','parent_pid','global_fd','campaign_fd')),
                'Step mode cannot supply child arguments')
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:return run((g,c))
    require(all(getattr(args,name) is not None for name in ('request','request_sha256','seal','seal_sha256','parent_pid','global_fd','campaign_fd')),
            'Complete inherited worker arguments required')
    return run((args.global_fd,args.campaign_fd))


if __name__=='__main__':
    main()

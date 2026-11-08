"""Correct one resource-root binding; leave scientific worker math immutable."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from research.byte_context_transform_pilot_v1 import ROOT,BASE as ORIGINAL,sha,put,read,runtime,pins
BASE=ORIGINAL/'quality-recovery001'
OUT=BASE/'resource-recovery001'
ORIGINAL_WEIGHTS=ROOT/'research_runs/byte-context-transform-001'
NEW_WEIGHTS=ROOT/'research_runs/byte-context-transform-recovery-001'
ROOTS=(ORIGINAL,ORIGINAL_WEIGHTS,NEW_WEIGHTS)
CAP=int(1.5*1024**3)


def actual_bytes():
    assert len({p.resolve() for p in ROOTS})==3
    return sum(p.stat().st_size for root in ROOTS for p in root.rglob('*') if p.is_file())


def process(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','state=,command='],capture_output=True,text=True)
    if r.returncode==1 and not r.stdout.strip():return None
    assert r.returncode==0,r.stderr
    state,command=r.stdout.strip().split(None,1)
    return dict(state=state,command=command)


def prepare():
    import shutil
    assert not OUT.exists()
    old=read(BASE/'quality-plan.json');pins(old)
    assert sha(BASE/'quality-plan.json')=='badc7b0daa9ff8455444d08fc5b7dcc6717033122077500253a66ae86cefd3cd'
    p=process(61922);w=process(61923)
    assert p and p['state'].startswith('T') and 'research.byte_context_transform_run_v2 run' in p['command']
    assert w and 'research.byte_context_transform_run_v2 worker 1' in w['command']
    sources={str(p.relative_to(ROOT)):sha(p) for f in ('research','src') for p in (ROOT/f).rglob('*.py')}
    sources[str((ORIGINAL/'resource-recovery-decision.md').relative_to(ROOT))]=sha(ORIGINAL/'resource-recovery-decision.md')
    measured=actual_bytes();free=shutil.disk_usage(ROOT).free
    assert free>=CAP-measured+10*1024**3
    plan=dict(kind='byte_context_resource_root_recovery_v1',sources=sources,assets=old['assets'],runtime=runtime(),
        scientific_plan_sha256=sha(BASE/'quality-plan.json'),controller_pid=61922,current_worker_pid=61923,current_slot=1,
        remaining_slots=list(range(2,9)),correct_roots=[str(p) for p in ROOTS],initial_bytes=measured,initial_free=free,
        output_cap=CAP,reserve=8*1024**3,extra_headroom=2*1024**3,
        scope='One corrected ORIGINAL_WEIGHTS constant supplied by a separately bound wrapper; no model, scorer, role, training or acceptance change. Current worker1 completes untouched.',
        original_dispatcher_suspended=True)
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def corrected_module(plan):
    pins(plan)
    from research import byte_context_transform_run_v2 as worker
    assert worker.ORIGINAL_WEIGHTS==worker.WEIGHTS==NEW_WEIGHTS
    assert worker.BASE==BASE and worker.ORIGINAL==ORIGINAL and worker.CAP==CAP
    assert sha(BASE/'quality-plan.json')==plan['scientific_plan_sha256']
    worker.ORIGINAL_WEIGHTS=ORIGINAL_WEIGHTS
    assert (worker.ORIGINAL,worker.ORIGINAL_WEIGHTS,worker.WEIGHTS)==ROOTS
    return worker


def record_slot(index,plan,kind):
    folder=BASE/f'{index:02d}';receipt=read(folder/'acceptance.json');result=read(folder/'result.json')
    assert receipt['accepted'] and receipt['result_sha256']==sha(folder/'result.json') and receipt['guard_sha256']==sha(folder/'guard.json')
    assert read(folder/'guard.json')['status']=='finished'
    for key in ('raw','h200'):
        r=result[key];assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['size']
    total=actual_bytes();assert total<CAP
    put(OUT/f'slot-{index:02d}.json',dict(kind=kind,slot=index,resource_plan_sha256=sha(OUT/'plan.json'),
        scientific_acceptance_sha256=sha(folder/'acceptance.json'),correct_total_retained_bytes=total,
        roots=plan['correct_roots'],weights=[result['raw'],result['h200']],
        limitation='Slot01 original guard used duplicate new root; this endpoint accounting plus immutable original weights and monotone new checkpoint publication bounds its actual output.' if index==1 else None))


def worker(index):
    plan=read(OUT/'plan.json');assert index in plan['remaining_slots']
    module=corrected_module(plan)
    module.worker(index);pins(plan)
    record_slot(index,plan,'corrected_root_worker')


def run():
    plan=read(OUT/'plan.json');module=corrected_module(plan)
    with (OUT/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(OUT/'controller-attempt.json',dict(pid=os.getpid(),resource_plan_sha256=sha(OUT/'plan.json')))
        started=time.monotonic()
        # The existing worker remains untouched, including its numerical guards.
        while True:
            state=process(plan['current_worker_pid'])
            if state is None or state['state'].startswith('Z'):break
            assert 'research.byte_context_transform_run_v2 worker 1' in state['command']
            assert time.monotonic()-started<3900
            assert actual_bytes()<CAP
            time.sleep(5)
        parent=process(plan['controller_pid'])
        assert parent and parent['state'].startswith('T') and 'research.byte_context_transform_run_v2 run' in parent['command']
        # Only the obsolete stopped dispatcher is terminated, after its child is terminal.
        os.kill(plan['controller_pid'],signal.SIGKILL)
        put(OUT/'dispatcher-superseded.json',dict(pid=plan['controller_pid'],worker_terminal=True,signal='SIGKILL',reason='Prevent duplicate dispatch; no scientific worker terminated'))
        if (BASE/'01/failure.json').exists():
            put(OUT/'current-worker-failure.json',dict(failure_sha256=sha(BASE/'01/failure.json')))
            return 1
        record_slot(1,plan,'existing_worker_completed_without_interruption')
        scientific=read(BASE/'quality-plan.json')
        # Charge completed execution durations including the imported baseline.
        charged=scientific['carried_elapsed_seconds']+read(BASE/'01/guard.json')['elapsed_seconds']
        for index in plan['remaining_slots']:
            assert scientific['total_seconds']-charged>=scientific['per_trial_seconds']
            assert not (BASE/f'{index:02d}').exists()
            child=subprocess.run([sys.executable,'-m','research.byte_context_resource_recovery_v1','worker',str(index)],cwd=ROOT)
            if child.returncode:
                put(OUT/'controller-failure.json',dict(slot=index,returncode=child.returncode));return child.returncode
            charged+=read(BASE/f'{index:02d}'/'guard.json')['elapsed_seconds']
        pins(plan);module.finish(scientific)
        put(BASE/'controller-complete.json',dict(completed=True,resource_plan_sha256=sha(OUT/'plan.json'),charged_guard_seconds=charged))
        print(json.dumps(read(BASE/'final.json')),flush=True);return 0


def review():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(OUT/'plan.json');pins(plan)
    from research import byte_context_transform_review_v2 as reviewer
    reviewer.PLAN_SHA=plan['scientific_plan_sha256']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(BASE/'final-review-guard.json',ROOT,timeout_seconds=300):
            result=reviewer.run()
            for index in range(1,9):
                r=read(OUT/f'slot-{index:02d}.json')
                assert r['resource_plan_sha256']==sha(OUT/'plan.json') and r['roots']==[str(p) for p in ROOTS]
                assert r['correct_total_retained_bytes']<CAP
                assert r['scientific_acceptance_sha256']==sha(BASE/f'{index:02d}'/'acceptance.json')
            assert actual_bytes()<CAP
            result.update(resource_root_correction_verified=True,resource_plan_sha256=sha(OUT/'plan.json'),
                actual_retained_bytes=actual_bytes(),slot01_resource_scope=read(OUT/'slot-01.json')['limitation'])
            pins(plan)
        put(BASE/'final-review.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker','review'));p.add_argument('index',nargs='?',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='worker':worker(a.index)
    elif a.action=='review':review()
    else:sys.exit(run())

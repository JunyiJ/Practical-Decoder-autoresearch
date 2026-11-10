"""Native horizon binding of the qualified time worker; training and publication unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'
RUNS=ROOT/'research_runs/general-rsi-native-horizon-001'
STOP=False


def put(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def replace(path,value):
    path=Path(path);temp=path.with_suffix('.json.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def stop(signum,frame):
    global STOP
    STOP=True


def stopped():
    return STOP or json.loads((ROOT/'research_state/general-rsi-state.json').read_text()).get('stop_requested',False)


def boot():
    return subprocess.check_output(['/usr/sbin/sysctl','-n','kern.boottime'],text=True).strip()


def retire(receipt,journal,dependency):
    from research.general_rsi_training_v2 import sha
    from research.general_rsi_time_journal_v1 import append
    p=Path(receipt['path'])
    assert p.resolve()==p and p.parent.parent==RUNS and p.name.startswith('restart-')
    assert not p.is_symlink() and p.stat().st_nlink==1
    st=p.stat();assert st.st_size==receipt['size'] and sha(p)==receipt['sha256']
    r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True)
    assert r.returncode==1 and not r.stdout and not r.stderr
    row=dict(receipt=receipt,inode=st.st_ino,closed_dependency=dependency,no_open_consumers=True)
    append(journal,dict(event='before_unlink',**row))
    assert p.stat().st_ino==st.st_ino and sha(p)==receipt['sha256'];p.unlink()
    append(journal,dict(event='unlinked',**row))


def execute(plan,slot,binding,attempt,out,weights):
    import torch
    from research.general_rsi_equal_time_training_v1 import Context,write_checkpoint,read_checkpoint,clone_tree,sha
    from research.general_rsi_training_v2 import score
    from research.general_rsi_time_journal_v1 import IntervalJournal,recovery_charge,append
    from research.general_rsi_generation_v2 import generate
    from research.byte_input_model_v6 import build_codebook
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
    config=dict(plan['slots'][slot])
    raw=[]
    for asset in plan['codebook_assets']:
        assert sha(asset['path'])==asset['sha256'];raw.append(Path(asset['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    context=Context(config['arm'],config['lr'],config['budget_ns'],plan['ema_half_life_ns'],config['roles'],book,binding)
    latest=None;checkpoint_seconds=0
    def checkpoint():
        nonlocal latest,checkpoint_seconds
        tick=time.perf_counter()
        receipt=write_checkpoint(weights/f'restart-a{attempt:02d}-s{context.step:06d}.pt',context.capture())
        payload=read_checkpoint(receipt)
        assert payload['step']==context.step and payload['clock']==context.clock.state_dict();del payload
        prior=latest;latest=dict(receipt=receipt,attempt=attempt,step=context.step,spent_ns=context.clock.spent_ns)
        replace(out/'latest.json',latest)
        if prior:retire(prior['receipt'],out/'deletion-journal.jsonl',latest)
        checkpoint_seconds+=time.perf_counter()-tick
    if (out/'latest.json').exists():
        latest=json.loads((out/'latest.json').read_text());assert 'receipt' in latest
        context.restore(read_checkpoint(latest['receipt']))
        # The previous attempt must own the last checkpoint; more complex gaps
        # require an explicit audited repair, never guessed elapsed time.
        assert latest['attempt']==attempt-1
        prior=attempt-1;exit_record=json.loads((out/f'supervisor-{prior:02d}-exit.json').read_text())
        assert exit_record['boot']==boot() and exit_record['binding']==binding
        path=out/f'intervals-{prior:02d}.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        charge=recovery_charge(rows,context.step,context.clock.spent_ns,exit_ns=exit_record['exit_ns'])
        put(out/f'recovery-{attempt:02d}.json',dict(charge=charge,source_sha256=sha(path) if path.exists() else None,supervisor_exit=exit_record,restored=latest))
        if charge['duration_ns']:context.charge_lost_work(charge['duration_ns'])
        checkpoint()
    else:
        assert attempt==1;checkpoint()
    last_checkpoint=context.clock.spent_ns
    # Diagnostics are prospective fixed time fractions, never model selection.
    completed_probes={int(p.stem.split('-')[-1]) for p in out.glob('probe-*.json')}
    while not context.clock.done:
        if stopped():
            if latest['step']!=context.step:checkpoint()
            return dict(status='stopped',step=context.step)
        journal=IntervalJournal(out/f'intervals-{attempt:02d}.jsonl',attempt,context.step+1,context.clock.spent_ns)
        row=context.update(now_ns=journal.tick)
        row=dict(row,mps_driver_bytes=torch.mps.driver_allocated_memory())
        assert row['mps_driver_bytes']<plan['mps_limit_bytes']
        append(out/f'steps-{attempt:02d}.jsonl',row)
        if context.step%128==0:
            progress=dict(pid=os.getpid(),slot=slot,step=context.step,spent_ns=context.clock.spent_ns,budget_ns=config['budget_ns'],loss=row['loss'])
            replace(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if context.clock.spent_ns-last_checkpoint>=plan['checkpoint_interval_ns']:
            checkpoint();last_checkpoint=context.clock.spent_ns
        for fraction in (1,2):
            if fraction not in completed_probes and context.clock.spent_ns>=config['budget_ns']*fraction/3:
                put(out/f'probe-{fraction}.json',score(context.model,context.data,plan['probe_documents']))
                completed_probes.add(fraction)
    if latest['step']!=context.step:checkpoint()
    assert context.ema and context.snapshots>0
    final=[]
    for endpoint in ('raw','time_ema'):
        if stopped():return dict(status='stopped',step=context.step)
        if endpoint=='time_ema':
            with torch.no_grad():
                for n,p in context.model.ordered_parameters():p.copy_(context.ema[n])
        receipt=write_checkpoint(weights/f'{endpoint}.pt',dict(kind='general_rsi_equal_time_inference_v1',binding=binding,slot=config,
            model_identity=context.model.identity,endpoint=endpoint,completed_steps=context.step,clock=context.clock.state_dict(),
            parameters=clone_tree(dict(context.model.ordered_parameters())),immutable_buffers=context.model._own_expected_buffer_hashes),cap=128*1024**2)
        tick=time.perf_counter();scored=score(context.model,context.data);score_seconds=time.perf_counter()-tick
        put(out/f'{endpoint}-validation.json',scored)
        generation_seconds=0
        if endpoint=='time_ema' and config['phase']=='confirmation':
            tick=time.perf_counter();generate(context.model,plan['generation_protocol'],out/'generation',receipt,binding);generation_seconds=time.perf_counter()-tick
        final.append(dict(endpoint=endpoint,weights=receipt,ce=scored['ce'],count=scored['count'],sources=scored['sources'],score_seconds=score_seconds,generation_seconds=generation_seconds))
    result=dict(status='completed',binding=binding,slot=config,steps=context.step,training_ns=context.clock.spent_ns,
        overshoot_ns=context.clock.spent_ns-config['budget_ns'],snapshots=context.snapshots,recovery_charges=context.recovery_charges,
        checkpoint_seconds_this_attempt=checkpoint_seconds,endpoints=final,test_opened=False)
    put(out/'result.json',result)
    for e in final:assert sha(e['weights']['path'])==e['weights']['sha256']
    retire(latest['receipt'],out/'deletion-journal.jsonl',dict(result=str(out/'result.json'),sha256=sha(out/'result.json')))
    replace(out/'latest.json',dict(status='retired_after_completion',result=str(out/'result.json')))
    return result


def remaining_guard_seconds(plan, slot, prior_slot_seconds, prior_tranche_seconds, attempt):
    assert 1 <= attempt <= plan['max_attempts_per_slot']
    remaining=min(plan['slots'][slot]['timeout_seconds']-prior_slot_seconds,
                  plan['tranche_seconds']-prior_tranche_seconds)
    assert remaining > 0, 'Admitted cumulative time budget exhausted'
    return remaining


def run(args):
    from research.general_rsi_training_v2 import sha
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(PANEL/'plan.json')==args.binding
    plan=json.loads((PANEL/'plan.json').read_text());assert 0<=args.slot<6
    out=PANEL/f'slot-{args.slot:02d}';weights=RUNS/f'slot-{args.slot:02d}'
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not stopped() and not (out/'result.json').exists()
        for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
        for i in range(args.slot):
            previous=PANEL/f'slot-{i:02d}'
            review=json.loads((previous/'review.json').read_text())
            assert review['accepted_execution'] and review['result_sha256']==sha(previous/'result.json')
        completed=sum(p.stat().st_size for p in RUNS.glob('slot-*/*.pt') if p.name in ('raw.pt','time_ema.pt')) if RUNS.exists() else 0
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-completed)
        spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in PANEL.glob('slot-*/guard-*.json'))
        slot_spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in out.glob('guard-*.json'))
        remaining=remaining_guard_seconds(plan,args.slot,slot_spent,spent,args.attempt)
        out.mkdir(exist_ok=True);weights.mkdir(parents=True,exist_ok=True)
        put(out/f'attempt-{args.attempt:02d}.json',dict(pid=os.getpid(),binding=args.binding,started=time.time(),boot=boot()))
        statepath=ROOT/'research_state/general-rsi-state.json';state=json.loads(statepath.read_text())
        state.update(status='ACTIVE_SCIENTIFIC_WORKER',phase='native_horizon_confirmation',active_worker_pid=os.getpid(),active_slot=args.slot,native_horizon_panel_launched=True)
        replace(statepath,state)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            with CpuPrerequisiteGuard(out/f'guard-{args.attempt:02d}.json',ROOT,timeout_seconds=remaining,rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
                result=execute(plan,args.slot,args.binding,args.attempt,out,weights)
            for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
            state=json.loads(statepath.read_text());state.update(status='ACTIVE_SLOT_COMPLETED' if result['status']=='completed' else 'STOPPED',active_worker_pid=None,next_action='Review native horizon endpoint, clock and checkpoint evidence, then continue frozen six-slot order. Preserve all failures and first generations.')
            replace(statepath,state)
        except BaseException:
            put(out/f'failure-{args.attempt:02d}.json',dict(traceback=traceback.format_exc()));raise


def supervise(args):
    from research.general_rsi_training_v2 import sha
    assert sha(PANEL/'plan.json')==args.binding and not stopped()
    out=PANEL/f'slot-{args.slot:02d}';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=1+len(list(out.glob('supervisor-*-start.json')))
    plan=json.loads((PANEL/'plan.json').read_text())
    remaining_guard_seconds(plan,args.slot,
        sum(json.loads(p.read_text())['elapsed_seconds'] for p in out.glob('guard-*.json')),
        sum(json.loads(p.read_text())['elapsed_seconds'] for p in PANEL.glob('slot-*/guard-*.json')),attempt)
    boot_id=boot();put(out/f'supervisor-{attempt:02d}-start.json',dict(binding=args.binding,boot=boot_id,started_ns=time.perf_counter_ns()))
    with (out/f'worker-{attempt:02d}.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-m','research.general_rsi_native_horizon_worker_v1','run',args.binding,str(args.slot),'--attempt',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        def forward(sig,frame):child.send_signal(sig)
        signal.signal(signal.SIGTERM,forward);signal.signal(signal.SIGINT,forward)
        code=child.wait();exit_ns=time.perf_counter_ns()
    put(out/f'supervisor-{attempt:02d}-exit.json',dict(binding=args.binding,boot=boot_id,pid=child.pid,returncode=code,exit_ns=exit_ns))
    if code:
        p=ROOT/'research_state/general-rsi-state.json';s=json.loads(p.read_text())
        s.update(status='ACTIVE_RECOVERY_REQUIRED',active_worker_pid=None,next_action='Inspect preserved native horizon failure and supervisor exit evidence; audit any uncheckpointed clock charge before a separately identified recovery. No seed replacement or free replay.')
        replace(p,s)
    print(json.dumps(dict(slot=args.slot,attempt=attempt,pid=child.pid,returncode=code)),flush=True)
    raise SystemExit(code)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','supervise']);p.add_argument('binding');p.add_argument('slot',type=int);p.add_argument('--attempt',type=int);args=p.parse_args()
    if args.action=='run':run(args)
    else:supervise(args)
if __name__=='__main__':main()

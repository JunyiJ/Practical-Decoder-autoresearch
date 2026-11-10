"""Bounded real-MPS time-budget integration and deterministic restart check."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/general-rsi-time-qualification-001'
WEIGHTS = ROOT/'research_runs/general-rsi-time-qualification-001'


def put(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def prepare():
    from research.general_rsi_training_v2 import sha
    prior = json.loads((ROOT/'research/tokenizer-evidence/general-rsi-exposure-001/plan.json').read_text())
    sources = dict(prior['sources'])
    for name in ('research/general_rsi_equal_time_training_v1.py',
                 'research/general_rsi_training_clock_v1.py',
                 'research/general_rsi_equal_time_qualification_v1.py',
                 'research/general_rsi_qualification_v1.py'):
        sources[name] = sha(ROOT/name)
    plan = dict(kind='equal_time_engineering_qualification_v1', sources=sources,
        codebook_assets=prior['codebook_assets'], torch_version=prior['torch_version'],
        roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'), range(62110600001,62110600005))),
        roles_audit='No existing research JSON/JSONL matches before assignment',
        arms=['native','raw_bytes'], lr=.0025, budget_ns=12_000_000_000, half_life_ns=1_000_000_000,
        restart_when='First averaged update whose end is below budget, or fail if none',
        exact_requirement='Complete final tensor/scalar state identical when replaying original measured clock durations after full restore; actual timer is measured on uninterrupted branch.',
        lost_work_requirement='Explicit positive charge advances budget and survives capture/restore; synthetic ns charge is not model-quality evidence.',
        timeout_seconds=600, rss_limit_bytes=4*1024**3, mps_limit_bytes=6*1024**3,
        reserve_bytes=10*1024**3, retained_cap_bytes=1024**3, metadata_cap_bytes=16*1024**2,
        scientific_selection=False, test_opened=False,
        scope='New time-based schedule/averaging/restart only. No repeat of qualified scoring oracle or old scientific fixtures.',
        disposition='Engineering restart payloads retained pending successful review and exact audited dependency closure.')
    assert not OUT.exists() and not WEIGHTS.exists()
    OUT.mkdir(); put(OUT/'plan.json', plan)
    print(sha(OUT/'plan.json'), flush=True)


def run(plan, binding):
    import torch
    from research.general_rsi_equal_time_training_v1 import Context, write_checkpoint, read_checkpoint, sha
    from research.general_rsi_qualification_v1 import equal
    from research.byte_input_model_v6 import build_codebook
    assert torch.__version__ == plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    torch.set_num_threads(4)
    raw=[]
    for asset in plan['codebook_assets']:
        assert sha(asset['path']) == asset['sha256']; raw.append(Path(asset['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    reports=[]
    for arm in plan['arms']:
        def new():
            return Context(arm,plan['lr'],plan['budget_ns'],plan['half_life_ns'],plan['roles'],book,binding)
        context=new(); receipt=None; checkpoint_step=None
        wall_start=time.perf_counter_ns()
        io_ns=0
        while not context.clock.done:
            assert not json.loads((ROOT/'research_state/general-rsi-state.json').read_text()).get('stop_requested',False)
            context.update()
            assert torch.mps.driver_allocated_memory() < plan['mps_limit_bytes']
            if context.snapshots and not context.clock.done and receipt is None:
                tick=time.perf_counter_ns(); before=context.clock.spent_ns
                receipt=write_checkpoint(WEIGHTS/f'{arm}-restart.pt',context.capture())
                checkpoint_step=context.step
                assert context.clock.spent_ns == before
                io_ns += time.perf_counter_ns()-tick
        assert receipt is not None and context.snapshots >= 2
        expected=context.capture()
        elapsed=time.perf_counter_ns()-wall_start
        last=expected['history'][-1]
        assert last['begin_ns'] < plan['budget_ns'] <= last['end_ns']
        assert 0 <= last['end_ns']-plan['budget_ns'] < last['end_ns']-last['begin_ns']
        try: context.update()
        except RuntimeError: pass
        else: raise AssertionError('budget allowed extra update')
        del context; gc.collect(); torch.mps.empty_cache()
        restored=new(); saved=read_checkpoint(receipt)
        tick=time.perf_counter_ns(); restored.restore(saved); restore_ns=time.perf_counter_ns()-tick
        equal(restored.capture(),saved); del saved
        while not restored.clock.done:
            row=expected['history'][restored.step]
            ticks=iter((0,row['end_ns']-row['begin_ns']))
            restored.update(now_ns=lambda: next(ticks))
        equal(restored.capture(),expected)
        # Audited lost-compute accounting persists even if it exhausts budget.
        restored.charge_lost_work(1_234_567)
        charged=restored.capture()
        assert charged['clock']['spent_ns'] == expected['clock']['spent_ns']+1_234_567
        del restored; gc.collect(); torch.mps.empty_cache()
        checked=new(); checked.restore(charged); equal(checked.capture(),charged)
        report=dict(arm=arm, steps=len(expected['history']), snapshots=expected['snapshots'],
            training_ns=expected['clock']['spent_ns'], overshoot_ns=last['end_ns']-plan['budget_ns'],
            actual_branch_wall_ns=elapsed, checkpoint_io_ns=io_ns, restore_ns=restore_ns,
            checkpoint_step=checkpoint_step, checkpoint=receipt, exact_restore=True,
            exact_replayed_complete_state=True, lost_work_charge_roundtrip=True,
            history=expected['history'], scientific_selection=False)
        put(OUT/f'{arm}-result.json',report); reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
        del checked,charged,expected;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    a,b=[r['history'] for r in reports]
    assert [r['batch_sha256'] for r in a[:min(len(a),len(b))]] == [r['batch_sha256'] for r in b[:min(len(a),len(b))]]
    return dict(qualified=True, arms=[r['arm'] for r in reports], paired_batch_prefix=True,
        scientific_selection=False,test_opened=False,
        limitation='Short timer/restart engineering runs. Production recovery must audit and charge any uncheckpointed compute; variable real clocks do not imply numerical replay equivalence.')


def main():
    from research.general_rsi_training_v2 import sha
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('binding',nargs='?');args=p.parse_args()
    if args.action=='prepare':return prepare()
    assert sha(OUT/'plan.json')==args.binding
    plan=json.loads((OUT/'plan.json').read_text())
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free >= plan['reserve_bytes']+plan['retained_cap_bytes']+plan['metadata_cap_bytes']
        for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
        assert not json.loads((ROOT/'research_state/general-rsi-state.json').read_text()).get('stop_requested',False)
        WEIGHTS.mkdir()
        put(OUT/'attempt.json',dict(pid=os.getpid(),binding=args.binding,started=time.time(),platform=platform.platform(),
            thermal=subprocess.run(['/usr/bin/pmset','-g','therm'],capture_output=True,text=True).stdout))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=plan['timeout_seconds'],rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
                result=run(plan,args.binding)
            for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
            assert sum(p.stat().st_size for p in WEIGHTS.iterdir())<=plan['retained_cap_bytes']
            assert sum(p.stat().st_size for p in OUT.iterdir())<=plan['metadata_cap_bytes']
            put(OUT/'result.json',result);print('QUALIFIED',flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()

"""Bounded qualification of new objective mixture, parent loading and restart state."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
from research.general_rsi_repetition_probability_v1 import sha,read,put
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-integration-001'
WEIGHTS=ROOT/'research_runs/general-rsi-pseudo-integration-001'
STOP=False

def stop(*_):
    global STOP
    STOP=True


def prepare():
    prior=read(ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001/plan.json')
    sources=dict(prior['sources'])
    for name in ['research/general_rsi_pseudo_spans_v1.py','research/general_rsi_pseudo_training_v1.py',
                 'research/general_rsi_pseudo_integration_check_v1.py','research/general_rsi_qualification_v1.py']:
        sources[name]=sha(ROOT/name)
    parent=read(ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001/slot-01/generation/result.json')['checkpoint']
    parent.update(slot=1,binding=sha(ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001/plan.json'))
    assert not OUT.exists() and not WEIGHTS.exists();OUT.mkdir(parents=True)
    plan=dict(sources=sources,parent=parent,torch_version=prior['torch_version'],arms=['native_ce','native_pseudo_repetition'],
        roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(62111205180001,62111205180005))),
        pseudo_seed=62111205180005,budget_ns=24_000_000_000,half_life_ns=1_000_000_000,
        checkpoint_when='First averaged odd-numbered update ending before budget; candidate next update is pseudo.',
        scope='New parent warm start,LR.00025,alternating objective,loss sampler and full restart only. No quality selection, scorer or generation prerequisite rerun.',
        component_check=sha(ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/component-attempt-01/result.json'),
        inventory_sha256=sha(ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/inventory.json'),
        total_seconds=900,reserve_bytes=10*2**30,retained_cap_bytes=2*2**30,metadata_cap_bytes=64*2**20,
        rss_limit_bytes=4*2**30,mps_limit_bytes=6*2**30,free_bytes=shutil.disk_usage(ROOT).free,test_opened=False)
    assert plan['free_bytes']>=plan['reserve_bytes']+plan['retained_cap_bytes']+plan['metadata_cap_bytes']
    put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'))


def run():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(OUT/'plan.json');binding=sha(OUT/'plan.json')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['retained_cap_bytes']+plan['metadata_cap_bytes']
        WEIGHTS.mkdir()
        with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=900,rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
            try:
                import torch
                from research.general_rsi_pseudo_training_v1 import Context
                from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint
                from research.general_rsi_qualification_v1 import equal
                for f,h in plan['sources'].items():assert sha(ROOT/f)==h,f
                assert sha(ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/inventory.json')==plan['inventory_sha256']
                assert sha(ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/component-attempt-01/result.json')==plan['component_check']
                assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
                def check():
                    assert not STOP and not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested',False),'human_stop'
                    assert torch.mps.driver_allocated_memory()<plan['mps_limit_bytes']
                reports=[]
                for arm in plan['arms']:
                    def new():return Context(arm,plan['budget_ns'],plan['half_life_ns'],plan['roles'],plan['pseudo_seed'],plan['parent'],binding)
                    check();context=new();receipt=None
                    while not context.clock.done:
                        check();context.update()
                        if context.snapshots and context.step%2==1 and not context.clock.done and receipt is None:
                            receipt=write_checkpoint(WEIGHTS/f'{arm}-restart.pt',context.capture())
                    assert receipt and context.snapshots>=2
                    expected=context.capture();del context;gc.collect();torch.mps.empty_cache()
                    context=new();saved=read_checkpoint(receipt);context.restore(saved);equal(context.capture(),saved);del saved
                    while not context.clock.done:
                        check();row=expected['history'][context.step];ticks=iter((0,row['end_ns']-row['begin_ns']))
                        context.update(now_ns=lambda:next(ticks))
                    equal(context.capture(),expected)
                    report=dict(arm=arm,checkpoint=receipt,steps=context.step,objective_counts=expected['objective_counts'],snapshots=context.snapshots,
                        training_ns=context.clock.spent_ns,exact_full_restore_and_replay=True,history=expected['history'],
                        peak_driver_bytes=torch.mps.driver_allocated_memory(),saturated_targets=sum(r['saturated_targets'] for r in expected['history']))
                    put(OUT/f'{arm}-result.json',report);reports.append(report)
                    print(arm,'passed',context.step,'updates',report['objective_counts'],flush=True)
                    del context,expected;gc.collect();torch.mps.empty_cache()
                streams=[[r['batch_sha256'] for r in x['history'] if r['objective']=='ce'] for x in reports]
                n=min(map(len,streams));assert streams[0][:n]==streams[1][:n]
                assert sum(p.stat().st_size for p in WEIGHTS.iterdir())<plan['retained_cap_bytes']
                put(OUT/'result.json',dict(qualified=True,plan_sha256=binding,paired_natural_batch_prefix=n,
                    complete_restart_exact=True,test_opened=False,quality_result=False))
            except BaseException as e:
                put(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),plan_sha256=binding));raise


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
